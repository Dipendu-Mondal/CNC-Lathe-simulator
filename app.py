
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re
import math

st.set_page_config(page_title="CNC Lathe G-Code Simulator", page_icon="⚙️", layout="wide")

st.title("⚙️ CNC Lathe G-Code Simulator")
st.caption("2D FANUC-style CNC turning toolpath simulator with corrected G02/G03 arc interpolation")

DEFAULT_GCODE = """%
O1001
G21 G90
G97 S1000 M03
G00 X40 Z5
G01 Z0 F0.2
G01 X30
G01 Z-20
G01 X20 Z-30
G03 X10 Z-40 R10
G01 Z-50
G02 X20 Z-60 R10
G01 X30
G00 X45
G00 Z5
M05
M30
%"""

# ============================================================
# Helpers
# ============================================================

def strip_comments(line):
    line = re.sub(r"\([^)]*\)", "", line)
    line = line.split(";")[0]
    return line.strip().upper()

def parse_words(line):
    """
    Parses words such as:
    G01 X20 Z-30 R10 F0.2
    """
    return [(m.group(1), float(m.group(2))) for m in
            re.finditer(r"([A-Z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", line)]

def arc_center_from_radius(x0, z0, x1, z1, radius, cw):
    """
    Find the two possible circle centers for an X-Z plane arc.

    CNC turning convention:
      X = diameter coordinate
      Z = axial coordinate

    Internally the geometry is converted to X-radius coordinates:
      r = X / 2

    For R arcs there are two possible circles. The sign of R selects
    the minor/major solution according to the common CNC convention:
      positive R -> minor arc
      negative R -> major arc

    cw=True corresponds to G02.
    """
    r0 = x0 / 2.0
    r1 = x1 / 2.0

    dx = r1 - r0
    dz = z1 - z0
    chord = math.hypot(dx, dz)

    R = abs(radius)

    if chord < 1e-12:
        raise ValueError("Arc start and end points are identical.")

    if chord > 2 * R + 1e-9:
        raise ValueError(
            f"Arc radius R{radius:g} is too small for the specified endpoints."
        )

    mx = (r0 + r1) / 2.0
    mz = (z0 + z1) / 2.0

    h = math.sqrt(max(R * R - (chord / 2.0) ** 2, 0.0))

    # Unit normal to chord
    nx = -dz / chord
    nz = dx / chord

    c1 = (mx + h * nx, mz + h * nz)
    c2 = (mx - h * nx, mz - h * nz)

    candidates = [c1, c2]

    def sweep_angle(center):
        cx, cz = center
        a0 = math.atan2(r0 - cx, z0 - cz)
        a1 = math.atan2(r1 - cx, z1 - cz)

        # Angle measured in the X-radius / Z plane.
        # For G02 choose clockwise; G03 choose counterclockwise.
        if cw:
            delta = (a0 - a1) % (2 * math.pi)
        else:
            delta = (a1 - a0) % (2 * math.pi)

        return delta

    sweeps = [sweep_angle(c) for c in candidates]

    # Positive R means minor arc, negative R means major arc.
    want_major = radius < 0

    if want_major:
        idx = int(np.argmax(sweeps))
    else:
        idx = int(np.argmin(sweeps))

    return candidates[idx], sweeps[idx]

def sample_arc(b, points=80):
    """
    Generate a mathematically correct circular G02/G03 path.

    Supports:
      G02/G03 X Z R

    X values are interpreted as FANUC diameter coordinates.
    """
    x0, z0 = b["x0"], b["z0"]
    x1, z1 = b["x1"], b["z1"]

    if "R" not in b["words"]:
        raise ValueError(
            "This simulator currently requires R for G02/G03 arcs."
        )

    radius = b["words"]["R"]
    cw = b["motion"] == "G02"

    center, sweep = arc_center_from_radius(
        x0, z0, x1, z1, radius, cw
    )

    cx, cz = center

    r0 = x0 / 2.0

    a0 = math.atan2(r0 - cx, z0 - cz)

    if cw:
        angles = np.linspace(a0, a0 - sweep, points)
    else:
        angles = np.linspace(a0, a0 + sweep, points)

    radial = math.hypot(r0 - cx, z0 - cz)

    radius_path = cx + radial * np.sin(angles)
    z_path = cz + radial * np.cos(angles)

    x_path = 2.0 * radius_path

    # Force exact endpoints to remove floating-point endpoint errors.
    x_path[0] = x0
    z_path[0] = z0
    x_path[-1] = x1
    z_path[-1] = z1

    return x_path, z_path, center, sweep

def parse_gcode(text):
    blocks = []

    x = 0.0
    z = 0.0

    modal_motion = "G00"
    absolute = True
    units = "mm"
    feed = 0.2
    spindle = 0.0

    source_line = 0

    for raw in text.splitlines():
        source_line += 1

        line = strip_comments(raw)

        if not line or line in ("%",):
            continue

        words_list = parse_words(line)
        words = {}

        for letter, value in words_list:
            words[letter] = value

        # Modal settings
        for letter, value in words_list:
            if letter == "G":
                g = int(value)

                if g in (0, 1, 2, 3):
                    modal_motion = f"G{g:02d}"

                elif g == 20:
                    units = "inch"

                elif g == 21:
                    units = "mm"

                elif g == 90:
                    absolute = True

                elif g == 91:
                    absolute = False

            elif letter == "F":
                feed = value

            elif letter == "S":
                spindle = value

        nx, nz = x, z

        if "X" in words:
            if absolute:
                nx = words["X"]
            else:
                nx = x + words["X"]

        if "Z" in words:
            if absolute:
                nz = words["Z"]
            else:
                nz = z + words["Z"]

        # Store only movement blocks.
        if modal_motion in ("G00", "G01", "G02", "G03"):
            if "X" in words or "Z" in words:

                block = {
                    "source_line": source_line,
                    "raw": raw.strip(),
                    "motion": modal_motion,
                    "x0": x,
                    "z0": z,
                    "x1": nx,
                    "z1": nz,
                    "feed": feed,
                    "spindle": spindle,
                    "words": words.copy(),
                    "units": units,
                    "absolute": absolute,
                }

                # Validate arc immediately.
                if modal_motion in ("G02", "G03"):
                    if "R" not in words:
                        block["error"] = "G02/G03 requires R in this version."
                    else:
                        try:
                            xs, zs, center, sweep = sample_arc(block)
                            block["arc_x"] = xs
                            block["arc_z"] = zs
                            block["center"] = center
                            block["sweep_deg"] = math.degrees(sweep)
                        except Exception as e:
                            block["error"] = str(e)

                blocks.append(block)

        x, z = nx, nz

    return blocks

def build_segments(blocks):
    segments = []

    for b in blocks:

        if b["motion"] in ("G00", "G01"):
            xs = np.array([b["x0"], b["x1"]], dtype=float)
            zs = np.array([b["z0"], b["z1"]], dtype=float)

        elif b["motion"] in ("G02", "G03"):
            if "error" in b:
                xs = np.array([b["x0"], b["x1"]], dtype=float)
                zs = np.array([b["z0"], b["z1"]], dtype=float)
            else:
                xs = b["arc_x"]
                zs = b["arc_z"]

        segments.append((b, xs, zs))

    return segments

def calculate_toolpath_length(segments):
    total = 0.0

    for b, xs, zs in segments:
        for i in range(len(xs) - 1):
            dx_radius = (xs[i + 1] - xs[i]) / 2.0
            dz = zs[i + 1] - zs[i]
            total += math.hypot(dx_radius, dz)

    return total

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("Simulation Settings")

    stock_diameter = st.number_input(
        "Stock diameter (mm)",
        min_value=1.0,
        value=50.0,
        step=1.0
    )

    stock_length = st.number_input(
        "Stock length (mm)",
        min_value=1.0,
        value=70.0,
        step=1.0
    )

    show_rapid = st.checkbox("Show G00 rapid moves", True)

    show_arc_centers = st.checkbox(
        "Show G02/G03 arc centers",
        False
    )

    show_labels = st.checkbox(
        "Show block labels",
        True
    )

    st.markdown("---")

    st.markdown("### Supported")

    st.write("""
    **Motion**
    - G00
    - G01
    - G02
    - G03

    **Coordinates**
    - X
    - Z
    - R

    **Modal**
    - G20 / G21
    - G90 / G91
    - F
    - S
    """)

    st.warning(
        "For G02/G03, use X, Z and R. "
        "X is interpreted as diameter."
    )

# ============================================================
# G-code editor
# ============================================================

gcode = st.text_area(
    "CNC Turning G-Code",
    DEFAULT_GCODE,
    height=330
)

simulate = st.button(
    "▶ Simulate G-Code",
    type="primary",
    use_container_width=True
)

blocks = parse_gcode(gcode)

if not blocks:
    st.error("No X/Z movement blocks were found.")
    st.stop()

segments = build_segments(blocks)

# ============================================================
# Errors
# ============================================================

arc_errors = [
    b for b in blocks
    if b["motion"] in ("G02", "G03") and "error" in b
]

if arc_errors:

    st.error("G02/G03 error detected")

    for b in arc_errors:
        st.write(
            f"Line {b['source_line']}: "
            f"`{b['raw']}` → {b['error']}"
        )

# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "Toolpath",
    "Machined Profile",
    "G-Code Blocks"
])

# ============================================================
# Toolpath
# ============================================================

with tab1:

    fig, ax = plt.subplots(figsize=(13, 6))

    stock_r = stock_diameter / 2.0

    # Stock boundaries in radius coordinates.
    ax.plot(
        [-stock_length, 0],
        [stock_r, stock_r],
        linewidth=2,
        label="Stock"
    )

    ax.plot(
        [-stock_length, 0],
        [-stock_r, -stock_r],
        linewidth=2
    )

    # Center line
    ax.axhline(0, linewidth=0.8)

    for b, xs, zs in segments:

        if b["motion"] == "G00" and not show_rapid:
            continue

        if b["motion"] == "G00":
            linestyle = "--"
            linewidth = 1.5
        else:
            linestyle = "-"
            linewidth = 2.5

        # Positive and negative radius sides.
        ax.plot(
            zs,
            xs / 2.0,
            linestyle=linestyle,
            linewidth=linewidth
        )

        if b["motion"] != "G00":
            ax.plot(
                zs,
                -xs / 2.0,
                linestyle=linestyle,
                linewidth=1
            )

        if show_labels:
            ax.annotate(
                b["motion"],
                (zs[-1], xs[-1] / 2.0),
                fontsize=8
            )

        if show_arc_centers and b["motion"] in ("G02", "G03") and "center" in b:

            cx, cz = b["center"]

            ax.plot(
                cz,
                cx,
                marker="x",
                markersize=8,
                markeredgewidth=2
            )

            ax.annotate(
                f"{b['motion']} center",
                (cz, cx),
                fontsize=8
            )

    ax.set_xlabel("Z axis (mm)")
    ax.set_ylabel("Radius (mm)")
    ax.set_title("CNC Lathe X-Z Toolpath")
    ax.grid(True, alpha=0.25)

    ax.set_aspect("equal", adjustable="box")

    st.pyplot(fig, clear_figure=True)

    cutting = sum(
        b["motion"] in ("G01", "G02", "G03")
        for b in blocks
    )

    rapid = sum(
        b["motion"] == "G00"
        for b in blocks
    )

    arcs = sum(
        b["motion"] in ("G02", "G03")
        for b in blocks
    )

    length = calculate_toolpath_length(segments)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Motion Blocks", len(blocks))
    c2.metric("Cutting Blocks", cutting)
    c3.metric("Arc Blocks", arcs)
    c4.metric("Path Length", f"{length:.2f} mm")

# ============================================================
# Machined profile
# ============================================================

with tab2:

    # A simple swept-tool profile visualization.
    z_samples = np.linspace(
        -stock_length,
        5,
        900
    )

    profile = np.full_like(
        z_samples,
        stock_r,
        dtype=float
    )

    for b, xs, zs in segments:

        if b["motion"] == "G00":
            continue

        # Ignore invalid arcs.
        if b["motion"] in ("G02", "G03") and "error" in b:
            continue

        # Convert path into radius.
        r_path = xs / 2.0

        z_low = min(zs)
        z_high = max(zs)

        mask = (
            (z_samples >= z_low) &
            (z_samples <= z_high)
        )

        if not np.any(mask):
            continue

        order = np.argsort(zs)

        z_sorted = zs[order]
        r_sorted = r_path[order]

        unique_z, unique_idx = np.unique(
            z_sorted,
            return_index=True
        )

        r_sorted = r_sorted[unique_idx]

        if len(unique_z) >= 2:

            local_r = np.interp(
                z_samples[mask],
                unique_z,
                r_sorted
            )

            profile[mask] = np.minimum(
                profile[mask],
                local_r
            )

    fig2, ax2 = plt.subplots(figsize=(13, 6))

    ax2.plot(
        z_samples,
        profile,
        linewidth=2.5
    )

    ax2.plot(
        z_samples,
        -profile,
        linewidth=2.5
    )

    ax2.axhline(0, linewidth=0.8)

    ax2.set_xlabel("Z axis (mm)")
    ax2.set_ylabel("Radius (mm)")
    ax2.set_title("Approximate Machined Workpiece Profile")
    ax2.grid(True, alpha=0.25)
    ax2.set_aspect("equal", adjustable="box")

    st.pyplot(fig2, clear_figure=True)

# ============================================================
# G-code blocks
# ============================================================

with tab3:

    table = []

    for i, b in enumerate(blocks, 1):

        arc_info = ""

        if b["motion"] in ("G02", "G03"):

            if "error" in b:
                arc_info = b["error"]

            else:
                arc_info = (
                    f"R={b['words'].get('R', '')}, "
                    f"sweep={b['sweep_deg']:.2f}°"
                )

        table.append({
            "Block": i,
            "Source line": b["source_line"],
            "Motion": b["motion"],
            "X Start": round(b["x0"], 3),
            "Z Start": round(b["z0"], 3),
            "X End": round(b["x1"], 3),
            "Z End": round(b["z1"], 3),
            "Feed": b["feed"],
            "Spindle": b["spindle"],
            "Arc": arc_info,
            "Original G-Code": b["raw"]
        })

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# G02/G03 explanation
# ============================================================

st.markdown("---")

st.subheader("G02 / G03 interpretation")

st.write("""
The simulator treats X as a **diameter coordinate** and Z as the axial
coordinate. Therefore an X20 position corresponds to a 10 mm radius.

For an arc such as:

`G03 X10 Z-40 R10`

the program calculates the actual circle center from the start point,
end point and radius, then generates the circular interpolation in the
X-Z plane. G02 and G03 use opposite directions.
""")

st.info(
    "This is an educational simulator. It is not a certified CNC "
    "controller or machine verification system. Controller-specific "
    "behavior can differ."
)
