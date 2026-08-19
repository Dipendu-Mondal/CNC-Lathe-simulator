
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import re

st.set_page_config(page_title="CNC Lathe G-Code Simulator", page_icon="⚙️", layout="wide")

st.title("⚙️ CNC Lathe G-Code Simulator")
st.caption("Educational 2D simulator for common FANUC-style turning G-code")

DEFAULT_GCODE = """%
O1001
G21
G90
G97 S1000 M03
G00 X40 Z5
G01 Z0 F0.2
G01 X30
G01 Z-20
G01 X20 Z-30
G01 Z-40
G00 X45
G00 Z5
M05
M30
%"""

# ----------------------------
# G-code parser
# ----------------------------
def strip_comments(line):
    line = re.sub(r"\([^)]*\)", "", line)
    line = line.split(";")[0]
    return line.strip()

def parse_gcode(text):
    blocks = []
    modal = {"motion": "G00", "feed": 0.2, "spindle": 1000}
    x = 0.0
    z = 0.0

    for raw in text.splitlines():
        line = strip_comments(raw.upper())
        if not line:
            continue

        words = {}
        for letter, value in re.findall(r"([A-Z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", line):
            try:
                words[letter] = float(value)
            except ValueError:
                pass

        if "F" in words:
            modal["feed"] = words["F"]
        if "S" in words:
            modal["spindle"] = words["S"]

        motion = modal["motion"]
        for g in re.findall(r"G\s*(\d+)", line):
            gcode = int(g)
            if gcode in (0, 1, 2, 3):
                motion = f"G{gcode:02d}"
            elif gcode == 90:
                modal["absolute"] = True
            elif gcode == 91:
                modal["absolute"] = False
            elif gcode == 21:
                modal["units"] = "mm"
            elif gcode == 20:
                modal["units"] = "inch"

        modal["motion"] = motion

        nx, nz = x, z
        if "X" in words:
            nx = words["X"] if modal.get("absolute", True) else x + words["X"]
        if "Z" in words:
            nz = words["Z"] if modal.get("absolute", True) else z + words["Z"]

        # Store a movement only when X/Z are present.
        if ("X" in words or "Z" in words) and motion in ("G00", "G01", "G02", "G03"):
            blocks.append({
                "raw": raw.strip(),
                "motion": motion,
                "x0": x, "z0": z,
                "x1": nx, "z1": nz,
                "feed": modal["feed"],
                "spindle": modal["spindle"],
                "line": len(blocks) + 1,
                "words": words
            })

        x, z = nx, nz

    return blocks

# ----------------------------
# Toolpath generation
# ----------------------------
def interpolate_arc(b, n=40):
    x0, z0, x1, z1 = b["x0"], b["z0"], b["x1"], b["z1"]
    words = b["words"]

    # Radius mode: simple circular interpolation.
    if "R" not in words:
        return np.array([x0, x1]), np.array([z0, z1])

    r = abs(words["R"])
    dx, dz = x1-x0, z1-z0
    chord = np.hypot(dx, dz)
    if chord == 0 or chord > 2*r:
        return np.array([x0, x1]), np.array([z0, z1])

    mx, mz = (x0+x1)/2, (z0+z1)/2
    h = np.sqrt(max(r*r - chord*chord/4, 0))
    # Work in X-Z plane.
    px, pz = -dz/chord, dx/chord
    sign = 1 if b["motion"] == "G02" else -1
    cx, cz = mx + sign*h*px, mz + sign*h*pz

    a0 = np.arctan2(z0-cz, x0-cx)
    a1 = np.arctan2(z1-cz, x1-cx)

    if b["motion"] == "G02":
        while a1 >= a0:
            a1 -= 2*np.pi
    else:
        while a1 <= a0:
            a1 += 2*np.pi

    ang = np.linspace(a0, a1, n)
    return cx + r*np.cos(ang), cz + r*np.sin(ang)

def make_toolpath(blocks):
    segments = []
    for b in blocks:
        if b["motion"] in ("G02", "G03"):
            xs, zs = interpolate_arc(b)
        else:
            xs = np.array([b["x0"], b["x1"]])
            zs = np.array([b["z0"], b["z1"]])
        segments.append((b, xs, zs))
    return segments

# ----------------------------
# Simple stock-removal model
# ----------------------------
def simulate_profile(blocks, stock_radius, zmin, zmax, resolution=500):
    zgrid = np.linspace(zmin, zmax, resolution)
    radius = np.full(resolution, stock_radius, dtype=float)

    # Approximate turning as a swept tool-center profile.
    # For each cutting move, the final tool X position becomes the local surface.
    for b in blocks:
        if b["motion"] not in ("G01", "G02", "G03"):
            continue
        # Only treat inward/turning moves as cutting.
        z0, z1 = b["z0"], b["z1"]
        lo, hi = min(z0, z1), max(z0, z1)
        mask = (zgrid >= lo) & (zgrid <= hi)
        if not np.any(mask):
            continue

        if b["motion"] == "G01":
            if abs(z1-z0) > 1e-9 and abs(b["x1"]-b["x0"]) > 1e-9:
                # Linear interpolation of tool radius along Z.
                rr = b["x0"]/2 + (zgrid[mask]-z0)/(z1-z0)*(b["x1"]-b["x0"])/2
                radius[mask] = np.minimum(radius[mask], rr)
            else:
                radius[mask] = np.minimum(radius[mask], min(b["x0"], b["x1"])/2)
        else:
            # Arc: use sampled path and interpolate tool radius.
            xs, zs = interpolate_arc(b, 80)
            order = np.argsort(zs)
            za = zs[order]
            ra = xs[order]/2
            valid = (zgrid[mask] >= za.min()) & (zgrid[mask] <= za.max())
            if np.any(valid):
                vals = np.interp(zgrid[mask][valid], za, ra)
                idx = np.where(mask)[0][valid]
                radius[idx] = np.minimum(radius[idx], vals)

    return zgrid, np.maximum(radius, 0)

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("Simulation Settings")
    stock_diameter = st.number_input("Stock diameter (mm)", min_value=1.0, value=50.0, step=1.0)
    stock_length = st.number_input("Stock length (mm)", min_value=1.0, value=60.0, step=1.0)
    show_rapid = st.checkbox("Show rapid moves (G00)", True)
    show_labels = st.checkbox("Show move labels", False)
    resolution = st.slider("Profile resolution", 200, 1200, 600, 100)

    st.markdown("---")
    st.markdown("**Supported basics**")
    st.write("G00, G01, G02, G03, G20/G21, G90/G91, X, Z, F, S, M03/M05")
    st.info("This is an educational 2D simulator. It is not a machine controller and does not verify every FANUC cycle.")

gcode = st.text_area("Paste CNC turning G-code", DEFAULT_GCODE, height=300)

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("▶ Simulate", type="primary", use_container_width=True)
with col2:
    clear = st.button("Reset Example", use_container_width=True)

if clear:
    gcode = DEFAULT_GCODE

blocks = parse_gcode(gcode)

if not blocks:
    st.warning("No supported X/Z motion blocks were found.")
    st.stop()

segments = make_toolpath(blocks)

# Determine display limits
all_z = [stock_length/2, -stock_length/2]
for b, xs, zs in segments:
    all_z.extend(zs.tolist())

z_lo = min(all_z)
z_hi = max(all_z)
# Keep the stock in a useful range around Z=0.
zmin = min(-stock_length, z_lo) - 5
zmax = max(5, z_hi) + 5

tabs = st.tabs(["Toolpath", "Machined Profile", "G-code Blocks"])

with tabs[0]:
    fig, ax = plt.subplots(figsize=(12, 5))

    # Stock
    stock_r = stock_diameter/2
    ax.add_patch(Rectangle((-stock_length, -stock_r), stock_length, stock_diameter,
                           fill=False, linewidth=2, label="Stock"))

    for b, xs, zs in segments:
        if b["motion"] == "G00" and not show_rapid:
            continue
        linestyle = "--" if b["motion"] == "G00" else "-"
        ax.plot(zs, xs/2, linestyle=linestyle, linewidth=2)
        if b["motion"] != "G00":
            ax.plot(zs, -xs/2, linestyle=linestyle, linewidth=1)

        if show_labels:
            ax.annotate(b["motion"], (zs[-1], xs[-1]/2), fontsize=8)

    # Centerline
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Z axis (mm)")
    ax.set_ylabel("Radius (mm)")
    ax.set_title("CNC Lathe Toolpath")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    st.pyplot(fig, clear_figure=True)

    rapid = sum(b["motion"] == "G00" for b in blocks)
    cutting = len(blocks) - rapid
    c1, c2, c3 = st.columns(3)
    c1.metric("Motion blocks", len(blocks))
    c2.metric("Cutting moves", cutting)
    c3.metric("Rapid moves", rapid)

with tabs[1]:
    zgrid, profile = simulate_profile(
        blocks, stock_r, zmin, zmax, resolution=resolution
    )

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(zgrid, profile, linewidth=2, label="Machined radius")
    ax2.plot(zgrid, -profile, linewidth=2)
    ax2.axhline(0, linewidth=0.8)
    ax2.set_xlabel("Z axis (mm)")
    ax2.set_ylabel("Radius (mm)")
    ax2.set_title("Approximate Machined Workpiece Profile")
    ax2.grid(True, alpha=0.25)
    ax2.set_aspect("equal", adjustable="box")
    st.pyplot(fig2, clear_figure=True)

    st.caption("The profile is an approximate swept-tool calculation, intended for learning and G-code visualization.")

with tabs[2]:
    rows = []
    for i, b in enumerate(blocks, 1):
        rows.append({
            "Block": i,
            "Motion": b["motion"],
            "X start": round(b["x0"], 3),
            "Z start": round(b["z0"], 3),
            "X end": round(b["x1"], 3),
            "Z end": round(b["z1"], 3),
            "Feed": b["feed"],
            "Spindle": b["spindle"],
            "Original": b["raw"]
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Important")
st.write(
    "For real CNC use, verify the program separately on the target controller. "
    "This simulator does not model machine limits, tool nose radius compensation, "
    "work offsets, canned cycles such as G71/G72, threading, collision detection, "
    "or controller-specific macro behavior."
)
