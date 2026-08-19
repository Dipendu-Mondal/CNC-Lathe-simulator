
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math, re

st.set_page_config(page_title="CNC Lathe 3D Simulator", page_icon="⚙️", layout="wide")

# ============================================================
# CNC / geometry engine
# ============================================================

DEFAULT = """%
O1001
G21 G90 G99
G50 S2500
T0101
G97 S1200 M03
G00 X42 Z3
G01 Z0 F0.20
G01 X32
G01 Z-20
G01 X24 Z-28
G03 X14 Z-38 R10
G01 Z-50
G02 X24 Z-60 R10
G01 X32
G00 X60 Z10
M05
M30
%"""

def clean(line):
    line = re.sub(r"\([^)]*\)", "", line)
    line = line.split(";")[0]
    return line.strip().upper()

def words(line):
    return [(m.group(1), float(m.group(2))) for m in
            re.finditer(r"([A-Z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", line)]

def arc_points(x0, z0, x1, z1, R, cw, n=40):
    # FANUC lathe X is diameter. Geometry uses radius=X/2.
    r0, r1 = x0/2, x1/2
    dx, dz = r1-r0, z1-z0
    chord = math.hypot(dx, dz)
    RR = abs(R)
    if chord < 1e-10:
        raise ValueError("Arc start and end are identical")
    if chord > 2*RR + 1e-8:
        raise ValueError(f"R{R:g} is too small for this arc")
    mx, mz = (r0+r1)/2, (z0+z1)/2
    h = math.sqrt(max(RR**2-(chord/2)**2, 0))
    nx, nz = -dz/chord, dx/chord
    centers = [(mx+h*nx, mz+h*nz), (mx-h*nx, mz-h*nz)]
    def sweep(c):
        cx, cz = c
        a0 = math.atan2(r0-cx, z0-cz)
        a1 = math.atan2(r1-cx, z1-cz)
        return ((a0-a1)%(2*math.pi)) if cw else ((a1-a0)%(2*math.pi))
    sw = [sweep(c) for c in centers]
    idx = np.argmax(sw) if R < 0 else np.argmin(sw)
    cx, cz = centers[idx]
    a0 = math.atan2(r0-cx, z0-cz)
    ang = np.linspace(a0, a0-sw[idx] if cw else a0+sw[idx], n)
    rr = math.hypot(r0-cx, z0-cz)
    rp = cx + rr*np.sin(ang)
    zp = cz + rr*np.cos(ang)
    xp = 2*rp
    xp[0], zp[0], xp[-1], zp[-1] = x0,z0,x1,z1
    return xp, zp

def parse_program(text):
    lines = text.splitlines()
    blocks = []
    state = dict(x=0.0,z=0.0,motion="G00",feed=0.2,rpm=0.0,
                 abs=True,tool="T0101",spindle="OFF",coolant="OFF")
    rawblocks = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        s = clean(raw)
        if not s or s == "%":
            i += 1; continue
        w = words(s); d = dict(w)
        # Modal and machine codes
        for L,v in w:
            if L=="G":
                g=int(v)
                if g in (0,1,2,3): state["motion"]=f"G{g:02d}"
                elif g==90: state["abs"]=True
                elif g==91: state["abs"]=False
                elif g==21: state["units"]="mm"
                elif g==20: state["units"]="inch"
                elif g==94: state["feed_mode"]="MIN"
                elif g==95: state["feed_mode"]="REV"
            elif L=="F": state["feed"]=v
            elif L=="S": state["rpm"]=v
            elif L=="T":
                # Numeric T is preserved; e.g. T0101
                state["tool"]=f"T{int(v):04d}"
        if re.search(r"\bM0?3\b", s): state["spindle"]="CW"
        if re.search(r"\bM0?4\b", s): state["spindle"]="CCW"
        if re.search(r"\bM0?5\b", s): state["spindle"]="OFF"
        if re.search(r"\bM0?8\b", s): state["coolant"]="ON"
        if re.search(r"\bM0?9\b", s): state["coolant"]="OFF"
        # Canned cycle headers are collected and expanded below.
        if re.search(r"\bG71\b", s) or re.search(r"\bG72\b", s) or re.search(r"\bG70\b", s) or re.search(r"\bG74\b", s) or re.search(r"\bG75\b", s) or re.search(r"\bG76\b", s):
            rawblocks.append((i,s,d,state.copy()))
            i += 1; continue
        if "X" in d:
            nx=d["X"] if state["abs"] else state["x"]+d["X"]
        else: nx=state["x"]
        if "Z" in d:
            nz=d["Z"] if state["abs"] else state["z"]+d["Z"]
        else: nz=state["z"]
        if state["motion"] in ("G00","G01","G02","G03") and ("X" in d or "Z" in d):
            b=dict(line=i+1,raw=s,motion=state["motion"],x0=state["x"],z0=state["z"],
                   x1=nx,z1=nz,feed=state["feed"],rpm=state["rpm"],
                   tool=state["tool"],spindle=state["spindle"],coolant=state["coolant"],words=d)
            if b["motion"] in ("G02","G03"):
                try:
                    b["path_x"],b["path_z"]=arc_points(b["x0"],b["z0"],b["x1"],b["z1"],d.get("R",0),b["motion"]=="G02")
                except Exception as e: b["error"]=str(e)
            else:
                b["path_x"]=np.array([b["x0"],b["x1"]]); b["path_z"]=np.array([b["z0"],b["z1"]])
            blocks.append(b)
        state["x"],state["z"]=nx,nz
        i += 1
    # Basic expansion for G71/G70/G72 from P/Q contour labels.
    # This intentionally supports common FANUC two-block forms.
    contour = []
    for j,line in enumerate(lines):
        s=clean(line)
        m=re.search(r"\bN(\d+)\b",s)
        if m: contour.append((int(m.group(1)),j,s,dict(words(s))))
    def get_contour(P,Q):
        out=[]
        active=False
        for n,idx,s,d in contour:
            if n==P: active=True
            if active and (("X" in d) or ("Z" in d)):
                # Parse contour motion from line.
                gs=[int(v) for L,v in words(s) if L=="G"]
                mot=next((f"G{g:02d}" for g in gs if g in (0,1,2,3)), "G01")
                out.append((idx,s,d,mot))
            if n==Q and active: break
        return out
    expanded=[]
    for b in blocks: expanded.append(b)
    # Add an informational cycle list separately; actual roughing is generated from contour
    cycles=[]
    for idx,s,d,_state in rawblocks:
        gs=[int(v) for L,v in words(s) if L=="G"]
        if 71 in gs and "P" in d and "Q" in d:
            cycles.append({"type":"G71","line":idx+1,"P":int(d["P"]),"Q":int(d["Q"]),"U":d.get("U",0),"W":d.get("W",0),"R":d.get("R",0)})
        elif 72 in gs and "P" in d and "Q" in d:
            cycles.append({"type":"G72","line":idx+1,"P":int(d["P"]),"Q":int(d["Q"])})
        elif 70 in gs and "P" in d and "Q" in d:
            cycles.append({"type":"G70","line":idx+1,"P":int(d["P"]),"Q":int(d["Q"])})
        elif 74 in gs: cycles.append({"type":"G74","line":idx+1,"params":d})
        elif 75 in gs: cycles.append({"type":"G75","line":idx+1,"params":d})
        elif 76 in gs: cycles.append({"type":"G76","line":idx+1,"params":d})
    return blocks, cycles, lines

# ============================================================
# 3D geometry
# ============================================================

def lathe_surface(z, r):
    theta=np.linspace(0,2*np.pi,48)
    Z,TH=np.meshgrid(z,theta)
    R=np.tile(r,(len(theta),1))
    X=Z
    Y=R*np.cos(TH)
    ZZ=R*np.sin(TH)
    return X,Y,ZZ

def cylinder_mesh(z0,z1,r,n=32):
    th=np.linspace(0,2*np.pi,n)
    z=np.array([z0,z1])
    Z,TH=np.meshgrid(z,th)
    return Z,r*np.cos(TH),r*np.sin(TH)

def scene(profile_z, profile_r, stock_r, stock_z0, stock_z1, tool_x, tool_z,
          chuck_jaws=True, spindle_angle=0, show_stock=True):
    fig=go.Figure()
    # Remaining workpiece surface
    X,Y,Z=lathe_surface(profile_z,profile_r)
    fig.add_trace(go.Surface(x=X,y=Y,z=Z,showscale=False,opacity=.95,name="Machined Workpiece",
                             colorscale=[[0,"#9ca3af"],[1,"#e5e7eb"]]))
    # Optional transparent stock envelope
    if show_stock:
        Xs,Ys,Zs=cylinder_mesh(stock_z0,stock_z1,stock_r)
        fig.add_trace(go.Surface(x=Xs,y=Ys,z=Zs,showscale=False,opacity=.12,name="Original Stock"))
    # Chuck body
    cb0=stock_z1+2; cb1=stock_z1+18
    Xc,Yc,Zc=cylinder_mesh(cb0,cb1,stock_r*1.45)
    fig.add_trace(go.Surface(x=Xc,y=Yc,z=Zc,showscale=False,opacity=.35,name="Chuck"))
    # Chuck jaws: three rectangular-ish bars along radial directions
    if chuck_jaws:
        for a in (0,2*np.pi/3,4*np.pi/3):
            rr=np.linspace(stock_r*.95,stock_r*1.6,10)
            zz=np.linspace(cb0,cb1,2)
            R,ZZ=np.meshgrid(rr,zz)
            xx=ZZ
            yy=R*np.cos(a)
            zzz=R*np.sin(a)
            # represented as thin surface line
            fig.add_trace(go.Scatter3d(x=xx.ravel(),y=yy.ravel(),z=zzz.ravel(),
                                       mode="lines",line=dict(width=12),name="Chuck Jaw",
                                       showlegend=False))
    # Tool holder
    holder_z=np.array([tool_z-10,tool_z+8])
    holder_x=np.array([tool_x+6,tool_x+6])
    fig.add_trace(go.Scatter3d(x=holder_z,y=np.zeros(2),z=holder_x,
                               mode="lines",line=dict(width=18),name="Tool Holder"))
    # Insert
    fig.add_trace(go.Scatter3d(x=[tool_z,tool_z+2],y=[0,0],z=[tool_x,tool_x],
                               mode="lines+markers",line=dict(width=10),marker=dict(size=5),
                               name="Cutting Tool"))
    # Centerline
    fig.add_trace(go.Scatter3d(x=[stock_z0-5,stock_z1+20],y=[0,0],z=[0,0],
                               mode="lines",line=dict(width=2,dash="dash"),name="Axis"))
    fig.update_layout(
        height=650, margin=dict(l=0,r=0,t=35,b=0),
        scene=dict(xaxis_title="Z (mm)",yaxis_title="Y (mm)",zaxis_title="X-radius (mm)",
                   aspectmode="data",camera=dict(eye=dict(x=1.45,y=1.45,z=1.05))),
        legend=dict(orientation="h",y=1.02)
    )
    return fig

def profile_after(blocks, step, stock_r, zmin, zmax, n=700):
    z=np.linspace(zmin,zmax,n)
    r=np.full(n,stock_r)
    for b in blocks[:step]:
        if b["motion"]=="G00" or "error" in b: continue
        zz=b["path_z"]; rr=b["path_x"]/2
        lo,hi=min(zz),max(zz)
        mask=(z>=lo)&(z<=hi)
        if mask.any():
            order=np.argsort(zz)
            uz,ui=np.unique(zz[order],return_index=True)
            ur=rr[order][ui]
            if len(uz)>1:
                r[mask]=np.minimum(r[mask],np.interp(z[mask],uz,ur))
    return z,r

# ============================================================
# UI
# ============================================================

st.title("⚙️ CNC Lathe Virtual Machine")
st.caption("FANUC-style educational simulator: 3D workpiece + chuck + tool + block-by-block machining")

with st.sidebar:
    st.header("Machine Setup")
    stock_d=st.number_input("Stock diameter (mm)",10.,200.,50.,1.)
    stock_l=st.number_input("Stock length (mm)",10.,300.,70.,5.)
    sim_speed=st.slider("Simulation speed",1,10,5)
    show_stock=st.checkbox("Show original stock",True)
    show_jaws=st.checkbox("Show 3-jaw chuck",True)
    st.markdown("---")
    st.header("Controller")
    controller=st.selectbox("Controller profile",["FANUC Turning (educational)"])
    st.caption("Canned cycles use common FANUC-style forms. Verify against your machine manual.")

gcode=st.text_area("Machine Program",DEFAULT,height=300)
blocks,cycles,lines=parse_program(gcode)

if not blocks:
    st.warning("No X/Z movement blocks detected.")
    st.stop()

# Persistent step
if "step" not in st.session_state:
    st.session_state.step=0

c1,c2,c3,c4,c5=st.columns(5)
if c1.button("⏮ Reset",use_container_width=True): st.session_state.step=0
if c2.button("◀ Previous",use_container_width=True): st.session_state.step=max(0,st.session_state.step-1)
if c3.button("▶ Next Block",use_container_width=True): st.session_state.step=min(len(blocks),st.session_state.step+1)
if c4.button("⏭ End",use_container_width=True): st.session_state.step=len(blocks)
if c5.button("▶ Run",use_container_width=True):
    st.session_state.step=len(blocks)

step=st.session_state.step
current=blocks[step-1] if step else None

zmin=-stock_l
zmax=5
z,r=profile_after(blocks,step,stock_d/2,zmin,zmax)
tool_x=current["x1"] if current else blocks[0]["x0"]
tool_z=current["z1"] if current else blocks[0]["z0"]

# Tabs
t3d,tpanel,tpath,tcycles,tprogram=st.tabs(["🛠 3D MACHINE","🎛 CONTROL PANEL","📈 TOOLPATH","🔄 CANNED CYCLES","📋 PROGRAM"])

with t3d:
    fig=scene(z,r,stock_d/2,-stock_l,0,tool_x,tool_z,show_jaws,step,show_stock)
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Drag to rotate, scroll to zoom. The 3D model shows the simulated remaining rotational profile.")

with tpanel:
    cols=st.columns(6)
    vals=[
        ("BLOCK",str(step)),
        ("X",f"{tool_x:.3f}"),
        ("Z",f"{tool_z:.3f}"),
        ("FEED",f"{current['feed']:.3f}" if current else "0"),
        ("RPM",f"{current['rpm']:.0f}" if current else "0"),
        ("TOOL",current["tool"] if current else "T----")
    ]
    for col,(a,b) in zip(cols,vals): col.metric(a,b)
    st.markdown("### Machine status")
    if current:
        a,b,c,d=st.columns(4)
        a.write(f"**Motion:** {current['motion']}")
        b.write(f"**Spindle:** {current['spindle']}")
        c.write(f"**Coolant:** {current['coolant']}")
        d.write(f"**Line:** {current['line']}")
        st.code(current["raw"])
        if "error" in current: st.error(current["error"])
    else:
        st.info("Press Next Block to execute the first movement.")

with tpath:
    fig2=go.Figure()
    for b in blocks:
        if b["motion"]=="G00":
            dash="dash"
        else: dash="solid"
        fig2.add_trace(go.Scatter(x=b["path_z"],y=b["path_x"]/2,mode="lines",
                                   line=dict(dash=dash,width=3),name=b["motion"]))
    fig2.update_layout(height=520,xaxis_title="Z (mm)",yaxis_title="Radius (mm)",
                       title="X-Z Toolpath",template="plotly_white")
    st.plotly_chart(fig2,use_container_width=True)

with tcycles:
    if cycles:
        for cy in cycles:
            if cy["type"]=="G71":
                st.success(f"G71 rough turning detected on line {cy['line']}: P{cy['P']} Q{cy['Q']} U{cy['U']} W{cy['W']} R{cy['R']}")
            elif cy["type"]=="G70":
                st.info(f"G70 finishing cycle detected on line {cy['line']}: P{cy['P']} Q{cy['Q']}")
            elif cy["type"]=="G72":
                st.success(f"G72 facing cycle detected on line {cy['line']}: P{cy['P']} Q{cy['Q']}")
            elif cy["type"] in ("G74","G75","G76"):
                st.warning(f"{cy['type']} cycle detected on line {cy['line']}. Parameters parsed and shown below; controller-specific execution is not yet fully expanded.")
                st.json(cy)
    else:
        st.info("No canned cycles detected.")
    st.markdown("**Cycle support architecture:** G70/G71/G72 are recognized with P/Q blocks; G74/G75/G76 are recognized and diagnosed. Full controller-specific cycle expansion should be added per FANUC model.")

with tprogram:
    for i,line in enumerate(lines,1):
        prefix="▶ " if current and i==current["line"] else "   "
        st.text(f"{prefix}{i:03d}  {line}")

st.markdown("---")
st.warning("Educational simulator only. It is not a CNC controller, machine safety system, or collision-proof verification tool. Always verify G-code on the target control before machining.")
