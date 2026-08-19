# CNC Lathe G-Code Simulator

A lightweight educational Streamlit app for visualizing CNC turning G-code.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Supported basics

- G00 rapid
- G01 linear interpolation
- G02/G03 radius-based arcs
- G20/G21 units indication
- G90/G91 positioning
- X and Z coordinates
- F feed
- S spindle speed

## Current limitations

This is a 2D educational simulator. It does not fully emulate a FANUC controller and does not yet support advanced turning cycles such as G71/G72, threading, tool nose radius compensation, work offsets, macros, or full collision detection.
