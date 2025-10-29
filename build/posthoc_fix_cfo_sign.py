#!/usr/bin/env python3
import sys, csv, numpy as np
from pathlib import Path

inp = Path(sys.argv[1] if len(sys.argv)>1 else "features.csv")
rows = list(csv.DictReader(open(inp, newline="")))
hdr  = rows and list(rows[0].keys())

def f(v):
    try: return float(v)
    except: return np.nan

# 1) Align signs per-row to a robust anchor (median of available CFOs)
def align_row(r):
    cq  = f(r.get("cfo_quick_hz"))
    cc  = f(r.get("cfo_centroid_hz"))
    c2  = f(r.get("cfo_two_stage_hz"))
    c2c = f(r.get("cfo_two_stage_coarse_hz"))
    vals = [x for x in (c2c, c2, cc, cq) if np.isfinite(x)]
    if not vals:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    anchor = np.nanmedian(vals)
    sgn = 1.0 if anchor >= 0 else -1.0
    def al(x): 
        if not np.isfinite(x): return np.nan
        return x if np.sign(x)==sgn or x==0 else -x
    cq2, cc2, c22, c2c2 = al(cq), al(cc), al(c2), al(c2c)
    cons = np.nanmedian([v for v in (c2c2, c22, cc2, cq2) if np.isfinite(v)])
    return cq2, cc2, c22, c2c2, cons

# First pass: sign-align
aligned = []
for r in rows:
    cq2, cc2, c22, c2c2, cons = align_row(r)
    aligned.append((cq2, cc2, c22, c2c2, cons))

# 2) Remove a *global median offset* between two-stage and coarse
#    This addresses your "handoff" bias: median(two_stage - coarse) should be ~0.
diffs = [ (a[2]-a[3]) for a in aligned if np.isfinite(a[2]) and np.isfinite(a[3]) ]
offset = float(np.nanmedian(diffs)) if diffs else 0.0  # ~ +1.57 kHz in your log

# 3) Overwrite originals with fixed values (and keep a consensus column)
out = []
for r, (cq2, cc2, c22, c2c2, cons) in zip(rows, aligned):
    r2 = dict(r)
    if np.isfinite(cq2):  r2["cfo_quick_hz"]      = f"{cq2:.6f}"
    if np.isfinite(cc2):  r2["cfo_centroid_hz"]   = f"{cc2:.6f}"
    if np.isfinite(c2c2): r2["cfo_two_stage_coarse_hz"] = f"{c2c2:.6f}"
    if np.isfinite(c22):  r2["cfo_two_stage_hz"]  = f"{(c22 - offset):.6f}"
    if np.isfinite(cons): r2["cfo_consensus_hz"]  = f"{cons - offset:.6f}"
    out.append(r2)

# Write to *_fixed_overwrite.csv
out_path = inp.with_name(inp.stem + "_fixed_overwrite.csv")
with open(out_path, "w", newline="") as f:
    fieldnames = list(out[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(out)

print(f"[OK] wrote {out_path}")
print(f"[info] global median offset removed from two-stage: {offset:+.2f} Hz")