"""Teaser: MOPD (trained student) vs CSPD (soft prompts on a frozen student) + in/out-of-domain scores."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'figures'
ORANGE = '#FF6A1F'; HATCH = '#9DB7D9'; GREY = '#666666'; LGREY = '#BBBBBB'
MONO = 'DejaVu Sans Mono'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'hatch.linewidth': 0.6})

EX = np.array([1.0, -0.33]); EZ = np.array([0.72, 0.36]); EY = np.array([0.0, 1.0])
ANG = np.degrees(np.arctan2(EX[1], EX[0]))
def P(x, y, z): return EX * x + EY * y + EZ * z

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.set_aspect('equal'); ax.axis('off')

def poly(pts, fc='white', ec='black', lw=1.1, hatch=None, z=2):
    if hatch:
        ax.add_patch(Polygon(pts, closed=True, facecolor='white', edgecolor=HATCH, hatch=hatch, lw=0, zorder=z))
        ax.add_patch(Polygon(pts, closed=True, facecolor='none', edgecolor=ec, lw=lw, zorder=z + .1))
    else:
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, lw=lw, zorder=z))

def box(x0, y0, z0, dx, dy, dz, ec='black', hatch=None, lw=1.1, z=2):
    a = [P(x0, y0, z0), P(x0 + dx, y0, z0), P(x0 + dx, y0 + dy, z0), P(x0, y0 + dy, z0)]            # front (ex, ey)
    b = [P(x0 + dx, y0, z0), P(x0 + dx, y0, z0 + dz), P(x0 + dx, y0 + dy, z0 + dz), P(x0 + dx, y0 + dy, z0)]  # side (ez, ey)
    t = [P(x0, y0 + dy, z0), P(x0 + dx, y0 + dy, z0), P(x0 + dx, y0 + dy, z0 + dz), P(x0, y0 + dy, z0 + dz)]  # top
    for f in (a, b, t): poly(f, ec=ec, hatch=hatch, lw=lw, z=z)

def plane(x0, x1, z0, z1, z=0):
    poly([P(x0, 0, z0), P(x1, 0, z0), P(x1, 0, z1), P(x0, 0, z1)], ec=LGREY, lw=.8, z=z)

def label(xy, s, size=10, rot=ANG, color=GREY, **kw):
    ax.text(*xy, s, rotation=rot, rotation_mode='anchor', fontsize=size, family=MONO, color=color, **kw)

# ---- MOPD: trained student
plane(-0.25, 1.95, -0.25, 1.95)
box(0, 0, 0, 1.6, 1.6, 1.6, hatch='////')
ax.text(*(P(0, 0, 0) + [-2.3, 4.0]), 'MOPD', fontsize=15, fontweight='bold', va='center')
label(P(-0.1, 0, -0.25) + [0, -0.33], 'TRAINED STUDENT', size=10.5)
label(P(0, 1.5, 0) + [-1.75, 0.55], 'EXPERTS', size=9.5, rot=0)
for i, s in enumerate('ISCM'):
    p0 = P(0, 1.5, 0) + [-1.25, +0.15 - 0.62 * i]
    ax.text(*p0, s, fontsize=13, family=MONO, va='center', ha='right')
    p1 = P(0, 1.3 - 0.33 * i, 0)
    xs = np.linspace(p0[0] + 0.1, p1[0], 50); t = (xs - xs[0]) / (xs[-1] - xs[0])
    ax.plot(xs, p0[1] + (p1[1] - p0[1]) * (3 * t ** 2 - 2 * t ** 3), color=GREY, lw=.8, zorder=1)

# ---- CSPD: soft prompts + frozen student
X0 = 4.4
plane(X0 - 0.3, 9.9, -0.2, 1.8)
slots = [X0 + 1.05 * i for i in range(4)]
hts = [1.55, 0.75, 1.3, 0.55]
for xs_, h, s in zip(slots, hts, 'ISCM'):
    poly([P(xs_, 0, .35), P(xs_ + .12, 0, .35), P(xs_ + .12, 0, 1.25), P(xs_, 0, 1.25)], ec=LGREY, lw=.8, z=1)
    box(xs_, h, .35, .12, 1.55, .9, ec=ORANGE, lw=1.4, z=3)
    c = P(xs_ + .12, h + .78, .8)
    ax.text(*c, s, color=ORANGE, fontsize=13, fontweight='bold', family=MONO, ha='center', va='center', zorder=4)
def arrow(p, up, text):
    a, b = (p, p + [0, 0.55]) if up else (p + [0, 0.55], p)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, color=ORANGE, lw=1.2, zorder=5))
    ax.text(p[0] + 0.25, p[1] + 0.3, text, fontsize=9.5, family=MONO, color=GREY, va='center')
arrow(P(slots[0] + .06, hts[0] + 1.55, .35) + [0.18, 0.12], False, 'ADD')
arrow(P(slots[2] + .06, hts[2] + 1.55, .35) + [0.18, 0.12], True, 'REMOVE')
box(X0 + 4.6, 0, 0, 1.6, 1.6, 1.6)
label(P(X0 + .1, 0, -.2) + [0, -0.33], 'SOFT PROMPTS', size=10.5)
label(P(X0 + 4.5, 0, -.2) + [0, -0.33], 'FROZEN STUDENT', size=10.5)
ax.text(P(X0, 0, 0)[0] - 0.4, P(0, 0, 0)[1] + 4.0, 'CSPD (ours)', fontsize=15, fontweight='bold', va='center')

# ---- Scores
BX, BY = 14.6, 3.0
ax.text(BX, P(0, 0, 0)[1] + 4.0, 'Scores', fontsize=15, fontweight='bold', va='center')
def bars(y, title, rows, vmin, vmax, ntask):
    ax.text(BX, y, title, fontsize=9.5, family=MONO, color=GREY, va='center')
    ax.text(BX + 7.9, y, ntask, fontsize=9.5, family=MONO, color='#999999', va='center', ha='right')
    for i, (name, v, style) in enumerate(rows):
        yy = y - 0.62 * (i + 1)
        ax.text(BX, yy, name, fontsize=9.5, family=MONO, color=GREY, va='center')
        w = 3.0 * v / 67.79
        x0 = BX + 3.9
        if style == 'hatch':
            ax.add_patch(Polygon([[x0, yy - .2], [x0 + w, yy - .2], [x0 + w, yy + .2], [x0, yy + .2]], facecolor='white', edgecolor=HATCH, hatch='////', lw=0, zorder=2))
            ax.add_patch(Polygon([[x0, yy - .2], [x0 + w, yy - .2], [x0 + w, yy + .2], [x0, yy + .2]], facecolor='none', edgecolor='black', lw=1, zorder=3))
        else:
            ax.add_patch(Polygon([[x0, yy - .2], [x0 + w, yy - .2], [x0 + w, yy + .2], [x0, yy + .2]], facecolor=ORANGE if style == 'ours' else 'white', edgecolor=ORANGE if style == 'ours' else 'black', lw=1, zorder=2))
        ax.text(x0 + w + 0.15, yy, f'{v}', fontsize=12, va='center')
bars(BY, 'IN-DOMAIN MEAN', [('STUDENT', 50.41, 'plain'), ('MOPD', 67.79, 'hatch'), ('CSPD (OURS)', 67.72, 'ours')], 50.41, 67.79, '4 TASKS')
ax.plot([BX, BX + 7.9], [BY - 2.3, BY - 2.3], color='#DDDDDD', lw=.8)
bars(BY - 2.85, 'OUT-OF-DOMAIN MEAN', [('STUDENT', 50.3, 'plain'), ('MOPD', 53.5, 'hatch'), ('CSPD (OURS)', 56.6, 'ours')], 50.3, 56.6, '6 TASKS')

ax.set_xlim(-2.6, 23.2); ax.set_ylim(-3.6, 5.3)
fig.savefig(OUT / 'teaser.pdf', bbox_inches='tight', pad_inches=0.05)
fig.savefig(OUT / 'teaser.png', dpi=200, bbox_inches='tight', pad_inches=0.05)
