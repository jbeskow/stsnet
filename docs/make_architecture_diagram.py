"""Generate architecture diagram for STS-Net."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

# ── Colour palette ────────────────────────────────────────────────────────────
C_INPUT   = "#dce8f7"
C_ENC     = "#c8e6c9"
C_FUSION  = "#fff9c4"
C_LSTM    = "#ffe0b2"
C_HEAD    = "#f3e5f5"
C_EDGE    = "#444444"
C_TEXT    = "#1a1a1a"

def box(ax, x, y, w, h, label, sublabel=None, color="#ffffff", fontsize=9, radius=0.12):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.04,rounding_size={radius}",
                          linewidth=0.8, edgecolor=C_EDGE, facecolor=color, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.07, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=C_TEXT, zorder=4)
        ax.text(x, y - 0.22, sublabel, ha="center", va="center", fontsize=7,
                color="#555555", zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=C_TEXT, zorder=4)

def arrow(ax, x0, y0, x1, y1, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE,
                                lw=0.9, mutation_scale=12),
                zorder=2)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.08, my, label, fontsize=7, color="#666666", va="center")

# ── Layout constants ─────────────────────────────────────────────────────────
STREAMS = [
    ("dom",    "dominant\n21 joints × 3",   1.5),
    ("nondom", "non-dominant\n21 joints × 3", 4.0),
    ("body",   "body\n12 joints × 3",         6.5),
    ("face",   "face\n25 joints × 3",          9.0),
]
HEADS = [
    ("state",        "4",  1.2),
    ("shape",        "42", 2.5),
    ("att",          "34", 3.8),
    ("hand_type",    "2",  5.1),
    ("contact_loc",  "22", 6.4),
    ("contact_type", "4",  7.7),
    ("motion",       "7",  9.0),
    ("nondom_shape", "42", 10.3),
    ("nondom_att",   "34", 11.6),
]

Y_INPUT  = 7.0
Y_ENC    = 5.5
Y_FUSION = 4.0
Y_LSTM   = 2.8
Y_HEADS  = 1.2

# ── Input boxes ──────────────────────────────────────────────────────────────
for name, sublbl, x in STREAMS:
    box(ax, x, Y_INPUT, 2.0, 0.7, name, sublbl, color=C_INPUT, fontsize=9)

# ── FrameEncoder boxes ───────────────────────────────────────────────────────
for name, _, x in STREAMS:
    box(ax, x, Y_ENC, 2.0, 0.7, "FrameEncoder",
        f"linear → conv×3\n(k=5, d={512})", color=C_ENC, fontsize=8)
    arrow(ax, x, Y_INPUT - 0.35, x, Y_ENC + 0.35)

# ── Fusion box ───────────────────────────────────────────────────────────────
fx = (STREAMS[0][2] + STREAMS[-1][2]) / 2
box(ax, fx, Y_FUSION, 9.5, 0.65, "Fusion",
    "concat → Linear(4D→D) → LayerNorm → ReLU → Dropout",
    color=C_FUSION, fontsize=9)

# Arrows from encoders to fusion
for _, _, x in STREAMS:
    arrow(ax, x, Y_ENC - 0.35, fx, Y_FUSION + 0.325)

# ── BiLSTM box ───────────────────────────────────────────────────────────────
box(ax, fx, Y_LSTM, 4.0, 0.65, "BiLSTM  (optional)",
    "1 layer × (D/2 → D), bidirectional", color=C_LSTM, fontsize=9)
arrow(ax, fx, Y_FUSION - 0.325, fx, Y_LSTM + 0.325)

# ── Head boxes ───────────────────────────────────────────────────────────────
for name, n_cls, x in HEADS:
    box(ax, x, Y_HEADS, 1.15, 0.62, name, f"{n_cls} classes",
        color=C_HEAD, fontsize=7.5)
    arrow(ax, fx, Y_LSTM - 0.325, x, Y_HEADS + 0.31)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,  edgecolor=C_EDGE, label="Pose input streams"),
    mpatches.Patch(facecolor=C_ENC,    edgecolor=C_EDGE, label="FrameEncoder  (shared weights per stream)"),
    mpatches.Patch(facecolor=C_FUSION, edgecolor=C_EDGE, label="Fusion layer"),
    mpatches.Patch(facecolor=C_LSTM,   edgecolor=C_EDGE, label="BiLSTM  (omitted for alignment bootstrap)"),
    mpatches.Patch(facecolor=C_HEAD,   edgecolor=C_EDGE, label="Classification heads  (CE loss)"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=7.5,
          framealpha=0.85, edgecolor="#aaaaaa", ncol=1)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(7.0, 7.75, "STS-Net Architecture  (hidden_dim = 512, 18.6 M parameters)",
        ha="center", va="center", fontsize=11, fontweight="bold", color=C_TEXT)

plt.tight_layout(pad=0.2)
out = "docs/architecture.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
