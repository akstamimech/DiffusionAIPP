"""
Standalone high-res image of the resolution_block_size(z) piecewise equation,
for direct use on a slide. No LaTeX install needed - uses matplotlib's
built-in "cm" (Computer Modern) mathtext fontset for a LaTeX-like look, and
stacked Unicode brace glyphs (U+23A7/23A8/23A9) for the piecewise brace since
mathtext itself has no multi-row \\begin{cases} support.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATH = SCRIPT_DIR / "resolution_block_equation.png"

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
})

FONT_SIZE = 34
BRACE_SIZE = 60
INK = "#0b0b0b"

rows = [
    ("1", r"$z \leq 20$"),
    ("2", r"$20 < z \leq 30$"),
    ("4", r"$z > 30$"),
]
brace_chars = ["⎧", "⎨", "⎩"]  # ⎧ ⎨ ⎩

fig, ax = plt.subplots(figsize=(7.5, 2.6))
ax.axis("off")

y_positions = [0.78, 0.5, 0.22]

ax.text(0.0, 0.5, r"$\mathrm{block}(z) =$", fontsize=FONT_SIZE, color=INK,
        ha="left", va="center", transform=ax.transAxes)

for y, ch in zip(y_positions, brace_chars):
    ax.text(0.40, y, ch, fontsize=BRACE_SIZE, color=INK,
            ha="left", va="center", transform=ax.transAxes, family="DejaVu Sans")

for y, (value, condition) in zip(y_positions, rows):
    ax.text(0.50, y, f"${value}$", fontsize=FONT_SIZE, color=INK,
            ha="left", va="center", transform=ax.transAxes)
    ax.text(0.62, y, condition, fontsize=FONT_SIZE, color=INK,
            ha="left", va="center", transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

fig.savefig(OUT_PATH, dpi=500, transparent=True, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print(f"Wrote {OUT_PATH}")
