"""
Renders word-level Shapley-value attributions as colour-highlighted text
(one panel per model condition)

Usage
-----
from shapley_text_plot import render_shapley_comparison, parse_shapley_block

RAW_FULL = '''
    word  position  shapley_value  variance  n_updates
  number         5         0.0917    0.0469        256
security         4         0.0512    0.0312        256
  ...
'''  # paste straight from the R console, headers/separator lines are ignored; these are from log.txt results files 

conditions = [
    {"title": "Full fine-tuning", "hct": 0, "words": parse_shapley_block(RAW_FULL)},
    {"title": "Rank 64",          "hct": 1, "words": parse_shapley_block(RAW_R64)},
    {"title": "Rank 1",           "hct": 1, "words": parse_shapley_block(RAW_R1)},
]

render_shapley_comparison(conditions, "out.png")

Disclosure: This script was partially created by AI in order to render images to match those in main text 
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

plt.rcParams["font.family"] = "Liberation Serif"

# Fixed to match dissertation style
WORD_FONT  = 14.25   # word font size (pt) -- midpoint chosen across earlier drafts
LABEL_FONT = 7.75    # value-label font size (pt)
PAD_IN = 0.11    # inches of padding either side of each word inside its box
GAP_IN = 0.14    # inches of gap between boxes
LEFT_IN = 0.14    # inches of left margin before the first box
RIGHT_IN = 0.18    # inches of right margin after the last box
DPI  = 300
FONT_FAMILY = "Liberation Serif"   # metric-compatible with Times New Roman

CMAP = mcolors.LinearSegmentedColormap.from_list(
    "seagreen_red4", ["#8FBC8F", "#FFFFFF", "#8B0000"] 
)

_ROW_RE = re.compile(
    r"^\s*(\S+)\s+(-?\d+)\s+(-?\d+\.\d+)\s+-?\d+\.\d+\s+\d+\s*$"
)


def parse_shapley_block(text):
    """
    Parse a block of pasted R console output with columns
    word / position / shapley_value / variance / n_updates
    into a list of (word, position, shapley_value) tuples.

    Header rows, '---' separator lines, and '[n/100] Index ...' lines are
    ignored automatically -- only lines matching the 5-column data format
    are kept, so you can paste the raw console output unedited.
    """
    rows = []
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if m:
            word, pos, val = m.group(1), int(m.group(2)), float(m.group(3))
            rows.append((word, pos, val))
    if not rows:
        raise ValueError("No data rows matched -- check the pasted block format.")
    return rows


def _measure_row_width_in(words):
    """Inches needed to lay out one row of words at WORD_FONT, so the figure
    can be sized to exactly fit its own content (all panels share the same
    prompt, so one measurement covers the whole figure)."""
    scratch = plt.figure(dpi=DPI)
    renderer = scratch.canvas.get_renderer()
    total = LEFT_IN
    for i, word in enumerate(words):
        t = scratch.text(0, 0, word, fontsize=WORD_FONT, family=FONT_FAMILY)
        scratch.canvas.draw()
        bbox = t.get_window_extent(renderer=renderer)
        total += bbox.width / DPI + 2 * PAD_IN
        if i < len(words) - 1:
            total += GAP_IN
    plt.close(scratch)
    return total + RIGHT_IN


def render_shapley_comparison(conditions, out_path):
    """
    conditions: list of dicts, each with
        "title": str                                    e.g. "Full fine-tuning"
        "hct":   0 or 1
        "words": list of (word, position, shapley_value) tuples
                 -- typically produced by parse_shapley_block()

    All conditions must be for the same prompt (same words/positions).
    Saves a PNG to out_path, sized to its own content, and returns out_path.
    """
    words_only = [w for w, _, _ in conditions[0]["words"]]
    fig_width = _measure_row_width_in(words_only)

    all_vals = [v for c in conditions for _, _, v in c["words"]]
    vmax = max(abs(v) for v in all_vals)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    n = len(conditions)
    fig = plt.figure(figsize=(fig_width, 2.55 * n + 0.9))
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[1] * n + [0.35], hspace=0.55,
                           left=0.0, right=1.0, top=1.0, bottom=0.0)

    pad = PAD_IN / fig_width
    gap = GAP_IN / fig_width
    x_start = LEFT_IN / fig_width

    for i, cond in enumerate(conditions):
        title = f'{cond["title"]} (HCT = {cond["hct"]})'
        rows_sorted = sorted(cond["words"], key=lambda r: r[1])
        words = [r[0] for r in rows_sorted]
        vals = [r[2] for r in rows_sorted]

        ax = fig.add_subplot(gs[i, 0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.text(0.5, 1.0, title, fontsize=12, fontweight="bold", color="#1a1a1a",
                 ha="center", va="bottom", family=FONT_FAMILY)
        ax.plot([0.0, 1.0], [0.88, 0.88], color="#1a1a1a", linewidth=0.8,
                 transform=ax.transAxes, clip_on=False)

        renderer = fig.canvas.get_renderer()
        x = x_start
        y = 0.55
        centers = []
        for word, val in zip(words, vals):
            color = CMAP(norm(val))
            box_x0 = x
            txt = ax.text(box_x0 + pad, y, word, fontsize=WORD_FONT, color="#1a1a1a",
                           ha="left", va="center", family=FONT_FAMILY)
            fig.canvas.draw()
            bbox = txt.get_window_extent(renderer=renderer)
            inv = ax.transData.inverted()
            (x0, y0), (x1, y1) = inv.transform(bbox)
            box_width = (x1 - x0) + 2 * pad
            rect = plt.Rectangle((box_x0, y0 - 0.06), box_width, (y1 - y0) + 0.12,
                                  facecolor=color, edgecolor="none", zorder=0)
            ax.add_patch(rect)
            txt.set_zorder(2)
            centers.append(box_x0 + box_width / 2)
            x = box_x0 + box_width + gap

        for center, val in zip(centers, vals):
            ax.text(center, y - 0.30, f"{val:+.3f}", fontsize=LABEL_FONT, color="#52514e",
                     ha="center", va="center", family=FONT_FAMILY)

    cax = fig.add_subplot(gs[n, 0])
    cax.set_position([0.15, cax.get_position().y0, 0.7, 0.03])
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Shapley value (word contribution to harmful classification)",
                  fontsize=9, color="#52514e", family=FONT_FAMILY)
    cb.ax.tick_params(labelsize=8, colors="#52514e")
    for label in cb.ax.get_xticklabels():
        label.set_family(FONT_FAMILY)
    cb.outline.set_visible(False)

    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
