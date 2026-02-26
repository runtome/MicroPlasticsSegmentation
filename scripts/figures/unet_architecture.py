#!/usr/bin/env python3
"""
Generate UNet Architecture Diagram for paper.

Layout (staircase U-shape):
  - Encoder: top-left → bottom-center (going right+down)
  - Bottleneck: bottom-center
  - Classification head: below bottleneck
  - Decoder: bottom-center → top-right (going right+up)
  - Output mask: top-right

Output: paper_figures/architecture/unet_architecture.png
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Output path ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
OUT_PATH = ROOT / "paper_figures" / "architecture" / "unet_architecture.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Color scheme ──────────────────────────────────────────────────────────────
C_FEAT   = "#5B9BD5"   # encoder/decoder feature blocks (blue)
C_FEAT2  = "#2E75B6"   # second/shadow feature block
C_EDGE   = "#1F4E79"   # block edge
C_INPUT  = "#D6E8F7"   # input block
C_SKIP   = "#A0A0A0"   # skip connection arrows (gray)
C_POOL   = "#C00000"   # max-pool arrows (red)
C_UP     = "#4CAF50"   # upsample arrows (green)
C_CONV   = "#1F3864"   # conv operation arrows (dark navy)
C_CLS    = "#7030A0"   # classification head (purple)
C_OUT    = "#00897B"   # output head (teal)
C_SEG    = "#C8E6C9"   # segmentation output block (light green)
C_TEXT   = "#1a1a1a"   # general text

FIG_W, FIG_H = 26, 16


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_feat(ax, cx, cy, w, h, fc=C_FEAT, ec=C_EDGE, alpha=0.88, zorder=3):
    """Draw a single feature-map rectangle centered at (cx, cy)."""
    r = mpatches.Rectangle(
        (cx - w / 2, cy - h / 2), w, h,
        facecolor=fc, edgecolor=ec, linewidth=0.9,
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(r)
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)  # l, b, r, t


def draw_double_block(ax, cx, cy, w, h, fc=C_FEAT, offset=0.18):
    """Draw two stacked-offset rectangles to represent a DoubleConv stage."""
    draw_feat(ax, cx + offset / 2, cy + offset / 2, w, h, fc=C_FEAT2,
              alpha=0.65, zorder=2)
    return draw_feat(ax, cx, cy, w, h, fc=fc, alpha=0.90, zorder=3)


def arrow(ax, x1, y1, x2, y2, color, lw=1.6, ms=13, style="->",
          conn="arc3,rad=0", zorder=5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        mutation_scale=ms, connectionstyle=conn),
        zorder=zorder,
    )


def label(ax, x, y, text, fs=6.5, ha="center", va="top",
          color=C_TEXT, bold=False, italic=False):
    ax.text(x, y, text,
            ha=ha, va=va, fontsize=fs, color=color,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal",
            zorder=6)


def cls_box(ax, cx, cy, w, h, text, fs=7):
    r = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor="#EDE7F6", edgecolor=C_CLS,
        linewidth=1.2, alpha=0.92, zorder=3,
    )
    ax.add_patch(r)
    label(ax, cx, cy, text, fs=fs, va="center", color="#4A235A", bold=True)


# ── Main diagram ──────────────────────────────────────────────────────────────

def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(-5.5, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Level layout: (enc_cx, enc_cy, dec_cx, dec_cy, ch, block_h, dim_str)
    # Block width = channels / 64 * 0.30  (capped at 1.4)
    # Block height = spatial_h decreasing with depth
    def bw(ch): return min(max(ch / 64 * 0.28, 0.40), 1.40)

    levels = [
        # enc_cx  enc_cy  dec_cx  dec_cy   ch     bh    dim
        (  2.8,   12.5,   21.8,   12.5,    64,   2.50, "640×640"),
        (  5.6,    9.5,   18.8,    9.5,   128,   2.00, "320×320"),
        (  8.4,    6.5,   16.0,    6.5,   256,   1.60, "160×160"),
        ( 11.2,    3.5,   14.0,    3.5,   512,   1.25,  "80×80"),
    ]

    bot_cx, bot_cy = 12.6, 1.0
    bot_ch, bot_bh = 512, 1.00

    # ── INPUT block ───────────────────────────────────────────────────────────
    inp_cx, inp_cy = 1.0, 12.5
    draw_feat(ax, inp_cx, inp_cy, 0.55, 2.50, fc=C_INPUT, ec="#4A90D9", alpha=0.85)
    label(ax, inp_cx, inp_cy + 1.40, "Input", fs=8, va="bottom", bold=True)
    label(ax, inp_cx, inp_cy - 1.38, "640×640×3", fs=6.5)

    # ── ENCODER ───────────────────────────────────────────────────────────────
    enc_right_edges = []

    for i, (ex, ey, dx, dy, ch, bh, dim) in enumerate(levels):
        w = bw(ch)
        l, b, r, t = draw_double_block(ax, ex, ey, w, bh)
        enc_right_edges.append(r)
        label(ax, ex, b - 0.18, f"{dim}×{ch}", fs=6.0)

        # Conv arrow from input / from previous encoder block to this block
        if i == 0:
            # input → level 0
            arrow(ax, inp_cx + 0.28, inp_cy, l - 0.05, ey, C_CONV, lw=1.5)
        else:
            # prev encoder right → max pool → diagonal to this level
            px, py = levels[i - 1][0], levels[i - 1][1]
            pw = bw(levels[i - 1][4])
            pr = px + pw / 2 + 0.18  # right edge of prev block
            # pool indicator dot
            ax.plot([pr + 0.15], [py - levels[i - 1][5] * 0.45],
                    marker="v", ms=8, color=C_POOL, zorder=5)
            # diagonal arrow to current level
            arrow(ax, pr + 0.18, py - levels[i - 1][5] * 0.5,
                  l - 0.05, ey, C_POOL, lw=1.6, ms=11)

    # encoder → bottleneck
    last_ex, last_ey = levels[-1][0], levels[-1][1]
    last_bh = levels[-1][5]
    last_w = bw(levels[-1][4])
    ax.plot([last_ex + last_w / 2 + 0.15],
            [last_ey - last_bh * 0.45], marker="v", ms=8, color=C_POOL, zorder=5)
    bl, bb, br, bt = draw_double_block(ax, bot_cx, bot_cy, bw(bot_ch), bot_bh,
                                       fc="#3A78B5")
    label(ax, bot_cx, bb - 0.18, f"40×40×{bot_ch}", fs=6.0)
    label(ax, bot_cx, bt + 0.10, "Bottleneck", fs=7, va="bottom",
          color="#1F4E79", italic=True)
    arrow(ax, last_ex + last_w / 2 + 0.18, last_ey - last_bh * 0.5,
          bl - 0.05, bot_cy, C_POOL, lw=1.6, ms=11)

    # ── SKIP CONNECTIONS ──────────────────────────────────────────────────────
    for i, (ex, ey, dx, dy, ch, bh, dim) in enumerate(levels):
        ew = bw(ch)
        dec_ch = [64, 64, 128, 256][i]
        dw = bw(dec_ch)
        # horizontal gray arrow from encoder right to decoder left
        ax.annotate(
            "", xy=(dx - dw / 2 - 0.05, ey),
            xytext=(ex + ew / 2 + 0.05, ey),
            arrowprops=dict(
                arrowstyle="-|>",
                color=C_SKIP, lw=2.8, mutation_scale=18,
                connectionstyle="arc3,rad=0",
            ),
            zorder=2,
        )
        # dim label on skip arrow (above midpoint)
        mid_x = (ex + ew / 2 + dx - dw / 2) / 2
        label(ax, mid_x, ey + bh / 2 + 0.12, dim, fs=5.8, color="#606060")

    # ── DECODER ───────────────────────────────────────────────────────────────
    dec_channels_out = [64, 64, 128, 256]  # indexed by level: [0]=640×640, [3]=80×80
    prev_dx, prev_dy = bot_cx, bot_cy
    prev_bw_val = bw(bot_ch)

    for i in range(len(levels) - 1, -1, -1):
        ex, ey, dx, dy, ch, bh, dim = levels[i]
        out_ch = dec_channels_out[i]
        dw = bw(out_ch)

        l, b, r, t = draw_double_block(ax, dx, dy, dw, bh, fc="#6AAFE6")
        label(ax, dx, b - 0.18, f"{dim}×{out_ch}", fs=6.0)

        # upsample arrow from previous decoder/bottleneck to this block
        arrow(ax, prev_dx + prev_bw_val / 2 + 0.18, prev_dy,
              l - 0.05, dy, C_UP, lw=1.7, ms=12,
              conn="arc3,rad=0")
        # upsample marker
        ax.plot([prev_dx + prev_bw_val / 2 + 0.1], [prev_dy + 0.15],
                marker="^", ms=8, color=C_UP, zorder=5)

        prev_dx, prev_dy = dx, dy
        prev_bw_val = dw

    # ── SEGMENTATION OUTPUT block ─────────────────────────────────────────────
    out_cx, out_cy = 23.8, 12.5
    draw_feat(ax, out_cx, out_cy, 0.55, 2.50, fc=C_SEG, ec="#2E7D32", alpha=0.85)
    label(ax, out_cx, out_cy + 1.40, "Seg Mask", fs=8, va="bottom",
          bold=True, color="#1B5E20")
    label(ax, out_cx, out_cy - 1.38, "640×640×1", fs=6.5)

    # final decoder → output (Conv 1×1)
    fin_dx, fin_dy = levels[0][2], levels[0][3]
    fin_dw = bw(dec_channels_out[0])
    arrow(ax, fin_dx + fin_dw / 2 + 0.05, fin_dy,
          out_cx - 0.28, out_cy, C_OUT, lw=2.0, ms=14)
    label(ax, (fin_dx + fin_dw / 2 + out_cx - 0.28) / 2,
          out_cy + 1.38, "Conv 1×1", fs=6.5, color=C_OUT)

    # ── CLASSIFICATION HEAD ───────────────────────────────────────────────────
    cls_x = bot_cx
    cls_stages = [
        (cls_x, -1.0, 1.80, 0.55, "AdaptiveAvgPool2d(1)  →  Flatten  →  (B, 512)"),
        (cls_x, -2.1, 1.80, 0.55, "Linear(512→256)  +  ReLU  +  Dropout(0.3)"),
        (cls_x, -3.2, 1.80, 0.55, "Linear(256→3)  +  Sigmoid  →  (B, 3)"),
    ]

    prev_cy = bot_cy - bot_bh / 2
    for cx, cy, cw_cls, ch_cls, text in cls_stages:
        arrow(ax, cx, prev_cy - 0.04, cx, cy + ch_cls / 2 + 0.04,
              C_CLS, lw=1.5, ms=10)
        cls_box(ax, cx, cy, cw_cls, ch_cls, text, fs=6.5)
        prev_cy = cy - ch_cls / 2

    # "Classification Head" side label (right of boxes, clear of text)
    label(ax, cls_x + 1.55, -2.1, "Classification\nHead", fs=8,
          ha="left", va="center", color=C_CLS, bold=True)

    # arrow from bottleneck down to first cls box
    # (already drawn inside the loop — this is just the first one)

    # ── LEGEND ────────────────────────────────────────────────────────────────
    lx, ly = 0.3, 8.5
    legend_items = [
        (C_CONV,  "●", "Conv 3×3 + BN + ReLU (DoubleConv)"),
        (C_SKIP,  "→", "Copy and concat (skip connection)"),
        (C_POOL,  "▼", "Max pool 2×2 (down-sample)"),
        (C_UP,    "▲", "Bilinear upsample 2×2"),
        (C_OUT,   "●", "Conv 1×1 (segmentation output)"),
        (C_CLS,   "■", "Classification head (GAP→FC→Sigmoid)"),
    ]
    box_h = len(legend_items) * 0.58 + 0.25
    bg = mpatches.FancyBboxPatch(
        (lx - 0.15, ly - box_h + 0.15), 3.50, box_h,
        boxstyle="round,pad=0.1",
        facecolor="#F8F8F8", edgecolor="#BBBBBB", linewidth=0.8,
        alpha=0.92, zorder=1,
    )
    ax.add_patch(bg)
    for idx, (color, sym, txt) in enumerate(legend_items):
        yy = ly - idx * 0.58
        ax.text(lx, yy, sym, ha="center", va="center",
                fontsize=9, color=color, fontweight="bold", zorder=6)
        label(ax, lx + 0.22, yy, txt, fs=6.8, ha="left", va="center")

    # ── TITLE ─────────────────────────────────────────────────────────────────
    ax.text(
        FIG_W / 2, FIG_H - 0.25,
        "U-Net Architecture — Dual Head (Segmentation + Classification)",
        ha="center", va="top", fontsize=13, fontweight="bold", color=C_TEXT,
        zorder=6,
    )

    # ── ENCODER / DECODER labels ──────────────────────────────────────────────
    ax.text(6.5, 13.8, "ENCODER", ha="center", va="bottom",
            fontsize=9, color="#1F4E79", fontweight="bold",
            style="italic", zorder=6)
    ax.text(18.5, 13.8, "DECODER", ha="center", va="bottom",
            fontsize=9, color="#1A5276", fontweight="bold",
            style="italic", zorder=6)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout(pad=0.3)
    fig.savefig(str(OUT_PATH), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
