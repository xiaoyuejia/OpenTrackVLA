#!/usr/bin/env python3
"""Draw the uncluttered five-stage AirGround-Coop V3 paper pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from draw_airground_v3_pipeline_paper import (
    BLUE,
    BLUE_LIGHT,
    GOLD,
    GOLD_LIGHT,
    GRAY,
    GRAY_DARK,
    GREEN,
    GREEN_LIGHT,
    GROUND,
    GROUND_LIGHT,
    INK,
    MUTED,
    ORANGE,
    ORANGE_LIGHT,
    PURPLE,
    PURPLE_LIGHT,
    RED,
    TEAL,
    TEAL_LIGHT,
    arrow,
    draw_drone,
    draw_frame,
    draw_person,
    draw_robotdog,
    label_chip,
    rounded,
    token,
    token_row,
)


ROOT = Path(__file__).resolve().parent
STEM = ROOT / "airground_v3_pipeline_clean"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.0,
    }
)


def stage_title(ax, x, y, number, title, color):
    rounded(ax, x, y - 0.18, 0.36, 0.36, fc=color, ec=color, lw=1.0, radius=0.18, z=8)
    ax.text(x + 0.18, y, str(number), ha="center", va="center", fontsize=10, color="white", weight="bold", zorder=9)
    ax.text(x + 0.48, y, title, ha="left", va="center", fontsize=12, color=INK, weight="bold", zorder=9)


def query_pair(ax, x, y, action_color):
    token(ax, x, y, ORANGE, size=0.23, label="V")
    token(ax, x + 0.36, y, action_color, size=0.23, label="A")


def main():
    fig, ax = plt.subplots(figsize=(18, 8.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8.2)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0.35, 7.88, "AirGround-Coop V3", fontsize=23, weight="bold", color=INK, va="center")
    ax.text(
        4.82,
        7.88,
        "two isolated self flows + one joint cooperative flow",
        fontsize=10,
        color=MUTED,
        va="center",
    )
    ax.plot([0.35, 17.65], [7.58, 7.58], color="#D7DEE3", lw=1.0)

    # ------------------------------------------------------------------
    # 1. Dual-view inputs.
    # ------------------------------------------------------------------
    stage_title(ax, 0.42, 7.22, 1, "Dual-view inputs", BLUE)
    rounded(ax, 0.35, 2.15, 2.50, 4.70, fc="#FAFCFD", ec="#A6B5C0", lw=1.2, radius=0.16)

    ax.text(0.58, 6.45, "Aerial history + current", fontsize=8.8, color=BLUE, weight="bold")
    for idx in range(3):
        draw_frame(
            ax,
            0.58 + idx * 0.61,
            5.62,
            0.54,
            0.65,
            aerial=True,
            target_x=0.44 + idx * 0.08,
            target_y=0.38,
        )
    draw_drone(ax, 2.48, 5.94, scale=0.55, color=INK)

    ax.text(0.58, 5.22, "Ground history + current", fontsize=8.8, color=GROUND, weight="bold")
    for idx in range(3):
        draw_frame(
            ax,
            0.58 + idx * 0.61,
            4.39,
            0.54,
            0.65,
            aerial=False,
            target_x=0.62 - idx * 0.07,
            target_y=0.34,
        )
    draw_robotdog(ax, 2.45, 4.70, scale=0.60, color=INK)

    rounded(ax, 0.57, 3.48, 2.05, 0.55, fc=GREEN_LIGHT, ec=GREEN, lw=1.0, radius=0.09)
    ax.text(1.60, 3.76, "tracking instruction", ha="center", va="center", fontsize=8.6, color=GREEN, weight="bold")
    rounded(ax, 0.57, 2.68, 2.05, 0.55, fc=PURPLE_LIGHT, ec=PURPLE, lw=1.0, radius=0.09)
    ax.text(1.60, 2.96, "two follower poses", ha="center", va="center", fontsize=8.6, color=PURPLE, weight="bold")
    ax.text(1.60, 2.48, "[x, y, sin(yaw), cos(yaw)]", ha="center", fontsize=7.2, color=MUTED)

    # ------------------------------------------------------------------
    # 2. Shared tokenization.
    # ------------------------------------------------------------------
    stage_title(ax, 3.28, 7.22, 2, "Tokenization", BLUE)
    rounded(ax, 3.20, 2.15, 2.48, 4.70, fc=BLUE_LIGHT, ec=BLUE, lw=1.3, radius=0.16)
    rounded(ax, 3.49, 5.67, 1.90, 0.67, fc="white", ec=BLUE, lw=1.0, radius=0.09)
    ax.text(4.44, 6.10, "DINOv3 + SigLIP", ha="center", fontsize=9.5, weight="bold")
    ax.text(4.44, 5.83, "coarse history / fine current", ha="center", fontsize=7.2, color=MUTED)
    rounded(ax, 3.49, 4.70, 1.90, 0.67, fc=ORANGE_LIGHT, ec=ORANGE, lw=1.0, radius=0.09)
    ax.text(4.44, 5.13, "YOLO + ROI", ha="center", fontsize=9.5, weight="bold")
    ax.text(4.44, 4.86, "proposal + scene grid", ha="center", fontsize=7.2, color=MUTED)
    rounded(ax, 3.49, 3.73, 1.90, 0.67, fc="white", ec="#778995", lw=1.0, radius=0.09)
    ax.text(4.44, 4.16, "Projector + TVI", ha="center", fontsize=9.5, weight="bold")
    ax.text(4.44, 3.89, "time / kind / agent", ha="center", fontsize=7.2, color=MUTED)

    token_row(ax, 3.58, 3.02, [GREEN, BLUE, BLUE, GROUND, GROUND, ORANGE, PURPLE], size=0.20, gap=0.065)
    ax.text(4.44, 2.69, "typed token packs", ha="center", fontsize=8.0, color=MUTED)

    arrow(ax, (2.85, 4.55), (3.20, 4.55), color=BLUE, lw=1.5, mutation=9)

    # ------------------------------------------------------------------
    # 3. Shared-weight VLM with three explicit rows.
    # ------------------------------------------------------------------
    stage_title(ax, 6.13, 7.22, 3, "Three information flows", PURPLE)
    rounded(ax, 6.05, 2.15, 4.55, 4.70, fc=GRAY, ec=GRAY_DARK, lw=1.5, radius=0.16)
    ax.text(8.33, 6.46, "Shared frozen Qwen3-0.6B", ha="center", fontsize=12, color=INK, weight="bold")
    ax.text(8.33, 6.16, "same parameters; different attention contexts", ha="center", fontsize=7.8, color=MUTED)

    rounded(ax, 6.37, 5.27, 3.91, 0.64, fc=BLUE_LIGHT, ec=BLUE, lw=1.1, radius=0.09)
    ax.text(6.58, 5.59, "SELF-D", va="center", fontsize=9.0, color=BLUE, weight="bold")
    ax.text(7.48, 5.59, "clean drone-only context", va="center", fontsize=8.2, color=INK)
    query_pair(ax, 9.36, 5.47, BLUE)

    rounded(ax, 6.37, 4.35, 3.91, 0.64, fc=GROUND_LIGHT, ec=GROUND, lw=1.1, radius=0.09)
    ax.text(6.58, 4.67, "SELF-G", va="center", fontsize=9.0, color=GROUND, weight="bold")
    ax.text(7.48, 4.67, "clean dog-only context", va="center", fontsize=8.2, color=INK)
    query_pair(ax, 9.36, 4.55, GROUND)

    rounded(ax, 6.37, 3.24, 3.91, 0.78, fc=PURPLE_LIGHT, ec=PURPLE, lw=1.2, radius=0.09)
    ax.text(6.58, 3.63, "COOP", va="center", fontsize=9.0, color=PURPLE, weight="bold")
    ax.text(7.36, 3.63, "drone + dog + two poses", va="center", fontsize=8.2, color=INK)
    token(ax, 9.43, 3.50, PURPLE, size=0.25, label="C")
    token(ax, 9.82, 3.50, PURPLE, size=0.25, label="C")

    ax.text(8.33, 2.72, "forward #1: SELF-D/G packed as 2B isolated rows", ha="center", fontsize=7.4, color=BLUE)
    ax.text(8.33, 2.45, "forward #2: one B joint cooperative row", ha="center", fontsize=7.4, color=PURPLE)
    arrow(ax, (5.68, 4.55), (6.05, 4.55), color=PURPLE, lw=1.5, mutation=9)

    # ------------------------------------------------------------------
    # 4. Prediction heads. One box per semantic branch.
    # ------------------------------------------------------------------
    stage_title(ax, 11.08, 7.22, 4, "Predictions", PURPLE)
    rounded(ax, 11.00, 4.52, 3.05, 2.33, fc="#FBFCFD", ec=BLUE, lw=1.3, radius=0.16)
    ax.text(11.28, 6.47, "SELF branch", fontsize=10.5, color=BLUE, weight="bold")
    rounded(ax, 11.27, 5.66, 1.18, 0.52, fc=ORANGE_LIGHT, ec=ORANGE, lw=1.0, radius=0.08)
    ax.text(11.86, 5.92, "VERIFY heads", ha="center", va="center", fontsize=8.0, weight="bold")
    ax.text(12.63, 5.92, "→  p_match", va="center", fontsize=8.0, color=ORANGE, weight="bold")
    rounded(ax, 11.27, 4.87, 1.18, 0.52, fc=BLUE_LIGHT, ec=BLUE, lw=1.0, radius=0.08)
    ax.text(11.86, 5.13, "action heads", ha="center", va="center", fontsize=8.0, weight="bold")
    ax.text(12.63, 5.13, "→  τ_self", va="center", fontsize=8.0, color=BLUE, weight="bold")
    ax.text(12.52, 4.68, "clean GT: L_self     IoU label: L_verify", ha="center", fontsize=7.0, color=MUTED)

    rounded(ax, 11.00, 2.15, 3.05, 2.05, fc="#FBF9FD", ec=PURPLE, lw=1.3, radius=0.16)
    ax.text(11.28, 3.84, "COOPERATIVE branch", fontsize=10.5, color=PURPLE, weight="bold")
    ax.text(11.30, 3.43, "Conditional JEPA  →  recovered latent", fontsize=8.1, color=INK)
    ax.text(11.30, 3.05, "K=4 decoders       →  τ_coop", fontsize=8.1, color=INK)
    ax.text(11.30, 2.62, "L_coop + L_mode + L_JEPA + auxiliary", fontsize=7.2, color=MUTED)

    arrow(ax, (10.28, 5.59), (11.00, 5.59), color=BLUE, lw=1.3, mutation=8)
    arrow(ax, (10.28, 3.63), (11.00, 3.63), color=PURPLE, lw=1.3, mutation=8)

    # ------------------------------------------------------------------
    # 5. Online router and control.
    # ------------------------------------------------------------------
    stage_title(ax, 14.55, 7.22, 5, "Online router", TEAL)
    rounded(ax, 14.47, 2.15, 3.18, 4.70, fc=TEAL_LIGHT, ec=TEAL, lw=1.5, radius=0.16)
    ax.text(16.06, 6.43, "verified = YOLO AND p_match", ha="center", fontsize=9.0, color=TEAL, weight="bold")
    ax.text(16.06, 6.10, "temporal hysteresis", ha="center", fontsize=7.5, color=MUTED)

    states = [
        ("both visible", "SELF / SELF", BLUE, BLUE_LIGHT),
        ("one invisible", "SELF / COOP", PURPLE, PURPLE_LIGHT),
        ("both lost", "BELIEF → SEARCH", GOLD, GOLD_LIGHT),
    ]
    for idx, (condition, route, color, fc) in enumerate(states):
        yy = 5.38 - idx * 0.69
        rounded(ax, 14.76, yy, 2.60, 0.49, fc=fc, ec=color, lw=0.9, radius=0.08)
        ax.text(14.92, yy + 0.245, condition, ha="left", va="center", fontsize=7.4, color=MUTED)
        ax.text(17.19, yy + 0.245, route, ha="right", va="center", fontsize=7.7, color=color, weight="bold")

    arrow(ax, (16.06, 3.62), (16.06, 3.20), color=TEAL, lw=1.2, mutation=8)
    rounded(ax, 14.80, 2.48, 2.52, 0.67, fc="white", ec=TEAL, lw=1.1, radius=0.10)
    ax.text(16.06, 2.82, "selected waypoints → actions", ha="center", va="center", fontsize=8.4, color=TEAL, weight="bold")
    draw_drone(ax, 15.47, 2.35, scale=0.46, color=INK)
    draw_robotdog(ax, 16.65, 2.33, scale=0.50, color=INK)

    arrow(ax, (14.05, 5.59), (14.47, 5.59), color=BLUE, lw=1.2, mutation=8)
    arrow(ax, (14.05, 3.63), (14.47, 3.63), color=PURPLE, lw=1.2, mutation=8)
    ax.text(16.06, 2.23, "inference: no synthetic corruption", ha="center", fontsize=7.2, color=TEAL, style="italic")

    # ------------------------------------------------------------------
    # Training-only band. This is subordinate to the main left-to-right flow.
    # ------------------------------------------------------------------
    rounded(ax, 6.05, 0.38, 8.00, 1.35, fc="#FFF9F6", ec=RED, lw=1.2, radius=0.14, ls="--")
    ax.text(6.30, 1.44, "TRAINING ONLY", fontsize=8.5, color=RED, weight="bold")
    ax.text(7.84, 1.44, "one receiver (D/G = 0.5/0.5)", fontsize=8.2, color=INK)
    ax.text(6.30, 1.06, "visual → [MISSING]      detection/ROI → 0      pose + (Δx, Δy, Δyaw)", fontsize=8.0, color=RED)
    ax.text(6.30, 0.70, "EMA latent → JEPA target          receiver-frame GT → cooperative target", fontsize=7.8, color=MUTED)
    arrow(ax, (8.65, 1.73), (8.65, 3.24), color=RED, lw=1.0, mutation=7, ls="--")
    arrow(ax, (12.45, 1.73), (12.45, 2.15), color=RED, lw=1.0, mutation=7, ls="--")

    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(STEM.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    main()
