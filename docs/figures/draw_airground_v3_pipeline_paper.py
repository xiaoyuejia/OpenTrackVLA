#!/usr/bin/env python3
"""Draw the compact paper-style AirGround-Coop V3 pipeline figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
)
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parent
STEM = ROOT / "airground_v3_pipeline_paper"

INK = "#20242A"
MUTED = "#65717C"
GRAY = "#F1F3F5"
GRAY_DARK = "#646A70"
BLUE = "#2F80B7"
BLUE_LIGHT = "#E7F4FB"
GROUND = "#249E8A"
GROUND_LIGHT = "#E6F6F2"
GREEN = "#2CB67D"
GREEN_LIGHT = "#E8FAF2"
ORANGE = "#F28E2B"
ORANGE_LIGHT = "#FFF2E4"
PURPLE = "#6D5195"
PURPLE_LIGHT = "#F2EDF8"
TEAL = "#168D7A"
TEAL_LIGHT = "#E6F7F3"
GOLD = "#C99518"
GOLD_LIGHT = "#FFF6D8"
RED = "#D7604D"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.0,
    }
)


def rounded(
    ax,
    x,
    y,
    w,
    h,
    *,
    fc="white",
    ec=INK,
    lw=1.4,
    radius=0.12,
    z=1,
    ls="-",
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    p0,
    p1,
    *,
    color=GRAY_DARK,
    lw=1.2,
    style="-|>",
    mutation=10,
    connection="arc3",
    z=3,
    ls="-",
):
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=connection,
        linestyle=ls,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def token(ax, x, y, color, *, size=0.23, label=None, lw=1.2):
    rounded(
        ax,
        x,
        y,
        size,
        size,
        fc="white",
        ec=color,
        lw=lw,
        radius=0.045,
        z=5,
    )
    if label:
        ax.text(
            x + size / 2,
            y + size / 2,
            label,
            ha="center",
            va="center",
            fontsize=6.5,
            color=color,
            weight="bold",
            zorder=6,
        )


def token_row(ax, x, y, colors, *, gap=0.07, size=0.21):
    for idx, color in enumerate(colors):
        token(ax, x + idx * (size + gap), y, color, size=size)


def label_chip(ax, x, y, w, text, color, fc, *, fs=8, lw=1.0):
    rounded(ax, x, y, w, 0.32, fc=fc, ec=color, lw=lw, radius=0.07, z=4)
    ax.text(
        x + w / 2,
        y + 0.16,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        weight="bold",
        zorder=5,
    )


def draw_person(ax, x, y, scale=1.0, color=INK):
    ax.add_patch(Circle((x, y + 0.35 * scale), 0.075 * scale, fc=color, ec="none", zorder=6))
    rounded(
        ax,
        x - 0.07 * scale,
        y + 0.02 * scale,
        0.14 * scale,
        0.29 * scale,
        fc=color,
        ec=color,
        lw=0.6,
        radius=0.04 * scale,
        z=6,
    )
    ax.plot(
        [x - 0.03 * scale, x - 0.08 * scale],
        [y + 0.03 * scale, y - 0.18 * scale],
        color=color,
        lw=2.0 * scale,
        solid_capstyle="round",
        zorder=6,
    )
    ax.plot(
        [x + 0.03 * scale, x + 0.08 * scale],
        [y + 0.03 * scale, y - 0.18 * scale],
        color=color,
        lw=2.0 * scale,
        solid_capstyle="round",
        zorder=6,
    )


def draw_drone(ax, x, y, scale=1.0, color=INK):
    ax.add_patch(Rectangle((x - 0.14 * scale, y - 0.05 * scale), 0.28 * scale, 0.10 * scale, fc=color, ec=color, zorder=6))
    ax.plot([x - 0.28 * scale, x + 0.28 * scale], [y, y], color=color, lw=1.8, zorder=6)
    ax.plot([x - 0.18 * scale, x - 0.27 * scale], [y, y + 0.10 * scale], color=color, lw=1.3, zorder=6)
    ax.plot([x + 0.18 * scale, x + 0.27 * scale], [y, y + 0.10 * scale], color=color, lw=1.3, zorder=6)
    for dx in (-0.28, 0.28):
        ax.add_patch(Circle((x + dx * scale, y + 0.10 * scale), 0.095 * scale, fill=False, ec=color, lw=1.2, zorder=6))
    ax.add_patch(Circle((x, y - 0.075 * scale), 0.045 * scale, fc=BLUE, ec=color, lw=0.7, zorder=6))


def draw_robotdog(ax, x, y, scale=1.0, color=INK):
    rounded(
        ax,
        x - 0.22 * scale,
        y - 0.08 * scale,
        0.38 * scale,
        0.19 * scale,
        fc="white",
        ec=color,
        lw=1.5,
        radius=0.03,
        z=6,
    )
    rounded(
        ax,
        x + 0.13 * scale,
        y + 0.01 * scale,
        0.15 * scale,
        0.12 * scale,
        fc="white",
        ec=color,
        lw=1.4,
        radius=0.025,
        z=6,
    )
    ax.add_patch(Circle((x + 0.25 * scale, y + 0.08 * scale), 0.025 * scale, fc=GROUND, ec="none", zorder=7))
    for dx in (-0.15, 0.10):
        ax.plot(
            [x + dx * scale, x + (dx - 0.04) * scale],
            [y - 0.07 * scale, y - 0.28 * scale],
            color=color,
            lw=1.8,
            zorder=6,
        )
        ax.plot(
            [x + (dx - 0.04) * scale, x + (dx + 0.02) * scale],
            [y - 0.28 * scale, y - 0.30 * scale],
            color=color,
            lw=1.6,
            zorder=6,
        )


def draw_wall(ax, x, y, w=0.35, h=0.8):
    ax.add_patch(Rectangle((x, y), w, h, fc="#D8D8D8", ec="#6E6E6E", lw=1.1, zorder=5))
    for yy in np.linspace(y + 0.15, y + h - 0.05, 4):
        ax.plot([x, x + w], [yy, yy], color="#9C9C9C", lw=0.7, zorder=6)
    ax.plot([x + w / 2, x + w / 2], [y, y + h], color="#9C9C9C", lw=0.7, zorder=6)


def trajectory(ax, points, color, *, lw=2.0, ls="-", alpha=1.0, arrowhead=True):
    points = np.asarray(points, dtype=float)
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * (len(points) - 1)
    path = MplPath(points, codes)
    ax.add_patch(PathPatch(path, fill=False, ec=color, lw=lw, ls=ls, alpha=alpha, zorder=6))
    if arrowhead:
        arrow(ax, points[-2], points[-1], color=color, lw=lw, mutation=9, z=7)


def draw_frame(ax, x, y, w, h, *, aerial=False, target_x=0.60, target_y=0.35):
    ax.add_patch(Rectangle((x, y), w, h, fc="#DCECF5", ec="#8796A2", lw=0.7, zorder=2))
    if aerial:
        ax.add_patch(Rectangle((x, y), w, h * 0.50, fc="#8FA7B8", ec="none", zorder=2))
        for k in range(3):
            ax.add_patch(
                Polygon(
                    [
                        (x + 0.05 + k * w * 0.30, y + 0.02),
                        (x + w * 0.25 + k * w * 0.25, y + 0.02),
                        (x + w * 0.20 + k * w * 0.25, y + h * 0.46),
                    ],
                    closed=True,
                    fc="#D9D2C4",
                    ec="none",
                    zorder=3,
                )
            )
    else:
        ax.add_patch(Rectangle((x, y), w, h * 0.42, fc="#A9B49A", ec="none", zorder=2))
        ax.add_patch(Polygon([(x, y), (x + w, y), (x + w * 0.62, y + h * 0.42), (x + w * 0.38, y + h * 0.42)], fc="#858B91", ec="none", zorder=3))
        ax.add_patch(Rectangle((x + 0.03, y + h * 0.42), w * 0.20, h * 0.50, fc="#C7B89D", ec="none", zorder=2))
        ax.add_patch(Rectangle((x + w * 0.77, y + h * 0.42), w * 0.20, h * 0.50, fc="#C2A78E", ec="none", zorder=2))
    px = x + target_x * w
    py = y + target_y * h
    ax.add_patch(Circle((px, py), max(0.018, 0.025 * h), fc=RED, ec="white", lw=0.5, zorder=4))


def panel_title(ax, x, y, letter, title, color):
    ax.text(x, y, f"({letter})", fontsize=11, color=color, weight="bold", va="center", zorder=8)
    title_offset = 0.55 if len(letter) > 1 else 0.34
    ax.text(x + title_offset, y, title, fontsize=12.5, color=INK, weight="bold", va="center", zorder=8)


def main():
    fig, ax = plt.subplots(figsize=(18, 10.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # Left: main backbone and dual-view tokenization.
    # ------------------------------------------------------------------
    rounded(ax, 0.22, 7.05, 7.62, 2.12, fc=GRAY, ec=GRAY_DARK, lw=1.8, radius=0.22)
    ax.text(0.48, 8.90, "②", ha="left", va="center", fontsize=17, color=PURPLE, weight="bold")
    ax.text(0.92, 8.90, "Shared-weight LLM backbone", ha="left", va="center", fontsize=17, color=INK, weight="bold")
    ax.text(7.51, 8.90, "frozen Qwen3-0.6B", ha="right", va="center", fontsize=8.3, color=MUTED)

    # Three information flows; the first two share one 2B call but never attend
    # across views, while the joint row is a separate cooperative forward.
    rounded(ax, 0.52, 8.34, 6.98, 0.34, fc=BLUE_LIGHT, ec=BLUE, lw=1.0, radius=0.07)
    ax.text(0.70, 8.51, "SELF-D", va="center", fontsize=8.0, color=BLUE, weight="bold")
    ax.text(1.52, 8.51, "clean drone tokens  |  VERIFY_D -> ACT_D", va="center", fontsize=8.0, color=INK)
    ax.text(7.30, 8.51, "forward #1: 2B isolated rows", ha="right", va="center", fontsize=7.0, color=BLUE)
    rounded(ax, 0.52, 7.88, 6.98, 0.34, fc=GROUND_LIGHT, ec=GROUND, lw=1.0, radius=0.07)
    ax.text(0.70, 8.05, "SELF-G", va="center", fontsize=8.0, color=GROUND, weight="bold")
    ax.text(1.52, 8.05, "clean dog tokens     |  VERIFY_G -> ACT_G", va="center", fontsize=8.0, color=INK)
    ax.text(7.30, 8.05, "same attention-isolated call", ha="right", va="center", fontsize=7.0, color=GROUND)
    rounded(ax, 0.52, 7.42, 6.98, 0.34, fc=PURPLE_LIGHT, ec=PURPLE, lw=1.0, radius=0.07)
    ax.text(0.70, 7.59, "COOP-DG", va="center", fontsize=8.0, color=PURPLE, weight="bold")
    ax.text(1.65, 7.59, "both streams + ABS/REL poses  |  COOP_ACT_D/G", va="center", fontsize=8.0, color=INK)
    ax.text(7.30, 7.59, "forward #2: one joint B row", ha="right", va="center", fontsize=7.0, color=PURPLE)

    # Token rows entering the backbone.
    token_row(ax, 0.55, 6.67, [GREEN] * 3, size=0.22)
    token_row(ax, 1.58, 6.67, [BLUE] * 6, size=0.22)
    token_row(ax, 3.62, 6.67, [GROUND] * 6, size=0.22)
    # Two YOLO/ROI tokens followed by four cooperative pose tokens:
    # two absolute poses and two directed receiver-local relative poses.
    token_row(ax, 5.46, 6.67, [ORANGE] * 2 + [PURPLE] * 4, size=0.20, gap=0.045)
    ax.text(0.84, 6.49, "text", ha="center", va="top", fontsize=7.5, color=GREEN)
    ax.text(2.31, 6.49, "drone visual", ha="center", va="top", fontsize=7.5, color=BLUE)
    ax.text(4.35, 6.49, "dog visual", ha="center", va="top", fontsize=7.5, color=GROUND)
    ax.text(6.18, 6.49, "YOLO + 4 pose tokens", ha="center", va="top", fontsize=7.3, color=PURPLE)
    for xx in (0.84, 2.31, 4.35, 6.18):
        arrow(ax, (xx, 6.93), (xx, 7.05), color=GRAY_DARK, lw=0.9, mutation=7)

    # Query tokens on the top edge, echoing the reference's visual grammar.
    token(ax, 5.30, 9.34, ORANGE, size=0.28, label="V")
    token(ax, 6.18, 9.34, BLUE, size=0.28, label="A")
    token(ax, 7.06, 9.34, PURPLE, size=0.28, label="C")
    ax.text(5.44, 9.76, "VERIFY", ha="center", fontsize=7.2, color=ORANGE, weight="bold")
    ax.text(6.32, 9.76, "SELF ACT", ha="center", fontsize=7.2, color=BLUE, weight="bold")
    ax.text(7.20, 9.76, "COOP ACT", ha="center", fontsize=7.2, color=PURPLE, weight="bold")
    arrow(ax, (5.44, 9.34), (5.44, 9.17), color=ORANGE, lw=1.0, mutation=7)
    arrow(ax, (6.32, 9.34), (6.32, 9.17), color=BLUE, lw=1.0, mutation=7)
    arrow(ax, (7.20, 9.34), (7.20, 9.17), color=PURPLE, lw=1.0, mutation=7)

    # Front-end blocks and dual video streams.
    ax.text(0.35, 6.13, "①  Inputs -> typed tokens", fontsize=11.3, color=INK, weight="bold")
    label_chip(ax, 0.35, 5.72, 2.75, "Follow the target without collision", GREEN, GREEN_LIGHT, fs=7.4)
    rounded(ax, 3.04, 5.53, 2.10, 0.68, fc=BLUE_LIGHT, ec=BLUE, lw=1.1, radius=0.10)
    ax.text(4.09, 5.87, "DINOv3 + SigLIP", ha="center", va="center", fontsize=10, color=INK, weight="bold")
    ax.text(4.09, 5.65, "frozen 1536-D visual tokens", ha="center", va="center", fontsize=7.4, color=MUTED)
    rounded(ax, 5.34, 5.53, 2.10, 0.68, fc=ORANGE_LIGHT, ec=ORANGE, lw=1.1, radius=0.10)
    ax.text(6.39, 5.87, "YOLO + ROI pooling", ha="center", va="center", fontsize=9.8, color=INK, weight="bold")
    ax.text(6.39, 5.65, "proposal + 8x8 scene grid", ha="center", va="center", fontsize=7.4, color=MUTED)

    # Drone filmstrip.
    ax.text(0.37, 5.22, "Aerial stream", color=BLUE, fontsize=9.5, weight="bold", va="center")
    for idx in range(5):
        draw_frame(ax, 0.38 + idx * 0.69, 4.32, 0.62, 0.72, aerial=True, target_x=0.42 + 0.07 * idx, target_y=0.38)
    ax.text(3.84, 4.67, "...", fontsize=15, color=MUTED, ha="center", va="center")
    arrow(ax, (0.38, 4.19), (3.74, 4.19), color=BLUE, lw=1.1, mutation=8)

    # Ground filmstrip.
    ax.text(0.37, 3.83, "Ground stream", color=GROUND, fontsize=9.5, weight="bold", va="center")
    for idx in range(5):
        draw_frame(ax, 0.38 + idx * 0.69, 2.93, 0.62, 0.72, aerial=False, target_x=0.64 - 0.05 * idx, target_y=0.34)
    ax.text(3.84, 3.28, "...", fontsize=15, color=MUTED, ha="center", va="center")
    arrow(ax, (0.38, 2.80), (3.74, 2.80), color=GROUND, lw=1.1, mutation=8)

    rounded(ax, 4.25, 4.20, 1.55, 0.75, fc="#F7FAFC", ec="#758692", lw=1.0, radius=0.10)
    ax.text(5.03, 4.63, "Grid pooling", ha="center", va="center", fontsize=9.5, weight="bold")
    ax.text(5.03, 4.38, "31x4 history\n64 current", ha="center", va="center", fontsize=7.5, color=MUTED, linespacing=1.05)
    rounded(ax, 5.92, 4.08, 1.57, 0.92, fc=PURPLE_LIGHT, ec=PURPLE, lw=1.0, radius=0.10)
    ax.text(6.71, 4.78, "Absolute + relative", ha="center", va="center", fontsize=8.6, weight="bold")
    ax.text(6.71, 4.52, "2× [x,y,sinψ,cosψ]", ha="center", va="center", fontsize=6.8, color=PURPLE)
    ax.text(6.71, 4.28, "2× [f,r,sinΔψ,cosΔψ,d]", ha="center", va="center", fontsize=6.4, color=PURPLE)

    arrow(ax, (3.78, 4.68), (4.25, 4.68), color=BLUE, lw=1.0, mutation=7)
    arrow(ax, (3.78, 3.30), (4.62, 4.20), color=GROUND, lw=1.0, mutation=7)
    arrow(ax, (5.03, 4.95), (4.10, 5.53), color=BLUE, lw=1.0, mutation=7)
    arrow(ax, (4.10, 6.21), (3.85, 6.67), color=BLUE, lw=1.0, mutation=7)
    arrow(ax, (6.39, 6.21), (5.90, 6.67), color=ORANGE, lw=1.0, mutation=7)
    arrow(ax, (6.71, 5.00), (6.43, 6.67), color=PURPLE, lw=1.0, mutation=7)

    ax.text(0.35, 2.36, "Typed token construction", fontsize=10.5, color=INK, weight="bold")
    ax.text(0.35, 2.08, "Cross-modal projection  +  time / kind / agent embeddings", fontsize=8.4, color=MUTED)
    label_chip(ax, 0.35, 1.55, 1.22, "language", GREEN, GREEN_LIGHT, fs=7.6)
    label_chip(ax, 1.70, 1.55, 1.22, "drone visual", BLUE, BLUE_LIGHT, fs=7.6)
    label_chip(ax, 3.05, 1.55, 1.22, "dog visual", GROUND, GROUND_LIGHT, fs=7.6)
    label_chip(ax, 4.40, 1.55, 1.22, "YOLO / ROI", ORANGE, ORANGE_LIGHT, fs=7.6)
    label_chip(ax, 5.75, 1.55, 1.22, "4 pose tokens", PURPLE, PURPLE_LIGHT, fs=7.3)
    ax.text(0.35, 1.10, "SELF rows use clean single-view tokens; COOPERATIVE uses both streams and both poses.", fontsize=8.3, color=INK)
    ax.text(0.35, 0.78, "Training corruption affects only the cooperative receiver row.", fontsize=8.3, color=RED, weight="bold")

    # ------------------------------------------------------------------
    # Right top: isolated self tracking and verification.
    # ------------------------------------------------------------------
    rounded(ax, 8.16, 6.92, 9.56, 3.02, fc="#FBF9FD", ec=PURPLE, lw=1.6, radius=0.22)
    panel_title(ax, 8.42, 9.66, "3a", "Isolated SELF outputs", PURPLE)
    ax.text(17.42, 9.66, "no cross-view attention", ha="right", va="center", fontsize=8.2, color=PURPLE, style="italic")

    # Two self token rows.
    ax.text(8.46, 8.94, "Drone row", color=BLUE, weight="bold", fontsize=9, va="center")
    token_row(ax, 9.38, 8.82, [GREEN] + [BLUE] * 5 + [ORANGE, ORANGE, BLUE], size=0.20, gap=0.055)
    ax.text(8.46, 7.75, "Dog row", color=GROUND, weight="bold", fontsize=9, va="center")
    token_row(ax, 9.38, 7.63, [GREEN] + [GROUND] * 5 + [ORANGE, ORANGE, GROUND], size=0.20, gap=0.055)
    ax.text(9.38, 8.48, "text | history/current | YOLO+ROI | VERIFY | ACT", fontsize=7.2, color=MUTED)
    ax.text(9.38, 7.42, "same weights, separate causal contexts (packed as 2B rows)", fontsize=7.0, color=MUTED)

    rounded(ax, 11.88, 7.47, 1.32, 1.74, fc=PURPLE_LIGHT, ec=PURPLE, lw=1.4, radius=0.14)
    ax.text(12.54, 8.49, "Qwen3", ha="center", va="center", fontsize=11.5, weight="bold", color=PURPLE)
    ax.text(12.54, 8.15, "shared", ha="center", va="center", fontsize=8, color=MUTED)
    ax.text(12.54, 7.83, "VERIFY -> ACT", ha="center", va="center", fontsize=7.5, color=INK)
    arrow(ax, (11.54, 8.92), (11.88, 8.92), color=BLUE, lw=1.0, mutation=7)
    arrow(ax, (11.54, 7.73), (11.88, 7.73), color=GROUND, lw=1.0, mutation=7)

    # Verification panel.
    rounded(ax, 13.48, 8.29, 1.54, 1.16, fc=ORANGE_LIGHT, ec=ORANGE, lw=1.15, radius=0.12)
    ax.text(14.25, 9.16, "VERIFY head", ha="center", fontsize=9.2, weight="bold")
    ax.text(14.25, 8.87, "YOLO candidate", ha="center", fontsize=7.4, color=MUTED)
    ax.add_patch(Rectangle((13.76, 8.50), 0.95, 0.13, fc="#E5E7E9", ec="#A8ADB2", lw=0.6, zorder=5))
    ax.add_patch(Rectangle((13.76, 8.50), 0.73, 0.13, fc=ORANGE, ec="none", zorder=6))
    ax.text(14.25, 8.37, r"$p_{match}$", ha="center", fontsize=8.5, color=ORANGE)

    # Self action heads and trajectories.
    rounded(ax, 13.48, 7.16, 1.54, 0.83, fc=BLUE_LIGHT, ec=BLUE, lw=1.15, radius=0.12)
    ax.text(14.25, 7.71, "Action heads", ha="center", fontsize=9.2, weight="bold")
    ax.text(14.25, 7.39, "two 3-layer MLPs", ha="center", fontsize=7.4, color=MUTED)
    arrow(ax, (13.20, 8.63), (13.48, 8.63), color=ORANGE, lw=1.0, mutation=7)
    arrow(ax, (13.20, 7.72), (13.48, 7.72), color=BLUE, lw=1.0, mutation=7)

    draw_drone(ax, 15.55, 8.75, scale=0.75, color=INK)
    draw_robotdog(ax, 15.55, 7.53, scale=0.82, color=INK)
    draw_person(ax, 17.06, 8.55, scale=0.72, color=INK)
    trajectory(ax, [(15.80, 8.70), (16.25, 8.95), (16.72, 8.62)], BLUE, lw=2.0)
    trajectory(ax, [(15.80, 7.46), (16.22, 7.58), (16.68, 8.12)], GROUND, lw=2.0)
    ax.text(16.34, 7.14, "clean self trajectories", ha="center", fontsize=7.8, color=MUTED)
    label_chip(ax, 8.45, 7.02, 1.52, r"clean GT -> $L_{self}$", BLUE, BLUE_LIGHT, fs=7.2)
    label_chip(ax, 10.08, 7.02, 1.58, r"IoU label -> $L_{verify}$", ORANGE, ORANGE_LIGHT, fs=7.0)

    # Backbone output connections.
    arrow(ax, (7.84, 8.46), (8.16, 8.46), color=PURPLE, lw=1.5, mutation=9)
    ax.text(7.95, 8.67, "self contexts", ha="center", fontsize=7.0, color=BLUE, weight="bold")

    # ------------------------------------------------------------------
    # Right middle: cooperative receiver recovery.
    # ------------------------------------------------------------------
    rounded(ax, 8.16, 3.26, 9.56, 3.38, fc="#FFFDFC", ec=ORANGE, lw=1.6, radius=0.22)
    panel_title(ax, 8.42, 6.36, "3b", "Cooperative receiver recovery", ORANGE)
    ax.text(17.42, 6.36, "training: one receiver is corrupted", ha="right", va="center", fontsize=8.2, color=RED, style="italic")

    # Source / receiver scene.
    draw_drone(ax, 8.90, 5.44, scale=0.82, color=INK)
    draw_robotdog(ax, 9.00, 4.12, scale=0.90, color=INK)
    draw_person(ax, 10.55, 5.15, scale=0.80, color=INK)
    draw_wall(ax, 9.78, 3.78, w=0.36, h=1.02)
    ax.add_patch(Polygon([(9.15, 5.44), (10.48, 5.02), (10.48, 5.42)], closed=True, fc=PURPLE_LIGHT, ec="none", alpha=0.9, zorder=2))
    arrow(ax, (9.20, 5.42), (10.36, 5.22), color=PURPLE, lw=1.2, mutation=8, ls="--")
    arrow(ax, (9.27, 4.18), (9.76, 4.32), color=RED, lw=1.1, mutation=8, ls="--")
    ax.text(8.54, 5.88, "clean source", fontsize=7.8, color=PURPLE, weight="bold")
    ax.text(8.46, 3.57, "masked receiver", fontsize=7.8, color=RED, weight="bold")
    ax.text(10.25, 4.16, "[MISSING]", fontsize=7.8, color=RED, weight="bold")
    ax.text(10.25, 3.85, "pose + Δx, Δy, Δψ", fontsize=7.5, color=RED)
    ax.text(8.47, 3.36, "D <-> G with probability 0.5 / 0.5", fontsize=7.1, color=MUTED)

    # Joint context.
    token_row(ax, 11.13, 5.77, [BLUE] * 3 + [GROUND] * 3 + [PURPLE] * 2, size=0.18, gap=0.045)
    rounded(ax, 11.12, 4.39, 1.49, 1.18, fc=PURPLE_LIGHT, ec=PURPLE, lw=1.4, radius=0.13)
    ax.text(11.87, 5.16, "Joint Qwen3", ha="center", va="center", fontsize=10.2, color=PURPLE, weight="bold")
    ax.text(11.87, 4.82, "both streams", ha="center", va="center", fontsize=7.6, color=MUTED)
    ax.text(11.87, 4.55, "+ 4 pose tokens", ha="center", va="center", fontsize=7.6, color=MUTED)
    arrow(ax, (10.70, 4.97), (11.12, 4.97), color=PURPLE, lw=1.2, mutation=8)

    # JEPA and teacher.
    rounded(ax, 12.94, 4.77, 1.43, 0.92, fc=PURPLE_LIGHT, ec=PURPLE, lw=1.15, radius=0.11)
    ax.text(13.66, 5.40, "Conditional JEPA", ha="center", fontsize=9.1, weight="bold")
    ax.text(13.66, 5.10, "recover receiver", ha="center", fontsize=7.2, color=MUTED)
    ax.text(13.66, 4.86, "fine-grid latent", ha="center", fontsize=7.2, color=MUTED)
    rounded(ax, 12.94, 3.66, 1.43, 0.67, fc=GOLD_LIGHT, ec=GOLD, lw=1.0, radius=0.10, ls="--")
    ax.text(13.66, 4.07, "EMA teacher", ha="center", fontsize=8.5, color=GOLD, weight="bold")
    ax.text(13.66, 3.82, "clean latent", ha="center", fontsize=7.0, color=MUTED)
    arrow(ax, (12.61, 5.16), (12.94, 5.16), color=PURPLE, lw=1.0, mutation=7)
    arrow(ax, (13.66, 4.33), (13.66, 4.77), color=GOLD, lw=1.0, mutation=7, ls="--")

    # Decoder and candidate trajectories.
    rounded(ax, 14.66, 4.40, 1.32, 1.28, fc=PURPLE_LIGHT, ec=PURPLE, lw=1.2, radius=0.12)
    ax.text(15.32, 5.28, "Agent-specific", ha="center", fontsize=8.6, weight="bold")
    ax.text(15.32, 5.00, "multimodal", ha="center", fontsize=8.0, color=MUTED)
    ax.text(15.32, 4.72, "trajectory", ha="center", fontsize=8.0, color=MUTED)
    ax.text(15.32, 4.48, "decoder", ha="center", fontsize=8.0, color=MUTED)
    arrow(ax, (14.37, 5.16), (14.66, 5.16), color=PURPLE, lw=1.0, mutation=7)

    # K trajectory modes and logits.
    draw_robotdog(ax, 16.25, 5.05, scale=0.68, color=INK)
    endpoints = [(17.27, 5.77), (17.35, 5.36), (17.30, 4.86), (17.17, 4.43)]
    mode_colors = [PURPLE, BLUE, ORANGE, GROUND]
    for idx, (end, color) in enumerate(zip(endpoints, mode_colors)):
        trajectory(
            ax,
            [(16.46, 5.04), (16.72, 5.20 + 0.12 * (1 - idx)), end],
            color,
            lw=1.55,
            alpha=0.95 if idx == 0 else 0.72,
        )
        ax.add_patch(Rectangle((16.31 + idx * 0.25, 3.83), 0.17, 0.10 + 0.10 * (4 - idx), fc=color, ec="none", alpha=0.85, zorder=5))
    ax.text(16.73, 5.95, "K=4 trajectory modes", ha="center", fontsize=7.1, color=PURPLE, weight="bold")
    ax.text(16.78, 3.84, "mode logits", ha="center", fontsize=6.8, color=MUTED)
    label_chip(ax, 11.12, 3.50, 0.84, r"$L_{coop}$", PURPLE, PURPLE_LIGHT, fs=7.7)
    label_chip(ax, 12.03, 3.50, 0.84, r"$L_{mode}$", PURPLE, PURPLE_LIGHT, fs=7.7)
    label_chip(ax, 14.47, 3.50, 0.84, r"$L_{JEPA}$", GOLD, GOLD_LIGHT, fs=7.7)
    label_chip(ax, 15.38, 3.50, 1.18, r"$L_{belief/unc}$", PURPLE, PURPLE_LIGHT, fs=7.3)
    ax.text(11.12, 3.31, "receiver-frame best-of-K supervision + smoothness / kinematics / diversity", fontsize=7.0, color=MUTED)

    arrow(ax, (7.84, 5.05), (8.16, 5.05), color=PURPLE, lw=1.5, mutation=9)
    ax.text(7.95, 5.26, "joint context", ha="center", fontsize=6.8, color=PURPLE, weight="bold")

    # ------------------------------------------------------------------
    # Right bottom: online routing and control.
    # ------------------------------------------------------------------
    rounded(ax, 8.16, 0.38, 9.56, 2.61, fc="#FBFEFD", ec=TEAL, lw=1.6, radius=0.22)
    panel_title(ax, 8.42, 2.72, "4", "Verified-visibility routing and closed-loop control", TEAL)
    ax.text(8.47, 2.45, "inputs: target-match probability + SELF / COOPERATIVE trajectories", ha="left", va="center", fontsize=6.9, color=MUTED)
    ax.text(17.42, 2.45, "inference: no synthetic corruption", ha="right", va="center", fontsize=7.5, color=TEAL, style="italic")

    rounded(ax, 8.43, 0.85, 2.35, 1.43, fc=TEAL_LIGHT, ec=TEAL, lw=1.1, radius=0.12)
    ax.text(9.60, 2.04, "Visibility hysteresis", ha="center", fontsize=9.8, weight="bold", color=TEAL)
    ax.text(9.60, 1.67, "YOLO valid/confident", ha="center", fontsize=8.1, color=INK)
    ax.text(9.60, 1.39, "AND  LLM VERIFY accepted", ha="center", fontsize=8.1, color=INK)
    ax.text(9.60, 1.08, "separate enter/exit + 2-frame confirmation", ha="center", fontsize=6.9, color=MUTED)

    # Four routing-state pills.
    states = [
        ("both visible", "SELF / SELF", BLUE, BLUE_LIGHT),
        ("one invisible", "SELF / COOP", PURPLE, PURPLE_LIGHT),
        ("both lost <=3", "BELIEF hold", GOLD, GOLD_LIGHT),
        ("prolonged loss", "+/-30 deg SEARCH", TEAL, TEAL_LIGHT),
    ]
    for idx, (condition, action_name, color, fc) in enumerate(states):
        yy = 2.12 - idx * 0.42
        rounded(ax, 11.08, yy - 0.25, 2.62, 0.32, fc=fc, ec=color, lw=0.9, radius=0.07)
        ax.text(11.20, yy - 0.09, condition, ha="left", va="center", fontsize=7.2, color=MUTED)
        ax.text(13.56, yy - 0.09, action_name, ha="right", va="center", fontsize=7.3, color=color, weight="bold")
    arrow(ax, (10.78, 1.58), (11.08, 1.58), color=TEAL, lw=1.0, mutation=7)

    rounded(ax, 13.98, 0.88, 1.38, 1.37, fc="#F4F7F9", ec="#607784", lw=1.0, radius=0.11)
    ax.text(14.67, 2.01, "selected", ha="center", fontsize=8.5, weight="bold")
    ax.text(14.67, 1.72, "10 × [x,y,yaw]", ha="center", fontsize=7.4, color=MUTED)
    ax.text(14.67, 1.41, "dog y adapter", ha="center", fontsize=7.2, color=MUTED)
    ax.text(14.67, 1.13, "inverse fixed-dt", ha="center", fontsize=7.2, color=MUTED)
    arrow(ax, (13.70, 1.56), (13.98, 1.56), color=TEAL, lw=1.0, mutation=7)

    rounded(ax, 15.65, 0.88, 1.73, 1.37, fc=TEAL_LIGHT, ec=TEAL, lw=1.1, radius=0.11)
    ax.text(16.52, 2.00, "Embodied actions", ha="center", fontsize=9.0, weight="bold", color=TEAL)
    draw_drone(ax, 16.13, 1.52, scale=0.55, color=INK)
    draw_robotdog(ax, 16.91, 1.49, scale=0.59, color=INK)
    ax.text(16.52, 1.04, "bbox residual + 2.5 m/s cap", ha="center", fontsize=6.8, color=MUTED)
    arrow(ax, (15.36, 1.56), (15.65, 1.56), color=TEAL, lw=1.0, mutation=7)

    # Temporal loop to reinforce closed-loop behavior without cluttering panels.
    arrow(
        ax,
        (17.38, 1.02),
        (17.56, 2.36),
        color=TEAL,
        lw=1.0,
        mutation=7,
        connection="arc3,rad=-0.42",
        ls="--",
    )
    ax.text(17.51, 1.69, "next RGB / pose", rotation=83, ha="center", va="center", fontsize=6.8, color=TEAL)

    # Explicitly merge the two model outputs into the online router.  The bus
    # stays outside the functional panels so it does not obscure their content.
    ax.plot([17.43, 17.83], [7.12, 7.12], color=BLUE, lw=1.2, zorder=8)
    ax.plot([17.43, 17.83], [4.86, 4.86], color=PURPLE, lw=1.2, zorder=8)
    ax.plot([17.83, 17.83], [7.12, 3.10], color="#71808B", lw=1.1, zorder=8)
    ax.add_patch(Circle((17.83, 7.12), 0.035, fc=BLUE, ec="white", lw=0.5, zorder=9))
    ax.add_patch(Circle((17.83, 4.86), 0.035, fc=PURPLE, ec="white", lw=0.5, zorder=9))
    arrow(ax, (17.83, 3.10), (17.58, 2.99), color=TEAL, lw=1.2, mutation=8, z=9)
    ax.text(17.93, 5.22, "trajectory bus", rotation=90, ha="center", va="center", fontsize=6.5, color=MUTED)

    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(STEM.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    main()
