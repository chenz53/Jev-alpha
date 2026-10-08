"""Render the documented Jev accuracy comparison using existing Matplotlib."""

import argparse
import json
import math
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/assets/jev-comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/jev-comparison.svg"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = data["tasks"]
    angles = [2 * math.pi * i / len(rows) for i in range(len(rows))]
    angles += angles[:1]
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "jev-alpha"}
    )
    fig, ax = plt.subplots(figsize=(11, 10), subplot_kw={"projection": "polar"}, facecolor="#f8fafc")
    ax.set_facecolor("#ffffff")
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    for key, name, color, style in [
        ("ours_accuracy", "Jev-alpha · Gemma + CE + KL 0.1", "#0369a1", "-"),
        ("jev_accuracy", "Jev · published in the Kev repository", "#c2410c", "--"),
    ]:
        values = [100 * row[key] for row in rows]
        ax.plot(angles, values + values[:1], color=color, linewidth=2.5, linestyle=style, marker="o", label=name)
        ax.fill(angles, values + values[:1], color=color, alpha=0.07)
    ax.set_xticks(angles[:-1], [r["label"] for r in rows])
    ax.tick_params(axis="x", pad=15)
    # Start at zero so the radial scale does not exaggerate small differences.
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="#64748b", fontsize=9)
    ax.set_rlabel_position(15)
    ax.grid(color="#cbd5e1", linewidth=0.7)
    ax.spines["polar"].set_color("#cbd5e1")
    fig.suptitle("Decision accuracy across 11 tasks", fontsize=22, fontweight="bold", y=0.97, color="#0f172a")
    fig.text(
        0.5,
        0.923,
        "Different evaluation samples · descriptive comparison, not a paired benchmark",
        ha="center",
        color="#475569",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=11)
    fig.text(
        0.5,
        0.065,
        "Jev-alpha: three-seed mean; 100 final questions/task. Jev: 80 or 150 published questions/task.",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.042,
        "Sources: jaredpalmer/kev @ fe64b1274ea7 · transfer-v4, decision-v4, breadth-v1, devtools-v1.",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.022,
        "Exact values, source URLs, and hashes: docs/assets/jev-comparison.json · Prepared 2026-10-08",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.84, bottom=0.22, left=0.14, right=0.86)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, metadata={"Date": None} if args.output.suffix == ".svg" else None)
    if args.output.suffix == ".svg":
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
