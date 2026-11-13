"""Charts of the README: training curves and evaluation results.

    python -m abcgpt.plots [--output_dir docs]

Reads `trainer_state.json` in the last checkpoint of each training run (the same records `trl` prints
while it trains) and the `tunes.jsonl` files written by `abcgpt.evaluate`.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from trl import TrlParser  # noqa: E402

from abcgpt.evaluate import REWARDS  # noqa: E402

# one color per checkpoint in every chart (a palette checked for color-blind readers)
COLORS = {"pretrain": "#2a78d6", "sft": "#eb6834", "grpo": "#1baf7a"}
LABELS = {"pretrain": "pretrain", "sft": "SFT", "grpo": "GRPO"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "savefig.dpi": 150, "savefig.bbox": "tight",
})


@dataclass
class PlotsConfig:
    pretrain_dir: str = "models/gpt2-abc-pretrain"
    finetune_dir: str = "models/gpt2-abc-sft"
    grpo_dir: str = "models/gpt2-abc-grpo"
    eval_dir: str = "outputs"  # output_dir of abcgpt.evaluate
    output_dir: str = "docs"


def history(run_dir: str) -> list[dict]:
    last = max(Path(run_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return json.loads((last / "trainer_state.json").read_text())["log_history"]


def mean_se(values: list[float]) -> tuple[float, float]:
    m = sum(values) / len(values)
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1) / len(values))


def losses(cfg: PlotsConfig, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    runs = (("pretrain", cfg.pretrain_dir, "Pretraining: loss on the whole tune"),
            ("sft", cfg.finetune_dir, "Finetuning: loss on the body only"))
    for ax, (name, run, title) in zip(axes, runs):
        h = history(run)
        train = [(x["epoch"], x["loss"]) for x in h if "loss" in x]
        val = [(x["epoch"], x["eval_loss"]) for x in h if "eval_loss" in x]
        ax.plot(*zip(*train), color=COLORS[name], lw=1, alpha=0.55, label="training")
        ax.plot(*zip(*val), color=COLORS[name], lw=2, marker="o", ms=5, label="validation")
        ax.annotate(f"{val[-1][1]:.3f}", val[-1], xytext=(6, 6), textcoords="offset points", color=INK)
        ax.set(title=title, xlabel="epoch", ylabel="cross-entropy loss")
        ax.legend(loc="upper right")
    axes[0].set_ylim(0.9, 2.0)  # the first steps start at 5.8: cut them so that the epochs stay readable
    axes[0].set_xticks(range(4))
    fig.savefig(path)
    plt.close(fig)


def grpo(cfg: PlotsConfig, path: Path, window: int = 5) -> None:
    h = [x for x in history(cfg.grpo_dir) if "reward" in x]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True)
    for ax, name in zip(axes.flat, REWARDS):
        steps, values = [x["step"] for x in h], [x[f"rewards/{name}/mean"] for x in h]
        smooth = [sum(values[max(0, i - window + 1) : i + 1]) / len(values[max(0, i - window + 1) : i + 1])
                  for i in range(len(values))]
        ax.plot(steps, values, color=COLORS["grpo"], lw=1, alpha=0.45)
        ax.plot(steps, smooth, color=COLORS["grpo"], lw=2)
        first, last = sum(values[:window]) / window, sum(values[-window:]) / window
        ax.set_title(f"{name}: {first:.3f} → {last:.3f}")
        ax.set_ylabel("mean reward")
    for ax in axes[1]:
        ax.set_xlabel("GRPO step (32 tunes each)")
    every = h[1]["step"] - h[0]["step"]  # trl averages the metrics of the steps between two logs
    fig.suptitle(f"Mean reward over {every} steps (thin) and over {window * every} steps (thick); in the titles, "
                 f"first {window * every} steps → last {window * every}", x=0.01, ha="left", color=MUTED)
    fig.savefig(path)
    plt.close(fig)


def bars(ax, groups: list[str], series: dict[str, list[tuple[float, float]]], fmt: str) -> None:
    width = 0.8 / len(series)
    for j, (name, values) in enumerate(series.items()):
        xs = [i + (j - (len(series) - 1) / 2) * width for i in range(len(groups))]
        ax.bar(xs, [v for v, _ in values], width - 0.02, yerr=[e for _, e in values], color=COLORS[name],
               label=LABELS[name], error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
        for x, (v, e) in zip(xs, values):
            ax.text(x, v + e + 0.015, fmt.format(v), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(groups)), groups)
    ax.grid(axis="x", visible=False)


def evaluation(cfg: PlotsConfig, path: Path) -> None:
    scores, adherence = {}, {}
    for d in sorted(Path(cfg.eval_dir).iterdir(), key=lambda p: list(COLORS).index(p.name.split("-")[-1])):
        rows = [json.loads(line) for line in (d / "tunes.jsonl").open(encoding="utf-8")]
        n = len(rows)
        exported = len(list(d.glob("tune_*.musicxml"))) / n
        scores[d.name.split("-")[-1]] = [mean_se([r[name] for r in rows]) for name in REWARDS] + [
            (exported, math.sqrt(exported * (1 - exported) / n))]
        same = [[], [], []]
        for r in rows:
            want, got = r["abc"].split("\n", 1)[0].lstrip("%").split(), r["control_code"].split()
            same[0].append(float(want[:1] == got[:1]))
            same[1].append(float([t for t in want if t[0] in "SB"] == [t for t in got if t[0] in "SB"]))
            same[2].append(float(want == got))
        adherence[d.name.split("-")[-1]] = [mean_se(s) for s in same]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [5, 3]})
    bars(axes[0], [*REWARDS, "exports to\nMusicXML"], scores, "{:.2f}")
    axes[0].set(title=f"Rewards of {n} generated tunes (mean ± standard error)", ylim=(0, 1.15), ylabel="score")
    bars(axes[1], ["same number\nof sections", "same sections\nand bars", "identical\ncontrol code"], adherence,
         "{:.0%}")
    axes[1].set(title="Control code of the tune vs the prompt", ylim=(0, 1.15), ylabel="share of tunes")
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    fig.savefig(path)
    plt.close(fig)


def main(cfg: PlotsConfig) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    losses(cfg, out / "losses.png")
    grpo(cfg, out / "grpo.png")
    evaluation(cfg, out / "eval.png")
    print(f"charts -> {out}/losses.png, grpo.png, eval.png")


if __name__ == "__main__":
    (cfg,) = TrlParser((PlotsConfig,)).parse_args_and_config()
    main(cfg)
