"""Stage 5: compare checkpoints on the GRPO rewards and export the generated tunes to MusicXML.

    python -m abcgpt.evaluate --config configs/evaluate.yaml [--num_samples 20 ...]

For each checkpoint, generates tunes from the same validation prompts, prints the mean of each reward
and writes `output_dir/<checkpoint>/tunes.jsonl` plus one MusicXML file per tune that parses.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from music21 import converter
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import TrlParser

from abcgpt import rewards

REWARDS = ("syntax", "meter", "note_order", "structure")


@dataclass
class EvaluateConfig:
    checkpoints: list[str] = field(default_factory=lambda: ["models/gpt2-abc-pretrain", "models/gpt2-abc-sft",
                                                             "models/gpt2-abc-grpo"])
    prompts_dir: str = "data/eval"  # JSONL files written by abcgpt.data
    num_samples: int = 100
    batch_size: int = 25
    max_new_tokens: int = 512
    temperature: float = 1.0
    output_dir: str = "outputs"
    seed: int = 42


@torch.no_grad()
def generate(model_path: str, prompts: list[str], cfg: EvaluateConfig) -> list[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
    completions = []
    for i in range(0, len(prompts), cfg.batch_size):
        batch = tokenizer(prompts[i : i + cfg.batch_size], return_tensors="pt", padding=True).to(device)
        out = model.generate(**batch, do_sample=True, temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
        completions += tokenizer.batch_decode(out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
    return completions


def export_musicxml(abc: str, path: Path) -> bool:
    try:
        with rewards.quiet():
            converter.parse(abc, format="abc").write("musicxml", fp=path)
    except Exception:  # an unparsable tune is simply not exported
        return False
    return True


def main(cfg: EvaluateConfig) -> None:
    prompts = [json.loads(line)["prompt"] for path in sorted(Path(cfg.prompts_dir).glob("*.jsonl"))
               for line in path.open(encoding="utf-8")]
    prompts = (prompts * (cfg.num_samples // len(prompts) + 1))[: cfg.num_samples]

    rows = []
    for checkpoint in cfg.checkpoints:
        if not Path(checkpoint).exists():
            print(f"skipping {checkpoint}: not found")
            continue
        set_seed(cfg.seed)
        completions = generate(checkpoint, prompts, cfg)
        scores = {name: getattr(rewards, name)(completions=completions, prompts=prompts) for name in REWARDS}

        out_dir = Path(cfg.output_dir) / Path(checkpoint).name
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("tune_*.musicxml"):  # left by a previous run
            old.unlink()
        exported = 0
        with (out_dir / "tunes.jsonl").open("w", encoding="utf-8") as f:
            for i, (prompt, completion) in enumerate(zip(prompts, completions)):
                abc = rewards.tune(prompt, completion)
                row_scores = {name: values[i] for name, values in scores.items()}
                code = " ".join(rewards.control_code(abc))  # what the tune has, to compare with the prompt's
                f.write(json.dumps({"abc": abc, "control_code": code, **row_scores}, ensure_ascii=False) + "\n")
                if row_scores["syntax"] == 1.0:
                    exported += export_musicxml(abc, out_dir / f"tune_{i:03d}.musicxml")
        means = {name: sum(values) / len(values) for name, values in scores.items()}
        rows.append((Path(checkpoint).name, means, exported))

    print("\n| checkpoint | " + " | ".join(REWARDS) + " | MusicXML files |\n|:--|" + "--:|" * (len(REWARDS) + 1))
    for name, means, exported in rows:
        print(f"| {name} | " + " | ".join(f"{means[r]:.3f}" for r in REWARDS) + f" | {exported} |")


if __name__ == "__main__":
    (cfg,) = TrlParser((EvaluateConfig,)).parse_args_and_config()
    main(cfg)
