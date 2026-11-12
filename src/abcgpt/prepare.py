"""Stage 1: data, tokenizer and a randomly initialised GPT-2.

    python -m abcgpt.prepare --config configs/prepare.yaml [--vocab_size 4096 ...]

Writes:
- `model_dir`: GPT-2 with random weights plus the BPE tokenizer, the starting point of `trl sft`;
- `data_dir/finetune/{train,validation}.jsonl`: prompt (control code + header) and completion (tune body);
- `data_dir/grpo/train.jsonl`: prompts for GRPO;
- `data_dir/eval/prompts.jsonl`: validation prompts for `abcgpt.evaluate`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import TrlParser

EOS, PAD = "<|endoftext|>", "<pad>"


@dataclass
class PrepareConfig:
    dataset_name: str = "sander-wood/irishman"
    text_field: str = "abc notation"
    control_field: str = "control code"
    vocab_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    n_positions: int = 1024
    model_dir: str = "models/gpt2-abc-init"
    data_dir: str = "data"
    num_grpo_prompts: int = 5000
    num_eval_prompts: int = 200
    seed: int = 42


def split_tune(abc: str) -> tuple[str, str] | None:
    """Header (up to and including the K: line) and body; None if there is no K: field."""
    start = abc.find("K:") if abc.startswith("K:") else abc.find("\nK:")
    if start < 0:
        return None
    end = abc.find("\n", start + 1)
    return (abc, "") if end < 0 else (abc[: end + 1], abc[end + 1 :])


def to_prompt_completion(example: dict, cfg: PrepareConfig) -> dict:
    header, body = split_tune(example[cfg.text_field]) or ("", "")
    # the control code goes in an ABC comment line, which ABC parsers ignore
    control = " ".join(example[cfg.control_field].split())  # the field spans several lines: keep one
    prompt = f"%{control}\n{header}" if header else ""
    return {"prompt": prompt, "completion": body}


def train_tokenizer(texts, vocab_size: int) -> PreTrainedTokenizerFast:
    """Byte-level BPE as in GPT-2, trained on ABC text."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOS, PAD],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token=EOS, eos_token=EOS, pad_token=PAD)


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(cfg: PrepareConfig) -> None:
    data = load_dataset(cfg.dataset_name)

    tokenizer = train_tokenizer(data["train"][cfg.text_field], cfg.vocab_size)
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=cfg.n_positions, n_embd=cfg.n_embd,
                        n_layer=cfg.n_layer, n_head=cfg.n_head, bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                        n_ctx=cfg.n_positions)  # n_ctx is no longer saved by default, but llama.cpp's converter reads it
    model = GPT2LMHeadModel(config)
    model.save_pretrained(cfg.model_dir)
    tokenizer.save_pretrained(cfg.model_dir)
    print(f"GPT-2 with {model.num_parameters() / 1e6:.1f}M parameters and {len(tokenizer)} tokens -> {cfg.model_dir}")

    data_dir = Path(cfg.data_dir)
    pairs = data.map(to_prompt_completion, fn_kwargs={"cfg": cfg}, remove_columns=data["train"].column_names)
    pairs = pairs.filter(lambda ex: ex["prompt"] != "" and ex["completion"].strip() != "")
    for split in ("train", "validation"):
        write_jsonl(pairs[split], data_dir / "finetune" / f"{split}.jsonl")

    grpo = pairs["train"].shuffle(seed=cfg.seed).select(range(min(cfg.num_grpo_prompts, len(pairs["train"]))))
    write_jsonl(({"prompt": p} for p in grpo["prompt"]), data_dir / "grpo" / "train.jsonl")
    evaluation = pairs["validation"].shuffle(seed=cfg.seed)
    evaluation = evaluation.select(range(min(cfg.num_eval_prompts, len(evaluation))))
    write_jsonl(({"prompt": p} for p in evaluation["prompt"]), data_dir / "eval" / "prompts.jsonl")
    print(f"finetune: {len(pairs['train'])} + {len(pairs['validation'])} tunes, "
          f"grpo: {len(grpo)} prompts, eval: {len(evaluation)} prompts -> {data_dir}")


if __name__ == "__main__":
    (cfg,) = TrlParser((PrepareConfig,)).parse_args_and_config()
    main(cfg)
