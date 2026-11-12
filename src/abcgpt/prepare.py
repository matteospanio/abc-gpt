"""Stage 1: BPE tokenizer and a randomly initialised GPT-2.

    python -m abcgpt.prepare --config configs/prepare.yaml [--vocab_size 4096 ...]

Writes `model_dir`: GPT-2 with random weights plus the tokenizer, the starting point of `trl sft`. The
data stage (`abcgpt.data`) uses the same tokenizer to drop the tunes that do not fit in the context.
"""

from dataclasses import dataclass

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import TrlParser

EOS, PAD = "<|endoftext|>", "<pad>"


@dataclass
class PrepareConfig:
    dataset_name: str = "sander-wood/irishman"
    text_field: str = "abc notation"
    vocab_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    n_positions: int = 1024
    model_dir: str = "models/gpt2-abc-init"


def train_tokenizer(texts, vocab_size: int) -> PreTrainedTokenizerFast:
    """Byte-level BPE as in GPT-2, trained on ABC text."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOS, PAD],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token=EOS, eos_token=EOS, pad_token=PAD)


def main(cfg: PrepareConfig) -> None:
    texts = load_dataset(cfg.dataset_name, split="train")[cfg.text_field]
    tokenizer = train_tokenizer(texts, cfg.vocab_size)
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=cfg.n_positions, n_embd=cfg.n_embd,
                        n_layer=cfg.n_layer, n_head=cfg.n_head, bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                        n_ctx=cfg.n_positions)  # n_ctx is no longer saved by default, but llama.cpp's converter reads it
    model = GPT2LMHeadModel(config)
    model.save_pretrained(cfg.model_dir)
    tokenizer.save_pretrained(cfg.model_dir)
    print(f"GPT-2 with {model.num_parameters() / 1e6:.1f}M parameters and {len(tokenizer)} tokens -> {cfg.model_dir}")


if __name__ == "__main__":
    (cfg,) = TrlParser((PrepareConfig,)).parse_args_and_config()
    main(cfg)
