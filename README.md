# abc-gpt

A small GPT-2 (~27M parameters) trained from scratch on Irish folk tunes written in
[ABC notation](https://abcnotation.com/wiki/abc:standard:v2.1), then finetuned with SFT and with
GRPO using music-aware rewards. It is companion code for a presentation on the Hugging Face ecosystem,
so every stage uses a library from that stack:

| Stage | What it does | Hugging Face tool |
|:--|:--|:--|
| `prepare` | downloads the data, trains a BPE tokenizer, builds a GPT-2 with random weights | `datasets`, `tokenizers`, `transformers` |
| `pretrain` | next-token prediction on raw ABC text | `trl sft` |
| `finetune` | control code + header → tune body (loss on the completion only); LoRA optional | `trl sft`, `peft` |
| `grpo` | reinforcement learning with three rewards computed with music21 | `trl grpo` |
| `evaluate` | compares the checkpoints on the rewards, exports the tunes to MusicXML | `transformers` |
| `export` | GGUF file for llama.cpp | `convert_hf_to_gguf.py` |
| `demo` | web app that composes a tune and draws the score | `gradio` |

Data: [IrishMAN](https://huggingface.co/datasets/sander-wood/irishman) (214,122 training and 2,162
validation tunes, MIT licence). Each tune comes with a *control code* (number of sections, bars per
section, similarity between sections) that the finetuned model learns to follow.

## Setup

```bash
uv sync --extra demo        # Python 3.12, PyTorch with CUDA from PyPI
uv run pytest               # reward tests, a few seconds
```

## Running

```bash
./run.sh prepare            # ~1 min on CPU: tokenizer, models/gpt2-abc-init, data/
./run.sh pretrain           # models/gpt2-abc-pretrain
./run.sh finetune           # models/gpt2-abc-sft
./run.sh grpo               # models/gpt2-abc-grpo
./run.sh evaluate           # reward table + outputs/<checkpoint>/*.musicxml
./run.sh demo               # http://127.0.0.1:7860
```

`./run.sh STAGE --quick` runs a few hundred steps instead of the full schedule, to check the whole
chain in minutes. Any other argument overrides the stage's YAML file:

```bash
./run.sh pretrain --learning_rate 3e-4 --num_train_epochs 1
./run.sh finetune --use_peft true --lora_r 8
./run.sh grpo --quick --num_generations 4 --max_completion_length 256
```

On a GPU without bfloat16 (before Ampere) add `--bf16 false`.
With `--use_peft true` the output folder holds only the LoRA adapter; `transformers` loads it on top of
the base model named in `adapter_config.json`, so `evaluate` and `demo` read it as it is (GRPO from an
adapter is not tested).

### Expected times on an RTX 3070 (8 GB)

Estimates, not measured: the pipeline was tested end to end only on CPU with a tiny model.

| Stage | Full | `--quick` |
|:--|--:|--:|
| prepare (CPU) | 1 min | — |
| pretrain (3 epochs, ~105M tokens) | ~1 h | ~2 min |
| finetune (1 epoch) | ~20 min | ~1 min |
| grpo (500 steps, 32 tunes per step) | ~45 min | ~2 min |
| evaluate (100 tunes × 3 checkpoints) | ~3 min | ~1 min |

## Configuration

Every stage reads a YAML file in `configs/` and accepts command-line overrides, through the parser
that TRL already uses (`TrlParser`, an extension of `transformers.HfArgumentParser`):

- `pretrain`, `finetune` and `grpo` are plain `trl sft` / `trl grpo` runs: the YAML files hold fields of
  `SFTConfig` / `GRPOConfig` (all of `TrainingArguments`), `ModelConfig` (model path, dtype, LoRA) and
  `ScriptArguments` (dataset). No training loop is written here.
- `prepare`, `evaluate`, `export` and `demo` are our scripts, each with a small dataclass parsed the same
  way (`python -m abcgpt.prepare --config configs/prepare.yaml --vocab_size 4096`).

The values actually used end up in `output_dir/training_args.bin`. Logging is off (`report_to: none`);
set it to `tensorboard` or `trackio` to get the curves.

Hydra would add composition of config groups and sweeps (`-m learning_rate=1e-4,3e-4`); here it would
be a second format next to the one TRL already reads, so it is not used.

## Rewards

`src/abcgpt/rewards.py` holds three functions that `configs/grpo.yaml` lists as dotted paths
(`abcgpt.rewards.syntax`) with weights `[1.0, 1.0, 0.5]`:

- `syntax`: 1 if music21 parses the tune without warnings and finds notes. music21 is lenient (it reads an
  unknown symbol as a C and only warns), so a warning counts as an error.
- `meter`: share of inner bars (not the first, which may be a pickup, nor the last) whose length matches
  `M:` and `L:`. Tuplets are scaled; a short bar next to a repeat or a double bar counts as correct.
- `note_order`: share of notes written in the canonical order accidental–pitch–octave–length
  (`^c'2`, not `c^2` or `2c`).

Parsing with music21 is the slow part, so `syntax` runs in a process pool. Scores on validation tunes and
on corrupted copies:

| | syntax | meter | note_order |
|:--|--:|--:|--:|
| real tunes | 0.997 | 0.951 | 1.000 |
| random characters | — | 0.607 | — |
| some notes removed from bars | — | 0.664 | — |
| length written before the pitch | — | — | 0.951 |
| output of a barely trained tiny model | 0.10 | 0.37 | 0.90 |

`note_order` is a weak signal (most generated notes are already in order): it has the lowest weight.
`meter` is the one that separates real tunes from plausible-looking ones.

## Slurm

```bash
./run.sh prepare                                   # on a login node: needs internet access
sbatch slurm/train.sbatch pretrain                 # then finetune, then grpo
sbatch --gres=gpu:2 slurm/train.sbatch grpo --max_steps 1000
```

`slurm/train.sbatch` uses all the GPUs of one node: it passes `--num_processes` (and `--multi_gpu`) to
`trl`, which hands them to `accelerate launch`. Set partition and account in the `#SBATCH` lines.
Compute nodes often have no internet, so the script sets `HF_HUB_OFFLINE=1` and reads the dataset from
the cache filled by `prepare` (set `HF_HOME` if the cache is on a shared filesystem). The batch sizes in
the YAML files are per GPU.

## GGUF export

```bash
git clone https://github.com/ggml-org/llama.cpp
LLAMA_CPP_DIR=llama.cpp ./run.sh export            # models/gpt2-abc-grpo.gguf
llama.cpp/build/bin/llama-cli -m models/gpt2-abc-grpo.gguf -p $'X:1\nL:1/8\nM:6/8\nK:D\n' -n 256
```

The script calls llama.cpp's own converter. That converter recognises a BPE tokenizer by hashing how it
encodes a test string, and a tokenizer trained here is not in its list. Ours uses the same byte-level
pre-tokenizer as GPT-2, so the script tells the converter so. llama.cpp pins its own versions of
`transformers` and `torch`: if the import fails, run the export in an environment made with
`pip install -r llama.cpp/requirements.txt` plus this package. **Not tested yet.**

## Layout

```
configs/                  one YAML file per stage
src/abcgpt/prepare.py     tokenizer, random GPT-2, finetuning and GRPO data
src/abcgpt/rewards.py     syntax, meter, note_order
src/abcgpt/evaluate.py    reward table and MusicXML export
src/abcgpt/export_gguf.py GGUF conversion
src/abcgpt/app.py         Gradio demo, score drawn by abcjs in the browser
slurm/train.sbatch        one node, all its GPUs
tests/test_rewards.py     real tunes vs corrupted ones
run.sh                    one entry point for all the stages
```

## Related work

- [NotaGen](https://github.com/ElectricAlexis/NotaGen) generates classical scores in ABC with the
  same three phases at a larger scale: pretraining on 1.6M pieces, finetuning on ~9k pieces conditioned
  on period, composer and instrumentation, then reinforcement learning (CLaMP-DPO, which uses a
  music-text model as judge instead of rules). Its model is a hierarchical GPT-2 (one level per bar, one
  per character) with a custom training loop.
- [MuPT](https://huggingface.co/m-a-p/MuPT-v1-8192-190M) is a family of Llama models pretrained on ABC
  (Apache-2.0). Starting `finetune` from MuPT instead of our pretrained model only needs
  `model_name_or_path` and `trust_remote_code: true` in `configs/finetune.yaml`, but the data must
  first be converted to its format (SMT-ABC, with `<n>` in place of newlines).
