"""Stage 2: the data pipeline, with DataTrove.

    python -m abcgpt.data --config configs/data.yaml [--tasks 16 --slurm_partition cpu]

For each split of IrishMAN, one DataTrove pipeline:

    Hub reader -> ABCFilter -> pretrain writer -> finetune writer -> sampler -> prompts writer

- `ABCFilter` drops tunes without a body, with symbols that music21 cannot read, or longer than
  `max_tokens` once the control code is added (`trl` would cut them, losing the end of the tune).
  Dropped tunes go to `data_dir/removed/<split>`, with the reason in `metadata.filter_reason`.
- `data_dir/pretrain/<split>`: kept tunes in the original IrishMAN columns, for `trl sft` pretraining.
- `data_dir/finetune/<split>`: prompt (control code as an ABC comment + header) and completion (body).
- a random share of the prompts: `data_dir/grpo` from train, `data_dir/eval` from validation.

Runs here with `tasks` processes, or as one Slurm job array per split if `slurm_partition` is set. A
second run skips the tasks already completed (delete `data_dir/logs` to redo them). The tokenizer comes
from `abcgpt.prepare`, so run that first.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from music21 import abcFormat
from tokenizers import Tokenizer
from trl import TrlParser

from abcgpt.rewards import quiet


@dataclass
class DataConfig:
    dataset_name: str = "sander-wood/irishman"
    text_field: str = "abc notation"
    control_field: str = "control code"
    tokenizer_file: str = "models/gpt2-abc-init/tokenizer.json"
    max_tokens: int = 512           # keep it equal to max_length in pretrain.yaml and finetune.yaml
    data_dir: str = "data"
    grpo_prompts_rate: float = 0.025  # share of the training tunes whose prompt goes to GRPO (~5,300)
    eval_prompts_rate: float = 0.1    # share of the validation tunes whose prompt goes to evaluate (~210)
    tasks: int = 8                  # shards of each split, processed in parallel
    slurm_partition: str | None = None  # set it to run each split as a Slurm job array
    slurm_time: str = "01:00:00"
    seed: int = 42


def split_tune(abc: str) -> tuple[str, str] | None:
    """Header (up to and including the K: line) and body; None if there is no K: field."""
    start = abc.find("K:") if abc.startswith("K:") else abc.find("\nK:")
    if start < 0:
        return None
    end = abc.find("\n", start + 1)
    return (abc, "") if end < 0 else (abc[: end + 1], abc[end + 1 :])


def readable(abc: str) -> bool:
    """music21's ABC tokenizer reads every symbol: on unknown ones it warns and guesses a C note.

    The `syntax` reward parses the whole score, but that is 60 times slower and also rejects some valid
    annotations ("^Slow, with expression"): to clean the data the tokenizer is enough.
    """
    with quiet() as err:
        try:
            handler = abcFormat.ABCHandler()
            handler.process(abc)
        except Exception:  # music21 raises many different exception types on malformed ABC
            return False
    has_notes = any(type(token).__name__ in ("ABCNote", "ABCChord") for token in handler.tokens)
    return has_notes and "WARNING" not in err.getvalue()


class ABCFilter(BaseFilter):
    name = "🎻 ABC"

    def __init__(self, tokenizer_file: str, max_tokens: int, control_field: str, exclusion_writer=None):
        super().__init__(exclusion_writer)
        self.tokenizer_file, self.max_tokens, self.control_field = tokenizer_file, max_tokens, control_field
        self.tokenizer = None  # loaded in each task, not pickled with the pipeline

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        parts = split_tune(doc.text)
        if parts is None or not parts[1].strip():
            return False, "no_body"
        if not readable(doc.text):
            return False, "unreadable"
        if self.tokenizer is None:
            self.tokenizer = Tokenizer.from_file(self.tokenizer_file)
        finetune_text = prompt(doc, self.control_field) + parts[1]  # the longest form the model sees
        if len(self.tokenizer.encode(finetune_text).ids) + 1 > self.max_tokens:  # + the end-of-text token
            return False, "too_long"
        return True


def prompt(doc: Document, control_field: str) -> str:
    """Control code in an ABC comment line (parsers ignore it), then the header."""
    control = " ".join(doc.metadata[control_field].split())  # the field spans several lines: keep one
    return f"%{control}\n{split_tune(doc.text)[0]}"


def main(cfg: DataConfig) -> None:
    text_field, control_field = cfg.text_field, cfg.control_field

    # adapters: how each writer turns a Document into a JSON line
    def original(self, doc):
        return {text_field: doc.text, control_field: doc.metadata[control_field]}

    def prompt_completion(self, doc):
        return {"prompt": prompt(doc, control_field), "completion": split_tune(doc.text)[1]}

    def prompt_only(self, doc):
        return {"prompt": prompt(doc, control_field)}

    data = Path(cfg.data_dir)
    for split, prompts_dir, rate in (("train", data / "grpo", cfg.grpo_prompts_rate),
                                     ("validation", data / "eval", cfg.eval_prompts_rate)):
        def writer(folder, adapter=None):
            folder.mkdir(parents=True, exist_ok=True)  # here, not in the Slurm tasks: see the NFS note below
            return JsonlWriter(str(folder), output_filename="${rank}.jsonl", compression=None, adapter=adapter)

        pipeline = [
            HuggingFaceDatasetReader(cfg.dataset_name, dataset_options={"split": split}, text_key=text_field),
            ABCFilter(cfg.tokenizer_file, cfg.max_tokens, control_field, exclusion_writer=writer(data / "removed" / split)),
            writer(data / "pretrain" / split, original),
            writer(data / "finetune" / split, prompt_completion),
            SamplerFilter(rate=rate, seed=cfg.seed),
            writer(prompts_dir, prompt_only),
        ]
        logging_dir = str(data / "logs" / split)
        if cfg.slurm_partition:
            # tasks on different nodes would create these folders at the same moment, and on NFS that can raise
            # FileExistsError even with exist_ok=True (a node does not see yet the folder made by another one)
            for folder in ("logs", "stats", "completions"):
                Path(logging_dir, folder).mkdir(parents=True, exist_ok=True)
            executor = SlurmPipelineExecutor(pipeline, tasks=cfg.tasks, time=cfg.slurm_time,
                                             partition=cfg.slurm_partition, logging_dir=logging_dir,
                                             qos=None,  # DataTrove defaults to "normal", which many clusters lack: use the account's default
                                             job_name=f"data-{split}", venv_path=f"{sys.prefix}/bin/activate")
        else:
            executor = LocalPipelineExecutor(pipeline, tasks=cfg.tasks, logging_dir=logging_dir)
        executor.run()


if __name__ == "__main__":
    # the pipeline must come from abcgpt.data, not __main__: dill pickles __main__'s globals by reference,
    # and in the Slurm job __main__ is launch_pickled_pipeline (NameError: name 'split_tune' is not defined)
    from abcgpt.data import DataConfig, main
    (cfg,) = TrlParser((DataConfig,)).parse_args_and_config()
    main(cfg)
