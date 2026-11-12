"""Stage 6: convert a checkpoint to GGUF for llama.cpp.

    LLAMA_CPP_DIR=/path/to/llama.cpp python -m abcgpt.export_gguf --config configs/export.yaml

Uses llama.cpp's own converter (convert_hf_to_gguf.py). The converter recognises BPE tokenizers by
hashing how they encode a test string, and a tokenizer trained by us is not in its list: since ours
uses the same byte-level pre-tokenizer as GPT-2, we tell the converter so.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from trl import TrlParser


@dataclass
class ExportConfig:
    model_path: str = "models/gpt2-abc-grpo"
    outfile: str = "models/gpt2-abc-grpo.gguf"
    outtype: str = "f16"  # f32, f16, bf16 or q8_0
    llama_cpp_dir: str = field(default_factory=lambda: os.environ.get("LLAMA_CPP_DIR", "llama.cpp"))


def main(cfg: ExportConfig) -> None:
    llama_cpp = Path(cfg.llama_cpp_dir).resolve()
    if not (llama_cpp / "convert_hf_to_gguf.py").exists():
        sys.exit(f"convert_hf_to_gguf.py not found in {llama_cpp}: clone https://github.com/ggml-org/llama.cpp "
                 "and set LLAMA_CPP_DIR or --llama_cpp_dir")
    sys.path.insert(0, str(llama_cpp))
    import convert_hf_to_gguf  # noqa: E402  (adds llama.cpp's gguf-py to the path)
    from conversion.base import TextModel  # noqa: E402

    TextModel.get_vocab_base_pre = lambda self, tokenizer: "gpt-2"
    sys.argv = ["convert_hf_to_gguf.py", cfg.model_path, "--outfile", cfg.outfile, "--outtype", cfg.outtype]
    convert_hf_to_gguf.main()


if __name__ == "__main__":
    (cfg,) = TrlParser((ExportConfig,)).parse_args_and_config()
    main(cfg)
