"""Stage 7: Gradio demo. Generates a tune and draws the score in the browser with abcjs.

    python -m abcgpt.app --config configs/app.yaml [--model_path models/gpt2-abc-sft]
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import TrlParser

from abcgpt import rewards
from abcgpt.evaluate import export_musicxml

ABCJS = '<script src="https://cdn.jsdelivr.net/npm/abcjs@6.4.4/dist/abcjs-basic-min.js"></script>'
DRAW = "(abc) => { ABCJS.renderAbc('score', abc, { responsive: 'resize' }); return []; }"


@dataclass
class AppConfig:
    model_path: str = "models/gpt2-abc-grpo"
    max_new_tokens: int = 384
    server_name: str = "127.0.0.1"
    server_port: int = 7860


def build(cfg: AppConfig) -> gr.Blocks:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path).to(device).eval()

    @torch.no_grad()
    def compose(meter: str, key: str, control: str, temperature: float):
        prompt = f"%{control.strip()}\nX:1\nL:1/8\nM:{meter}\nK:{key}\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs, do_sample=True, temperature=temperature, max_new_tokens=cfg.max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
        body = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        abc = prompt + body
        path = Path(tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False).name)
        # check the tune with the syntax reward first: it has a timeout, and music21 never returns on some tunes
        ok = rewards.syntax(completions=[body], prompts=[prompt])[0] == 1.0 and export_musicxml(abc, path)
        return abc, str(path) if ok else None  # the tune is shown anyway, only the MusicXML file is missing

    with gr.Blocks(title="abc-gpt") as demo:
        gr.Markdown("## abc-gpt: Irish tunes from a small GPT-2")
        with gr.Row():
            meter = gr.Dropdown(["4/4", "6/8", "3/4", "2/4", "9/8"], value="6/8", label="Meter (M:)")
            key = gr.Dropdown(["D", "G", "A", "C", "F", "Emin", "Amin", "Bmin", "Dmix", "Ador"], value="D", label="Key (K:)")
            control = gr.Textbox(value="S:2 B:8 E:5 B:8", label="Control code (sections, bars, similarity)")
            temperature = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Temperature")
        button = gr.Button("Compose", variant="primary")
        score = gr.HTML('<div id="score"></div>')
        abc = gr.Textbox(label="ABC", lines=8)
        musicxml = gr.File(label="MusicXML")
        button.click(compose, [meter, key, control, temperature], [abc, musicxml]).then(None, abc, None, js=DRAW)
    return demo


if __name__ == "__main__":
    (cfg,) = TrlParser((AppConfig,)).parse_args_and_config()
    build(cfg).launch(server_name=cfg.server_name, server_port=cfg.server_port, head=ABCJS)
