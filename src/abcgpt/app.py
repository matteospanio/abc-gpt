"""Stage 7: Gradio demo. Generates a tune, then draws and plays the score in the browser with abcjs.

    python -m abcgpt.app --config configs/app.yaml [--model_path models/gpt2-abc-sft]
"""

import os
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
# draws the score, then abcjs synthesizes it into a WAV for the browser's own <audio> player
# (abcjs's player buttons are tiny and vanish in Gradio's dark theme; its soundfont comes from the abcjs site)
DRAW = """async (abc) => {
  const tune = ABCJS.renderAbc('score', abc, { responsive: 'resize' })[0];
  const audio = document.getElementById('audio');
  if (audio.src) URL.revokeObjectURL(audio.src);
  audio.removeAttribute('src');
  const synth = new ABCJS.synth.CreateSynth();
  await synth.init({ visualObj: tune, options: {} });  // prime() fails without options (it reads options.swing)
  await synth.prime();
  audio.src = synth.download();
  return [];
}"""


@dataclass
class AppConfig:
    model_path: str = "models/gpt2-abc-grpo"
    max_new_tokens: int = 512
    server_name: str = "127.0.0.1"
    server_port: int = 7860


def build(cfg: AppConfig) -> gr.Blocks:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # shortcut: half the CPUs, measured on one laptop (8 cores, 4 of them performance cores): 2-4 threads give
        # ~180 tok/s and 8 give ~110, a model this small gains nothing from more threads; measure on other CPUs
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path).to(device).eval()
    # fork the music21 workers of the syntax reward now, before Gradio opens its sockets: workers forked later
    # inherit the browser's connections, which then never close, and the next Compose hangs
    rewards.syntax(completions=["|:A2 B2:|\n"], prompts=["X:1\nL:1/8\nM:2/4\nK:D\n"])

    @torch.no_grad()
    def compose(meter: str, key: str, control: str, temperature: float):
        """Write an Irish folk tune in ABC notation with a small GPT-2, and export it to MusicXML.

        Args:
            meter: time signature, the ABC M: field (e.g. 6/8 for a jig, 4/4 for a reel, 3/4, 2/4, 9/8)
            key: key and mode, the ABC K: field (e.g. D, G, Emin, Amin, Dmix, Ador)
            control: structure code, e.g. "S:2 B:8 E:5 B:8" = 2 sections of 8 bars, the 2nd 50% similar to the 1st (E:0-10)
            temperature: sampling temperature, from 0.5 (safer, more repetitive) to 1.5 (more varied, more mistakes)

        Returns:
            the tune in ABC; the MusicXML file, or None if music21 cannot read the tune; the tune's own control code
        """
        # Gradio builds the MCP tool description from this docstring: one line per parameter, Google style
        prompt = f"%{control.strip()}\nX:1\nL:1/8\nM:{meter}\nK:{key}\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs, do_sample=True, temperature=temperature, max_new_tokens=cfg.max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
        body = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        abc = prompt + body
        path = Path(tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False).name)
        # check the tune with the syntax reward first: it has a timeout, and music21 never returns on some tunes
        ok = rewards.syntax(completions=[body], prompts=[prompt])[0] == 1.0 and export_musicxml(abc, path)
        code = " ".join(rewards.control_code(abc))
        score = rewards.structure(completions=[body], prompts=[prompt])[0]
        return abc, str(path) if ok else None, f"{code}   (structure reward {score:.2f})"

    with gr.Blocks(title="abc-gpt") as demo:
        gr.Markdown("## abc-gpt: Irish tunes from a small GPT-2")
        with gr.Row():
            meter = gr.Dropdown(["4/4", "6/8", "3/4", "2/4", "9/8"], value="6/8", label="Meter (M:)")
            key = gr.Dropdown(["D", "G", "A", "C", "F", "Emin", "Amin", "Bmin", "Dmix", "Ador"], value="D", label="Key (K:)")
            control = gr.Textbox(value="S:2 B:8 E:5 B:8", label="Control code (sections, bars, similarity)")
            temperature = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Temperature")
        button = gr.Button("Compose", variant="primary")
        score = gr.HTML('<div id="score"></div><audio id="audio" controls style="width: 100%"></audio>')
        abc = gr.Textbox(label="ABC", lines=8)
        obtained = gr.Textbox(label="Control code of the generated tune")
        musicxml = gr.File(label="MusicXML")
        button.click(compose, [meter, key, control, temperature], [abc, musicxml, obtained]).then(
            None, abc, None, js=DRAW)
    return demo


if __name__ == "__main__":
    (cfg,) = TrlParser((AppConfig,)).parse_args_and_config()
    build(cfg).launch(server_name=cfg.server_name, server_port=cfg.server_port, head=ABCJS, mcp_server=True)
