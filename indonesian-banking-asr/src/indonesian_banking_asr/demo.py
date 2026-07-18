from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from indonesian_banking_asr.evaluation.router import (
    DEFAULT_BANKING_MODEL,
    transcribe_routed_audio,
)
from indonesian_banking_asr.evaluation.whisper import DEFAULT_MODEL

DEMO_AUDIO_PATH = Path(__file__).resolve().parents[2] / "examples/audio/banking-installment-rp11m.wav"

_CSS = """
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }
.hero { text-align: center; margin: 2.5rem auto 1.5rem; max-width: 760px; }
.hero h1 { font-size: clamp(2rem, 6vw, 3.75rem); letter-spacing: -0.05em; margin-bottom: .5rem; }
.hero p { color: var(--body-text-color-subdued); font-size: 1.05rem; }
.output-card { border: 1px solid var(--border-color-primary); border-radius: 18px; padding: 8px 16px 16px; }
.output-card textarea { font-size: 1.15rem !important; line-height: 1.55 !important; }
.meta-row { gap: 12px; }
footer { display: none !important; }
"""


def build_demo(*, general_model: str, banking_model: str) -> gr.Blocks:
    def transcribe(audio_path: str | None):
        if not audio_path:
            raise gr.Error("Upload or record audio first.")
        result = transcribe_routed_audio(
            audio_path,
            general_model=general_model,
            banking_model=banking_model,
        )
        return (
            result["hypothesis"],
            result["baseline_hypothesis"],
            result["route"],
            result["matched_keyword"] or "—",
            result["model"],
        )

    with gr.Blocks(title="Indonesian Banking ASR") as demo:
        gr.Markdown(
            """
            # Indonesian Banking ASR
            **Keyword-routed Whisper for banking entities.** A general model transcribes first; the Step37 checkpoint runs only when the normalized transcript contains a banking keyword.
            """,
            elem_classes="hero",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                audio = gr.Audio(
                    label="Indonesian audio",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                submit = gr.Button("Transcribe", variant="primary", size="lg")
                gr.Examples(examples=[[str(DEMO_AUDIO_PATH)]], inputs=[audio], label="Try the banking example")
            with gr.Column(scale=7, elem_classes="output-card"):
                final_transcript = gr.Textbox(label="Final transcript", lines=3, interactive=False)
                baseline_transcript = gr.Textbox(
                    label="Postprocessed baseline",
                    lines=2,
                    interactive=False,
                )
                with gr.Row(elem_classes="meta-row"):
                    route = gr.Textbox(label="Route", interactive=False)
                    keyword = gr.Textbox(label="Matched keyword", interactive=False)
                model = gr.Textbox(label="Selected model", interactive=False)

        submit.click(
            transcribe,
            inputs=[audio],
            outputs=[final_transcript, baseline_transcript, route, keyword, model],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local Indonesian banking ASR demo.")
    parser.add_argument("--general-model", default=DEFAULT_MODEL)
    parser.add_argument("--banking-model", default=DEFAULT_BANKING_MODEL)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", default=7860, type=int)
    args = parser.parse_args()

    if not Path(args.banking_model).is_dir():
        parser.error("--banking-model must point to the local Step37 checkpoint")

    build_demo(general_model=args.general_model, banking_model=args.banking_model).launch(
        share=False,
        css=_CSS,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
