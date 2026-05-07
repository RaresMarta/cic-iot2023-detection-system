"""Gradio demo — upload a PCAP or a pre-extracted CSV, get per-flow IDS verdicts.

Run:
    python demo/app.py

To expose to the internet for thesis defense:
    cloudflared tunnel --url http://localhost:7860
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import polars as pl

from demo.cicflowmeter_runner import CICFlowMeterError, run_cicflowmeter
from demo.inference import IDSPredictor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'

# Load each (split, mode) the user trained. Probe filesystem to avoid hard-coding.
def discover_predictors() -> dict[str, IDSPredictor]:
    found: dict[str, IDSPredictor] = {}
    for ckpt in MODELS_DIR.glob('ids_dnn_*_*class.pth'):
        # stem: ids_dnn_<split>_<mode>class — split may itself contain underscores (per_csv)
        parts = ckpt.stem.split('_')
        if len(parts) < 4 or not parts[-1].endswith('class'):
            continue
        mode = parts[-1].removesuffix('class')
        split = '_'.join(parts[2:-1])
        try:
            found[f'{split} / {mode}-class'] = IDSPredictor(MODELS_DIR, split=split, mode=mode)
        except FileNotFoundError:
            continue
    return found


PREDICTORS = discover_predictors()
if not PREDICTORS:
    raise SystemExit(
        f'No trained models found in {MODELS_DIR}. '
        f'Run the notebook end-to-end before launching the demo.'
    )


def classify_csv(csv_path: Path, predictor_key: str) -> tuple[str, list[list]]:
    df = pl.read_csv(str(csv_path))
    pred = PREDICTORS[predictor_key].predict(df)

    rows = []
    for i, (label, conf) in enumerate(zip(pred['labels'], pred['confidences'])):
        rows.append([i, str(label), f'{conf:.4f}'])

    summary_counts = {}
    for label in pred['labels']:
        summary_counts[str(label)] = summary_counts.get(str(label), 0) + 1
    summary = ' | '.join(f'{k}: {v}' for k, v in sorted(
        summary_counts.items(), key=lambda kv: -kv[1]
    ))
    return f'{len(pred["labels"])} flows classified — {summary}', rows


def handle_csv(file, predictor_key: str):
    if file is None:
        return 'Upload a CSV first.', []
    return classify_csv(Path(file.name), predictor_key)


def handle_pcap(file, predictor_key: str):
    if file is None:
        return 'Upload a PCAP first.', []
    pcap_path = Path(file.name)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            csv_path = run_cicflowmeter(pcap_path, Path(tmp))
        except CICFlowMeterError as e:
            return f'CICFlowMeter error: {e}', []
        return classify_csv(csv_path, predictor_key)


with gr.Blocks(title='RT-IDS for CIC-IoT-2023') as app:
    gr.Markdown('# RT-IDS demo — CIC-IoT-2023')
    gr.Markdown(
        'Upload a **CSV** (already in CIC 39-feature format) or a **PCAP** (will be processed '
        'by CICFlowMeter first). Each row is classified independently.'
    )

    predictor_choice = gr.Dropdown(
        choices=list(PREDICTORS.keys()),
        value=list(PREDICTORS.keys())[0],
        label='Model (split / granularity)',
    )

    with gr.Tab('CSV upload'):
        csv_in = gr.File(label='CIC-format CSV', file_types=['.csv'])
        csv_btn = gr.Button('Classify CSV')
        csv_summary = gr.Textbox(label='Summary', interactive=False)
        csv_table = gr.Dataframe(headers=['#', 'Predicted', 'Confidence'], interactive=False)
        csv_btn.click(handle_csv, inputs=[csv_in, predictor_choice],
                      outputs=[csv_summary, csv_table])

    with gr.Tab('PCAP upload'):
        pcap_in = gr.File(label='PCAP file', file_types=['.pcap', '.pcapng'])
        pcap_btn = gr.Button('Extract flows + classify')
        pcap_summary = gr.Textbox(label='Summary', interactive=False)
        pcap_table = gr.Dataframe(headers=['#', 'Predicted', 'Confidence'], interactive=False)
        pcap_btn.click(handle_pcap, inputs=[pcap_in, predictor_choice],
                       outputs=[pcap_summary, pcap_table])


if __name__ == '__main__':
    app.launch(server_name='0.0.0.0', server_port=7860)
