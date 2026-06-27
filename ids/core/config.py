"""Configuration for CIC-IoT-2023 detection pipeline."""

import json
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = PROJECT_ROOT / 'data' / 'CSV'
PARQUET_PATH = PROJECT_ROOT / 'data' / 'cic_iot_2023.parquet'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)
HPARAMS_PATH = PROJECT_ROOT / 'hparams.json'

MAX_ROWS_PER_CLASS = 200_000

X_COLUMNS = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP',
    'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Variance',
]
Y_COLUMN = 'Label'
N_FEATURES = len(X_COLUMNS)
FLAG_COLUMNS = [
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP',
    'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC',
]
LOG_COLUMNS = [c for c in X_COLUMNS if c not in FLAG_COLUMNS]

DROPPED_REDUNDANT = [
    'Tot size',
    'Variance',
]
DROPPED_LOW_VAR = [
    'ece_flag_number',
    'cwr_flag_number',
    'IGMP',
    'IRC',
    'Telnet',
    'SMTP',
    'DHCP',
    'SSH',
    'ICMP',
    'ARP',
    'LLC',
    'IPv',
]
DROPPED_FEATURES = set(DROPPED_REDUNDANT + DROPPED_LOW_VAR)

X_COLUMNS_SELECTED = [c for c in X_COLUMNS if c not in DROPPED_FEATURES]
FLAG_COLUMNS_SELECTED = [c for c in FLAG_COLUMNS if c not in DROPPED_FEATURES]
LOG_COLUMNS_SELECTED = [c for c in X_COLUMNS_SELECTED if c not in FLAG_COLUMNS_SELECTED]
N_FEATURES_SELECTED = len(X_COLUMNS_SELECTED)

MODES_TO_RUN = ['2', '8']
SPLITS_TO_RUN = ['random']

BATCH_SIZE = 4096
N_EPOCHS = 50
PATIENCE = 5
LR = 1e-3
SEED = 42

BATCH_SIZES = [1, 32, 256, 1024]
N_WARMUP = 100
N_RUNS = 1000

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def load_hparams(model: str, mode: str) -> dict:
    """Read the tuned hyperparameters for a ``(model, mode)`` cell from hparams.json.

    hparams.json is the single source of truth shared by training and serving;
    ``ids.training.tune`` is its only writer. Raises a clear error (pointing at the
    command that generates the cell) if the file or the requested cell is missing."""
    cmd = f'python -m ids.training.tune --model {model} --mode {mode} --split random'
    if not HPARAMS_PATH.exists():
        raise FileNotFoundError(f'{HPARAMS_PATH} not found. Generate it with `{cmd}`.')
    data = json.loads(HPARAMS_PATH.read_text())
    try:
        return data[model][str(mode)]
    except KeyError:
        raise KeyError(
            f'No hyperparameters for model={model!r} mode={mode!r} in {HPARAMS_PATH}. '
            f'Generate them with `{cmd}`.')
