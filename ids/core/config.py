"""Configuration for CIC-IoT-2023 detection pipeline."""

from pathlib import Path
import numpy as np
import torch

# Paths — anchored to the repo root (this file is ids/core/config.py), so they
# resolve the same regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = PROJECT_ROOT / 'data' / 'CSV'
PARQUET_PATH = PROJECT_ROOT / 'data' / 'cic_iot_2023.parquet'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# Data
MAX_ROWS_PER_CLASS = 200_000

# 39-feature public CICIoT2023 column set
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

# Feature selection (39 -> 25), justified by the notebook's EDA
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
N_FEATURES_SELECTED = len(X_COLUMNS_SELECTED)  # 25

# Tasks & splits
MODES_TO_RUN = ['2', '8']
SPLITS_TO_RUN = ['random', 'per_csv']

# Training
BATCH_SIZE = 4096
N_EPOCHS = 50
PATIENCE = 5
LR = 1e-3
SEED = 42

# Benchmarking
BATCH_SIZES = [1, 32, 256, 1024]
N_WARMUP = 100
N_RUNS = 1000

# RNG initialization
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
