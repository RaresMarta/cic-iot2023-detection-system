"""Configuration for CIC-IoT-2023 detection pipeline."""

from pathlib import Path
import numpy as np
import torch

# Paths
DATASET_DIRECTORY = Path('data/CSV')
PARQUET_PATH = Path('data/cic_iot_2023.parquet')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

# Data
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

# Tasks & splits
MODES_TO_RUN = ['2']
SPLITS_TO_RUN = ['temporal']

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
