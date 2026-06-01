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

# The 25 retained features after EDA-driven pruning of the 39 public CICIoT2023
# columns: 12 near-zero-variance binary flags (ece, cwr, Telnet, SMTP, SSH, IRC,
# DHCP, ARP, ICMP, IGMP, IPv, LLC) and 2 mathematically redundant continuous
# features (Tot size ~ AVG, Variance ~ Std) are dropped. Order matches the
# cached parquet and the fitted scaler/model input ordering.
X_COLUMNS = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ack_count', 'syn_count',
    'fin_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'TCP', 'UDP',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'IAT', 'Number',
]
Y_COLUMN = 'Label'
N_FEATURES = len(X_COLUMNS)  # 25 after feature selection

FLAG_COLUMNS = [
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'HTTP', 'HTTPS', 'DNS',
    'TCP', 'UDP',
]
LOG_COLUMNS = [c for c in X_COLUMNS if c not in FLAG_COLUMNS]

# Tasks & splits
MODES_TO_RUN = ['2', '8']
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
