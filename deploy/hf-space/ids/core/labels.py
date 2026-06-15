"""Label mappings for 34 → 8 → 2 class granularities."""

import numpy as np

DICT_8CLASSES = {
    # DDoS (12)
    'DDoS-ACK_Fragmentation': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-ICMP_Flood': 'DDoS', 'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-PSHACK_FLOOD': 'DDoS', 'DDoS-RSTFINFLOOD': 'DDoS',
    'DDoS-SlowLoris': 'DDoS', 'DDoS-SYN_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS', 'DDoS-UDP_Fragmentation': 'DDoS',
    # DoS (4)
    'DoS-HTTP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS', 'DoS-UDP_Flood': 'DoS',
    # Mirai (3)
    'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    # Recon (5)
    'Recon-HostDiscovery': 'Recon', 'Recon-OSScan': 'Recon',
    'Recon-PingSweep': 'Recon', 'Recon-PortScan': 'Recon',
    'VulnerabilityScan': 'Recon',
    # Spoofing (2)
    'DNS_Spoofing': 'Spoofing', 'MITM-ArpSpoofing': 'Spoofing',
    # Web (6)
    'Backdoor_Malware': 'Web', 'BrowserHijacking': 'Web',
    'CommandInjection': 'Web', 'SqlInjection': 'Web',
    'Uploading_Attack': 'Web', 'XSS': 'Web',
    # BruteForce (1)
    'DictionaryBruteForce': 'BruteForce',
    # Benign (1)
    'Benign_Final': 'Benign',
}

DICT_2CLASSES = {
    k: ('Benign' if v == 'Benign' else 'Attack')
    for k, v in DICT_8CLASSES.items()
}


def remap_labels(y_34_str: np.ndarray, mode: str) -> np.ndarray:
    """Remap 34-class labels to 8-class or 2-class granularity."""
    if mode == '34':
        return y_34_str

    mapping = DICT_8CLASSES if mode == '8' else DICT_2CLASSES

    return np.array([mapping[x] for x in y_34_str])
