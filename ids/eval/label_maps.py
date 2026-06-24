"""Pluggable per-dataset label maps for cross-dataset evaluation.

The CIC-IoT-2023 2-class model emits ``Benign`` / ``Attack``. To compare its
predictions against a *foreign* dataset's ground-truth labels, those raw labels
must be normalised to the same binary vocabulary. Different datasets spell
"benign" differently (``Normal``, ``BENIGN``, ``benign``, ``-`` ...), so this
module centralises the normalisation per dataset.

Public API
----------
``to_binary(dataset_name, raw_label) -> {'benign', 'attack'}``
    Lower-cased, whitespace-trimmed binary label.

Design
------
* The DEFAULT rule is the honest, conservative one: *any non-benign string is an
  attack*. This is correct for cross-dataset binary detection because the foreign
  datasets are attack-vs-benign corpora — anything not explicitly benign is, by
  construction, malicious.
* ``_BENIGN_ALIASES`` is the only place we enumerate per-dataset benign spellings.
  Add a dataset's benign token(s) here and the default rule handles the rest.
* The 8-class FAMILY maps below are intentionally LEFT EMPTY. Mapping a foreign
  dataset's attack taxonomy onto the CIC-IoT-2023 8-family scheme (DDoS / DoS /
  Mirai / Recon / Spoofing / Web / BruteForce / Benign) is a research decision,
  not a mechanical one, and is deferred to a human. See the TODO stubs.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Benign aliases per dataset (extend as datasets are provisioned).
# Keys are lower-cased dataset names; values are sets of lower-cased benign
# tokens that should normalise to 'benign'. The special key '*' applies to
# every dataset (common spellings seen across IDS corpora).
# ---------------------------------------------------------------------------
_BENIGN_ALIASES: dict[str, set[str]] = {
    '*': {'benign', 'benign_final', 'normal', 'background', 'legitimate', '-', ''},
    # CIC-IoT-2023 itself (useful for sanity round-trips / in-distribution eval).
    'ciciot2023': {'benign', 'benign_final'},
    # Foreign datasets this harness is wired to accept. Benign tokens taken from
    # each dataset's published label vocabulary.
    'bot-iot':  {'normal'},          # Bot-IoT: category 'Normal'
    'ton-iot':  {'normal', '0'},     # ToN-IoT: 'type'=='normal' / label 0
    'iot-23':   {'benign', '-'},     # IoT-23 Zeek 'detailed-label': '-' == benign
}


def _benign_tokens(dataset_name: str) -> set[str]:
    name = (dataset_name or '').strip().lower()
    return _BENIGN_ALIASES['*'] | _BENIGN_ALIASES.get(name, set())


def to_binary(dataset_name: str, raw_label) -> str:
    """Normalise a foreign dataset's raw label to ``'benign'`` or ``'attack'``.

    Rule: the label is ``'benign'`` iff its lower-cased / trimmed form is in the
    dataset's benign-alias set (union of the dataset-specific set and the global
    ``'*'`` set). Everything else — every attack family, every unknown string —
    maps to ``'attack'``. This is the deliberate default: a non-benign label is
    an attack.

    Args:
        dataset_name: dataset identifier, e.g. ``'bot-iot'`` (case-insensitive).
        raw_label:    the dataset's own label value (str / int / anything
                      stringifiable).

    Returns:
        ``'benign'`` or ``'attack'``.
    """
    token = str(raw_label).strip().lower()
    return 'benign' if token in _benign_tokens(dataset_name) else 'attack'


# ---------------------------------------------------------------------------
# 8-class FAMILY maps — DEFERRED. Do NOT fill these in here.
#
# These would map each foreign dataset's raw attack label onto the CIC-IoT-2023
# 8-family taxonomy used by the 8-class model:
#     {'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web', 'BruteForce', 'Benign'}
# (see ids/core/labels.py::DICT_8CLASSES for the in-distribution mapping).
#
# Leaving them empty is intentional: the 8-class cross-dataset path is out of
# scope for the current harness (binary is the clean path). A human must decide
# how, e.g., Bot-IoT's 'DDoS'/'DoS'/'Reconnaissance'/'Theft' subcategories map
# onto our families, and document the judgement calls in the thesis.
#
# Each dict's keys should be the dataset's raw labels; values one of the 8
# family strings above. When a map is populated, add a `to_family()` resolver
# that mirrors `to_binary()`.
# ---------------------------------------------------------------------------

# TODO(human): populate Bot-IoT raw-label -> CIC-IoT-2023 family.
BOT_IOT_FAMILY_MAP: dict[str, str] = {}

# TODO(human): populate ToN-IoT raw-label -> CIC-IoT-2023 family.
TON_IOT_FAMILY_MAP: dict[str, str] = {}

# TODO(human): populate IoT-23 raw-label -> CIC-IoT-2023 family.
IOT_23_FAMILY_MAP: dict[str, str] = {}

# Registry so a future to_family() can look the map up by --dataset name.
FAMILY_MAPS: dict[str, dict[str, str]] = {
    'bot-iot': BOT_IOT_FAMILY_MAP,
    'ton-iot': TON_IOT_FAMILY_MAP,
    'iot-23':  IOT_23_FAMILY_MAP,
}

# Datasets whose benign vocabulary is wired (binary path ready).
WIRED_DATASETS: tuple[str, ...] = ('ciciot2023', 'bot-iot', 'ton-iot', 'iot-23')
