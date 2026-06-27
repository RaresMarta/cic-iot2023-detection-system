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

_BENIGN_ALIASES: dict[str, set[str]] = {
    '*': {'benign', 'benign_final', 'normal', 'background', 'legitimate', '-', ''},
    'ciciot2023': {'benign', 'benign_final'},
    'bot-iot':  {'normal'},
    'ton-iot':  {'normal', '0'},
    'iot-23':   {'benign', '-'},
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



BOT_IOT_FAMILY_MAP: dict[str, str] = {}

TON_IOT_FAMILY_MAP: dict[str, str] = {}

IOT_23_FAMILY_MAP: dict[str, str] = {}

FAMILY_MAPS: dict[str, dict[str, str]] = {
    'bot-iot': BOT_IOT_FAMILY_MAP,
    'ton-iot': TON_IOT_FAMILY_MAP,
    'iot-23':  IOT_23_FAMILY_MAP,
}

WIRED_DATASETS: tuple[str, ...] = ('ciciot2023', 'bot-iot', 'ton-iot', 'iot-23')
