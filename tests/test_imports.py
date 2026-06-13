"""#6 Smoke test — every core library module imports cleanly.

Banal, but it's the first thing to catch a broken path after a package restructure.
Side-effecting entrypoints (analyzer.app, live capture) are intentionally excluded —
they need trained models / Linux privileges and are covered elsewhere.
"""
import importlib

import pytest

CORE_MODULES = [
    'ids.core.config',
    'ids.core.models',
    'ids.core.labels',
    'ids.data.preprocessing',
    'ids.data.ingest',
    'ids.data.sampler',
    'ids.runtime.extractor',
    'ids.runtime.predictor',
    'ids.runtime.explain',
    'ids.apps.monitor.windower',
]


@pytest.mark.parametrize('mod', CORE_MODULES)
def test_module_imports(mod):
    importlib.import_module(mod)
