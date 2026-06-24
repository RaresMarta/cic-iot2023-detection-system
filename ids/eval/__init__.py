"""Cross-dataset evaluation harness for the frozen CIC-IoT-2023 2-class model.

Inference-only: loads the existing frozen scaler + model and scores foreign-dataset
pcaps through the same packet-window extractor used for the CIC-IoT-2023 demo. No
retraining. See ``cross_dataset_eval`` for the public API and CLI.
"""
