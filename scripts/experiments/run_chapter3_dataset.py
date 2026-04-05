"""Run Chapter 3 dataset construction."""
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from thesis_rebuild.modeling.dataset_builder import build_chapter3_datasets


if __name__ == "__main__":
    manifest, manifest_path = build_chapter3_datasets()
    print("=" * 60)
    print("Desktop thesis_rebuild: Chapter 3 datasets completed")
    print(f"manifest_path: {manifest_path}")
    for condition_name, condition_info in manifest["conditions"].items():
        print("-" * 60)
        print(condition_name)
        print(f"selected_count: {condition_info['selected_count']}")
        print(f"selected_features: {condition_info['selected_features']}")
        print(f"split_summary: {condition_info['split_summary']}")
