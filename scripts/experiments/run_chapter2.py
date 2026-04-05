"""Run the rebuilt Chapter 2 pipeline."""
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from thesis_rebuild.data_processing.chapter2_pipeline import run_chapter2_pipeline


if __name__ == "__main__":
    report, _, report_path = run_chapter2_pipeline()
    print("=" * 60)
    print("Desktop thesis_rebuild: Chapter 2 completed")
    print(f"report_path: {report_path}")
    print(f"selected_count: {report['selection']['selected_count']}")
    print(f"selected_features: {report['selection']['selected_features']}")
    print(f"fpt_summary: {report['fpt_summary']}")
