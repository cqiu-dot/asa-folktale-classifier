#!/usr/bin/env python
"""
Test script to verify JSON data loading works with the actual data files
"""

import sys
sys.path.insert(0, '..')

from src.data_loader import FolktaleDataLoader
import logging

logging.basicConfig(level=logging.INFO)

def test_json_loading():
    """Test loading the JSON files from Downloads folder"""
    print("Testing JSON data loading...")

    loader = FolktaleDataLoader('../config/config.yaml')

    try:
        asian_df = loader.load_asian_tales()
        print(f"✓ Successfully loaded {len(asian_df)} tales")

        if len(asian_df) > 0:
            print(f"Columns: {asian_df.columns.tolist()}")
            print(f"Regions: {asian_df['region'].value_counts()}")
            print(f"Sample tale text (first 200 chars):")
            print(asian_df['text'].iloc[0][:200])
        else:
            print("✗ No tales loaded. Check file paths and JSON structure.")

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Check that JSON files exist in the Downloads folder")
        print("Expected files: china_china_fables_dataset*.json, korea_*.json, japan_*.json")

if __name__ == '__main__':
    test_json_loading()