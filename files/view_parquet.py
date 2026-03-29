#!/usr/bin/env python3
"""
Quick Parquet file viewer
Usage: python view_parquet.py <file.parquet>
"""

import pandas as pd
import sys
from pathlib import Path

def view_parquet(file_path):
    try:
        # Read parquet file
        df = pd.read_parquet(file_path)
        
        print(f"\n=== Parquet File: {file_path} ===")
        print(f"Shape: {df.shape} (rows, columns)")
        print(f"Columns: {list(df.columns)}")
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nFirst 10 rows:")
        print(df.head(10))
        
        if len(df) > 10:
            print(f"\nLast 5 rows:")
            print(df.tail(5))
            
        print(f"\nBasic statistics:")
        print(df.describe())
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python view_parquet.py <file.parquet>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    view_parquet(file_path)
