#!/usr/bin/env python3
try:
    from datasets import load_dataset
    print("datasets imported successfully")
    import torch
    print("torch imported successfully")
    import transformers
    print("transformers imported successfully")
    import sklearn
    print("sklearn imported successfully")
    print("All key dependencies installed!")
except ImportError as e:
    print(f"Import error: {e}")
