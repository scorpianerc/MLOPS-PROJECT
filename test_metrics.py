#!/usr/bin/env python3
"""
Test script to verify metrics.json loading
"""
import json
from pathlib import Path

def test_metrics():
    metrics_path = Path("models/metrics.json")
    
    print("=" * 60)
    print("METRICS.JSON LOADING TEST")
    print("=" * 60)
    
    # Check if file exists
    if not metrics_path.exists():
        print(f"❌ ERROR: File not found at {metrics_path.absolute()}")
        return False
    
    print(f"✅ File exists at: {metrics_path.absolute()}")
    
    # Try to load
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        print("\n✅ Successfully loaded metrics.json")
    except Exception as e:
        print(f"\n❌ ERROR loading file: {e}")
        return False
    
    # Display values
    print("\n" + "=" * 60)
    print("METRICS VALUES (Raw from JSON)")
    print("=" * 60)
    
    keys = [
        'test_accuracy', 'test_f1', 'test_precision', 'test_recall',
        'train_accuracy', 'train_f1', 'train_precision', 'train_recall'
    ]
    
    for key in keys:
        value = metrics.get(key, 'NOT FOUND')
        if isinstance(value, (int, float)):
            print(f"{key:20s}: {value:.6f}  ({value:.2%})")
        else:
            print(f"{key:20s}: {value}")
    
    print("\n" + "=" * 60)
    print("WHAT STREAMLIT SHOULD DISPLAY")
    print("=" * 60)
    
    test_acc = metrics.get('test_accuracy', 0)
    test_f1 = metrics.get('test_f1', 0)
    
    print(f"Model Accuracy (top card): {test_acc:.1%}")
    print(f"F1 Score (top card):       {test_f1:.1%}")
    print(f"\nTest Accuracy (perf):      {test_acc:.1%}")
    print(f"Test F1 Score (perf):      {test_f1:.1%}")
    print(f"Test Precision (perf):     {metrics.get('test_precision', 0):.1%}")
    print(f"Test Recall (perf):        {metrics.get('test_recall', 0):.1%}")
    
    print("\n" + "=" * 60)
    return True

if __name__ == "__main__":
    success = test_metrics()
    print("\n" + ("✅ TEST PASSED" if success else "❌ TEST FAILED"))
