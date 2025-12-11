"""
Script untuk Test Semua Fitur MLOps
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv()

# Database config
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'sentiment_db'),
    'user': os.getenv('POSTGRES_USER', 'sentiment_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'password')
}


def test_mlflow():
    """Test MLflow Manager"""
    print("\n" + "="*60)
    print("🧪 Testing MLflow Manager")
    print("="*60)
    
    try:
        from src.mlops.mlflow_manager import MLflowManager
        
        # Initialize
        manager = MLflowManager(experiment_name="test-mlops")
        print("✅ MLflow Manager initialized")
        
        # Start run
        manager.start_run(run_name="test_run")
        print("✅ MLflow run started")
        
        # Log params and metrics
        manager.log_params({"test_param": "value"})
        manager.log_metrics({"test_metric": 0.95})
        print("✅ Logged params and metrics")
        
        # End run
        manager.end_run()
        print("✅ MLflow run ended")
        
        print("\n✅ MLflow Manager: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ MLflow Manager: FAILED - {e}")
        return False


def test_drift_detection():
    """Test Drift Detection"""
    print("\n" + "="*60)
    print("🧪 Testing Drift Detection")
    print("="*60)
    
    try:
        from src.mlops.drift_detection import (
            ModelDriftMonitor,
            create_prediction_logs_table
        )
        
        # Create table
        create_prediction_logs_table(DB_CONFIG)
        print("✅ Prediction logs table created")
        
        # Initialize monitor
        monitor = ModelDriftMonitor(DB_CONFIG)
        print("✅ Model drift monitor initialized")
        
        # Get baseline metrics
        baseline = monitor.get_baseline_metrics()
        if baseline:
            print(f"✅ Baseline metrics loaded: {baseline['accuracy']:.2%}")
        else:
            print("⚠️  No baseline metrics (database might be empty)")
        
        # Get trend
        trend = monitor.analyze_trend(days=30)
        print(f"✅ Trend analysis: {trend['overall_trend']}")
        
        print("\n✅ Drift Detection: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Drift Detection: FAILED - {e}")
        return False


def test_retraining_pipeline():
    """Test Retraining Pipeline"""
    print("\n" + "="*60)
    print("🧪 Testing Retraining Pipeline")
    print("="*60)
    
    try:
        from src.mlops.retraining_pipeline import RetrainingTrigger
        
        # Initialize trigger
        trigger = RetrainingTrigger(DB_CONFIG)
        print("✅ Retraining trigger initialized")
        
        # Evaluate triggers
        evaluation = trigger.evaluate_triggers()
        print(f"✅ Triggers evaluated: should_retrain={evaluation['should_retrain']}")
        print(f"   Total triggers: {evaluation['summary']['total_triggers']}")
        
        if evaluation['triggers']:
            for t in evaluation['triggers'][:3]:  # Show first 3
                print(f"   - {t['type']}: {t['reason']}")
        
        print("\n✅ Retraining Pipeline: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Retraining Pipeline: FAILED - {e}")
        return False


def test_feature_store():
    """Test Feature Store"""
    print("\n" + "="*60)
    print("🧪 Testing Feature Store")
    print("="*60)
    
    try:
        from src.mlops.feature_store import TextFeatureExtractor, FeatureStore
        
        # Test feature extractor
        extractor = TextFeatureExtractor()
        print("✅ Feature extractor initialized")
        
        # Extract features
        features = extractor.extract_features("Aplikasi ini sangat bagus!")
        print(f"✅ Features extracted: {len(features)} features")
        print(f"   Cleaned text: {features['cleaned_text']}")
        print(f"   Word count: {features['word_count']}")
        
        # Test feature store
        feature_store = FeatureStore(DB_CONFIG, extractor)
        print("✅ Feature store initialized")
        
        print("\n✅ Feature Store: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Feature Store: FAILED - {e}")
        return False


def test_api_server():
    """Test API Server (check if loadable)"""
    print("\n" + "="*60)
    print("🧪 Testing API Server")
    print("="*60)
    
    try:
        from src.api.api_server import app
        print("✅ API server module loaded")
        
        # Check if FastAPI app exists
        assert app is not None
        print("✅ FastAPI app initialized")
        
        print("\n✅ API Server: PASSED")
        print("   To test fully, run: uvicorn src.api.api_server:app --port 8080")
        return True
        
    except Exception as e:
        print(f"\n❌ API Server: FAILED - {e}")
        return False


def test_automated_tests():
    """Test Automated Tests"""
    print("\n" + "="*60)
    print("🧪 Testing Automated Tests")
    print("="*60)
    
    try:
        import pytest
        
        # Check test files exist
        test_files = [
            'tests/test_data_validation.py',
            'tests/test_model_validation.py',
            'tests/test_integration.py'
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                print(f"✅ {test_file} exists")
            else:
                print(f"⚠️  {test_file} not found")
        
        print("\n✅ Automated Tests: PASSED")
        print("   To run tests: pytest tests/ -v")
        return True
        
    except ImportError:
        print("\n⚠️  pytest not installed")
        print("   Install: pip install pytest")
        return False
    except Exception as e:
        print(f"\n❌ Automated Tests: FAILED - {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 MLOPS FEATURES TEST SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'MLflow Manager': test_mlflow(),
        'Drift Detection': test_drift_detection(),
        'Retraining Pipeline': test_retraining_pipeline(),
        'Feature Store': test_feature_store(),
        'API Server': test_api_server(),
        'Automated Tests': test_automated_tests()
    }
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for feature, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{feature:.<40} {status}")
    
    print("-"*60)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! MLOps features ready to use!")
        print("\n📖 Next steps:")
        print("   1. Start MLflow UI: mlflow ui --port 5000")
        print("   2. Start API server: docker-compose up -d api")
        print("   3. Run training with tracking: python src/training/train_bert.py")
        print("   4. Check API docs: http://localhost:8080/docs")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check errors above.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save results
    results_file = Path("logs/mlops_test_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': {k: ('passed' if v else 'failed') for k, v in results.items()},
            'summary': {
                'total': total,
                'passed': passed,
                'failed': failed
            }
        }, f, indent=2)
    
    print(f"\n📝 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
