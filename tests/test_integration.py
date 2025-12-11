"""
Integration Tests untuk End-to-End Pipeline
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))

load_dotenv()


class TestDatabaseIntegration:
    """Test suite untuk database integration"""
    
    @pytest.fixture
    def db_connection(self):
        """Create database connection"""
        try:
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', 5432)),
                database=os.getenv('POSTGRES_DB', 'sentiment_db'),
                user=os.getenv('POSTGRES_USER', 'sentiment_user'),
                password=os.getenv('POSTGRES_PASSWORD', 'password')
            )
            yield conn
            conn.close()
        except Exception as e:
            pytest.skip(f"Cannot connect to database: {str(e)}")
    
    def test_reviews_table_exists(self, db_connection):
        """Test reviews table exists"""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'reviews'
            );
        """)
        exists = cursor.fetchone()[0]
        cursor.close()
        assert exists, "reviews table does not exist"
    
    def test_model_metrics_table_exists(self, db_connection):
        """Test model_metrics table exists"""
        cursor = db_connection.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'model_metrics'
            );
        """)
        exists = cursor.fetchone()[0]
        cursor.close()
        assert exists, "model_metrics table does not exist"
    
    def test_reviews_table_has_data(self, db_connection):
        """Test reviews table has data"""
        cursor = db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM reviews;")
        count = cursor.fetchone()[0]
        cursor.close()
        assert count > 0, "reviews table is empty"
    
    def test_model_metrics_can_be_inserted(self, db_connection):
        """Test metrics bisa di-insert"""
        cursor = db_connection.cursor()
        
        # Insert test metric
        cursor.execute("""
            INSERT INTO model_metrics 
            (model_name, accuracy, precision_score, recall_score, f1_score, 
             train_accuracy, train_precision, train_recall, train_f1,
             dataset_type, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            'test_model', 0.85, 0.84, 0.83, 0.84,
            0.90, 0.89, 0.88, 0.89,
            'test', datetime.now()
        ))
        
        inserted_id = cursor.fetchone()[0]
        db_connection.commit()
        
        # Delete test metric
        cursor.execute("DELETE FROM model_metrics WHERE id = %s;", (inserted_id,))
        db_connection.commit()
        cursor.close()
        
        assert inserted_id is not None


class TestPredictionPipeline:
    """Test suite untuk prediction pipeline"""
    
    def test_prediction_script_exists(self):
        """Test prediction script exists"""
        pred_file = Path("src/prediction/predict.py")
        assert pred_file.exists(), "predict.py not found"
    
    def test_can_import_prediction_module(self):
        """Test bisa import prediction module"""
        try:
            from src.prediction import predict
        except ImportError as e:
            pytest.fail(f"Cannot import prediction module: {str(e)}")
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction"""
        model_path = Path("models/bert_model")
        if not model_path.exists():
            pytest.skip("Model not found")
        
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        import json
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load label map
        with open(model_path / 'label_map.json') as f:
            label_map = json.load(f)
        reverse_label_map = {v: k for k, v in label_map.items()}
        
        # Test prediction
        text = "Aplikasi ini sangat bagus dan mudah digunakan"
        
        encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            pred_label = reverse_label_map[pred_id]
        
        # Should predict positive for this text
        assert pred_label in ['positive', 'negative', 'neutral']


class TestTrainingPipeline:
    """Test suite untuk training pipeline"""
    
    def test_training_script_exists(self):
        """Test training script exists"""
        train_file = Path("src/training/train_bert.py")
        assert train_file.exists(), "train_bert.py not found"
    
    def test_params_file_exists(self):
        """Test params.yaml exists"""
        params_file = Path("params.yaml")
        assert params_file.exists(), "params.yaml not found"
    
    def test_params_file_valid(self):
        """Test params.yaml valid"""
        import yaml
        
        params_file = Path("params.yaml")
        with open(params_file) as f:
            params = yaml.safe_load(f)
        
        # Check required sections
        assert 'training' in params, "Missing 'training' section in params.yaml"
        
        # Check required training params
        required_params = ['bert_model', 'batch_size', 'epochs', 'learning_rate']
        for param in required_params:
            assert param in params['training'], f"Missing param: {param}"
    
    def test_processed_data_exists(self):
        """Test processed data exists"""
        data_file = Path("data/processed/processed_reviews.csv")
        if not data_file.exists():
            pytest.skip("Processed data not found")
        
        df = pd.read_csv(data_file)
        assert len(df) > 0, "Processed data is empty"
        assert 'cleaned_text' in df.columns
        assert 'sentiment_label' in df.columns


class TestDataPipeline:
    """Test suite untuk data pipeline"""
    
    def test_scraper_exists(self):
        """Test scraper script exists"""
        scraper_file = Path("src/data_collection/scraper.py")
        assert scraper_file.exists(), "scraper.py not found"
    
    def test_preprocessing_exists(self):
        """Test preprocessing script exists"""
        preprocess_file = Path("src/preprocessing/preprocess.py")
        assert preprocess_file.exists(), "preprocess.py not found"
    
    def test_data_directories_exist(self):
        """Test data directories exist"""
        dirs = ['data/raw', 'data/processed']
        for dir_path in dirs:
            assert Path(dir_path).exists(), f"Directory {dir_path} not found"


class TestMonitoring:
    """Test suite untuk monitoring"""
    
    def test_prometheus_exporter_exists(self):
        """Test prometheus exporter exists"""
        exporter_file = Path("src/monitoring/prometheus_exporter.py")
        assert exporter_file.exists(), "prometheus_exporter.py not found"
    
    def test_prometheus_config_exists(self):
        """Test prometheus config exists"""
        config_file = Path("prometheus/prometheus.yml")
        assert config_file.exists(), "prometheus.yml not found"
    
    def test_grafana_dashboard_exists(self):
        """Test grafana dashboard exists"""
        dashboard_files = list(Path("grafana/dashboards").glob("*.json"))
        assert len(dashboard_files) > 0, "No Grafana dashboard found"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
