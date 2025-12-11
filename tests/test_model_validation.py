"""
Unit Tests untuk Model Validation
"""

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


class TestModelStructure:
    """Test suite untuk model structure"""
    
    @pytest.fixture
    def model_path(self):
        return Path("models/bert_model")
    
    def test_model_files_exist(self, model_path):
        """Test semua required model files ada"""
        required_files = [
            'config.json',
            'label_map.json',
            'pytorch_model.bin'  # or model.safetensors
        ]
        
        for file in required_files:
            if file == 'pytorch_model.bin':
                # Check either pytorch_model.bin or model.safetensors
                assert (model_path / 'pytorch_model.bin').exists() or \
                       (model_path / 'model.safetensors').exists(), \
                       "Model weights file not found"
            else:
                assert (model_path / file).exists(), f"Missing {file}"
    
    def test_label_map_valid(self, model_path):
        """Test label map valid"""
        label_map_file = model_path / 'label_map.json'
        if label_map_file.exists():
            with open(label_map_file) as f:
                label_map = json.load(f)
            
            # Should have 3 labels for sentiment
            assert len(label_map) == 3, f"Expected 3 labels, got {len(label_map)}"
            
            # Labels should be continuous integers starting from 0
            expected_ids = set(range(len(label_map)))
            actual_ids = set(label_map.values())
            assert expected_ids == actual_ids, "Label IDs not continuous"
    
    def test_model_loadable(self, model_path):
        """Test model bisa di-load"""
        if model_path.exists():
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                assert model is not None
            except Exception as e:
                pytest.fail(f"Failed to load model: {str(e)}")
    
    def test_model_num_labels(self, model_path):
        """Test model memiliki correct number of labels"""
        if model_path.exists():
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            assert model.num_labels == 3, f"Expected 3 labels, got {model.num_labels}"


class TestModelInference:
    """Test suite untuk model inference"""
    
    @pytest.fixture
    def model_and_tokenizer(self):
        model_path = Path("models/bert_model")
        if not model_path.exists():
            pytest.skip("Model not found")
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    
    def test_single_prediction(self, model_and_tokenizer):
        """Test single text prediction"""
        model, tokenizer = model_and_tokenizer
        model.eval()
        
        text = "Aplikasi ini sangat bagus"
        
        encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=1)
        
        assert predictions.shape[0] == 1
        assert 0 <= predictions.item() < 3
    
    def test_batch_prediction(self, model_and_tokenizer):
        """Test batch predictions"""
        model, tokenizer = model_and_tokenizer
        model.eval()
        
        texts = [
            "Aplikasi bagus",
            "Sangat buruk",
            "Biasa saja"
        ]
        
        encodings = tokenizer(
            texts,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=1)
        
        assert predictions.shape[0] == len(texts)
        assert all(0 <= p < 3 for p in predictions.tolist())
    
    def test_prediction_probabilities_sum_to_one(self, model_and_tokenizer):
        """Test prediction probabilities sum to 1"""
        model, tokenizer = model_and_tokenizer
        model.eval()
        
        text = "Test text"
        
        encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=1)
        
        prob_sum = probs.sum().item()
        assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities sum to {prob_sum}, not 1.0"
    
    def test_consistent_predictions(self, model_and_tokenizer):
        """Test predictions consistent untuk input yang sama"""
        model, tokenizer = model_and_tokenizer
        model.eval()
        
        text = "Aplikasi ini bagus"
        
        # Predict twice
        predictions = []
        for _ in range(2):
            encoding = tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = model(**encoding)
                pred = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(pred)
        
        assert predictions[0] == predictions[1], "Inconsistent predictions"


class TestModelPerformance:
    """Test suite untuk model performance"""
    
    def test_minimum_accuracy(self):
        """Test model accuracy minimal threshold"""
        metrics_file = Path("models/bert_metrics.json")
        
        if not metrics_file.exists():
            pytest.skip("Metrics file not found")
        
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        test_metrics = metrics.get('test', {})
        accuracy = test_metrics.get('accuracy', 0)
        
        min_accuracy = 0.70  # 70% minimum
        assert accuracy >= min_accuracy, \
            f"Model accuracy {accuracy:.2%} below threshold {min_accuracy:.2%}"
    
    def test_no_severe_overfitting(self):
        """Test tidak ada overfitting parah"""
        metrics_file = Path("models/bert_metrics.json")
        
        if not metrics_file.exists():
            pytest.skip("Metrics file not found")
        
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        train_acc = metrics.get('train', {}).get('accuracy', 0)
        test_acc = metrics.get('test', {}).get('accuracy', 0)
        
        gap = train_acc - test_acc
        max_gap = 0.15  # 15% maximum gap
        
        assert gap <= max_gap, \
            f"Overfitting detected: train {train_acc:.2%}, test {test_acc:.2%}, gap {gap:.2%}"
    
    def test_balanced_precision_recall(self):
        """Test precision dan recall balanced"""
        metrics_file = Path("models/bert_metrics.json")
        
        if not metrics_file.exists():
            pytest.skip("Metrics file not found")
        
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        test_metrics = metrics.get('test', {})
        precision = test_metrics.get('precision', 0)
        recall = test_metrics.get('recall', 0)
        
        # Difference should be < 0.10
        diff = abs(precision - recall)
        assert diff < 0.10, \
            f"Precision-Recall imbalance: {precision:.2%} vs {recall:.2%}"


class TestModelRobustness:
    """Test suite untuk model robustness"""
    
    @pytest.fixture
    def model_and_tokenizer(self):
        model_path = Path("models/bert_model")
        if not model_path.exists():
            pytest.skip("Model not found")
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    
    def test_handles_empty_string(self, model_and_tokenizer):
        """Test model handles empty string"""
        model, tokenizer = model_and_tokenizer
        model.eval()
        
        text = ""
        
        try:
            encoding = tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = model(**encoding)
                predictions = torch.argmax(outputs.logits, dim=1)
            
            assert predictions.shape[0] == 1
        except Exception as e:
            pytest.fail(f"Failed to handle empty string: {str(e)}")
    
    def test_handles_long_text(self, model_and_tokenizer):
        """Test model handles very long text"""
        model, tokenizer = model_and_tokenizer
        model.eval()
        
        text = "Bagus " * 1000  # Very long text
        
        try:
            encoding = tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = model(**encoding)
                predictions = torch.argmax(outputs.logits, dim=1)
            
            assert predictions.shape[0] == 1
        except Exception as e:
            pytest.fail(f"Failed to handle long text: {str(e)}")
    
    def test_handles_special_characters(self, model_and_tokenizer):
        """Test model handles special characters"""
        model, tokenizer = model_and_tokenizer
        model.eval()
        
        texts = [
            "Aplikasi!!! @#$%",
            "🔥🔥🔥 TOP",
            "Very.... good???",
        ]
        
        for text in texts:
            try:
                encoding = tokenizer(
                    text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                with torch.no_grad():
                    outputs = model(**encoding)
                    predictions = torch.argmax(outputs.logits, dim=1)
                
                assert 0 <= predictions.item() < 3
            except Exception as e:
                pytest.fail(f"Failed on text '{text}': {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
