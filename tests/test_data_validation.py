"""
Unit Tests untuk Data Validation
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class TestDataValidation:
    """Test suite untuk data validation"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing"""
        return pd.DataFrame({
            'review_text': ['Aplikasi bagus', 'Sangat buruk', 'Lumayan'],
            'cleaned_text': ['aplikasi bagus', 'sangat buruk', 'lumayan'],
            'sentiment_label': ['positive', 'negative', 'neutral'],
            'rating': [5, 1, 3]
        })
    
    def test_required_columns(self, sample_dataframe):
        """Test bahwa required columns ada"""
        required_cols = ['review_text', 'cleaned_text', 'sentiment_label']
        for col in required_cols:
            assert col in sample_dataframe.columns, f"Missing required column: {col}"
    
    def test_no_null_values(self, sample_dataframe):
        """Test tidak ada null values di columns penting"""
        important_cols = ['review_text', 'sentiment_label']
        for col in important_cols:
            assert sample_dataframe[col].notna().all(), f"Null values found in {col}"
    
    def test_sentiment_labels_valid(self, sample_dataframe):
        """Test sentiment labels hanya positive, negative, neutral"""
        valid_labels = ['positive', 'negative', 'neutral']
        assert sample_dataframe['sentiment_label'].isin(valid_labels).all(), \
            "Invalid sentiment labels found"
    
    def test_text_not_empty(self, sample_dataframe):
        """Test text tidak kosong"""
        assert (sample_dataframe['review_text'].str.len() > 0).all(), \
            "Empty text found"
    
    def test_rating_range(self, sample_dataframe):
        """Test rating dalam range 1-5"""
        if 'rating' in sample_dataframe.columns:
            assert sample_dataframe['rating'].between(1, 5).all(), \
                "Rating outside valid range (1-5)"
    
    def test_data_types(self, sample_dataframe):
        """Test data types correct"""
        assert sample_dataframe['review_text'].dtype == object
        assert sample_dataframe['sentiment_label'].dtype == object
    
    def test_no_duplicates(self, sample_dataframe):
        """Test tidak ada duplicate reviews"""
        assert not sample_dataframe['review_text'].duplicated().any(), \
            "Duplicate reviews found"
    
    def test_balanced_distribution(self, sample_dataframe):
        """Test distribusi sentiment tidak terlalu imbalanced"""
        counts = sample_dataframe['sentiment_label'].value_counts()
        max_ratio = counts.max() / counts.min()
        assert max_ratio < 10, f"Data too imbalanced (ratio: {max_ratio:.2f})"
    
    def test_minimum_samples(self, sample_dataframe):
        """Test minimum jumlah samples per class"""
        min_samples_per_class = 2
        counts = sample_dataframe['sentiment_label'].value_counts()
        assert counts.min() >= min_samples_per_class, \
            f"Some classes have fewer than {min_samples_per_class} samples"


class TestDataQuality:
    """Test suite untuk data quality checks"""
    
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'review_text': [
                'Aplikasi bagus sekali!', 
                'Buruk', 
                'OK lah',
                'Sangat recommended',
                'Tidak suka'
            ],
            'cleaned_text': [
                'aplikasi bagus sekali', 
                'buruk', 
                'ok lah',
                'sangat recommended',
                'tidak suka'
            ],
            'sentiment_label': ['positive', 'negative', 'neutral', 'positive', 'negative']
        })
    
    def test_text_length_reasonable(self, sample_dataframe):
        """Test panjang text dalam range reasonable"""
        min_length = 2
        max_length = 5000
        
        lengths = sample_dataframe['review_text'].str.len()
        assert (lengths >= min_length).all(), f"Text too short (< {min_length} chars)"
        assert (lengths <= max_length).all(), f"Text too long (> {max_length} chars)"
    
    def test_cleaned_text_lowercase(self, sample_dataframe):
        """Test cleaned text dalam lowercase"""
        assert (sample_dataframe['cleaned_text'] == 
                sample_dataframe['cleaned_text'].str.lower()).all(), \
            "Cleaned text not in lowercase"
    
    def test_no_special_chars_in_cleaned(self, sample_dataframe):
        """Test tidak ada karakter special yang tidak perlu"""
        # Allow alphanumeric and spaces only
        pattern = r'^[a-z0-9\s]+$'
        assert sample_dataframe['cleaned_text'].str.match(pattern).all(), \
            "Special characters found in cleaned text"
    
    def test_data_freshness(self):
        """Test data tidak terlalu lama (< 1 tahun)"""
        # This would check scraped_at or created_at timestamp
        # For now, just placeholder
        pass


class TestModelInput:
    """Test suite untuk validasi model input"""
    
    def test_tokenizer_output_shape(self):
        """Test tokenizer output has correct shape"""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased')
        text = "Aplikasi ini bagus"
        max_length = 128
        
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        assert encoding['input_ids'].shape[1] == max_length
        assert encoding['attention_mask'].shape[1] == max_length
    
    def test_batch_processing(self):
        """Test batch processing works"""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-base-uncased')
        texts = ["Text 1", "Text 2", "Text 3"]
        
        encodings = tokenizer(
            texts,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        assert encodings['input_ids'].shape[0] == len(texts)


def validate_processed_data(file_path: str) -> dict:
    """
    Validate processed data file
    
    Args:
        file_path: Path to processed CSV file
    
    Returns:
        Dictionary dengan validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        results['stats']['total_samples'] = len(df)
        
        # Check required columns
        required_cols = ['review_text', 'cleaned_text', 'sentiment_label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing columns: {missing_cols}")
            return results
        
        # Check null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            results['warnings'].append(f"Null values found: {null_counts.to_dict()}")
        
        # Check sentiment distribution
        sentiment_dist = df['sentiment_label'].value_counts()
        results['stats']['sentiment_distribution'] = sentiment_dist.to_dict()
        
        # Check balance
        max_ratio = sentiment_dist.max() / sentiment_dist.min()
        if max_ratio > 5:
            results['warnings'].append(f"Data imbalanced (ratio: {max_ratio:.2f})")
        
        # Check text quality
        avg_length = df['review_text'].str.len().mean()
        results['stats']['avg_text_length'] = avg_length
        
        if avg_length < 10:
            results['warnings'].append(f"Average text length very short: {avg_length:.1f}")
        
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Error loading/validating data: {str(e)}")
    
    return results


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
