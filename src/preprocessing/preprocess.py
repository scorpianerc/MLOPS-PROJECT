"""
Text Preprocessing untuk Bahasa Indonesia
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from pathlib import Path
import pickle

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class IndonesianTextPreprocessor:
    """Preprocessor untuk teks Bahasa Indonesia"""
    
    def __init__(self, params: Dict = None):
        """
        Initialize preprocessor
        
        Args:
            params: Dictionary berisi parameter preprocessing
        """
        self.params = params or {}
        
        # Initialize Sastrawi
        self.stemmer_factory = StemmerFactory()
        self.stemmer = self.stemmer_factory.create_stemmer()
        
        self.stopword_factory = StopWordRemoverFactory()
        self.stopword_remover = self.stopword_factory.create_stop_word_remover()
        
        # Additional Indonesian stopwords
        self.custom_stopwords = set([
            'yg', 'dg', 'rt', 'dgn', 'ny', 'dr', 'jg', 'tdk', 'krn', 'pd', 'kl',
            'pake', 'gak', 'gk', 'banget', 'bgt', 'sih', 'dong', 'kok', 'nya'
        ])
        
        # Slang dictionary (contoh)
        self.slang_dict = {
            'gak': 'tidak',
            'gk': 'tidak',
            'ga': 'tidak',
            'tdk': 'tidak',
            'gpp': 'tidak apa apa',
            'bgt': 'banget',
            'bgt': 'sekali',
            'org': 'orang',
            'yg': 'yang',
            'dgn': 'dengan',
            'dg': 'dengan',
            'krn': 'karena',
            'utk': 'untuk',
            'trs': 'terus',
            'jd': 'jadi',
            'jdnya': 'jadinya',
            'hrs': 'harus',
            'tp': 'tapi',
            'klo': 'kalau',
            'kl': 'kalau',
            'udh': 'sudah',
            'sdh': 'sudah',
            'blm': 'belum',
            'bs': 'bisa',
            'bisa': 'dapat',
            'emg': 'memang',
            'emng': 'memang',
        }
        
        logger.info("Indonesian Text Preprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text dari karakter yang tidak diinginkan
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string
        text = str(text)
        
        # Lowercase
        if self.params.get('lowercase', True):
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (optional)
        if self.params.get('remove_numbers', False):
            text = re.sub(r'\d+', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (optional)
        if self.params.get('remove_punctuation', True):
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Strip
        text = text.strip()
        
        return text
    
    def normalize_slang(self, text: str) -> str:
        """
        Normalize slang words
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        words = text.split()
        normalized_words = [self.slang_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords
        
        Args:
            text: Input text
        
        Returns:
            Text without stopwords
        """
        if not self.params.get('remove_stopwords', True):
            return text
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords
        indonesian_stopwords = set(stopwords.words('indonesian'))
        all_stopwords = indonesian_stopwords.union(self.custom_stopwords)
        
        filtered_words = [word for word in words if word not in all_stopwords]
        
        return ' '.join(filtered_words)
    
    def stem_text(self, text: str) -> str:
        """
        Apply stemming
        
        Args:
            text: Input text
        
        Returns:
            Stemmed text
        """
        if not self.params.get('stem', True):
            return text
        
        return self.stemmer.stem(text)
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        # Clean
        text = self.clean_text(text)
        
        # Check minimum length
        if len(text) < self.params.get('min_text_length', 10):
            return ''
        
        # Normalize slang
        text = self.normalize_slang(text)
        
        # Remove stopwords
        text = self.remove_stopwords(text)
        
        # Stemming
        text = self.stem_text(text)
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'review_text') -> pd.DataFrame:
        """
        Preprocess entire DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Nama kolom yang berisi text
        
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Preprocessing {len(df)} reviews...")
        
        df = df.copy()
        
        # Apply preprocessing
        df['cleaned_text'] = df[text_column].apply(self.preprocess)
        
        # Remove empty texts
        df = df[df['cleaned_text'] != '']
        
        # Create sentiment label based on rating
        df['sentiment_label'] = df['rating'].apply(self._rating_to_sentiment)
        
        logger.info(f"Preprocessing complete. {len(df)} reviews remaining.")
        
        return df
    
    def _rating_to_sentiment(self, rating: int) -> str:
        """
        Convert rating to sentiment label
        
        Args:
            rating: Rating (1-5)
        
        Returns:
            Sentiment label (positive, negative, neutral)
        """
        if rating >= 4:
            return 'positive'
        elif rating <= 2:
            return 'negative'
        else:
            return 'neutral'


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Text length features
    df['text_length'] = df['cleaned_text'].apply(len)
    df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    
    # Rating features
    df['is_high_rating'] = (df['rating'] >= 4).astype(int)
    df['is_low_rating'] = (df['rating'] <= 2).astype(int)
    
    # Thumbs up feature (popularity)
    df['thumbs_up_log'] = np.log1p(df['thumbs_up'])
    
    # Time features
    if 'review_date' in df.columns:
        df['review_date'] = pd.to_datetime(df['review_date'])
        df['review_year'] = df['review_date'].dt.year
        df['review_month'] = df['review_date'].dt.month
        df['review_day_of_week'] = df['review_date'].dt.dayofweek
    
    return df


def main():
    """Main function untuk preprocessing"""
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    preprocessing_params = params['preprocessing']
    
    # Load raw data
    logger.info("Loading raw data...")
    df = pd.read_csv('data/raw/reviews.csv')
    logger.info(f"Loaded {len(df)} reviews")
    
    # Initialize preprocessor
    preprocessor = IndonesianTextPreprocessor(preprocessing_params)
    
    # Preprocess
    df_processed = preprocessor.preprocess_dataframe(df)
    
    # Create features
    df_processed = create_features(df_processed)
    
    # Save processed data
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'processed_reviews.csv'
    df_processed.to_csv(output_file, index=False)
    logger.info(f"Processed data saved to {output_file}")
    
    # Save preprocessor
    preprocessor_file = output_dir / 'preprocessor.pkl'
    with open(preprocessor_file, 'wb') as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Preprocessor saved to {preprocessor_file}")
    
    # Print statistics
    print("\n=== Preprocessing Statistics ===")
    print(f"Total reviews: {len(df_processed)}")
    print(f"\nSentiment distribution:")
    print(df_processed['sentiment_label'].value_counts())
    print(f"\nRating distribution:")
    print(df_processed['rating'].value_counts().sort_index())


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
