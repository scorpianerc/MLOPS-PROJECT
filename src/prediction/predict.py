"""
Prediction Pipeline
"""

import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing.preprocess import IndonesianTextPreprocessor
from src.data_collection.database import DatabaseManager

logger = logging.getLogger(__name__)


class SentimentPredictor:
    """Class untuk prediksi sentiment"""
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize predictor
        
        Args:
            model_dir: Directory berisi model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.vectorizer = None
        self.preprocessor = None
        self.label_decoder = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, vectorizer, dan preprocessor"""
        try:
            logger.info("Loading model artifacts...")
            
            # Load model
            with open(self.model_dir / 'sentiment_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load vectorizer
            with open(self.model_dir / 'vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load preprocessor
            preprocessor_path = Path('data/processed/preprocessor.pkl')
            if preprocessor_path.exists():
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
            else:
                # Create new preprocessor if not exists
                import yaml
                with open('config/params.yaml', 'r') as f:
                    params = yaml.safe_load(f)
                self.preprocessor = IndonesianTextPreprocessor(params['preprocessing'])
            
            logger.info("All artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess single text"""
        return self.preprocessor.preprocess(text)
    
    def predict_single(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for single text
        
        Args:
            text: Input text
        
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        # Preprocess
        cleaned_text = self.preprocess_text(text)
        
        if not cleaned_text:
            return 'neutral', 0.33
        
        # Vectorize
        text_vec = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.model.predict(text_vec)[0]
        proba = self.model.predict_proba(text_vec)[0]
        
        sentiment = self.label_decoder[prediction]
        confidence = float(proba[prediction])
        
        return sentiment, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for batch of texts
        
        Args:
            texts: List of texts
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for text in texts:
            sentiment, confidence = self.predict_single(text)
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence
            })
        
        return results
    
    def predict_dataframe(self, df: pd.DataFrame, text_column: str = 'review_text') -> pd.DataFrame:
        """
        Predict sentiment for DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Column name containing text
        
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Predicting sentiment for {len(df)} reviews...")
        
        df = df.copy()
        
        # Preprocess texts
        df['cleaned_text'] = df[text_column].apply(self.preprocess_text)
        
        # Filter out empty texts
        valid_mask = df['cleaned_text'] != ''
        
        if valid_mask.sum() == 0:
            logger.warning("No valid texts to predict")
            df['sentiment'] = 'neutral'
            df['sentiment_score'] = 0.33
            return df
        
        # Vectorize
        texts_vec = self.vectorizer.transform(df.loc[valid_mask, 'cleaned_text'])
        
        # Predict
        predictions = self.model.predict(texts_vec)
        probas = self.model.predict_proba(texts_vec)
        
        # Map predictions
        df.loc[valid_mask, 'sentiment'] = [self.label_decoder[p] for p in predictions]
        df.loc[valid_mask, 'sentiment_score'] = [float(proba[pred]) for pred, proba in zip(predictions, probas)]
        
        # Fill invalid texts
        df.loc[~valid_mask, 'sentiment'] = 'neutral'
        df.loc[~valid_mask, 'sentiment_score'] = 0.33
        
        logger.info("Prediction complete!")
        
        return df


class PredictionPipeline:
    """Pipeline untuk prediksi otomatis"""
    
    def __init__(self):
        """Initialize pipeline"""
        self.predictor = SentimentPredictor()
        self.db_manager = DatabaseManager()
        
    def run(self, batch_size: int = 100):
        """
        Run prediction pipeline
        
        Args:
            batch_size: Number of reviews to process at once
        """
        try:
            logger.info("Starting prediction pipeline...")
            
            # Get unpredicted reviews from database
            df_reviews = self.db_manager.get_unpredicted_reviews(limit=batch_size)
            
            if len(df_reviews) == 0:
                logger.info("No unpredicted reviews found")
                return
            
            logger.info(f"Found {len(df_reviews)} unpredicted reviews")
            
            # Predict
            df_predicted = self.predictor.predict_dataframe(df_reviews)
            
            # Prepare predictions for database
            predictions = []
            for _, row in df_predicted.iterrows():
                predictions.append({
                    'review_id': row['review_id'],
                    'sentiment': row['sentiment'],
                    'sentiment_score': row['sentiment_score']
                })
            
            # Update database
            updated_count = self.db_manager.update_predictions(predictions)
            
            logger.info(f"Updated {updated_count} predictions in database")
            
            # Get and log stats
            stats = self.db_manager.get_sentiment_stats()
            logger.info(f"Current sentiment distribution:")
            logger.info(f"  Positive: {stats['positive']} ({stats['positive_pct']:.2f}%)")
            logger.info(f"  Neutral: {stats['neutral']} ({stats['neutral_pct']:.2f}%)")
            logger.info(f"  Negative: {stats['negative']} ({stats['negative_pct']:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {str(e)}")
            raise


def main():
    """Main function untuk prediction"""
    from dotenv import load_dotenv
    import argparse
    
    load_dotenv()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Sentiment Prediction Pipeline')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for prediction')
    parser.add_argument('--mode', choices=['pipeline', 'test'], default='pipeline', 
                       help='Run mode: pipeline or test')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # Test mode - predict sample texts
        predictor = SentimentPredictor()
        
        test_texts = [
            "Aplikasi bagus banget, sangat membantu!",
            "Jelek aplikasinya, sering error",
            "Biasa aja sih, tidak ada yang spesial",
            "Mantap sekali, fiturnya lengkap dan mudah digunakan",
            "Kecewa, banyak bug dan lambat"
        ]
        
        print("\n=== Testing Predictor ===")
        for text in test_texts:
            sentiment, confidence = predictor.predict_single(text)
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment} (confidence: {confidence:.4f})")
            print()
    
    else:
        # Pipeline mode - process database
        pipeline = PredictionPipeline()
        pipeline.run(batch_size=args.batch_size)


if __name__ == '__main__':
    main()
