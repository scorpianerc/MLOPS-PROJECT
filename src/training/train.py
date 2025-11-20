"""
Model Training dengan DVC dan MLflow tracking
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import yaml

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


class SentimentModelTrainer:
    """Class untuk training sentiment analysis model"""
    
    def __init__(self, params: Dict):
        """
        Initialize trainer
        
        Args:
            params: Dictionary berisi training parameters
        """
        self.params = params
        self.model = None
        self.vectorizer = None
        self.label_encoder = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load processed data"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} samples")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for training...")
        
        # Encode labels
        df['sentiment_encoded'] = df['sentiment_label'].map(self.label_encoder)
        
        # Split data
        X = df['cleaned_text']
        y = df['sentiment_encoded']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.params['test_size'],
            random_state=self.params['random_state'],
            stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def create_vectorizer(self) -> TfidfVectorizer:
        """Create TF-IDF vectorizer"""
        vectorizer = TfidfVectorizer(
            max_features=self.params['max_features'],
            ngram_range=tuple(self.params['ngram_range']),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        return vectorizer
    
    def create_model(self) -> object:
        """Create model based on model_type parameter"""
        model_type = self.params['model_type']
        
        logger.info(f"Creating model: {model_type}")
        
        if model_type == 'logistic_regression':
            model = LogisticRegression(
                random_state=self.params['random_state'],
                max_iter=2000,
                C=10.0,  # Increase regularization strength
                class_weight='balanced',  # Handle imbalanced data
                solver='saga'  # Better for large datasets
            )
        elif model_type == 'naive_bayes':
            model = MultinomialNB(alpha=0.5)
        elif model_type == 'svm':
            model = LinearSVC(
                random_state=self.params['random_state'],
                max_iter=2000,
                C=1.0,
                class_weight='balanced'
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.params['random_state'],
                n_jobs=-1,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def train(self, X_train, X_test, y_train, y_test) -> Dict:
        """
        Train model
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
        
        Returns:
            Dictionary berisi metrics
        """
        logger.info("Starting training...")
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.params)
            
            # Create and fit vectorizer
            self.vectorizer = self.create_vectorizer()
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            logger.info(f"Vectorized features shape: {X_train_vec.shape}")
            
            # Create and train model
            self.model = self.create_model()
            self.model.fit(X_train_vec, y_train)
            
            logger.info("Training completed!")
            
            # Make predictions
            y_train_pred = self.model.predict(X_train_vec)
            y_test_pred = self.model.predict(X_test_vec)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                y_train, y_train_pred,
                y_test, y_test_pred
            )
            
            # Log metrics to MLflow
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            
            logger.info("Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
            
            return metrics
    
    def _calculate_metrics(
        self, 
        y_train_true, y_train_pred,
        y_test_true, y_test_pred
    ) -> Dict:
        """Calculate training and test metrics"""
        
        metrics = {
            # Training metrics
            'train_accuracy': accuracy_score(y_train_true, y_train_pred),
            'train_precision': precision_score(y_train_true, y_train_pred, average='weighted'),
            'train_recall': recall_score(y_train_true, y_train_pred, average='weighted'),
            'train_f1': f1_score(y_train_true, y_train_pred, average='weighted'),
            
            # Test metrics
            'test_accuracy': accuracy_score(y_test_true, y_test_pred),
            'test_precision': precision_score(y_test_true, y_test_pred, average='weighted'),
            'test_recall': recall_score(y_test_true, y_test_pred, average='weighted'),
            'test_f1': f1_score(y_test_true, y_test_pred, average='weighted'),
        }
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_test_true, y_test_pred,
            target_names=['negative', 'neutral', 'positive']
        )
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, output_path: str):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
    
    def save_model(self, model_dir: str = 'models'):
        """Save model and vectorizer"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / 'sentiment_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {model_path}")
        
        # Save vectorizer
        vectorizer_path = model_dir / 'vectorizer.pkl'
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"Vectorizer saved to {vectorizer_path}")
        
        # Save label encoder
        encoder_path = model_dir / 'label_encoder.json'
        with open(encoder_path, 'w') as f:
            json.dump(self.label_encoder, f)
        logger.info(f"Label encoder saved to {encoder_path}")
    
    def save_metrics(self, metrics: Dict, output_path: str = 'models/metrics.json'):
        """Save metrics to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Filter only numeric metrics for JSON
        json_metrics = {
            k: v for k, v in metrics.items() 
            if isinstance(v, (int, float))
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")


def main():
    """Main function untuk training"""
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    training_params = params['training']
    
    # Initialize trainer
    trainer = SentimentModelTrainer(training_params)
    
    # Load data
    df = trainer.load_data('data/processed/processed_reviews.csv')
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Train model
    metrics = trainer.train(X_train, X_test, y_train, y_test)
    
    # Plot confusion matrix
    X_test_vec = trainer.vectorizer.transform(X_test)
    y_test_pred = trainer.model.predict(X_test_vec)
    trainer.plot_confusion_matrix(y_test, y_test_pred, 'models/confusion_matrix.png')
    
    # Save model
    trainer.save_model()
    
    # Save metrics
    trainer.save_metrics(metrics)
    
    # Print classification report
    print("\n=== Classification Report ===")
    print(metrics['classification_report'])
    
    print("\n=== Training Complete ===")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test F1 Score: {metrics['test_f1']:.4f}")


if __name__ == '__main__':
    main()
