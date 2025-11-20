"""
Model Evaluation
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import Dict
import yaml

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class untuk evaluasi model"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_decoder = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
    def load_model(self, model_dir: str = 'models'):
        """Load trained model"""
        model_dir = Path(model_dir)
        
        # Load model
        with open(model_dir / 'sentiment_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load vectorizer
        with open(model_dir / 'vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        logger.info("Model loaded successfully")
    
    def evaluate(self, df: pd.DataFrame) -> Dict:
        """Evaluate model on test data"""
        logger.info("Evaluating model...")
        
        # Prepare data
        X = df['cleaned_text']
        y_true = df['sentiment_encoded']
        
        # Vectorize
        X_vec = self.vectorizer.transform(X)
        
        # Predict
        y_pred = self.model.predict(X_vec)
        y_pred_proba = self.model.predict_proba(X_vec)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Per-class metrics
        for i, label in self.label_decoder.items():
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            metrics[f'{label}_precision'] = precision_score(y_true_binary, y_pred_binary)
            metrics[f'{label}_recall'] = recall_score(y_true_binary, y_pred_binary)
            metrics[f'{label}_f1'] = f1_score(y_true_binary, y_pred_binary)
        
        logger.info("Evaluation complete")
        
        return metrics, y_true, y_pred, y_pred_proba
    
    def plot_roc_curve(self, y_true, y_pred_proba, output_path: str):
        """Plot ROC curve for each class"""
        plt.figure(figsize=(10, 8))
        
        for i, label in self.label_decoder.items():
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Multi-class')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {output_path}")
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, output_path: str):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=(10, 8))
        
        for i, label in self.label_decoder.items():
            y_true_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_proba[:, i])
            
            plt.plot(recall, precision, label=label)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-Recall curve saved to {output_path}")
    
    def save_metrics(self, metrics: Dict, output_path: str = 'models/evaluation_metrics.json'):
        """Save evaluation metrics"""
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation metrics saved to {output_path}")


def main():
    """Main evaluation function"""
    from dotenv import load_dotenv
    
    load_dotenv()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load model
    evaluator.load_model()
    
    # Load test data
    df = pd.read_csv('data/processed/processed_reviews.csv')
    
    # Evaluate
    metrics, y_true, y_pred, y_pred_proba = evaluator.evaluate(df)
    
    # Plot curves
    evaluator.plot_roc_curve(y_true, y_pred_proba, 'models/roc_curve.png')
    evaluator.plot_precision_recall_curve(y_true, y_pred_proba, 'models/precision_recall_curve.png')
    
    # Save metrics
    evaluator.save_metrics(metrics)
    
    # Print results
    print("\n=== Evaluation Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()
