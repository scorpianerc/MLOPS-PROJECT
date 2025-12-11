"""
BERT Training Module for Sentiment Analysis
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import yaml
import logging
import json
from pathlib import Path
from tqdm import tqdm
import pickle
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.mlops.mlflow_manager import MLflowManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables for database
load_dotenv()


class SentimentDataset(Dataset):
    """Dataset untuk sentiment analysis dengan BERT"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTTrainer:
    """Trainer untuk IndoBERT"""
    
    def __init__(self, params, mlflow_manager=None):
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(params['bert_model'])
        
        # Initialize model
        self.model = None
        self.label_map = {}
        self.epoch_train_metrics = {}
        
        # MLflow manager
        self.mlflow_manager = mlflow_manager
        self.use_mlflow = mlflow_manager is not None
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Create label mapping
        unique_labels = sorted(df['sentiment_label'].unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}
        
        # Convert labels to integers
        df['label_id'] = df['sentiment_label'].map(self.label_map)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'].values,
            df['label_id'].values,
            test_size=self.params['test_size'],
            random_state=self.params['random_state'],
            stratify=df['label_id']
        )
        
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_dataloaders(self, X_train, X_test, y_train, y_test):
        """Create PyTorch DataLoaders"""
        train_dataset = SentimentDataset(
            X_train, y_train,
            self.tokenizer,
            self.params['max_length']
        )
        
        test_dataset = SentimentDataset(
            X_test, y_test,
            self.tokenizer,
            self.params['max_length']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params['batch_size'],
            shuffle=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.params['batch_size'],
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def train(self, train_loader, test_loader):
        """Train the model"""
        num_labels = len(self.label_map)
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.params['bert_model'],
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # Log model info to MLflow
        if self.use_mlflow:
            self.mlflow_manager.log_model_info(self.model)
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.params['learning_rate'])
        )
        
        total_steps = len(train_loader) * self.params['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_accuracy = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(self.params['epochs']):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.params['epochs']}")
            logger.info(f"{'='*50}")
            
            # Training
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in tqdm(train_loader, desc="Training"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                preds = torch.argmax(outputs.logits, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            # Calculate training metrics
            train_accuracy = accuracy_score(train_labels, train_preds)
            train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=0)
            train_recall = recall_score(train_labels, train_preds, average='weighted', zero_division=0)
            train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
            avg_train_loss = train_loss / len(train_loader)
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}")
            logger.info(f"Train Accuracy: {train_accuracy:.4f}")
            logger.info(f"Train Precision: {train_precision:.4f}")
            logger.info(f"Train Recall: {train_recall:.4f}")
            logger.info(f"Train F1: {train_f1:.4f}")
            
            # Store train metrics for this epoch
            self.epoch_train_metrics = {
                'accuracy': train_accuracy,
                'precision': train_precision,
                'recall': train_recall,
                'f1': train_f1
            }
            
            # Log to MLflow
            if self.use_mlflow:
                self.mlflow_manager.log_metrics({
                    f'train_loss': avg_train_loss,
                    f'train_accuracy': train_accuracy,
                    f'train_precision': train_precision,
                    f'train_recall': train_recall,
                    f'train_f1': train_f1
                }, step=epoch)
            
            # Evaluation
            test_accuracy, test_preds, test_labels = self.evaluate(test_loader)
            test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
            test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
            test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
            
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"Test Precision: {test_precision:.4f}")
            logger.info(f"Test Recall: {test_recall:.4f}")
            logger.info(f"Test F1: {test_f1:.4f}")
            
            # Log to MLflow
            if self.use_mlflow:
                self.mlflow_manager.log_metrics({
                    f'test_accuracy': test_accuracy,
                    f'test_precision': test_precision,
                    f'test_recall': test_recall,
                    f'test_f1': test_f1
                }, step=epoch)
            
            # Track for plots
            train_losses.append(avg_train_loss)
            val_accuracies.append(test_accuracy)
            
            # Save best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                self.save_model()
                logger.info(f"✓ Best model saved! Accuracy: {best_accuracy:.4f}")
        
        # Log training plot
        if self.use_mlflow:
            self.mlflow_manager.log_training_plot(train_losses)
        
        return best_accuracy
    
    def evaluate(self, test_loader):
        """Evaluate the model"""
        self.model.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(test_labels, test_preds)
        
        return accuracy, test_preds, test_labels
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate all metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=[self.reverse_label_map[i] for i in sorted(self.reverse_label_map.keys())]
        )
        
        return metrics, report
    
    def save_model(self):
        """Save model and tokenizer"""
        model_dir = Path('models') / 'bert_model'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # Save label map
        with open(model_dir / 'label_map.json', 'w') as f:
            json.dump(self.label_map, f)
        
        logger.info(f"Model saved to {model_dir}")


def main():
    """Main training function"""
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    training_params = params['training']
    
    # Initialize MLflow
    mlflow_manager = MLflowManager(experiment_name="sentiment-analysis-indobert")
    run_name = f"indobert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_manager.start_run(
        run_name=run_name,
        tags={
            "model_type": "IndoBERT",
            "dataset": "pintu_reviews"
        }
    )
    
    # Log parameters
    mlflow_manager.log_params({
        "bert_model": training_params['bert_model'],
        "batch_size": training_params['batch_size'],
        "learning_rate": training_params['learning_rate'],
        "epochs": training_params['epochs'],
        "max_length": training_params['max_length'],
        "test_size": training_params['test_size']
    })
    
    # Load data
    logger.info("Loading processed data...")
    df = pd.read_csv('data/processed/processed_reviews.csv')
    logger.info(f"Loaded {len(df)} samples")
    
    # Log dataset info
    mlflow_manager.log_metrics({
        "dataset_size": len(df),
        "num_classes": df['sentiment_label'].nunique()
    })
    
    # Initialize trainer
    trainer = BERTTrainer(training_params, mlflow_manager=mlflow_manager)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Create dataloaders
    train_loader, test_loader = trainer.create_dataloaders(
        X_train, X_test, y_train, y_test
    )
    
    # Train
    logger.info("\nStarting BERT training...")
    best_accuracy = trainer.train(train_loader, test_loader)
    
    # Final evaluation
    logger.info("\n" + "="*50)
    logger.info("FINAL EVALUATION")
    logger.info("="*50)
    
    test_accuracy, test_preds, test_labels = trainer.evaluate(test_loader)
    metrics, report = trainer.calculate_metrics(test_labels, test_preds)
    
    # Print results
    print(f"\n{'='*50}")
    print("FINAL METRICS")
    print(f"{'='*50}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1-Score: {metrics['f1']:.4f}")
    print(f"\n{report}")
    
    # Log confusion matrix to MLflow
    mlflow_manager.log_confusion_matrix(
        test_labels, 
        test_preds,
        labels=[trainer.reverse_label_map[i] for i in sorted(trainer.reverse_label_map.keys())]
    )
    
    # Log final metrics to MLflow
    mlflow_manager.log_metrics({
        "final_test_accuracy": metrics['accuracy'],
        "final_test_precision": metrics['precision'],
        "final_test_recall": metrics['recall'],
        "final_test_f1": metrics['f1']
    })
    
    # Get train metrics from last epoch
    train_metrics = trainer.epoch_train_metrics
    
    mlflow_manager.log_metrics({
        "final_train_accuracy": train_metrics['accuracy'],
        "final_train_precision": train_metrics['precision'],
        "final_train_recall": train_metrics['recall'],
        "final_train_f1": train_metrics['f1']
    })
    
    # Save both train and test metrics to JSON
    all_metrics = {
        'test': metrics,
        'train': train_metrics
    }
    with open('models/bert_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save metrics to PostgreSQL for Grafana dashboard
    try:
        logger.info("\n💾 Saving metrics to database for Grafana...")
        
        db_config = {
            'host': os.getenv('POSTGRES_HOST') if os.getenv('POSTGRES_HOST') != 'postgres' else 'localhost',
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'sentiment_db'),
            'user': os.getenv('POSTGRES_USER', 'sentiment_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'password')
        }
        
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Insert test metrics
        cursor.execute("""
            INSERT INTO model_metrics 
            (model_name, accuracy, precision_score, recall_score, f1_score, 
             train_accuracy, train_precision, train_recall, train_f1, 
             dataset_type, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            'bert-indonesian-base', 
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            train_metrics['accuracy'],
            train_metrics['precision'],
            train_metrics['recall'],
            train_metrics['f1'],
            'test',
            datetime.now()
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ Train & Test metrics saved to database!")
        logger.info(f"   📊 Test  - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        logger.info(f"   📊 Train - Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        logger.info("📈 Dashboard akan menampilkan kedua metrics dalam 30 detik")
        logger.info("🔗 Dashboard: http://localhost:3000/d/sentiment-dashboard-v3/sentiment-analysis-dashboard")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not save metrics to database: {e}")
        logger.warning("Metrics tetap tersimpan di models/bert_metrics.json")
    
    # Log model to MLflow
    try:
        logger.info("\n🚀 Logging model to MLflow...")
        
        # Prepare sample input for signature
        sample_texts = df['cleaned_text'].head(5).values
        sample_encoding = trainer.tokenizer(
            sample_texts.tolist(),
            add_special_tokens=True,
            max_length=training_params['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Log model with registry
        mlflow_manager.log_model_pytorch(
            model=trainer.model,
            artifact_path="model",
            registered_model_name="indobert-sentiment-analysis"
        )
        
        # Transition to production if accuracy > 0.80
        if metrics['accuracy'] > 0.80:
            latest_version = mlflow_manager.get_latest_model_version("indobert-sentiment-analysis")
            if latest_version:
                mlflow_manager.transition_model_stage(
                    model_name="indobert-sentiment-analysis",
                    version=latest_version,
                    stage="Production",
                    archive_existing_versions=True
                )
                logger.info(f"✅ Model v{latest_version} promoted to Production!")
        
        logger.info("✅ Model logged to MLflow successfully!")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not log model to MLflow: {e}")
    
    # End MLflow run
    mlflow_manager.end_run()
    
    logger.info("\n✓ Training complete!")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    logger.info(f"📊 MLflow UI: http://localhost:5000")
    logger.info(f"   Run: mlflow ui --port 5000")


if __name__ == '__main__':
    main()
