"""
MLflow Manager untuk Model Versioning, Registry, dan Experiment Tracking
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import torch
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Any, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowManager:
    """Manager untuk MLflow tracking, model registry, dan versioning"""
    
    def __init__(self, experiment_name: str = "sentiment-analysis", tracking_uri: str = None):
        """
        Initialize MLflow Manager
        
        Args:
            experiment_name: Nama experiment di MLflow
            tracking_uri: URI untuk MLflow tracking server (default: local mlruns/)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to local mlruns directory
            mlflow_dir = Path("mlruns").absolute()
            mlflow_dir.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file:///{mlflow_dir}")
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"MLflow experiment: {experiment_name}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """
        Start MLflow run
        
        Args:
            run_name: Nama run (default: timestamp)
            tags: Tags untuk run
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if tags is None:
            tags = {}
        
        # Add default tags
        tags.update({
            "model_type": "IndoBERT",
            "task": "sentiment_analysis",
            "framework": "pytorch",
            "timestamp": datetime.now().isoformat()
        })
        
        mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run_name}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Logged {len(metrics)} metrics")
    
    def log_metric(self, key: str, value: float, step: int = None):
        """Log single metric"""
        mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log artifact file"""
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")
    
    def log_dict(self, dictionary: Dict, artifact_file: str):
        """Log dictionary as JSON artifact"""
        mlflow.log_dict(dictionary, artifact_file)
        logger.info(f"Logged dict as: {artifact_file}")
    
    def log_model_pytorch(
        self, 
        model, 
        artifact_path: str = "model",
        registered_model_name: str = None,
        input_example: Any = None,
        signature = None,
        metadata: Dict = None
    ):
        """
        Log PyTorch model dengan model registry
        
        Args:
            model: PyTorch model
            artifact_path: Path untuk save artifact
            registered_model_name: Nama untuk register model
            input_example: Contoh input untuk model
            signature: Model signature
            metadata: Metadata untuk model
        """
        try:
            # Log model
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
                metadata=metadata
            )
            
            logger.info(f"Model logged to MLflow: {artifact_path}")
            
            if registered_model_name:
                logger.info(f"Model registered as: {registered_model_name}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
            raise
    
    def register_model(self, model_uri: str, model_name: str):
        """
        Register model ke MLflow Model Registry
        
        Args:
            model_uri: URI dari logged model
            model_name: Nama untuk register
        
        Returns:
            ModelVersion object
        """
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model: {model_name} version {model_version.version}")
            return model_version
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def transition_model_stage(
        self, 
        model_name: str, 
        version: int, 
        stage: str,
        archive_existing_versions: bool = False
    ):
        """
        Transition model stage (None, Staging, Production, Archived)
        
        Args:
            model_name: Nama model
            version: Version number
            stage: Target stage (Staging, Production, Archived)
            archive_existing_versions: Archive existing versions di stage yang sama
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Error transitioning model stage: {str(e)}")
            raise
    
    def get_latest_model_version(self, model_name: str, stage: str = None) -> Optional[int]:
        """
        Get latest model version
        
        Args:
            model_name: Nama model
            stage: Filter by stage (None, Staging, Production)
        
        Returns:
            Latest version number atau None
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if stage:
                versions = [v for v in versions if v.current_stage == stage]
            
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                return int(latest.version)
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest model version: {str(e)}")
            return None
    
    def load_model_from_registry(
        self, 
        model_name: str, 
        stage: str = "Production",
        version: int = None
    ):
        """
        Load model dari registry
        
        Args:
            model_name: Nama model
            stage: Model stage (Production, Staging, None)
            version: Specific version (override stage)
        
        Returns:
            Loaded model
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/{stage}"
            
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded model from registry: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def compare_models(self, run_ids: list) -> Dict:
        """
        Compare metrics dari multiple runs
        
        Args:
            run_ids: List of run IDs
        
        Returns:
            Dictionary dengan comparison
        """
        comparison = {}
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time
            }
        
        return comparison
    
    def get_best_run(self, metric_name: str, maximize: bool = True):
        """
        Get best run berdasarkan metric
        
        Args:
            metric_name: Nama metric untuk compare
            maximize: True untuk maximize, False untuk minimize
        
        Returns:
            Best run object
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'DESC' if maximize else 'ASC'}"],
            max_results=1
        )
        
        if runs:
            best_run = runs[0]
            logger.info(f"Best run: {best_run.info.run_id} with {metric_name}={best_run.data.metrics.get(metric_name)}")
            return best_run
        
        return None
    
    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def log_confusion_matrix(self, y_true, y_pred, labels: list):
        """Log confusion matrix sebagai artifact"""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save temporarily
        temp_path = Path("temp_confusion_matrix.png")
        plt.savefig(temp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(str(temp_path), "plots")
        
        # Remove temp file
        temp_path.unlink()
        
        logger.info("Logged confusion matrix")
    
    def log_training_plot(self, train_losses: list, val_losses: list = None):
        """Log training loss plot"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', marker='o')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        temp_path = Path("temp_training_plot.png")
        plt.savefig(temp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(str(temp_path), "plots")
        temp_path.unlink()
        
        logger.info("Logged training plot")
    
    def log_model_info(self, model, input_shape: tuple = None):
        """Log model architecture info"""
        info = {
            "model_class": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        
        if input_shape:
            info["input_shape"] = input_shape
        
        mlflow.log_dict(info, "model_info.json")
        logger.info(f"Logged model info: {info['num_parameters']:,} parameters")


def create_model_signature(input_example, output_example):
    """
    Create MLflow model signature
    
    Args:
        input_example: Example input tensor
        output_example: Example output tensor
    
    Returns:
        ModelSignature
    """
    from mlflow.models.signature import infer_signature
    import numpy as np
    
    # Convert tensors to numpy if needed
    if isinstance(input_example, torch.Tensor):
        input_example = input_example.cpu().numpy()
    if isinstance(output_example, torch.Tensor):
        output_example = output_example.cpu().numpy()
    
    signature = infer_signature(input_example, output_example)
    return signature


if __name__ == "__main__":
    # Test MLflow Manager
    manager = MLflowManager(experiment_name="test-experiment")
    
    manager.start_run(run_name="test_run")
    manager.log_params({"learning_rate": 0.001, "batch_size": 32})
    manager.log_metrics({"accuracy": 0.85, "loss": 0.3})
    manager.end_run()
    
    print("✅ MLflow Manager initialized successfully!")
