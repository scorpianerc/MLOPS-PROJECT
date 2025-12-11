"""
Data Drift Detection & Model Drift Monitoring
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
import logging
import psycopg2
from typing import Dict, Tuple, List
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detect data drift menggunakan statistical tests
    """
    
    def __init__(self, reference_data: pd.DataFrame, significance_level: float = 0.05):
        """
        Initialize drift detector
        
        Args:
            reference_data: Baseline/training data untuk comparison
            significance_level: P-value threshold untuk detect drift
        """
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.drift_results = {}
    
    def detect_numerical_drift(
        self, 
        feature_name: str,
        current_data: pd.DataFrame,
        test_type: str = 'ks'
    ) -> Dict:
        """
        Detect drift untuk numerical features menggunakan statistical tests
        
        Args:
            feature_name: Nama feature
            current_data: Current/production data
            test_type: 'ks' (Kolmogorov-Smirnov) atau 'chi2'
        
        Returns:
            Dictionary dengan drift results
        """
        ref_values = self.reference_data[feature_name].dropna()
        cur_values = current_data[feature_name].dropna()
        
        if test_type == 'ks':
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(ref_values, cur_values)
            test_name = "Kolmogorov-Smirnov"
        else:
            # Chi-square test (for categorical or binned numerical)
            ref_bins = pd.cut(ref_values, bins=10).value_counts()
            cur_bins = pd.cut(cur_values, bins=10).value_counts()
            statistic, p_value = stats.chisquare(cur_bins, ref_bins)
            test_name = "Chi-Square"
        
        drift_detected = p_value < self.significance_level
        
        result = {
            'feature': feature_name,
            'test': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'drift_detected': drift_detected,
            'significance_level': self.significance_level,
            'ref_mean': float(ref_values.mean()),
            'cur_mean': float(cur_values.mean()),
            'ref_std': float(ref_values.std()),
            'cur_std': float(cur_values.std()),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def detect_categorical_drift(
        self,
        feature_name: str,
        current_data: pd.DataFrame
    ) -> Dict:
        """
        Detect drift untuk categorical features
        
        Args:
            feature_name: Nama feature
            current_data: Current data
        
        Returns:
            Dictionary dengan drift results
        """
        ref_dist = self.reference_data[feature_name].value_counts(normalize=True)
        cur_dist = current_data[feature_name].value_counts(normalize=True)
        
        # Ensure same categories
        all_categories = set(ref_dist.index) | set(cur_dist.index)
        ref_dist = ref_dist.reindex(all_categories, fill_value=0)
        cur_dist = cur_dist.reindex(all_categories, fill_value=0)
        
        # Chi-square test
        # Convert to counts (approximate)
        ref_counts = (ref_dist * len(self.reference_data)).astype(int)
        cur_counts = (cur_dist * len(current_data)).astype(int)
        
        statistic, p_value = stats.chisquare(cur_counts, ref_counts)
        drift_detected = p_value < self.significance_level
        
        # Calculate distribution shift
        dist_shift = np.abs(ref_dist - cur_dist).max()
        
        result = {
            'feature': feature_name,
            'test': 'Chi-Square',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'drift_detected': drift_detected,
            'max_distribution_shift': float(dist_shift),
            'ref_distribution': ref_dist.to_dict(),
            'cur_distribution': cur_dist.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def detect_text_length_drift(
        self,
        text_column: str,
        current_data: pd.DataFrame
    ) -> Dict:
        """
        Detect drift dalam text length distribution
        """
        ref_lengths = self.reference_data[text_column].str.len()
        cur_lengths = current_data[text_column].str.len()
        
        return self.detect_numerical_drift('text_length', 
                                          pd.DataFrame({'text_length': cur_lengths}))
    
    def detect_all_drifts(
        self,
        current_data: pd.DataFrame,
        numerical_features: List[str] = None,
        categorical_features: List[str] = None
    ) -> Dict:
        """
        Detect drift untuk semua features
        
        Returns:
            Dictionary dengan all drift results
        """
        results = {
            'overall_drift_detected': False,
            'features': [],
            'summary': {}
        }
        
        # Detect numerical drifts
        if numerical_features:
            for feature in numerical_features:
                if feature in self.reference_data.columns and feature in current_data.columns:
                    drift_result = self.detect_numerical_drift(feature, current_data)
                    results['features'].append(drift_result)
                    if drift_result['drift_detected']:
                        results['overall_drift_detected'] = True
        
        # Detect categorical drifts
        if categorical_features:
            for feature in categorical_features:
                if feature in self.reference_data.columns and feature in current_data.columns:
                    drift_result = self.detect_categorical_drift(feature, current_data)
                    results['features'].append(drift_result)
                    if drift_result['drift_detected']:
                        results['overall_drift_detected'] = True
        
        # Summary statistics
        total_features = len(results['features'])
        drifted_features = sum(1 for f in results['features'] if f['drift_detected'])
        
        results['summary'] = {
            'total_features_checked': total_features,
            'features_with_drift': drifted_features,
            'drift_percentage': (drifted_features / total_features * 100) if total_features > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return results


class ModelDriftMonitor:
    """
    Monitor model performance drift over time
    """
    
    def __init__(self, db_config: Dict):
        """
        Initialize model drift monitor
        
        Args:
            db_config: Database configuration
        """
        self.db_config = db_config
        self.baseline_metrics = None
    
    def get_baseline_metrics(self) -> Dict:
        """Get baseline metrics dari database (latest training metrics)"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT accuracy, precision_score, recall_score, f1_score
                FROM model_metrics
                WHERE dataset_type = 'test'
                ORDER BY created_at DESC
                LIMIT 1;
            """)
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return {
                    'accuracy': result[0],
                    'precision': result[1],
                    'recall': result[2],
                    'f1': result[3]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting baseline metrics: {e}")
            return None
    
    def calculate_current_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Calculate metrics dari current predictions"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def detect_performance_drift(
        self,
        current_metrics: Dict,
        threshold: float = 0.05
    ) -> Dict:
        """
        Detect performance drift
        
        Args:
            current_metrics: Current performance metrics
            threshold: Degradation threshold (default 5%)
        
        Returns:
            Dictionary dengan drift detection results
        """
        if self.baseline_metrics is None:
            self.baseline_metrics = self.get_baseline_metrics()
        
        if self.baseline_metrics is None:
            return {
                'drift_detected': False,
                'reason': 'No baseline metrics available'
            }
        
        results = {
            'drift_detected': False,
            'degraded_metrics': [],
            'baseline': self.baseline_metrics,
            'current': current_metrics,
            'differences': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            baseline_val = self.baseline_metrics[metric_name]
            current_val = current_metrics[metric_name]
            diff = baseline_val - current_val
            
            results['differences'][metric_name] = {
                'absolute': float(diff),
                'relative': float(diff / baseline_val * 100) if baseline_val > 0 else 0
            }
            
            # Check if degradation exceeds threshold
            if diff > threshold:
                results['drift_detected'] = True
                results['degraded_metrics'].append({
                    'metric': metric_name,
                    'baseline': float(baseline_val),
                    'current': float(current_val),
                    'degradation': float(diff)
                })
        
        return results
    
    def get_metrics_trend(self, days: int = 30) -> pd.DataFrame:
        """
        Get metrics trend dari database
        
        Args:
            days: Number of days untuk trend analysis
        
        Returns:
            DataFrame dengan metrics over time
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            
            query = f"""
                SELECT 
                    created_at,
                    accuracy,
                    precision_score as precision,
                    recall_score as recall,
                    f1_score as f1,
                    dataset_type
                FROM model_metrics
                WHERE created_at >= NOW() - INTERVAL '{days} days'
                ORDER BY created_at ASC;
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting metrics trend: {e}")
            return pd.DataFrame()
    
    def analyze_trend(self, days: int = 30) -> Dict:
        """
        Analyze performance trend
        
        Returns:
            Dictionary dengan trend analysis
        """
        df = self.get_metrics_trend(days)
        
        if df.empty:
            return {
                'trend': 'unknown',
                'reason': 'Insufficient data'
            }
        
        # Calculate linear regression slope untuk each metric
        results = {
            'period_days': days,
            'num_datapoints': len(df),
            'trends': {},
            'overall_trend': 'stable',
            'timestamp': datetime.now().isoformat()
        }
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in df.columns:
                values = df[metric].values
                x = np.arange(len(values))
                
                # Linear regression
                slope, intercept = np.polyfit(x, values, 1)
                
                # Determine trend
                if slope > 0.001:
                    trend = 'improving'
                elif slope < -0.001:
                    trend = 'degrading'
                else:
                    trend = 'stable'
                
                results['trends'][metric] = {
                    'slope': float(slope),
                    'trend': trend,
                    'current_value': float(values[-1]),
                    'min_value': float(values.min()),
                    'max_value': float(values.max()),
                    'mean_value': float(values.mean())
                }
                
                # Update overall trend
                if trend == 'degrading':
                    results['overall_trend'] = 'degrading'
        
        return results


class PredictionLogger:
    """
    Log predictions untuk monitoring
    """
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def log_prediction(
        self,
        review_id: int,
        text: str,
        predicted_sentiment: str,
        confidence: float,
        model_version: str = None
    ):
        """Log single prediction"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO prediction_logs 
                (review_id, text, predicted_sentiment, confidence, model_version, created_at)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (
                review_id, text, predicted_sentiment, confidence,
                model_version, datetime.now()
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
    
    def get_recent_predictions(self, hours: int = 24) -> pd.DataFrame:
        """Get recent predictions"""
        try:
            conn = psycopg2.connect(**self.db_config)
            
            query = f"""
                SELECT *
                FROM prediction_logs
                WHERE created_at >= NOW() - INTERVAL '{hours} hours'
                ORDER BY created_at DESC;
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return pd.DataFrame()


def create_prediction_logs_table(db_config: Dict):
    """Create prediction_logs table jika belum ada"""
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                review_id INTEGER,
                text TEXT,
                predicted_sentiment VARCHAR(50),
                confidence FLOAT,
                model_version VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_prediction_logs_created 
            ON prediction_logs(created_at);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ prediction_logs table created/verified")
        
    except Exception as e:
        logger.error(f"Error creating table: {e}")


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'database': os.getenv('POSTGRES_DB', 'sentiment_db'),
        'user': os.getenv('POSTGRES_USER', 'sentiment_user'),
        'password': os.getenv('POSTGRES_PASSWORD', 'password')
    }
    
    # Create prediction logs table
    create_prediction_logs_table(db_config)
    
    # Test model drift monitor
    monitor = ModelDriftMonitor(db_config)
    trend = monitor.analyze_trend(days=30)
    print(json.dumps(trend, indent=2))
