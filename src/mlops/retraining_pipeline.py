"""
Automated Retraining Pipeline dengan Data-Driven dan Performance-Based Triggers
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import psycopg2
import pandas as pd
from typing import Dict, Tuple
import os
from dotenv import load_dotenv
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.mlops.drift_detection import ModelDriftMonitor, DataDriftDetector

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """
    Evaluate retraining triggers
    """
    
    def __init__(self, db_config: Dict):
        """
        Initialize retraining trigger
        
        Args:
            db_config: Database configuration
        """
        self.db_config = db_config
        self.drift_monitor = ModelDriftMonitor(db_config)
        
        # Thresholds
        self.min_days_between_training = 7
        self.max_days_between_training = 30
        self.new_data_threshold = 500
        self.error_rate_threshold = 0.12  # 12%
        self.accuracy_drop_threshold = 0.03  # 3%
        self.drift_score_threshold = 0.3
    
    def get_last_training_date(self) -> datetime:
        """Get tanggal training terakhir"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT MAX(created_at)
                FROM model_metrics
                WHERE dataset_type = 'test';
            """)
            
            result = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return result if result else datetime.now() - timedelta(days=365)
            
        except Exception as e:
            logger.error(f"Error getting last training date: {e}")
            return datetime.now() - timedelta(days=365)
    
    def get_new_data_count(self, since_date: datetime) -> int:
        """Get jumlah data baru sejak last training"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*)
                FROM reviews
                WHERE scraped_at > %s;
            """, (since_date,))
            
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return count
            
        except Exception as e:
            logger.error(f"Error getting new data count: {e}")
            return 0
    
    def get_user_feedback_error_rate(self, days: int = 7) -> float:
        """Get error rate dari user feedback"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check if user_feedback table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'user_feedback'
                );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                cursor.close()
                conn.close()
                return 0.0
            
            # Get error rate
            cursor.execute(f"""
                SELECT 
                    COUNT(*) FILTER (WHERE is_correct = false) AS incorrect,
                    COUNT(*) AS total
                FROM user_feedback
                WHERE created_at >= NOW() - INTERVAL '{days} days';
            """)
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result and result[1] > 0:
                return result[0] / result[1]
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting feedback error rate: {e}")
            return 0.0
    
    def check_performance_degradation(self) -> Tuple[bool, Dict]:
        """Check apakah ada performance degradation"""
        baseline_metrics = self.drift_monitor.get_baseline_metrics()
        
        if not baseline_metrics:
            return False, {'reason': 'No baseline metrics'}
        
        # Get recent predictions untuk calculate current performance
        # For now, skip actual calculation (would need labeled recent data)
        
        return False, {'reason': 'Not implemented - requires labeled recent data'}
    
    def check_data_drift(self) -> Tuple[bool, Dict]:
        """Check data drift"""
        # Would need reference data and current data
        # For now, return False
        return False, {'reason': 'Not implemented - requires data comparison'}
    
    def evaluate_triggers(self) -> Dict:
        """
        Evaluate semua retraining triggers
        
        Returns:
            Dictionary dengan trigger evaluation results
        """
        results = {
            'should_retrain': False,
            'triggers': [],
            'evaluations': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Time-based trigger
        last_training = self.get_last_training_date()
        days_since_training = (datetime.now() - last_training).days
        
        results['evaluations']['time_based'] = {
            'last_training_date': last_training.isoformat(),
            'days_since_training': days_since_training,
            'max_days_threshold': self.max_days_between_training,
            'min_days_threshold': self.min_days_between_training
        }
        
        if days_since_training >= self.max_days_between_training:
            results['should_retrain'] = True
            results['triggers'].append({
                'type': 'time_based',
                'reason': f'Maximum interval reached ({days_since_training} days)',
                'priority': 'high'
            })
        
        # Don't trigger if too soon
        if days_since_training < self.min_days_between_training:
            results['evaluations']['too_soon'] = True
            results['should_retrain'] = False
            results['triggers'] = [{
                'type': 'blocked',
                'reason': f'Too soon to retrain (only {days_since_training} days)',
                'priority': 'info'
            }]
            return results
        
        # 2. New data trigger
        new_data_count = self.get_new_data_count(last_training)
        results['evaluations']['new_data'] = {
            'new_reviews_count': new_data_count,
            'threshold': self.new_data_threshold
        }
        
        if new_data_count >= self.new_data_threshold:
            results['should_retrain'] = True
            results['triggers'].append({
                'type': 'new_data',
                'reason': f'New data threshold reached ({new_data_count} reviews)',
                'priority': 'medium'
            })
        
        # 3. User feedback error rate
        error_rate = self.get_user_feedback_error_rate(days=7)
        results['evaluations']['user_feedback'] = {
            'error_rate': error_rate,
            'threshold': self.error_rate_threshold
        }
        
        if error_rate > self.error_rate_threshold:
            results['should_retrain'] = True
            results['triggers'].append({
                'type': 'user_feedback',
                'reason': f'High error rate ({error_rate:.2%})',
                'priority': 'high'
            })
        
        # 4. Performance degradation
        perf_degraded, perf_info = self.check_performance_degradation()
        results['evaluations']['performance'] = perf_info
        
        if perf_degraded:
            results['should_retrain'] = True
            results['triggers'].append({
                'type': 'performance_degradation',
                'reason': 'Model performance degraded',
                'priority': 'high'
            })
        
        # 5. Data drift
        data_drift, drift_info = self.check_data_drift()
        results['evaluations']['data_drift'] = drift_info
        
        if data_drift:
            results['should_retrain'] = True
            results['triggers'].append({
                'type': 'data_drift',
                'reason': 'Significant data drift detected',
                'priority': 'medium'
            })
        
        # Summary
        results['summary'] = {
            'total_triggers': len([t for t in results['triggers'] if t['type'] != 'blocked']),
            'high_priority_triggers': len([t for t in results['triggers'] if t.get('priority') == 'high']),
            'should_retrain': results['should_retrain']
        }
        
        return results


class RetrainingPipeline:
    """
    Execute retraining pipeline
    """
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.trigger = RetrainingTrigger(db_config)
    
    def prepare_training_data(self) -> pd.DataFrame:
        """Prepare data untuk training"""
        try:
            conn = psycopg2.connect(**self.db_config)
            
            # Load reviews with sentiment
            query = """
                SELECT 
                    review_text,
                    sentiment
                FROM reviews
                WHERE sentiment IS NOT NULL
                AND review_text IS NOT NULL
                ORDER BY scraped_at DESC;
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} samples for training")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame()
    
    def execute_training(self) -> Dict:
        """Execute training script"""
        import subprocess
        
        try:
            logger.info("🚀 Starting model retraining...")
            
            # Run training script
            result = subprocess.run(
                ["python", "src/training/train_bert.py"],
                capture_output=True,
                text=True,
                timeout=7200  # 2 hours timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Training completed successfully!")
                return {
                    'status': 'success',
                    'message': 'Training completed',
                    'stdout': result.stdout[-500:]  # Last 500 chars
                }
            else:
                logger.error(f"❌ Training failed with code {result.returncode}")
                return {
                    'status': 'failed',
                    'message': 'Training failed',
                    'stderr': result.stderr[-500:]
                }
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Training timeout (> 2 hours)")
            return {
                'status': 'timeout',
                'message': 'Training timeout'
            }
        except Exception as e:
            logger.error(f"❌ Error executing training: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def validate_new_model(self) -> bool:
        """Validate model baru sebelum deployment"""
        try:
            metrics_file = Path("models/bert_metrics.json")
            
            if not metrics_file.exists():
                logger.warning("Metrics file not found")
                return False
            
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            test_accuracy = metrics.get('test', {}).get('accuracy', 0)
            
            # Minimum accuracy threshold
            min_accuracy = 0.75
            
            if test_accuracy >= min_accuracy:
                logger.info(f"✅ Model validation passed (accuracy: {test_accuracy:.2%})")
                return True
            else:
                logger.warning(f"❌ Model validation failed (accuracy: {test_accuracy:.2%} < {min_accuracy:.2%})")
                return False
                
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return False
    
    def send_notification(self, message: str, status: str = 'info'):
        """Send notification (placeholder untuk Slack/Email integration)"""
        logger.info(f"[{status.upper()}] {message}")
        
        # TODO: Implement actual notification
        # - Slack webhook
        # - Email
        # - MS Teams
    
    def run(self) -> Dict:
        """
        Run full retraining pipeline
        
        Returns:
            Dictionary dengan pipeline results
        """
        results = {
            'started_at': datetime.now().isoformat(),
            'trigger_evaluation': None,
            'training_result': None,
            'validation_result': None,
            'status': 'not_started'
        }
        
        try:
            # 1. Evaluate triggers
            logger.info("📊 Evaluating retraining triggers...")
            trigger_eval = self.trigger.evaluate_triggers()
            results['trigger_evaluation'] = trigger_eval
            
            if not trigger_eval['should_retrain']:
                results['status'] = 'skipped'
                results['reason'] = 'No triggers met'
                logger.info("⏭️  No retraining needed")
                return results
            
            # Send notification
            self.send_notification(
                f"🔄 Retraining triggered: {', '.join([t['reason'] for t in trigger_eval['triggers']])}",
                'info'
            )
            
            # 2. Execute training
            logger.info("🏋️  Executing training...")
            training_result = self.execute_training()
            results['training_result'] = training_result
            
            if training_result['status'] != 'success':
                results['status'] = 'failed'
                self.send_notification(
                    f"❌ Training failed: {training_result['message']}",
                    'error'
                )
                return results
            
            # 3. Validate model
            logger.info("✅ Validating new model...")
            validation_passed = self.validate_new_model()
            results['validation_result'] = {
                'passed': validation_passed
            }
            
            if validation_passed:
                results['status'] = 'success'
                self.send_notification(
                    "✅ Retraining completed successfully! New model deployed.",
                    'success'
                )
            else:
                results['status'] = 'validation_failed'
                self.send_notification(
                    "⚠️  Retraining completed but model validation failed. Old model kept.",
                    'warning'
                )
            
            results['completed_at'] = datetime.now().isoformat()
            
            # Save results
            self.save_pipeline_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
            self.send_notification(
                f"❌ Pipeline error: {str(e)}",
                'error'
            )
            return results
    
    def save_pipeline_results(self, results: Dict):
        """Save pipeline results untuk tracking"""
        try:
            results_dir = Path("logs/retraining")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = results_dir / f"retraining_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"📝 Pipeline results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving pipeline results: {e}")


def main():
    """Main function untuk automated retraining"""
    
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'database': os.getenv('POSTGRES_DB', 'sentiment_db'),
        'user': os.getenv('POSTGRES_USER', 'sentiment_user'),
        'password': os.getenv('POSTGRES_PASSWORD', 'password')
    }
    
    pipeline = RetrainingPipeline(db_config)
    results = pipeline.run()
    
    print("\n" + "="*50)
    print("RETRAINING PIPELINE RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2))
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
