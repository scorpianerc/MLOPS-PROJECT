"""
Scheduler untuk menjalankan pipeline secara otomatis
"""

import schedule
import time
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_collection.scraper import PlayStoreReviewScraper
from src.data_collection.database import DatabaseManager
from src.prediction.predict import PredictionPipeline
from google_play_scraper import Sort
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Scheduler untuk automated tasks"""
    
    def __init__(self):
        """Initialize scheduler"""
        # Load params
        with open('params.yaml', 'r') as f:
            self.params = yaml.safe_load(f)
        
        self.scraping_params = self.params['scraping']
        self.scheduler_params = self.params['scheduler']
        
        self.db_manager = DatabaseManager()
        
        logger.info("Task Scheduler initialized")
    
    def scrape_and_store(self):
        """Task: Scrape reviews dan simpan ke database"""
        try:
            logger.info("=" * 50)
            logger.info("Starting scraping task...")
            logger.info("=" * 50)
            
            # Initialize scraper
            scraper = PlayStoreReviewScraper(
                app_id=self.scraping_params['app_id'],
                country=self.scraping_params['country'],
                lang=self.scraping_params['lang']
            )
            
            # Scrape reviews
            df_reviews = scraper.scrape_reviews(
                max_reviews=self.scraping_params['max_reviews'],
                sort_by=Sort.NEWEST
            )
            
            # Clean data
            df_clean = scraper.clean_dataframe(df_reviews)
            
            # Save to database
            postgres_count = self.db_manager.save_reviews_to_postgres(df_clean)
            mongo_count = self.db_manager.save_reviews_to_mongodb(df_clean)
            
            logger.info(f"Saved {postgres_count} new reviews to PostgreSQL")
            logger.info(f"Saved {mongo_count} reviews to MongoDB")
            
            # Save to CSV as backup
            scraper.save_data(df_clean, filename='reviews.csv')
            scraper.save_metrics(df_clean)
            
            logger.info("Scraping task completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in scraping task: {str(e)}", exc_info=True)
    
    def run_predictions(self):
        """Task: Run prediction pipeline"""
        try:
            logger.info("=" * 50)
            logger.info("Starting prediction task...")
            logger.info("=" * 50)
            
            pipeline = PredictionPipeline()
            pipeline.run(batch_size=100)
            
            logger.info("Prediction task completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in prediction task: {str(e)}", exc_info=True)
    
    def check_and_retrain(self):
        """Task: Check if model needs retraining"""
        try:
            logger.info("=" * 50)
            logger.info("Checking if retraining is needed...")
            logger.info("=" * 50)
            
            # Get stats
            stats = self.db_manager.get_sentiment_stats()
            
            # Check if we have enough new data
            if stats['unpredicted_reviews'] > 1000:
                logger.info("Sufficient new data found. Retraining recommended.")
                # TODO: Implement auto-retraining logic
                # For now, just log the recommendation
            else:
                logger.info(f"Not enough new data for retraining ({stats['unpredicted_reviews']} reviews)")
            
        except Exception as e:
            logger.error(f"Error in retraining check: {str(e)}", exc_info=True)
    
    def log_stats(self):
        """Task: Log statistics"""
        try:
            stats = self.db_manager.get_sentiment_stats()
            
            logger.info("=" * 50)
            logger.info("Current Statistics")
            logger.info("=" * 50)
            logger.info(f"Total Reviews: {stats['total_reviews']}")
            logger.info(f"Predicted: {stats['predicted_reviews']}")
            logger.info(f"Unpredicted: {stats['unpredicted_reviews']}")
            logger.info(f"Sentiment Distribution:")
            logger.info(f"  Positive: {stats['positive']} ({stats['positive_pct']:.2f}%)")
            logger.info(f"  Neutral: {stats['neutral']} ({stats['neutral_pct']:.2f}%)")
            logger.info(f"  Negative: {stats['negative']} ({stats['negative_pct']:.2f}%)")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Error logging stats: {str(e)}")
    
    def setup_schedule(self):
        """Setup scheduled tasks"""
        logger.info("Setting up scheduled tasks...")
        
        # Scraping task - every N hours
        scraping_interval = self.scheduler_params['scraping_interval_hours']
        schedule.every(scraping_interval).hours.do(self.scrape_and_store)
        logger.info(f"  - Scraping: every {scraping_interval} hours")
        
        # Prediction task - every N hours
        prediction_interval = self.scheduler_params['prediction_interval_hours']
        schedule.every(prediction_interval).hours.do(self.run_predictions)
        logger.info(f"  - Prediction: every {prediction_interval} hours")
        
        # Stats logging - every hour
        schedule.every(1).hours.do(self.log_stats)
        logger.info(f"  - Stats logging: every 1 hour")
        
        # Retraining check - every N days
        retrain_days = self.scheduler_params['model_retrain_days']
        schedule.every(retrain_days).days.do(self.check_and_retrain)
        logger.info(f"  - Retraining check: every {retrain_days} days")
        
        logger.info("All tasks scheduled!")
    
    def run_initial_tasks(self):
        """Run initial tasks on startup"""
        logger.info("Running initial tasks...")
        
        # Run scraping
        self.scrape_and_store()
        
        # Run prediction
        self.run_predictions()
        
        # Log stats
        self.log_stats()
    
    def start(self, run_immediately: bool = True):
        """
        Start scheduler
        
        Args:
            run_immediately: If True, run all tasks immediately on startup
        """
        logger.info("Starting Task Scheduler...")
        
        # Setup schedule
        self.setup_schedule()
        
        # Run initial tasks if requested
        if run_immediately:
            self.run_initial_tasks()
        
        # Start scheduler loop
        logger.info("Scheduler is now running. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        finally:
            self.db_manager.close()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Task Scheduler for Sentiment Analysis')
    parser.add_argument('--no-initial-run', action='store_true',
                       help='Skip initial task execution')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/scheduler.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Initialize and start scheduler
    scheduler = TaskScheduler()
    scheduler.start(run_immediately=not args.no_initial_run)


if __name__ == '__main__':
    main()
