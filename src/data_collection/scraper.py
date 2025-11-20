"""
Data Collection Module
Scraper untuk mengambil review dari Google Play Store
"""

from google_play_scraper import app, reviews, Sort
import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path
import logging
from typing import List, Dict
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlayStoreReviewScraper:
    """Class untuk scraping review dari Google Play Store"""
    
    def __init__(self, app_id: str, country: str = 'id', lang: str = 'id'):
        """
        Initialize scraper
        
        Args:
            app_id: Google Play Store app ID
            country: Country code (default: 'id' untuk Indonesia)
            lang: Language code (default: 'id' untuk Indonesia)
        """
        self.app_id = app_id
        self.country = country
        self.lang = lang
        self.app_info = None
        
    def get_app_info(self) -> Dict:
        """Ambil informasi aplikasi"""
        try:
            logger.info(f"Mengambil info aplikasi: {self.app_id}")
            self.app_info = app(
                self.app_id,
                lang=self.lang,
                country=self.country
            )
            return self.app_info
        except Exception as e:
            logger.error(f"Error mengambil info aplikasi: {str(e)}")
            raise
    
    def scrape_reviews(
        self, 
        max_reviews: int = 500,
        sort_by: Sort = Sort.NEWEST,
        filter_score_with: int = None
    ) -> pd.DataFrame:
        """
        Scrape reviews dari Google Play Store
        
        Args:
            max_reviews: Jumlah maksimal review yang diambil
            sort_by: Sorting method (NEWEST, RATING, HELPFULNESS)
            filter_score_with: Filter berdasarkan rating (1-5)
        
        Returns:
            DataFrame berisi review
        """
        try:
            logger.info(f"Mulai scraping reviews untuk {self.app_id}")
            logger.info(f"Target: {max_reviews} reviews")
            
            all_reviews = []
            continuation_token = None
            
            while len(all_reviews) < max_reviews:
                try:
                    result, continuation_token = reviews(
                        self.app_id,
                        lang=self.lang,
                        country=self.country,
                        sort=sort_by,
                        count=min(200, max_reviews - len(all_reviews)),
                        filter_score_with=filter_score_with,
                        continuation_token=continuation_token
                    )
                    
                    if not result:
                        logger.info("Tidak ada review lagi yang bisa diambil")
                        break
                    
                    all_reviews.extend(result)
                    logger.info(f"Progress: {len(all_reviews)}/{max_reviews} reviews")
                    
                    if not continuation_token:
                        logger.info("Sudah mencapai akhir review")
                        break
                    
                    # Delay untuk menghindari rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Error saat scraping batch: {str(e)}")
                    break
            
            # Convert ke DataFrame
            df = pd.DataFrame(all_reviews)
            logger.info(f"Berhasil scraping {len(df)} reviews")
            
            return df
            
        except Exception as e:
            logger.error(f"Error saat scraping reviews: {str(e)}")
            raise
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dan format DataFrame"""
        try:
            logger.info("Membersihkan dan memformat data")
            
            # Pilih kolom yang penting
            columns_to_keep = [
                'reviewId', 'userName', 'userImage', 'content', 'score',
                'thumbsUpCount', 'reviewCreatedVersion', 'at', 'replyContent',
                'repliedAt'
            ]
            
            df_clean = df[columns_to_keep].copy()
            
            # Rename kolom
            df_clean = df_clean.rename(columns={
                'reviewId': 'review_id',
                'userName': 'user_name',
                'userImage': 'user_image',
                'content': 'review_text',
                'score': 'rating',
                'thumbsUpCount': 'thumbs_up',
                'reviewCreatedVersion': 'app_version',
                'at': 'review_date',
                'replyContent': 'reply_text',
                'repliedAt': 'reply_date'
            })
            
            # Tambah timestamp scraping
            df_clean['scraped_at'] = datetime.now()
            
            # Tambah app_id
            df_clean['app_id'] = self.app_id
            
            # Sort by date
            df_clean = df_clean.sort_values('review_date', ascending=False)
            
            # Reset index
            df_clean = df_clean.reset_index(drop=True)
            
            logger.info(f"Data berhasil dibersihkan: {len(df_clean)} rows")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error saat cleaning data: {str(e)}")
            raise
    
    def save_data(
        self, 
        df: pd.DataFrame, 
        output_dir: str = 'data/raw',
        filename: str = None
    ) -> str:
        """
        Simpan data ke CSV
        
        Args:
            df: DataFrame yang akan disimpan
            output_dir: Directory output
            filename: Nama file (jika None, akan generate otomatis)
        
        Returns:
            Path file yang disimpan
        """
        try:
            # Buat directory jika belum ada
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate filename jika tidak diberikan
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'reviews_{timestamp}.csv'
            
            filepath = os.path.join(output_dir, filename)
            
            # Save ke CSV
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Data berhasil disimpan ke: {filepath}")
            
            # Also save as reviews.csv (untuk DVC pipeline)
            main_filepath = os.path.join(output_dir, 'reviews.csv')
            df.to_csv(main_filepath, index=False, encoding='utf-8')
            logger.info(f"Data juga disimpan ke: {main_filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saat menyimpan data: {str(e)}")
            raise
    
    def save_metrics(
        self, 
        df: pd.DataFrame,
        output_dir: str = 'data/raw'
    ) -> None:
        """Simpan metrics dari hasil scraping"""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            metrics = {
                'total_reviews': len(df),
                'scraped_at': datetime.now().isoformat(),
                'app_id': self.app_id,
                'rating_distribution': df['rating'].value_counts().to_dict(),
                'avg_rating': float(df['rating'].mean()),
                'date_range': {
                    'earliest': df['review_date'].min().isoformat(),
                    'latest': df['review_date'].max().isoformat()
                }
            }
            
            filepath = os.path.join(output_dir, 'collection_metrics.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Metrics berhasil disimpan ke: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saat menyimpan metrics: {str(e)}")


def main():
    """Main function untuk menjalankan scraper"""
    import yaml
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    scraping_params = params['scraping']
    
    # Initialize scraper
    scraper = PlayStoreReviewScraper(
        app_id=scraping_params['app_id'],
        country=scraping_params['country'],
        lang=scraping_params['lang']
    )
    
    # Get app info
    app_info = scraper.get_app_info()
    logger.info(f"Aplikasi: {app_info['title']}")
    logger.info(f"Rating: {app_info['score']} ({app_info['ratings']} reviews)")
    
    # Scrape reviews
    df_reviews = scraper.scrape_reviews(
        max_reviews=scraping_params['max_reviews'],
        sort_by=Sort.NEWEST
    )
    
    # Clean data
    df_clean = scraper.clean_dataframe(df_reviews)
    
    # Save data
    scraper.save_data(df_clean, filename='reviews.csv')
    
    # Save metrics
    scraper.save_metrics(df_clean)
    
    logger.info("Scraping selesai!")


if __name__ == '__main__':
    main()
