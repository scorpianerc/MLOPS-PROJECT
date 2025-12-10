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
    
    def load_existing_review_ids(self, filepath: str = 'data/raw/reviews.csv') -> set:
        """
        Load existing review IDs dari file CSV
        
        Args:
            filepath: Path ke file CSV yang berisi review
        
        Returns:
            Set berisi review_id yang sudah ada
        """
        try:
            if os.path.exists(filepath):
                df_existing = pd.read_csv(filepath)
                if 'review_id' in df_existing.columns:
                    existing_ids = set(df_existing['review_id'].tolist())
                    logger.info(f"Loaded {len(existing_ids)} existing review IDs")
                    return existing_ids
            logger.info("No existing reviews found, starting fresh")
            return set()
        except Exception as e:
            logger.warning(f"Error loading existing reviews: {str(e)}")
            return set()
    
    def filter_new_reviews(self, df: pd.DataFrame, existing_ids: set) -> pd.DataFrame:
        """
        Filter hanya review yang belum pernah diambil
        
        Args:
            df: DataFrame hasil scraping
            existing_ids: Set berisi review_id yang sudah ada
        
        Returns:
            DataFrame berisi hanya review baru
        """
        try:
            if 'review_id' not in df.columns:
                logger.warning("Column review_id not found, returning all reviews")
                return df
            
            df_new = df[~df['review_id'].isin(existing_ids)].copy()
            logger.info(f"Found {len(df_new)} new reviews out of {len(df)} scraped")
            return df_new
        except Exception as e:
            logger.error(f"Error filtering reviews: {str(e)}")
            return df
    
    def save_data(
        self, 
        df: pd.DataFrame, 
        output_dir: str = 'data/raw',
        filename: str = None,
        append: bool = True
    ) -> str:
        """
        Simpan data ke CSV
        
        Args:
            df: DataFrame yang akan disimpan
            output_dir: Directory output
            filename: Nama file (jika None, akan generate otomatis)
            append: Jika True, append ke file existing; jika False, overwrite
        
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
            
            # Save ke CSV with timestamp
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Data berhasil disimpan ke: {filepath}")
            
            # Also save as reviews.csv (untuk DVC pipeline)
            main_filepath = os.path.join(output_dir, 'reviews.csv')
            
            if append and os.path.exists(main_filepath):
                # Append mode: load existing, combine, remove duplicates
                df_existing = pd.read_csv(main_filepath)
                df_combined = pd.concat([df_existing, df], ignore_index=True)
                # Remove duplicates based on review_id
                df_combined = df_combined.drop_duplicates(subset=['review_id'], keep='last')
                # Convert review_date to datetime untuk sorting
                df_combined['review_date'] = pd.to_datetime(df_combined['review_date'], errors='coerce')
                df_combined = df_combined.sort_values('review_date', ascending=False)
                df_combined.to_csv(main_filepath, index=False, encoding='utf-8')
                logger.info(f"Data di-append ke: {main_filepath} (Total: {len(df_combined)} reviews)")
            else:
                # Overwrite mode atau file belum ada
                df.to_csv(main_filepath, index=False, encoding='utf-8')
                logger.info(f"Data disimpan ke: {main_filepath} (Total: {len(df)} reviews)")
            
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
    
    # Load existing review IDs
    existing_ids = scraper.load_existing_review_ids()
    logger.info(f"Review yang sudah tersimpan: {len(existing_ids)}")
    
    # Scrape reviews per rating untuk mendapatkan lebih banyak data
    all_reviews = []
    ratings_to_scrape = [5, 4, 3, 2, 1]  # Scrape dari rating tinggi ke rendah
    
    for rating in ratings_to_scrape:
        logger.info(f"\n{'='*50}")
        logger.info(f"Scraping reviews dengan rating: {rating} ⭐")
        logger.info(f"{'='*50}")
        
        try:
            # Ambil lebih banyak review per rating (1000 per rating)
            max_per_rating = 1000  # Fixed: 1000 review per rating
            df_reviews = scraper.scrape_reviews(
                max_reviews=max_per_rating,
                sort_by=Sort.NEWEST,
                filter_score_with=rating
            )
            
            if not df_reviews.empty:
                all_reviews.append(df_reviews)
                logger.info(f"✓ Berhasil scraping {len(df_reviews)} reviews untuk rating {rating}")
            else:
                logger.info(f"✗ Tidak ada review untuk rating {rating}")
                
        except Exception as e:
            logger.warning(f"Error scraping rating {rating}: {str(e)}")
            continue
    
    # Gabungkan semua reviews
    if all_reviews:
        df_combined = pd.concat(all_reviews, ignore_index=True)
        logger.info(f"\n{'='*50}")
        logger.info(f"Total reviews dari semua rating: {len(df_combined)}")
        logger.info(f"{'='*50}")
    else:
        logger.error("Tidak ada review yang berhasil di-scrape")
        return
    
    # Clean data
    df_clean = scraper.clean_dataframe(df_combined)
    
    # Remove duplicates berdasarkan review_id
    df_clean = df_clean.drop_duplicates(subset=['review_id'], keep='first')
    logger.info(f"Setelah deduplikasi: {len(df_clean)} reviews unik")
    
    # Filter hanya review baru
    df_new = scraper.filter_new_reviews(df_clean, existing_ids)
    
    if len(df_new) == 0:
        logger.info("Tidak ada review baru yang ditemukan")
    else:
        logger.info(f"Ditemukan {len(df_new)} review baru")
        
        # Save data (append mode)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        scraper.save_data(df_new, filename=f'reviews_{timestamp}.csv', append=True)
        
        # Save metrics untuk review baru
        scraper.save_metrics(df_new)
    
    # Load total reviews untuk info
    total_reviews_path = 'data/raw/reviews.csv'
    if os.path.exists(total_reviews_path):
        df_total = pd.read_csv(total_reviews_path)
        logger.info(f"Total review tersimpan sekarang: {len(df_total)}")
    
    logger.info("Scraping selesai!")


if __name__ == '__main__':
    main()
