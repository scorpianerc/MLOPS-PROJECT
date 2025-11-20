"""
Utility functions for the sentiment analysis project
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def ensure_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        'data/raw',
        'data/processed',
        'data/predictions',
        'models',
        'logs',
        'notebooks'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {dir_path}")


def load_latest_reviews(data_dir: str = 'data/raw') -> pd.DataFrame:
    """Load the most recent reviews file"""
    data_path = Path(data_dir)
    csv_files = list(data_path.glob('reviews*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No review files found in {data_dir}")
    
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading latest reviews from: {latest_file}")
    
    return pd.read_csv(latest_file)


def split_into_batches(df: pd.DataFrame, batch_size: int = 100):
    """Split DataFrame into batches"""
    n_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        yield df.iloc[start_idx:end_idx].copy()


def get_sentiment_distribution(df: pd.DataFrame, sentiment_col: str = 'sentiment') -> dict:
    """Get sentiment distribution statistics"""
    dist = df[sentiment_col].value_counts()
    total = len(df)
    
    stats = {
        'total': total,
        'distribution': dist.to_dict(),
        'percentages': (dist / total * 100).to_dict()
    }
    
    return stats


def print_sentiment_stats(df: pd.DataFrame):
    """Print sentiment statistics"""
    stats = get_sentiment_distribution(df)
    
    print("\n" + "=" * 50)
    print("SENTIMENT STATISTICS")
    print("=" * 50)
    print(f"Total Reviews: {stats['total']}")
    print("\nDistribution:")
    for sentiment, count in stats['distribution'].items():
        pct = stats['percentages'][sentiment]
        print(f"  {sentiment.capitalize()}: {count} ({pct:.2f}%)")
    print("=" * 50 + "\n")


def create_word_frequency(texts, top_n: int = 20):
    """Create word frequency from list of texts"""
    from collections import Counter
    
    all_words = []
    for text in texts:
        if isinstance(text, str):
            words = text.lower().split()
            all_words.extend(words)
    
    word_freq = Counter(all_words)
    return dict(word_freq.most_common(top_n))


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    """Plot confusion matrix"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def save_predictions(df: pd.DataFrame, output_dir: str = 'data/predictions'):
    """Save predictions with timestamp"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f'predictions_{timestamp}.csv'
    
    filepath = output_path / filename
    df.to_csv(filepath, index=False)
    
    logger.info(f"Predictions saved to: {filepath}")
    return str(filepath)


def get_project_stats():
    """Get overall project statistics"""
    stats = {
        'data_files': 0,
        'models': 0,
        'logs': 0
    }
    
    # Count data files
    if Path('data/raw').exists():
        stats['data_files'] = len(list(Path('data/raw').glob('*.csv')))
    
    # Count models
    if Path('models').exists():
        stats['models'] = len(list(Path('models').glob('*.pkl')))
    
    # Count logs
    if Path('logs').exists():
        stats['logs'] = len(list(Path('logs').glob('*.log')))
    
    return stats
