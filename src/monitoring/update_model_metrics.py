"""
Update Model Metrics in PostgreSQL

Script ini untuk menyimpan metrics model (accuracy, precision, recall, F1) 
ke database agar bisa ditampilkan di Grafana.

Usage:
    python update_model_metrics.py --accuracy 0.92 --precision 0.91 --recall 0.93 --f1 0.92 --model "bert-base-multilingual"
"""

import argparse
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def update_metrics(model_name, accuracy, precision, recall, f1_score):
    """Update model metrics in PostgreSQL"""
    
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'sentiment_db'),
        user=os.getenv('POSTGRES_USER', 'sentiment_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'password')
    )
    
    cursor = conn.cursor()
    
    # Insert new metrics
    cursor.execute("""
        INSERT INTO model_metrics 
        (model_name, accuracy, precision_score, recall_score, f1_score, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (model_name, accuracy, precision, recall, f1_score, datetime.now()))
    
    conn.commit()
    
    print(f"✓ Metrics updated successfully!")
    print(f"  Model: {model_name}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  Timestamp: {datetime.now()}")
    
    cursor.close()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description='Update model metrics in PostgreSQL')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--accuracy', type=float, required=True, help='Model accuracy (0-1)')
    parser.add_argument('--precision', type=float, required=True, help='Precision score (0-1)')
    parser.add_argument('--recall', type=float, required=True, help='Recall score (0-1)')
    parser.add_argument('--f1', type=float, required=True, help='F1 score (0-1)')
    
    args = parser.parse_args()
    
    # Validate scores are between 0 and 1
    if not all(0 <= score <= 1 for score in [args.accuracy, args.precision, args.recall, args.f1]):
        print("ERROR: All scores must be between 0 and 1")
        return
    
    update_metrics(
        model_name=args.model,
        accuracy=args.accuracy,
        precision=args.precision,
        recall=args.recall,
        f1_score=args.f1
    )

if __name__ == "__main__":
    main()
