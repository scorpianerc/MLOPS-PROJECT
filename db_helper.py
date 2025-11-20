#!/usr/bin/env python3
"""
🔧 Database Helper Script
Quick access to PostgreSQL and MongoDB databases
"""

import sys
from datetime import datetime
from sqlalchemy import func
from src.data_collection.database import DatabaseManager, Review

class DBHelper:
    def __init__(self):
        """Initialize database connections"""
        print("🔌 Connecting to databases...")
        self.db = DatabaseManager()
        self.session = self.db.Session()
        print("✅ Connected!\n")
    
    def postgres_stats(self):
        """Show PostgreSQL statistics"""
        print("=" * 60)
        print("📊 POSTGRESQL STATISTICS")
        print("=" * 60)
        
        # Total reviews
        total = self.session.query(Review).count()
        print(f"📝 Total Reviews: {total}")
        
        # Reviews by sentiment
        print("\n🎯 Sentiment Distribution:")
        sentiments = self.session.query(
            Review.sentiment, 
            func.count(Review.id).label('count')
        ).group_by(Review.sentiment).all()
        
        for sentiment, count in sentiments:
            emoji = "✅" if sentiment == "positive" else "❌" if sentiment == "negative" else "➖"
            percentage = (count / total * 100) if total > 0 else 0
            print(f"   {emoji} {sentiment or 'unpredicted'}: {count} ({percentage:.1f}%)")
        
        # Average rating
        avg_rating = self.session.query(func.avg(Review.rating)).scalar()
        print(f"\n⭐ Average Rating: {avg_rating:.2f}" if avg_rating else "\n⭐ Average Rating: N/A")
        
        # Predicted vs unpredicted
        predicted = self.session.query(Review).filter(Review.sentiment.isnot(None)).count()
        unpredicted = total - predicted
        print(f"\n🤖 Predicted: {predicted}")
        print(f"⏳ Unpredicted: {unpredicted}")
        
        # Recent reviews
        print("\n📅 Latest 5 Reviews:")
        recent = self.session.query(Review).order_by(
            Review.scraped_at.desc()
        ).limit(5).all()
        
        for i, review in enumerate(recent, 1):
            sentiment_emoji = "✅" if review.sentiment == "positive" else "❌" if review.sentiment == "negative" else "➖" if review.sentiment == "neutral" else "⏳"
            text_preview = review.review_text[:50] + "..." if len(review.review_text) > 50 else review.review_text
            print(f"   {i}. {sentiment_emoji} [{review.rating}⭐] {text_preview}")
        
        print("\n")
    
    def mongodb_stats(self):
        """Show MongoDB statistics"""
        print("=" * 60)
        print("📊 MONGODB STATISTICS")
        print("=" * 60)
        
        # Reviews collection
        reviews_count = self.db.reviews_collection.count_documents({})
        print(f"📝 Reviews Collection: {reviews_count} documents")
        
        # Predictions collection
        predictions_count = self.db.predictions_collection.count_documents({})
        print(f"🤖 Predictions Collection: {predictions_count} documents")
        
        # Recent predictions
        if predictions_count > 0:
            print("\n📅 Latest 5 Predictions:")
            predictions = self.db.predictions_collection.find().sort(
                "predicted_at", -1
            ).limit(5)
            
            for i, pred in enumerate(predictions, 1):
                sentiment_emoji = "✅" if pred.get('sentiment') == "positive" else "❌" if pred.get('sentiment') == "negative" else "➖"
                text = pred.get('review_text', '')[:40] + "..." if len(pred.get('review_text', '')) > 40 else pred.get('review_text', '')
                score = pred.get('sentiment_score', 0)
                print(f"   {i}. {sentiment_emoji} [{score:.2f}] {text}")
        
        print("\n")
    
    def show_sample_data(self, limit=5):
        """Show sample data from PostgreSQL"""
        print("=" * 60)
        print(f"🔍 SAMPLE DATA (Latest {limit} Reviews)")
        print("=" * 60)
        
        reviews = self.session.query(Review).order_by(
            Review.scraped_at.desc()
        ).limit(limit).all()
        
        for i, review in enumerate(reviews, 1):
            print(f"\n📝 Review #{i}")
            print(f"   User: {review.user_name}")
            print(f"   Rating: {'⭐' * review.rating}")
            print(f"   Text: {review.review_text[:100]}...")
            print(f"   Sentiment: {review.sentiment or 'Not predicted'}")
            if review.sentiment_score:
                print(f"   Score: {review.sentiment_score:.4f}")
            print(f"   Scraped: {review.scraped_at}")
        
        print("\n")
    
    def export_to_csv(self, filename="database_export.csv"):
        """Export PostgreSQL data to CSV"""
        import pandas as pd
        
        print(f"💾 Exporting to {filename}...")
        
        reviews = self.session.query(Review).all()
        data = [{
            'review_id': r.review_id,
            'user_name': r.user_name,
            'rating': r.rating,
            'review_text': r.review_text,
            'sentiment': r.sentiment,
            'sentiment_score': r.sentiment_score,
            'scraped_at': r.scraped_at,
            'predicted_at': r.predicted_at
        } for r in reviews]
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"✅ Exported {len(df)} reviews to {filename}\n")
    
    def search_reviews(self, keyword, limit=10):
        """Search reviews by keyword"""
        print(f"🔍 Searching for '{keyword}'...")
        print("=" * 60)
        
        reviews = self.session.query(Review).filter(
            Review.review_text.ilike(f"%{keyword}%")
        ).limit(limit).all()
        
        print(f"Found {len(reviews)} results:\n")
        
        for i, review in enumerate(reviews, 1):
            sentiment_emoji = "✅" if review.sentiment == "positive" else "❌" if review.sentiment == "negative" else "➖" if review.sentiment == "neutral" else "⏳"
            print(f"{i}. {sentiment_emoji} [{review.rating}⭐] {review.review_text[:80]}...")
        
        print("\n")
    
    def close(self):
        """Close database connections"""
        self.session.close()
        print("👋 Connections closed!")


def main():
    """Main menu"""
    helper = DBHelper()
    
    while True:
        print("\n" + "=" * 60)
        print("🗄️  DATABASE HELPER MENU")
        print("=" * 60)
        print("1. PostgreSQL Statistics")
        print("2. MongoDB Statistics")
        print("3. Show Sample Data")
        print("4. Search Reviews")
        print("5. Export to CSV")
        print("6. Show All Stats")
        print("0. Exit")
        print("=" * 60)
        
        choice = input("\n👉 Choose option (0-6): ").strip()
        
        if choice == "1":
            helper.postgres_stats()
        elif choice == "2":
            helper.mongodb_stats()
        elif choice == "3":
            limit = input("How many reviews? (default 5): ").strip()
            limit = int(limit) if limit.isdigit() else 5
            helper.show_sample_data(limit)
        elif choice == "4":
            keyword = input("Enter keyword: ").strip()
            if keyword:
                helper.search_reviews(keyword)
        elif choice == "5":
            filename = input("Filename (default: database_export.csv): ").strip()
            filename = filename if filename else "database_export.csv"
            helper.export_to_csv(filename)
        elif choice == "6":
            helper.postgres_stats()
            helper.mongodb_stats()
        elif choice == "0":
            helper.close()
            break
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure Docker containers are running: docker-compose ps")
