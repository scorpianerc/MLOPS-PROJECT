# 📊 Panduan MongoDB & PostgreSQL

## 🎯 **Fungsi Database**

### **PostgreSQL (Relational Database)**
- ✅ Menyimpan **structured data** (tabel reviews)
- ✅ Support **SQL queries** kompleks
- ✅ Untuk **analitik** dan **reporting**
- ✅ ACID compliance (data consistency)

### **MongoDB (NoSQL Database)**
- ✅ Menyimpan **raw JSON** dari scraping
- ✅ **Flexible schema** (tidak perlu migrasi)
- ✅ **Fast writes** untuk real-time data
- ✅ Menyimpan **predictions history**

---

## 🔌 **Koneksi Database**

### **1. Cek Status Container**
```powershell
docker-compose ps
```

Pastikan `sentiment_postgres` dan `sentiment_mongodb` status **Healthy**.

### **2. Akses PostgreSQL**
```powershell
# Via Docker
docker exec -it sentiment_postgres psql -U sentiment_user -d sentiment_db

# SQL Commands:
\dt                    # List tables
\d reviews             # Describe reviews table
SELECT COUNT(*) FROM reviews;
SELECT sentiment, COUNT(*) FROM reviews GROUP BY sentiment;
```

### **3. Akses MongoDB**
```powershell
# Via Docker
docker exec -it sentiment_mongodb mongosh

# MongoDB Commands:
use sentiment_reviews
show collections
db.reviews.countDocuments()
db.reviews.find().limit(5)
db.predictions.find().sort({predicted_at: -1}).limit(10)
```

---

## 🐍 **Cara Pakai dari Python**

### **Contoh 1: Query PostgreSQL**
```python
from src.data_collection.database import DatabaseManager

# Initialize
db = DatabaseManager()
session = db.Session()

# Count reviews by sentiment
from sqlalchemy import func
result = session.query(
    Review.sentiment, 
    func.count(Review.id)
).group_by(Review.sentiment).all()

for sentiment, count in result:
    print(f"{sentiment}: {count}")
```

### **Contoh 2: Query MongoDB**
```python
from src.data_collection.database import DatabaseManager

# Initialize
db = DatabaseManager()

# Get all reviews
reviews = db.reviews_collection.find().limit(10)
for review in reviews:
    print(review['review_text'][:50])

# Get predictions
predictions = db.predictions_collection.find(
    {"sentiment": "positive"}
).limit(5)
```

---

## 📝 **Schema Database**

### **PostgreSQL - Table `reviews`**
```sql
CREATE TABLE reviews (
    id SERIAL PRIMARY KEY,
    review_id VARCHAR(255) UNIQUE,
    app_id VARCHAR(100),
    user_name VARCHAR(255),
    review_text TEXT,
    rating INTEGER,
    thumbs_up INTEGER,
    app_version VARCHAR(50),
    review_date TIMESTAMP,
    scraped_at TIMESTAMP,
    
    -- Prediction results
    sentiment VARCHAR(20),      -- positive/negative/neutral
    sentiment_score FLOAT,      -- confidence score
    predicted_at TIMESTAMP
);
```

### **MongoDB - Collection `reviews`**
```json
{
  "_id": "ObjectId",
  "review_id": "unique_id",
  "app_id": "com.valar.pintu",
  "user_name": "John Doe",
  "review_text": "Aplikasi bagus!",
  "rating": 5,
  "thumbs_up": 10,
  "app_version": "1.0.0",
  "review_date": "2025-11-15",
  "scraped_at": "2025-11-15 10:00:00"
}
```

### **MongoDB - Collection `predictions`**
```json
{
  "_id": "ObjectId",
  "review_id": "unique_id",
  "review_text": "Aplikasi bagus!",
  "sentiment": "positive",
  "sentiment_score": 0.95,
  "predicted_at": "2025-11-15 10:05:00",
  "model_version": "v1.0"
}
```

---

## 🚀 **Quick Commands**

### **PostgreSQL**
```sql
-- Total reviews
SELECT COUNT(*) FROM reviews;

-- Reviews by sentiment
SELECT sentiment, COUNT(*) as count 
FROM reviews 
GROUP BY sentiment;

-- Average rating
SELECT AVG(rating) FROM reviews;

-- Top users
SELECT user_name, COUNT(*) as review_count 
FROM reviews 
GROUP BY user_name 
ORDER BY review_count DESC 
LIMIT 10;

-- Recent reviews
SELECT review_text, rating, sentiment, predicted_at 
FROM reviews 
WHERE predicted_at IS NOT NULL 
ORDER BY predicted_at DESC 
LIMIT 10;
```

### **MongoDB**
```javascript
// Total reviews
db.reviews.countDocuments()

// Reviews by rating
db.reviews.aggregate([
  { $group: { _id: "$rating", count: { $sum: 1 } } }
])

// Search reviews
db.reviews.find({ 
  review_text: { $regex: "bagus", $options: "i" } 
})

// Recent predictions
db.predictions.find()
  .sort({ predicted_at: -1 })
  .limit(10)
```

---

## 🔧 **Troubleshooting**

### **Connection Error?**
```powershell
# Restart databases
docker-compose restart postgres mongodb

# Check logs
docker logs sentiment_postgres
docker logs sentiment_mongodb
```

### **Empty Tables?**
```powershell
# Run scraper manually
docker exec -it sentiment_scheduler python src/data_collection/scraper.py

# Check if data saved
docker exec -it sentiment_postgres psql -U sentiment_user -d sentiment_db -c "SELECT COUNT(*) FROM reviews;"
```

### **Reset Database?**
```powershell
# WARNING: This will delete all data!
docker-compose down -v
docker-compose up -d
```

---

## 📚 **Resources**

- **PostgreSQL Docs**: https://www.postgresql.org/docs/
- **MongoDB Docs**: https://docs.mongodb.com/
- **SQLAlchemy**: https://docs.sqlalchemy.org/
- **PyMongo**: https://pymongo.readthedocs.io/
