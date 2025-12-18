# 🐳 Local Docker Deployment Guide

## ✅ DEPLOYMENT SUCCESS!

Your complete MLOps stack is now running locally on Docker! 🎉

---

## 🌐 Access Your Services

All services are now accessible:

| Service | URL | Description |
|---------|-----|-------------|
| **API Server** | http://localhost:8080 | REST API for predictions |
| **API Documentation** | http://localhost:8080/docs | Interactive Swagger UI |
| **Streamlit Dashboard** | http://localhost:8501 | Interactive web interface |
| **Grafana** | http://localhost:3000 | Monitoring dashboards |
| **Prometheus** | http://localhost:9090 | Metrics collection |
| **PostgreSQL** | localhost:5432 | Database |
| **MongoDB** | localhost:27017 | NoSQL database |
| **Metrics Exporter** | http://localhost:8000 | Custom metrics |

---

## 🚀 Quick Start

### **1. Start Services**
```powershell
# Start Docker Desktop first, then run:
docker-compose up -d
```

### **2. Check Status**
```powershell
docker-compose ps
```

### **3. View Logs**
```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f streamlit
```

### **4. Stop Services**
```powershell
docker-compose down
```

### **5. Restart Services**
```powershell
docker-compose restart
```

---

## 🧪 Testing API

### **Health Check**
```powershell
curl http://localhost:8080/health
```

### **Predict Sentiment (Positive)**
```powershell
$body = @{ text = "Aplikasi ini sangat bagus dan mudah digunakan!" } | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:8080/predict -Method POST -Body $body -ContentType "application/json"
```

### **Predict Sentiment (Negative)**
```powershell
$body = @{ text = "Aplikasi ini buruk sekali, sering error dan lambat!" } | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:8080/predict -Method POST -Body $body -ContentType "application/json"
```

### **Batch Prediction**
```powershell
$body = @{
    texts = @(
        "Produk berkualitas tinggi",
        "Pengiriman lambat sekali",
        "Harga terjangkau"
    )
} | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:8080/predict/batch -Method POST -Body $body -ContentType "application/json"
```

---

## 📊 Monitoring & Observability

### **Grafana Setup**

1. Open http://localhost:3000
2. Login: `admin` / `admin`
3. Navigate to Dashboards → Browse
4. Import dashboards from `grafana/dashboards/`

**Available Dashboards:**
- Model Performance
- API Metrics
- System Resources
- Drift Detection

### **Prometheus Metrics**

Access metrics at:
- http://localhost:9090/graph
- http://localhost:8000/metrics (Custom metrics)

**Key Metrics:**
```
sentiment_predictions_total
sentiment_prediction_duration_seconds
sentiment_drift_detected_total
sentiment_model_accuracy
```

---

## 🗄️ Database Access

### **PostgreSQL**

```powershell
# Connect to PostgreSQL
docker exec -it sentiment_postgres psql -U sentiment_user -d sentiment_db

# Useful queries
\dt                           # List tables
SELECT * FROM reviews LIMIT 5;
SELECT * FROM predictions LIMIT 5;
SELECT * FROM drift_reports ORDER BY report_date DESC LIMIT 5;
```

### **MongoDB**

```powershell
# Connect to MongoDB
docker exec -it sentiment_mongodb mongosh sentiment_reviews

# Useful queries
show collections
db.reviews.find().limit(5)
db.stats()
```

---

## 🔄 Common Commands

### **View Real-time Logs**
```powershell
docker-compose logs -f --tail=100
```

### **Check Resource Usage**
```powershell
docker stats
```

### **Restart Specific Service**
```powershell
docker-compose restart api
docker-compose restart streamlit
```

### **Rebuild and Restart**
```powershell
docker-compose up -d --build
```

### **Clean Everything**
```powershell
# Stop and remove containers, networks, volumes
docker-compose down -v

# Remove all images
docker-compose down --rmi all
```

---

## 🐛 Troubleshooting

### **Service Won't Start**

1. Check logs:
   ```powershell
   docker-compose logs service_name
   ```

2. Verify environment variables:
   ```powershell
   cat .env
   ```

3. Restart service:
   ```powershell
   docker-compose restart service_name
   ```

### **Port Already in Use**

```powershell
# Check what's using the port
netstat -ano | findstr :8080
netstat -ano | findstr :8501

# Kill the process (replace PID)
Stop-Process -Id PID -Force
```

### **Database Connection Failed**

```powershell
# Check if PostgreSQL is healthy
docker-compose ps postgres

# Restart PostgreSQL
docker-compose restart postgres

# Wait for health check
Start-Sleep -Seconds 10
```

### **Out of Disk Space**

```powershell
# Clean unused Docker resources
docker system prune -a --volumes

# Check disk usage
docker system df
```

### **Container Keeps Restarting**

```powershell
# Check logs for errors
docker-compose logs --tail=50 service_name

# Inspect container
docker inspect sentiment_api
```

---

## 📈 Resource Requirements

**Minimum Requirements:**
- RAM: 4 GB
- Disk: 10 GB free space
- CPU: 2 cores

**Recommended:**
- RAM: 8 GB
- Disk: 20 GB free space
- CPU: 4 cores

**Current Usage:**
```
Service         RAM    CPU
─────────────────────────
PostgreSQL      256MB  10%
MongoDB         256MB  5%
API             512MB  15%
Streamlit       256MB  10%
Grafana         128MB  5%
Prometheus      256MB  10%
Exporter        128MB  5%
─────────────────────────
Total           ~2GB   60%
```

---

## 🔐 Security Notes

### **Change Default Passwords**

Edit `.env` file:
```env
POSTGRES_PASSWORD=your_secure_password
GRAFANA_ADMIN_PASSWORD=your_secure_password
```

Then restart:
```powershell
docker-compose down
docker-compose up -d
```

### **Network Isolation**

All services run in isolated network `sentiment_network`. Only exposed ports are accessible from host.

---

## 🎯 Next Steps

### **1. Explore Streamlit Dashboard**
- Open http://localhost:8501
- Try sentiment prediction
- View model metrics
- Monitor drift detection

### **2. Review API Documentation**
- Open http://localhost:8080/docs
- Test all endpoints interactively
- See request/response schemas

### **3. Setup Monitoring**
- Open Grafana at http://localhost:3000
- Import pre-configured dashboards
- Setup alerts (optional)

### **4. Test Automated Retraining**
- Add new data to database
- Trigger retraining via scheduler
- Monitor model version changes

### **5. Integrate with Your App**
```python
import requests

# Example integration
response = requests.post(
    "http://localhost:8080/predict",
    json={"text": "Your review text here"}
)

sentiment = response.json()
print(f"Sentiment: {sentiment['sentiment']}")
print(f"Confidence: {sentiment['confidence']:.2%}")
```

---

## 📚 Additional Resources

- **Quick Reference**: See `MLOPS_QUICK_REFERENCE.md`
- **Monitoring Guide**: See `MONITORING_GUIDE.md`
- **GitHub Actions**: See `GITHUB_ACTIONS_GUIDE.md`

---

## 🆘 Need Help?

1. Check logs: `docker-compose logs -f`
2. Verify .env configuration
3. Ensure Docker Desktop is running
4. Check port availability
5. Create GitHub issue for bugs

---

**🎉 Congratulations!** Your complete MLOps pipeline is running locally!

Now you can:
- ✅ Make predictions via API
- ✅ Monitor model performance
- ✅ Track drift detection
- ✅ View interactive dashboards
- ✅ Test retraining pipeline
- ✅ Access all MLOps features locally

**Total Cost: $0** 💰 (runs on your PC!)
