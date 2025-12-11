# 🚀 Quick Access Links - Local Deployment

## 📋 Service URLs

Open these links in your browser:

### **Main Services**
- 🌐 **API Documentation**: http://localhost:8080/docs
- 📊 **Streamlit Dashboard**: http://localhost:8501
- 📈 **Grafana Monitoring**: http://localhost:3000
- 🔍 **Prometheus Metrics**: http://localhost:9090

### **API Endpoints**
- ✅ **Health Check**: http://localhost:8080/health
- 📝 **All Reviews**: http://localhost:8080/reviews
- 📊 **Statistics**: http://localhost:8080/stats
- 🎯 **Model Info**: http://localhost:8080/model/info
- 📉 **Drift Report**: http://localhost:8080/drift/report

### **Metrics**
- 📊 **Custom Metrics**: http://localhost:8000/metrics
- 🔍 **Prometheus Targets**: http://localhost:9090/targets

---

## ⚡ Quick Commands

### Start Services
```powershell
docker-compose up -d
```

### Stop Services
```powershell
docker-compose down
```

### View Logs
```powershell
docker-compose logs -f
```

### Check Status
```powershell
docker-compose ps
```

### Test API
```powershell
curl http://localhost:8080/health
```

---

## 🔑 Default Credentials

**Grafana:**
- Username: `admin`
- Password: `admin`

**PostgreSQL:**
- Host: `localhost`
- Port: `5432`
- User: `sentiment_user`
- Password: Check `.env` file
- Database: `sentiment_db`

**MongoDB:**
- Host: `localhost`
- Port: `27017`
- Database: `sentiment_reviews`

---

## 📖 Full Documentation

- [Local Deployment Guide](LOCAL_DEPLOYMENT_GUIDE.md) - Complete setup instructions
- [MLOps Architecture](MLOPS_ARCHITECTURE.md) - System architecture
- [Quick Reference](MLOPS_QUICK_REFERENCE.md) - Common commands

---

**Status**: ✅ All services running on Docker
**Updated**: December 12, 2025
