# Fitur Enhancement & Retraining Strategy

## 📝 **Fitur Komentar & Star Rating**

### **Apakah Perlu Ditambahkan?**
✅ **YA, SANGAT DIREKOMENDASIKAN!** Berikut alasannya:

### **Manfaat Menambahkan Fitur Komentar & Rating:**

#### 1. **User Feedback Loop**
- User dapat memberikan feedback apakah prediksi sentiment **akurat atau tidak**
- Membangun **ground truth data** untuk evaluasi model
- Meningkatkan **engagement** user dengan dashboard

#### 2. **Data Labeling Gratis**
- User memberikan **label manual** untuk data yang salah diprediksi
- **Crowdsourcing labeling** dari user sebenarnya
- Mengurangi biaya manual labeling

#### 3. **Model Improvement Tracking**
- Melihat **error patterns** dari feedback user
- Identifikasi **edge cases** yang model belum handle
- **Prioritas retraining** berdasarkan feedback terbanyak

#### 4. **Trust & Transparency**
- User merasa **dilibatkan** dalam proses improvement
- **Transparansi** tentang akurasi model
- Meningkatkan **trust** terhadap sistem

---

## 🎯 **Implementasi Fitur Komentar & Rating**

### **Schema Database Baru:**

```sql
-- Tabel untuk user feedback
CREATE TABLE user_feedback (
    id SERIAL PRIMARY KEY,
    review_id INTEGER REFERENCES reviews(id),
    user_name VARCHAR(100),
    predicted_sentiment VARCHAR(20),
    actual_sentiment VARCHAR(20),  -- feedback dari user
    confidence_score FLOAT,
    user_rating INTEGER CHECK (user_rating BETWEEN 1 AND 5),  -- 1-5 stars
    user_comment TEXT,
    is_correct BOOLEAN,  -- apakah prediksi benar
    feedback_type VARCHAR(50),  -- 'correction', 'validation', 'question'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(50),
    user_agent TEXT
);

-- Index untuk query cepat
CREATE INDEX idx_feedback_review ON user_feedback(review_id);
CREATE INDEX idx_feedback_correct ON user_feedback(is_correct);
CREATE INDEX idx_feedback_created ON user_feedback(created_at);
```

### **UI Components (Streamlit):**

```python
# Di tab Details, tambahkan feedback form untuk setiap review
with st.expander("📝 Berikan Feedback untuk Review Ini"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Prediksi Model:** {sentiment}")
        user_sentiment = st.radio(
            "Apakah sentiment ini benar?",
            ["✓ Benar", "✗ Seharusnya Positive", "✗ Seharusnya Negative", "✗ Seharusnya Neutral"],
            key=f"sentiment_{row['id']}"
        )
    
    with col2:
        user_rating = st.slider(
            "Seberapa yakin Anda?",
            1, 5, 3,
            help="1 = Tidak yakin, 5 = Sangat yakin",
            key=f"rating_{row['id']}"
        )
    
    user_comment = st.text_area(
        "Komentar (opsional):",
        placeholder="Jelaskan kenapa prediksi salah...",
        key=f"comment_{row['id']}"
    )
    
    if st.button("Kirim Feedback", key=f"submit_{row['id']}"):
        # Save feedback ke database
        save_user_feedback(
            review_id=row['id'],
            predicted=sentiment,
            actual=parse_user_sentiment(user_sentiment),
            rating=user_rating,
            comment=user_comment
        )
        st.success("✓ Terima kasih atas feedback Anda!")
```

### **Analytics Dashboard untuk Feedback:**

```python
# Tab baru: Feedback Analytics
with tab5:
    st.markdown("### 📊 User Feedback Analytics")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Feedback", f"{total_feedback:,}")
    with col2:
        st.metric("Prediksi Benar", f"{correct_predictions:.1%}")
    with col3:
        st.metric("Avg User Rating", f"{avg_rating:.1f} ⭐")
    with col4:
        st.metric("Top Error Type", most_common_error)
    
    # Confusion Matrix dari user feedback
    feedback_df = load_user_feedback()
    confusion_matrix = pd.crosstab(
        feedback_df['predicted_sentiment'],
        feedback_df['actual_sentiment'],
        normalize='index'
    )
    
    fig = px.imshow(
        confusion_matrix,
        labels=dict(x="Actual (User)", y="Predicted (Model)", color="Percentage"),
        title="Model Performance dari User Feedback"
    )
    st.plotly_chart(fig, use_container_width=True)
```

---

## 🔄 **Mekanisme Retraining Otomatis**

### **Kapan Retraining Dilakukan?**

#### **1. Trigger Berdasarkan Waktu (Time-Based)**
```python
# Retraining setiap minggu
schedule.every().monday.at("02:00").do(retrain_model)

# Atau setiap bulan
schedule.every().month.do(retrain_model)
```

**Keuntungan:**
- Predictable dan mudah di-schedule
- Tidak overload system
- Cocok untuk production

**Kekurangan:**
- Tidak adaptive terhadap perubahan data
- Bisa terlalu sering atau terlalu jarang

#### **2. Trigger Berdasarkan Data (Data-Driven)**

```python
def check_retraining_condition():
    """Cek apakah perlu retraining berdasarkan kondisi data"""
    
    # Kondisi 1: Jumlah data baru mencapai threshold
    new_reviews_count = get_new_reviews_since_last_training()
    if new_reviews_count >= 1000:  # 1000 review baru
        return True, "New data threshold reached"
    
    # Kondisi 2: Feedback negatif > threshold
    negative_feedback_rate = get_negative_feedback_rate()
    if negative_feedback_rate > 0.15:  # 15% prediksi salah
        return True, "High error rate detected"
    
    # Kondisi 3: Model accuracy drop
    current_accuracy = calculate_recent_accuracy()
    baseline_accuracy = get_baseline_accuracy()
    if current_accuracy < baseline_accuracy - 0.05:  # Drop 5%
        return True, "Accuracy degradation detected"
    
    # Kondisi 4: Drift detection
    data_drift_score = calculate_data_drift()
    if data_drift_score > 0.3:  # Significant drift
        return True, "Data drift detected"
    
    return False, "No retraining needed"
```

**Keuntungan:**
- Adaptive dan intelligent
- Optimal resource usage
- Responsive terhadap perubahan

**Kekurangan:**
- Lebih kompleks untuk implement
- Perlu monitoring metrics

#### **3. Trigger Berdasarkan Performance (Performance-Based)**

```python
def monitor_model_performance():
    """Monitor real-time performance dan trigger retraining"""
    
    # Hitung metrics 7 hari terakhir
    metrics_7d = calculate_metrics_last_7_days()
    
    triggers = []
    
    # Precision drop
    if metrics_7d['precision'] < 0.80:
        triggers.append("Low precision")
    
    # Recall drop
    if metrics_7d['recall'] < 0.75:
        triggers.append("Low recall")
    
    # User feedback agreement rate
    feedback_agreement = get_user_agreement_rate()
    if feedback_agreement < 0.85:
        triggers.append("Low user agreement")
    
    if triggers:
        send_alert_to_team(triggers)
        schedule_retraining()
```

---

## 🎯 **Rekomendasi Strategi Retraining**

### **Strategi Hybrid (Best Practice):**

```python
class RetrainingScheduler:
    def __init__(self):
        self.last_training_date = None
        self.baseline_accuracy = 0.84
        self.min_days_between_training = 7
        self.max_days_between_training = 30
    
    def should_retrain(self):
        """Kombinasi time-based dan data-driven"""
        
        days_since_training = (datetime.now() - self.last_training_date).days
        
        # Force retraining setelah 30 hari
        if days_since_training >= self.max_days_between_training:
            return True, "Monthly scheduled retraining"
        
        # Minimal 7 hari sebelum retraining berikutnya
        if days_since_training < self.min_days_between_training:
            return False, "Too soon to retrain"
        
        # Cek kondisi data-driven
        new_data = get_new_labeled_data_count()
        if new_data >= 500:  # 500 review baru dengan feedback
            return True, f"New labeled data: {new_data}"
        
        # Cek performance degradation
        current_perf = get_current_performance()
        if current_perf['accuracy'] < self.baseline_accuracy - 0.03:
            return True, "Performance degradation"
        
        # Cek user feedback
        error_rate = get_user_feedback_error_rate()
        if error_rate > 0.12:  # 12% error rate
            return True, f"High error rate: {error_rate:.1%}"
        
        return False, "All metrics healthy"
    
    def execute_retraining(self):
        """Execute full retraining pipeline"""
        
        # 1. Prepare data
        train_data = prepare_training_data(
            include_new_reviews=True,
            include_user_feedback=True
        )
        
        # 2. Train model
        new_model = train_bert_model(train_data)
        
        # 3. Validate model
        test_metrics = validate_model(new_model)
        
        # 4. A/B Testing (optional)
        if test_metrics['accuracy'] > self.baseline_accuracy:
            deploy_new_model(new_model)
            self.baseline_accuracy = test_metrics['accuracy']
            self.last_training_date = datetime.now()
            
            # Save metrics to database
            save_model_metrics(test_metrics)
            
            return True, "Model deployed successfully"
        else:
            return False, "New model performance not better"
```

### **Implementasi di Docker/Scheduler:**

```python
# src/scheduler/retraining_scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

# Cek setiap hari jam 2 pagi
@scheduler.scheduled_job('cron', hour=2, minute=0)
def check_and_retrain():
    """Daily check untuk retraining"""
    
    retrain_manager = RetrainingScheduler()
    should_retrain, reason = retrain_manager.should_retrain()
    
    if should_retrain:
        logger.info(f"Retraining triggered: {reason}")
        
        # Send notification
        send_slack_notification(
            f"🔄 Model retraining started\nReason: {reason}"
        )
        
        # Execute retraining
        success, message = retrain_manager.execute_retraining()
        
        if success:
            send_slack_notification(f"✅ {message}")
        else:
            send_slack_notification(f"❌ {message}")
    else:
        logger.info(f"No retraining needed: {reason}")

scheduler.start()
```

---

## 📊 **Monitoring Dashboard untuk Retraining**

### **Metrics to Track:**

1. **Model Performance Over Time**
   - Test Accuracy, Precision, Recall, F1
   - Train vs Test gap (overfitting detection)

2. **Data Quality**
   - New reviews count
   - Labeled data growth
   - Data distribution changes

3. **User Feedback**
   - Feedback rate (% users giving feedback)
   - Agreement rate (% correct predictions)
   - Common error patterns

4. **Retraining History**
   - Last retraining date
   - Reason for retraining
   - Performance improvement
   - Training duration

---

## 🚀 **Kesimpulan & Rekomendasi**

### **Untuk Fitur Komentar & Rating:**
✅ **IMPLEMENTASI SEGERA** - High ROI untuk:
- Data labeling gratis
- User engagement
- Model improvement tracking
- Continuous learning

### **Untuk Retraining Otomatis:**
📅 **Strategi Hybrid Recommended:**

```
IF days_since_training >= 30 THEN retrain  # Monthly maximum
ELSE IF new_feedback_data >= 500 THEN retrain  # Data threshold
ELSE IF error_rate > 12% THEN retrain  # Performance threshold
ELSE IF accuracy_drop > 3% THEN retrain  # Degradation detection
ELSE wait  # All healthy
```

### **Priority Implementation Order:**

1. **Phase 1 (Immediate):**
   - ✅ Tambah tabel `user_feedback`
   - ✅ UI untuk submit feedback
   - ✅ Basic analytics dashboard

2. **Phase 2 (Week 2):**
   - ✅ Retraining scheduler (time-based)
   - ✅ Email/Slack notification
   - ✅ Model versioning

3. **Phase 3 (Month 1):**
   - ✅ Data-driven retraining
   - ✅ A/B testing framework
   - ✅ Advanced monitoring

### **Estimated Impact:**
- **Model Accuracy**: +3-5% improvement per month
- **User Engagement**: +25% active users
- **Data Labeling Cost**: -80% (crowdsourcing)
- **Time to Production**: -50% (automated pipeline)

---

**Next Steps:**
1. Implement user_feedback table
2. Add feedback UI to dashboard
3. Set up weekly retraining schedule
4. Monitor and iterate! 🎯
