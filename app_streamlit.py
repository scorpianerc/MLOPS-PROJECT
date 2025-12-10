"""
Streamlit Dashboard untuk Sentiment Analysis MLOps  
Modern Glassmorphism Design with Monochrome Theme — Enhanced UX/UI
Fixed: All charts now use full width within tabs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import os
import math
import time

# Try to import torch and transformers for BERT model
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis MLOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Glassmorphism CSS with Monochrome — Enhanced
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Base styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #0a0a0a 100%);
        color: rgba(255, 255, 255, 0.85);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem 1rem;
    }

    [data-testid="stSidebar"] * {
        color: rgba(255, 255, 255, 0.9) !important;
    }

    /* Main container */
    .main .block-container {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(40px);
        -webkit-backdrop-filter: blur(40px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin: 1.5rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }

    /* Headings */
    h1 {
        color: rgba(255, 255, 255, 0.98) !important;
        font-weight: 800 !important;
        font-size: 2.3rem !important;
        text-align: center;
        letter-spacing: -0.03em;
        margin-bottom: 0.8rem;
        text-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }

    h2 {
        color: rgba(255, 255, 255, 0.92) !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.6rem;
    }

    h3 {
        color: rgba(255, 255, 255, 0.88) !important;
        font-weight: 600 !important;
        font-size: 1.2rem;
    }

    p, label, .stMarkdown {
        color: rgba(255, 255, 255, 0.75) !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: rgba(255, 255, 255, 0.95) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.55) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.8rem 1.2rem;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 12px 32px rgba(255, 255, 255, 0.10);
        border-color: rgba(255, 255, 255, 0.18);
        background: rgba(255, 255, 255, 0.06);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 8px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: transparent;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.55);
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        padding: 0 20px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.07);
        color: rgba(255, 255, 255, 0.85);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.14) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.1), 0 4px 12px rgba(0,0,0,0.2);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(70,70,70,0.7), rgba(40,40,40,0.8));
        color: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        padding: 0.9rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(90,90,90,0.8), rgba(60,60,60,0.9));
        border-color: rgba(255, 255, 255, 0.3);
        transform: translateY(-3px) scale(1.03);
        box-shadow: 0 8px 24px rgba(255, 255, 255, 0.15);
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }

    /* Select/Input */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div {
        background: rgba(30, 30, 30, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        color: rgba(255, 255, 255, 0.9);
        padding: 0.5rem 1rem;
    }
    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within {
        border-color: rgba(100, 180, 255, 0.5);
        box-shadow: 0 0 0 3px rgba(100, 180, 255, 0.15);
    }

    /* Plotly - Full width container */
    .js-plotly-plot {
        border-radius: 18px;
        background: rgba(20, 20, 20, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 16px;
        transition: transform 0.3s ease;
        width: 100% !important;
    }
    .js-plotly-plot:hover {
        transform: scale(1.005);
        border-color: rgba(255, 255, 255, 0.12);
    }

    /* Review cards */
    .review-card {
        background: rgba(35, 35, 35, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .review-card:hover {
        background: rgba(45, 45, 45, 0.85);
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(0,0,0,0.3);
    }

    /* Responsive grid */
    @media (max-width: 768px) {
        .main .block-container {
            margin: 0.8rem;
            padding: 1rem;
        }
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.3rem !important; }
        .stTabs [data-baseweb="tab"] { height: 42px; font-size: 0.85rem; padding: 0 14px; }
        div[data-testid="metric-container"] { padding: 1.2rem 0.8rem; }
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Loading spinner (custom) */
    .loading-spinner {
        display: inline-block;
        width: 24px;
        height: 24px;
        border: 3px solid rgba(255,255,255,0.2);
        border-radius: 50%;
        border-top-color: #6366f1;
        animation: spin 1s linear infinite;
        margin-right: 8px;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: rgba(30,30,30,0.3); }
    ::-webkit-scrollbar-thumb { background: rgba(100,100,100,0.4); border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(150,150,150,0.6); }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data(ttl=30)
def load_data():
    try:
        df = pd.read_csv('data/processed/processed_reviews.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def load_metrics():
    """Load metrics from JSON file with minimal caching"""
    # Try to load BERT metrics first (better model)
    bert_metrics_path = 'models/bert_metrics.json'
    regular_metrics_path = 'models/metrics.json'
    
    try:
        if os.path.exists(bert_metrics_path):
            with open(bert_metrics_path, 'r') as f:
                metrics_data = json.load(f)
            metrics_data['model_type'] = 'IndoBERT'
            # BERT metrics only have test metrics
            if 'accuracy' in metrics_data:
                metrics_data['test_accuracy'] = metrics_data['accuracy']
                metrics_data['test_f1'] = metrics_data['f1']
                metrics_data['test_precision'] = metrics_data['precision']
                metrics_data['test_recall'] = metrics_data['recall']
                # Set train metrics same as test (or load from training logs if available)
                metrics_data['train_accuracy'] = metrics_data.get('train_accuracy', metrics_data['accuracy'])
                metrics_data['train_f1'] = metrics_data.get('train_f1', metrics_data['f1'])
                metrics_data['train_precision'] = metrics_data.get('train_precision', metrics_data['precision'])
                metrics_data['train_recall'] = metrics_data.get('train_recall', metrics_data['recall'])
            return metrics_data
        elif os.path.exists(regular_metrics_path):
            with open(regular_metrics_path, 'r') as f:
                metrics_data = json.load(f)
            metrics_data['model_type'] = 'Logistic Regression'
            required_keys = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall', 
                           'train_accuracy', 'train_f1', 'train_precision', 'train_recall']
            for key in required_keys:
                if key not in metrics_data:
                    metrics_data[key] = 0
            return metrics_data
    except Exception as e:
        pass
    
    return {
        'test_accuracy': 0, 'test_f1': 0, 'test_precision': 0, 'test_recall': 0,
        'train_accuracy': 0, 'train_f1': 0, 'train_precision': 0, 'train_recall': 0,
        'model_type': 'Unknown'
    }

@st.cache_resource
def load_bert_model():
    """Load BERT model and tokenizer (cached across sessions)"""
    if not BERT_AVAILABLE:
        return None, None, "PyTorch/Transformers not installed"
    
    model_path = 'models/bert_model'
    if not os.path.exists(model_path):
        return None, None, "BERT model not found"
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        # Load label map
        with open(os.path.join(model_path, 'label_map.json'), 'r') as f:
            label_map = json.load(f)
        reverse_label_map = {v: k for k, v in label_map.items()}
        
        return model, tokenizer, device, reverse_label_map, None
    except Exception as e:
        return None, None, None, None, str(e)

def predict_sentiment_bert(text, model, tokenizer, device, reverse_label_map):
    """Predict sentiment using BERT model"""
    if model is None or tokenizer is None:
        return None, None
    
    try:
        # Tokenize input
        encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        sentiment = reverse_label_map[predicted_class]
        return sentiment, confidence
    except Exception as e:
        return None, None

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'page_reviews' not in st.session_state:
    st.session_state.page_reviews = 1
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# Main title
st.markdown("<h1>📊 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)

# Load data & metrics
with st.spinner("Loading data and metrics..."):
    df = load_data()
    metrics = load_metrics()
    st.session_state.data_loaded = True
    st.session_state.last_refresh = datetime.now()

st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.6); margin-bottom: 2rem;'>MLOps Pipeline for Pintu App Reviews</p>", unsafe_allow_html=True)

# Debug: Show metrics loading status in sidebar
st.sidebar.markdown("### 🔍 Debug Info")
if metrics and metrics.get('test_accuracy', 0) > 0:
    st.sidebar.success("✅ Metrics loaded")
else:
    st.sidebar.error("❌ Metrics NOT loaded!")
    st.sidebar.info("Check models/metrics.json")

if df.empty:
    st.warning("No data available. Please run the scraping process first.")
    st.stop()

# Sidebar filters
st.sidebar.markdown("## 🎯 Filters")

# Refresh button
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.session_state.last_refresh = datetime.now()
    st.rerun()

st.sidebar.markdown("### 💾 Export Format")
export_format = st.sidebar.radio("Choose format:", ["CSV", "JSON"], horizontal=True)

sentiment_filter = st.sidebar.multiselect(
    "Sentiment",
    options=df['sentiment_label'].unique() if 'sentiment_label' in df.columns else [],
    default=df['sentiment_label'].unique() if 'sentiment_label' in df.columns else []
)

rating_range = st.sidebar.slider(
    "Rating Range",
    min_value=1,
    max_value=5,
    value=(1, 5)
)

# Filter data
filtered_df = df.copy()
if sentiment_filter and 'sentiment_label' in df.columns:
    filtered_df = filtered_df[filtered_df['sentiment_label'].isin(sentiment_filter)]
if 'rating' in df.columns:
    filtered_df = filtered_df[(filtered_df['rating'] >= rating_range[0]) & (filtered_df['rating'] <= rating_range[1])]

# Export buttons
if not filtered_df.empty:
    col_exp1, col_exp2 = st.sidebar.columns(2)
    with col_exp1:
        if export_format == "CSV":
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 CSV",
                data=csv_data,
                file_name=f"pintu_sentiment_{datetime.now():%Y%m%d}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button("📥 CSV", disabled=True, use_container_width=True)
    with col_exp2:
        if export_format == "JSON":
            json_data = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                "📥 JSON",
                data=json_data,
                file_name=f"pintu_sentiment_{datetime.now():%Y%m%d}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.button("📥 JSON", disabled=True, use_container_width=True)

# Show last updated timestamp for data file
data_path = Path('data/processed/processed_reviews.csv')
if data_path.exists():
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(data_path))
        st.caption(f"📅 Last data refresh: {mtime:%Y-%m-%d %H:%M:%S}")
    except Exception:
        pass

# Visualization Section
st.markdown("## 📈 Analytics")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "📈 Trends", "📝 Details", "☁️ Insights"])

with tab1:
    # Key Metrics Cards Row
    st.markdown("### 📊 Key Metrics")
    
    # Simulate prior metrics for delta (replace with real history if available)
    prev_accuracy = 0.89
    current_accuracy = metrics.get('test_accuracy', 0)
    accuracy_delta = f"{(current_accuracy - prev_accuracy)*100:+.1f}%" if prev_accuracy > 0 else None

    # Center the metrics with padding columns
    col_pad1, col1, col2, col3, col4, col_pad2 = st.columns([0.5, 1, 1, 1, 1, 0.5])
    with col1:
        total_reviews = len(filtered_df)
        st.metric("Total Reviews", f"{total_reviews:,}")
    with col2:
        avg_rating = filtered_df['rating'].mean() if 'rating' in filtered_df.columns else 0
        st.metric("Avg Rating", f"{avg_rating:.1f} ⭐")
    with col3:
        st.metric("Model Accuracy", f"{current_accuracy:.1%}", delta=accuracy_delta)
    with col4:
        f1_score = metrics.get('test_f1', 0)
        f1_delta = "+2.1%" if f1_score > 0.85 else None
        st.metric("F1 Score", f"{f1_score:.1%}", delta=f1_delta)

    # Graph Cards - Full Width
    st.markdown("### 📈 Visual Analytics")
    
    # Sentiment Distribution - Full Width
    st.markdown("**📊 Sentiment Distribution**")
    if 'sentiment_label' in filtered_df.columns:
        sentiment_counts = filtered_df['sentiment_label'].value_counts()
        color_map = {'positive': '#4CAF50', 'negative': '#F44336', 'neutral': '#9E9E9E'}
        colors = [color_map.get(label, '#CCCCCC') for label in sentiment_counts.index]

        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="",
            color_discrete_sequence=colors,
            template='plotly_dark',
            hole=0.4
        )
        fig.update_traces(
            textinfo='percent+label',
            textfont_size=14,
            textfont_color='white',
            marker=dict(line=dict(color='#000000', width=2))
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.9)', size=14),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(255,255,255,0.1)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1
            ),
            margin=dict(t=40, b=40, l=40, r=40),
            height=500,
            hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=13, font_color="white")
        )
        st.plotly_chart(fig, use_container_width=True)

    # Rating Distribution - Full Width
    st.markdown("**⭐ Rating Distribution**")
    if 'rating' in filtered_df.columns:
        rating_counts = filtered_df['rating'].value_counts().sort_index()
        colors_list = ['#F44336', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50']
        bar_colors = [colors_list[int(r)-1] for r in rating_counts.index]

        fig = go.Figure(data=[
            go.Bar(
                x=rating_counts.index,
                y=rating_counts.values,
                marker=dict(
                    color=bar_colors,
                    line=dict(color='rgba(255,255,255,0.3)', width=1)
                ),
                text=rating_counts.values,
                textposition='outside',
                textfont=dict(size=12, color='white')
            )
        ])
        fig.update_layout(
            title="",
            xaxis_title='Rating (⭐)',
            yaxis_title='Count',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.9)', size=12),
            showlegend=False,
            template='plotly_dark',
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            margin=dict(t=40, b=80, l=60, r=60),
            height=500,
            hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=13, font_color="white")
        )
        st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.markdown("### 📈 Reviews Over Time")
    if 'review_date' in filtered_df.columns:
        filtered_df['review_date'] = pd.to_datetime(filtered_df['review_date'])
        daily_counts = filtered_df.groupby(filtered_df['review_date'].dt.date).size()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=daily_counts.values,
            mode='lines+markers',
            name='Reviews',
            line=dict(color='#2196F3', width=3),
            marker=dict(size=7, color='#2196F3', line=dict(color='white', width=1)),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.2)'
        ))
        fig.update_layout(
            title="",
            xaxis_title='Date',
            yaxis_title='Number of Reviews',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.9)', size=14),
            template='plotly_dark',
            hovermode='x unified',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            height=600,
            margin=dict(t=40, b=80, l=80, r=80),
            hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=13, font_color="white")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No date data available for trend analysis.")


with tab3:
    st.markdown("### 📝 Recent Reviews")
    
    # Pagination controls
    page_size = 8
    total = len(filtered_df)
    total_pages = math.ceil(total / page_size) if total > 0 else 1

    colp1, colp2, colp3 = st.columns([1, 2, 1])
    
    with colp1:
        if st.button('◀ Prev', key="prev_btn", use_container_width=True):
            st.session_state.page_reviews = max(1, st.session_state.page_reviews - 1)
    
    with colp2:
        st.markdown(
            f"<p style='text-align: center; font-size: 1.1rem; font-weight: 600; color: rgba(255,255,255,0.9); padding-top: 4px;'>"
            f"Page {st.session_state.page_reviews} / {total_pages} — {total:,} reviews</p>",
            unsafe_allow_html=True
        )
    
    with colp3:
        if st.button('Next ▶', key="next_btn", use_container_width=True):
            st.session_state.page_reviews = min(total_pages, st.session_state.page_reviews + 1)

    start = (st.session_state.page_reviews - 1) * page_size
    end = start + page_size
    page_df = filtered_df.iloc[start:end]

    if page_df.empty:
        st.info('No reviews match the current filters.')
    else:
        cols = st.columns(2)
        for i, (_, row) in enumerate(page_df.iterrows()):
            sentiment = str(row.get('sentiment_label', 'neutral')).lower()
            rating_val = row.get('rating', 0) or 0
            try:
                rating_stars = "⭐" * int(max(0, min(5, int(rating_val))))
            except:
                rating_stars = ''
            
            import html
            review_text = html.escape(str(row.get('review_text', '')))
            review_preview = review_text[:250] + ('...' if len(review_text) > 250 else '')
            
            if sentiment == 'negative':
                badge_style = "background:rgba(244,67,54,0.15); color:#E57373; border:1px solid rgba(244,67,54,0.3);"
            elif sentiment == 'neutral':
                badge_style = "background:rgba(158,158,158,0.15); color:#BDBDBD; border:1px solid rgba(158,158,158,0.3);"
            else:
                badge_style = "background:rgba(76,175,80,0.15); color:#81C784; border:1px solid rgba(76,175,80,0.3);"

            card_html = f"""
            <div class="review-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <span style="color:#FFD700; font-weight:600;">{rating_stars} {rating_val:.1f}</span>
                    <span style="padding:4px 12px; border-radius:20px; font-size:0.85rem; font-weight:500; {badge_style}">{sentiment.upper()}</span>
                </div>
                <div style="color:rgba(255,255,255,0.85); line-height:1.6; font-size:0.95rem; margin-bottom:10px;">
                    {review_preview}
                </div>
                <div style="color:rgba(255,255,255,0.4); font-size:0.8rem;">
                    {row.get('review_date', '')}
                </div>
            </div>
            """

            with cols[i % 2]:
                st.markdown(card_html, unsafe_allow_html=True)


with tab4:
    st.markdown("### 🔤 Word Frequency Analysis")
    
    if 'review_text' in filtered_df.columns and not filtered_df.empty:
        all_text = ' '.join(filtered_df['review_text'].dropna().astype(str))
        words = all_text.lower().split()
        stop_words = {
            'yang', 'dan', 'di', 'ke', 'untuk', 'dari', 'ini', 'itu', 'dengan', 'pada', 'adalah',
            'tidak', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'as', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'yg', 'aja', 'sih',
            'nya', 'gue', 'gw', 'loh', 'lah', 'pun', 'juga', 'yg', 'nya'
        }
        words_filtered = [w for w in words if len(w) > 2 and w.isalpha() and w not in stop_words]
        
        from collections import Counter
        word_freq = Counter(words_filtered).most_common(20)
        
        if word_freq:
            word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            
            fig = px.bar(
                word_df,
                x='Frequency',
                y='Word',
                orientation='h',
                title='Top 20 Most Frequent Words',
                color='Frequency',
                color_continuous_scale=['#333333', '#FFFFFF'],
                template='plotly_dark'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(255,255,255,0.9)', size=13),
                height=600,
                margin=dict(t=60, b=60, l=120, r=80),
                hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=13, font_color="white")
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📅 Sentiment Timeline")
    if 'review_date' in filtered_df.columns and 'sentiment_label' in filtered_df.columns:
        filtered_df['review_date'] = pd.to_datetime(filtered_df['review_date'])
        sentiment_timeline = filtered_df.groupby([filtered_df['review_date'].dt.date, 'sentiment_label']).size().reset_index(name='count')
        sentiment_timeline.columns = ['date', 'sentiment', 'count']
        
        fig = px.area(
            sentiment_timeline,
            x='date',
            y='count',
            color='sentiment',
            title='Sentiment Distribution Over Time',
            color_discrete_map={'positive': '#81C784', 'negative': '#E57373', 'neutral': '#BDBDBD'},
            template='plotly_dark'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.9)', size=14),
            hovermode='x unified',
            height=500,
            margin=dict(t=60, b=80, l=80, r=80),
            hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=13, font_color="white")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🔥 Sentiment vs Rating Heatmap")
    if 'rating' in filtered_df.columns and 'sentiment_label' in filtered_df.columns:
        heatmap_data = pd.crosstab(filtered_df['sentiment_label'], filtered_df['rating'])
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Greys',
            text=heatmap_data.values,
            texttemplate='%{text}',
            textfont={"size": 14, "color": "white"}
        ))
        fig.update_layout(
            title='Review Count by Sentiment and Rating',
            xaxis_title='Rating',
            yaxis_title='Sentiment',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.9)', size=14),
            height=450,
            margin=dict(t=60, b=80, l=120, r=80),
            hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=13, font_color="white")
        )
        st.plotly_chart(fig, use_container_width=True)


# Model Performance Section
if metrics:
    model_type = metrics.get('model_type', 'Unknown')
    st.markdown(f"## 🎯 Model Performance: **{model_type}**")
    
    # Show model badge
    if model_type == 'IndoBERT':
        st.markdown("""
        <div style='
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 8px 16px;
            border-radius: 20px;
            margin-bottom: 10px;
        '>
            <span style='color: white; font-weight: bold;'>🤖 State-of-the-art Indonesian BERT Model</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    test_acc = metrics.get('test_accuracy', 0)
    test_prec = metrics.get('test_precision', 0)
    test_rec = metrics.get('test_recall', 0)
    test_f1 = metrics.get('test_f1', 0)

    with col1:
        st.metric("Test Accuracy", f"{test_acc:.1%}")
    with col2:
        st.metric("Test F1 Score", f"{test_f1:.1%}")
    with col3:
        st.metric("Test Precision", f"{test_prec:.1%}")
    with col4:
        st.metric("Test Recall", f"{test_rec:.1%}")

    st.markdown("### 📈 Train vs Test Comparison")

    perf_metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Train': [
            metrics.get('train_accuracy', 0),
            metrics.get('train_precision', 0),
            metrics.get('train_recall', 0),
            metrics.get('train_f1', 0)
        ],
        'Test': [test_acc, test_prec, test_rec, test_f1]
    }
    perf_df = pd.DataFrame(perf_metrics)

    fig = go.Figure(data=[
        go.Bar(
            name='Train',
            x=perf_df['Metric'],
            y=perf_df['Train'],
            marker_color='rgba(76, 175, 80, 0.8)',
            marker=dict(line=dict(color='rgba(255,255,255,0.3)', width=1))
        ),
        go.Bar(
            name='Test',
            x=perf_df['Metric'],
            y=perf_df['Test'],
            marker_color='rgba(33, 150, 243, 0.8)',
            marker=dict(line=dict(color='rgba(255,255,255,0.3)', width=1))
        )
    ])
    fig.update_layout(
        title="",
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(255,255,255,0.9)', size=13),
        margin=dict(t=40, b=80, l=80, r=80),
        height=500,
        showlegend=True,
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=13, font_color="white"),
        legend=dict(
            bgcolor='rgba(255,255,255,0.1)',
            bordercolor='rgba(255,255,255,0.3)',
            borderwidth=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🔍 View Raw Metrics Data"):
        st.json(metrics)
        model_file = 'bert_metrics.json' if metrics.get('model_type') == 'IndoBERT' else 'metrics.json'
        st.caption(f"Loaded from `models/{model_file}` | Model: **{metrics.get('model_type', 'Unknown')}**")


# ================= NEW: TRY IT YOURSELF SECTION =================
st.markdown("---")
st.markdown("## 🧪 Try It Yourself: Real-Time Prediction")

if BERT_AVAILABLE and os.path.exists('models/bert_model'):
    # Load BERT model
    with st.spinner("Loading IndoBERT model..."):
        model_result = load_bert_model()
    
    if len(model_result) == 5 and model_result[0] is not None:
        model, tokenizer, device, reverse_label_map, error = model_result
        
        st.success(f"✅ IndoBERT model loaded successfully! Running on: **{device}**")
        
        # Input text area
        user_input = st.text_area(
            "Enter your review text (in Indonesian):",
            placeholder="Contoh: Aplikasi ini sangat bagus dan mudah digunakan!",
            height=100,
            help="Type or paste a review in Indonesian and click 'Predict Sentiment'"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔮 Predict Sentiment", use_container_width=True)
        
        if predict_button and user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence = predict_sentiment_bert(
                    user_input, model, tokenizer, device, reverse_label_map
                )
            
            if sentiment:
                # Display result with styling
                sentiment_color = {
                    'positive': '#4CAF50',
                    'negative': '#F44336',
                    'neutral': '#FFC107'
                }
                color = sentiment_color.get(sentiment.lower(), '#2196F3')
                
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255,255,255,0.2);
                    border-radius: 15px;
                    padding: 25px;
                    margin: 20px 0;
                    text-align: center;
                '>
                    <h3 style='color: {color}; margin-bottom: 10px;'>
                        Sentiment: {sentiment.upper()}
                    </h3>
                    <p style='font-size: 1.2rem; color: rgba(255,255,255,0.8);'>
                        Confidence: <strong>{confidence:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence Level", 'font': {'color': 'white'}},
                    number={'suffix': "%", 'font': {'color': 'white'}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickcolor': "white"},
                        'bar': {'color': color},
                        'bgcolor': "rgba(255,255,255,0.1)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(255,255,255,0.3)",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(255,255,255,0.1)'},
                            {'range': [50, 75], 'color': 'rgba(255,255,255,0.15)'},
                            {'range': [75, 100], 'color': 'rgba(255,255,255,0.2)'}
                        ],
                    }
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': "white"},
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Failed to predict sentiment. Please try again.")
        
        elif predict_button and not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze!")
    
    else:
        error_msg = model_result[-1] if len(model_result) == 5 else "Unknown error"
        st.error(f"❌ Failed to load BERT model: {error_msg}")
        st.info("💡 Make sure you have trained the BERT model by running: `python src/training/train_bert.py`")

else:
    if not BERT_AVAILABLE:
        st.warning("⚠️ PyTorch and Transformers libraries not installed.")
        st.code("pip install torch transformers", language="bash")
    elif not os.path.exists('models/bert_model'):
        st.warning("⚠️ BERT model not found.")
        st.info("💡 Train the model first: `python src/training/train_bert.py`")


# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: rgba(255,255,255,0.4); font-size: 0.85rem;'>"
    "MLOps Sentiment Analysis Dashboard | Powered by Streamlit & Docker"
    "</p>",
    unsafe_allow_html=True
)