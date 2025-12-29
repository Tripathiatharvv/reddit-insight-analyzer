"""
Reddit Product Insight Analyzer - Streamlit App
A free, optimized tool for analyzing Reddit product discussions
"""

import streamlit as st
import time
from datetime import datetime

# Import modules
from reddit_fetcher import RedditFetcher
from text_processor import TextProcessor
from nlp_analyzer import NLPAnalyzer
from report_generator import ReportGenerator
from ollama_insights import AIInsights


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Reddit Product Insight Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# SEO META TAGS FOR SOCIAL PREVIEWS
# ============================================================================
st.markdown("""
<meta name="description" content="Transform Reddit discussions into actionable product insights using AI-powered sentiment analysis.">
<meta property="og:title" content="Reddit Product Insight Analyzer">
<meta property="og:description" content="AI-powered tool to analyze Reddit discussions and extract product insights, sentiment trends, and actionable recommendations.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://reddit-insight-analyzer.streamlit.app">
<meta property="og:image" content="https://raw.githubusercontent.com/Tripathiatharvv/reddit-insight-analyzer/main/preview.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Reddit Product Insight Analyzer">
<meta name="twitter:description" content="AI-powered sentiment analysis for Reddit product discussions.">
""", unsafe_allow_html=True)

# ============================================================================
# CUSTOM CSS - AWWWARDS LEVEL DESIGN
# ============================================================================
st.markdown("""
<style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root variables */
    :root {
        --bg-primary: #050510;
        --bg-secondary: #0a0a1a;
        --accent-1: #6366f1;
        --accent-2: #8b5cf6;
        --accent-3: #a855f7;
        --success: #22c55e;
        --warning: #f59e0b;
        --error: #ef4444;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
        --glow: rgba(99, 102, 241, 0.4);
    }
    
    /* Global styles */
    .stApp {
        background: var(--bg-primary);
        background-image: 
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.15), transparent),
            radial-gradient(ellipse 60% 40% at 100% 100%, rgba(139, 92, 246, 0.1), transparent);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding and sidebar */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    
    /* Config Panel Styling */
    .config-panel {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.04) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
    }
    
    .config-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .config-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .config-subtitle {
        font-size: 0.85rem;
        color: var(--text-muted);
    }
    
    .config-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }
    
    .config-item {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .config-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    
    /* Style Streamlit inputs */
    .stTextInput > div > div {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
    }
    
    .stTextInput input {
        color: var(--text-primary) !important;
    }
    
    .stSelectbox > div > div {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent-1);
        margin: 1.5rem 0 0.5rem 0;
    }
    
    /* Premium header */
    .hero-section {
        text-align: center;
        padding: 3rem 0 2rem 0;
        position: relative;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        background: linear-gradient(135deg, #fff 0%, #e2e8f0 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-title .accent {
        background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-3) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-title {
        font-size: 1.125rem;
        font-weight: 400;
        color: var(--text-secondary);
        letter-spacing: 0.01em;
    }
    
    .divider {
        width: 100%;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--glass-border), transparent);
        margin: 2rem 0;
    }
    
    /* Glass cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 0 40px rgba(99, 102, 241, 0.1);
    }
    
    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-item {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-item:hover {
        transform: translateY(-2px);
        border-color: var(--accent-1);
    }
    
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2.5rem 0 1.5rem 0;
    }
    
    .section-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    
    /* Sentiment visualization */
    .sentiment-bar-container {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 12px;
        padding: 3px;
        margin: 1rem 0;
    }
    
    .sentiment-bar-inner {
        display: flex;
        height: 40px;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .sentiment-segment {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
    }
    
    .sentiment-segment.positive {
        background: linear-gradient(135deg, #22c55e, #16a34a);
    }
    
    .sentiment-segment.neutral {
        background: linear-gradient(135deg, #64748b, #475569);
    }
    
    .sentiment-segment.negative {
        background: linear-gradient(135deg, #ef4444, #dc2626);
    }
    
    .sentiment-legend {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    
    .legend-dot.positive { background: #22c55e; }
    .legend-dot.neutral { background: #64748b; }
    .legend-dot.negative { background: #ef4444; }
    
    /* Theme cards */
    .theme-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .theme-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: var(--accent-1);
    }
    
    .theme-card.negative::before { background: var(--error); }
    .theme-card.positive::before { background: var(--success); }
    .theme-card.mixed::before { background: var(--warning); }
    
    .theme-card:hover {
        transform: translateX(4px);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .theme-name {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .theme-desc {
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    
    .theme-count {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: var(--accent-1);
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin-top: 0.75rem;
    }
    
    /* Impact cards */
    .impact-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        position: relative;
        transition: all 0.3s ease;
    }
    
    .impact-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-1), var(--accent-3));
    }
    
    .impact-card.negative::before { background: linear-gradient(90deg, #ef4444, #f87171); }
    .impact-card.positive::before { background: linear-gradient(90deg, #22c55e, #4ade80); }
    
    .impact-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    .impact-title {
        font-size: 0.95rem;
        font-weight: 500;
        color: var(--text-primary);
        line-height: 1.4;
        margin-bottom: 0.75rem;
    }
    
    .impact-meta {
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Action items */
    .action-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        position: relative;
        padding-left: 3rem;
    }
    
    .action-number {
        position: absolute;
        left: 1rem;
        top: 50%;
        transform: translateY(-50%);
        width: 24px;
        height: 24px;
        background: var(--accent-1);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        color: white;
    }
    
    .action-text {
        font-size: 0.9rem;
        color: var(--text-primary);
        line-height: 1.5;
    }
    
    /* Summary box */
    .summary-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.04) 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 1.75rem;
        font-size: 1rem;
        line-height: 1.7;
        color: var(--text-secondary);
    }
    
    /* Insights grid */
    .insights-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        margin: 1rem 0;
    }
    
    .insight-column h4 {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .insight-item {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 8px;
        padding: 0.875rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: var(--text-secondary);
        transition: all 0.2s ease;
    }
    
    .insight-item:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .insight-item.positive { border-left: 3px solid var(--success); }
    .insight-item.negative { border-left: 3px solid var(--error); }
    .insight-item.improving { border-left: 3px solid #22c55e; }
    .insight-item.worsening { border-left: 3px solid #f59e0b; }
    
    /* AI badge */
    .ai-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 100px;
        padding: 0.5rem 1rem;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--accent-1);
        margin: 1.5rem 0;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 6rem 2rem;
        color: var(--text-muted);
    }
    
    .empty-state h2 {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 0.75rem;
    }
    
    .empty-state p {
        font-size: 1rem;
        max-width: 400px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: var(--accent-1) !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 0;
        margin-top: 4rem;
        border-top: 1px solid var(--glass-border);
    }
    
    .footer-text {
        font-size: 0.8rem;
        color: var(--text-muted);
    }
    
    .footer-accent {
        color: var(--accent-1);
    }
    
    /* ============================================
       MOBILE RESPONSIVE STYLES
       ============================================ */
    
    /* Tablet (768px and below) */
    @media screen and (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        
        .sub-title {
            font-size: 1rem;
        }
        
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }
        
        .metric-item {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.25rem;
        }
        
        .insights-grid {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .section-header {
            margin: 2rem 0 1rem 0;
        }
        
        .section-title {
            font-size: 1.1rem;
        }
        
        .summary-box {
            padding: 1.25rem;
            font-size: 0.95rem;
        }
        
        .sentiment-legend {
            gap: 1rem;
            flex-wrap: wrap;
        }
        
        .hero-section {
            padding: 2rem 0 1.5rem 0;
        }
    }
    
    /* Mobile (480px and below) */
    @media screen and (max-width: 480px) {
        .main-title {
            font-size: 1.75rem;
            letter-spacing: -0.02em;
        }
        
        .sub-title {
            font-size: 0.875rem;
            padding: 0 0.5rem;
        }
        
        .metric-grid {
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }
        
        .metric-item {
            padding: 0.875rem;
        }
        
        .metric-label {
            font-size: 0.6rem;
        }
        
        .metric-value {
            font-size: 1rem;
        }
        
        .section-header {
            margin: 1.5rem 0 1rem 0;
            gap: 0.5rem;
        }
        
        .section-icon {
            width: 28px;
            height: 28px;
            font-size: 0.875rem;
        }
        
        .section-title {
            font-size: 1rem;
        }
        
        .summary-box {
            padding: 1rem;
            font-size: 0.9rem;
            line-height: 1.6;
            border-radius: 12px;
        }
        
        .sentiment-bar-inner {
            height: 36px;
        }
        
        .sentiment-segment {
            font-size: 0.7rem;
        }
        
        .sentiment-legend {
            gap: 0.75rem;
        }
        
        .legend-item {
            font-size: 0.7rem;
        }
        
        .theme-card {
            padding: 1rem;
        }
        
        .theme-name {
            font-size: 0.9rem;
        }
        
        .theme-desc {
            font-size: 0.8rem;
        }
        
        .theme-count {
            font-size: 0.65rem;
        }
        
        .insight-column h4 {
            font-size: 0.7rem;
        }
        
        .insight-item {
            padding: 0.75rem;
            font-size: 0.8rem;
        }
        
        .impact-card {
            padding: 1rem;
        }
        
        .impact-title {
            font-size: 0.85rem;
        }
        
        .impact-meta {
            font-size: 0.65rem;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        
        .action-card {
            padding: 1rem;
            padding-left: 2.5rem;
        }
        
        .action-number {
            width: 20px;
            height: 20px;
            font-size: 0.65rem;
            left: 0.75rem;
        }
        
        .action-text {
            font-size: 0.85rem;
        }
        
        .ai-badge {
            font-size: 0.7rem;
            padding: 0.4rem 0.8rem;
        }
        
        .empty-state {
            padding: 3rem 1rem;
        }
        
        .empty-state h2 {
            font-size: 1.25rem;
        }
        
        .empty-state p {
            font-size: 0.875rem;
        }
        
        .footer {
            padding: 2rem 0;
            margin-top: 2rem;
        }
        
        .footer-text {
            font-size: 0.7rem;
        }
        
        .divider {
            margin: 1.5rem 0;
        }
        
        .hero-section {
            padding: 1.5rem 0 1rem 0;
        }
        
        /* Touch-friendly button sizing */
        .stButton>button {
            padding: 0.875rem 1.25rem !important;
            font-size: 0.85rem !important;
            min-height: 48px !important;
        }
    }
    
    /* Small mobile (360px and below) */
    @media screen and (max-width: 360px) {
        .main-title {
            font-size: 1.5rem;
        }
        
        .metric-grid {
            grid-template-columns: 1fr;
        }
        
        .sentiment-legend {
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Hero Header
st.markdown('''
<div class="hero-section">
    <h1 class="main-title">
        Reddit <span class="accent">Insight</span> Analyzer
    </h1>
    <p class="sub-title">Transform community discussions into actionable product intelligence</p>
</div>
''', unsafe_allow_html=True)

# Initialize AI to check provider
ai_insights = AIInsights()
provider_icon = "☁️" if ai_insights.provider == "groq" else "💻"
provider_name = "Groq Cloud (Free)" if ai_insights.provider == "groq" else "Local Ollama"
provider_color = "#f59e0b" if ai_insights.provider == "groq" else "#22c55e" # Orange for Cloud, Green for Local

# Configuration Panel (embedded in main page)
st.markdown(f'''
<div class="config-panel">
    <div class="config-header">
        <div class="config-title">⚙️ Analysis Configuration</div>
        <div class="config-subtitle">Configure your analysis parameters below</div>
        <div style="margin-top: 0.5rem; font-size: 0.8rem; color: {provider_color}; background: rgba(255,255,255,0.05); padding: 4px 12px; border-radius: 20px; display: inline-block;">
            {provider_icon} AI Provider: <strong>{provider_name}</strong>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# Config inputs using columns
col1, col2, col3 = st.columns(3)

with col1:
    subreddit = st.text_input(
        "🎯 Subreddit",
        value="apple",
        placeholder="e.g., apple, iphone, Android",
        help="Enter subreddit name without r/"
    )
    
    use_ml = st.checkbox(
        "🧠 ML Sentiment",
        value=True,
        help="VADER + TextBlob for better sentiment accuracy"
    )

with col2:
    post_limit = st.slider(
        "📊 Posts to Analyze",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        help="More posts = more comprehensive analysis"
    )
    
    use_ai = st.checkbox(
        "🤖 AI Deep Insights",
        value=True,
        help=f"Use {provider_name} for AI-generated insights"
    )

with col3:
    comments_per_post = st.slider(
        "💬 Comments per Post",
        min_value=0,
        max_value=50,
        value=10,
        step=5,
        help="0 = no comments"
    )
    
    analysis_mode = st.selectbox(
        "📈 Analysis Mode",
        options=[
            "summary + sentiment + themes",
            "summary + sentiment",
            "summary_only"
        ],
        index=0
    )

# Analyze button
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_button = st.button(
        "🚀 Analyze Subreddit",
        type="primary",
        use_container_width=True
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ============================================================================
# ANALYSIS LOGIC
# ============================================================================

@st.cache_resource
def get_analyzer(use_ml: bool):
    """Cache the NLP analyzer so ML models are loaded only once"""
    return NLPAnalyzer(use_ml_models=use_ml)


@st.cache_resource
def get_report_generator(use_ai: bool):
    """Cache the report generator with AI settings"""
    return ReportGenerator(use_ai=use_ai)


def run_analysis(subreddit: str, post_limit: int, comments_per_post: int, mode: str, use_ml: bool, use_ai: bool):
    """Run the full analysis pipeline with optional AI enhancement and comment fetching"""
    
    start_time = time.time()
    
    # Initialize components (analyzer is cached if using ML)
    fetcher = RedditFetcher()
    processor = TextProcessor()
    analyzer = get_analyzer(use_ml)
    report_gen = get_report_generator(use_ai)
    
    # Progress tracking
    progress = st.progress(0, text="Initializing...")
    
    try:
        # Step 1: Fetch posts (with comments if requested)
        if comments_per_post > 0:
            progress.progress(10, text=f"📡 Fetching Reddit posts and {comments_per_post} comments each...")
        else:
            progress.progress(10, text="📡 Fetching Reddit posts...")
        raw_posts = fetcher.fetch_posts(subreddit, post_limit, comments_per_post=comments_per_post)
        
        # Step 2: Process text (including comments)
        progress.progress(30, text="🧹 Processing text and comments...")
        processed_posts = processor.process_posts(raw_posts)
        
        if not processed_posts:
            st.error("No valid posts found after processing.")
            return None
        
        # Step 3: Analyze sentiment
        progress.progress(50, text="🔍 Analyzing sentiment...")
        analyzed_posts = analyzer.analyze_posts(processed_posts)
        
        # Calculate sentiment distribution
        sentiment_dist = analyzer.calculate_sentiment_distribution(analyzed_posts)
        
        # Step 4: Extract themes (if mode requires)
        progress.progress(70, text="🏷️ Extracting themes...")
        themes = []
        if "themes" in mode:
            themes = analyzer.extract_themes(analyzed_posts)
        
        # Step 5: Get high-impact issues
        progress.progress(85, text="⚡ Identifying high-impact issues...")
        high_impact = analyzer.get_high_impact_issues(analyzed_posts)
        
        # Step 6: Generate report (with AI if enabled)
        if use_ai:
            progress.progress(90, text="🤖 Generating AI insights...")
        else:
            progress.progress(95, text="📝 Generating report...")
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        report = report_gen.generate_full_report(
            subreddit=subreddit,
            analyzed_posts=analyzed_posts,
            themes=themes,
            sentiment_dist=sentiment_dist,
            high_impact=high_impact,
            analysis_mode=mode,
            processing_time_ms=processing_time
        )
        
        progress.progress(100, text="✅ Analysis complete!")
        time.sleep(0.5)
        progress.empty()
        
        return report
        
    except ValueError as e:
        progress.empty()
        st.error(f"❌ {str(e)}")
        return None
    except ConnectionError as e:
        progress.empty()
        st.error(f"🌐 {str(e)}")
        return None
    except Exception as e:
        progress.empty()
        st.error(f"⚠️ Unexpected error: {str(e)}")
        return None


# ============================================================================
# DISPLAY RESULTS
# ============================================================================
def display_report(report):
    """Display the analysis report with premium styling"""
    
    # Metric Grid
    st.markdown(f'''
    <div class="metric-grid">
        <div class="metric-item">
            <div class="metric-label">Subreddit</div>
            <div class="metric-value">r/{report.subreddit}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Posts Analyzed</div>
            <div class="metric-value">{report.posts_analyzed}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Processing Time</div>
            <div class="metric-value">{report.processing_time_ms:.0f}ms</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Analysis Date</div>
            <div class="metric-value">{report.timestamp[:10]}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Executive Summary
    st.markdown('''
    <div class="section-header">
        <div class="section-icon">📝</div>
        <div class="section-title">Executive Summary</div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown(f'''
    <div class="summary-box">
        {report.executive_summary}
    </div>
    ''', unsafe_allow_html=True)
    
    # Sentiment Section
    if report.sentiment_distribution:
        st.markdown('''
        <div class="section-header">
            <div class="section-icon">📊</div>
            <div class="section-title">Sentiment Analysis</div>
        </div>
        ''', unsafe_allow_html=True)
        
        pos = report.sentiment_distribution['positive']
        neu = report.sentiment_distribution['neutral']
        neg = report.sentiment_distribution['negative']
        
        st.markdown(f'''
        <div class="sentiment-bar-container">
            <div class="sentiment-bar-inner">
                <div class="sentiment-segment positive" style="width: {pos}%">{pos:.0f}%</div>
                <div class="sentiment-segment neutral" style="width: {neu}%">{neu:.0f}%</div>
                <div class="sentiment-segment negative" style="width: {neg}%">{neg:.0f}%</div>
            </div>
        </div>
        <div class="sentiment-legend">
            <div class="legend-item"><div class="legend-dot positive"></div>Positive</div>
            <div class="legend-item"><div class="legend-dot neutral"></div>Neutral</div>
            <div class="legend-item"><div class="legend-dot negative"></div>Negative</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Themes Section
    if report.themes:
        st.markdown('''
        <div class="section-header">
            <div class="section-icon">🏷️</div>
            <div class="section-title">Key Themes</div>
        </div>
        ''', unsafe_allow_html=True)
        
        cols = st.columns(2)
        for i, theme in enumerate(report.themes):
            with cols[i % 2]:
                mood_class = theme.get('mood', 'neutral')
                st.markdown(f'''
                <div class="theme-card {mood_class}">
                    <div class="theme-name">{theme['name']}</div>
                    <div class="theme-desc">{theme['explanation']}</div>
                    <div class="theme-count">{theme['count']} mentions</div>
                </div>
                ''', unsafe_allow_html=True)
    
    # Product Insights
    st.markdown('''
    <div class="section-header">
        <div class="section-icon">💡</div>
        <div class="section-title">Product Insights</div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''<div class="insights-grid">''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''<div class="insight-column"><h4>✅ What Users Like</h4></div>''', unsafe_allow_html=True)
        for item in report.product_insights['likes']:
            st.markdown(f'''<div class="insight-item positive">{item}</div>''', unsafe_allow_html=True)
        
        st.markdown('''<div class="insight-column" style="margin-top: 1.5rem;"><h4>📈 Improving</h4></div>''', unsafe_allow_html=True)
        for item in report.product_insights['improving']:
            st.markdown(f'''<div class="insight-item improving">↑ {item}</div>''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''<div class="insight-column"><h4>❌ What Frustrates Users</h4></div>''', unsafe_allow_html=True)
        for item in report.product_insights['frustrations']:
            st.markdown(f'''<div class="insight-item negative">{item}</div>''', unsafe_allow_html=True)
        
        st.markdown('''<div class="insight-column" style="margin-top: 1.5rem;"><h4>📉 Worsening</h4></div>''', unsafe_allow_html=True)
        for item in report.product_insights['worsening']:
            st.markdown(f'''<div class="insight-item worsening">↓ {item}</div>''', unsafe_allow_html=True)
    
    st.markdown('''</div>''', unsafe_allow_html=True)
    
    # High Impact Issues
    if report.high_impact_issues:
        st.markdown('''
        <div class="section-header">
            <div class="section-icon">⚡</div>
            <div class="section-title">High-Impact Issues</div>
        </div>
        ''', unsafe_allow_html=True)
        
        for i, issue in enumerate(report.high_impact_issues, 1):
            st.markdown(f'''
            <div class="impact-card {issue['sentiment']}">
                <div class="impact-title"><strong>#{i}</strong> — {issue['title']}</div>
                <div class="impact-meta">
                    <span>Score: {issue['score']}</span>
                    <span>Comments: {issue['comments']}</span>
                    <span>Impact: {issue['impact_score']}</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    # AI-Generated Action Items
    if report.ai_enhanced and report.action_items:
        st.markdown('''
        <div class="section-header">
            <div class="section-icon">🎯</div>
            <div class="section-title">AI-Generated Action Items</div>
        </div>
        ''', unsafe_allow_html=True)
        
        for i, action in enumerate(report.action_items, 1):
            # Handle both dict and string action items
            action_text = action.get('action', str(action)) if isinstance(action, dict) else str(action)
            st.markdown(f'''
            <div class="action-card">
                <div class="action-number">{i}</div>
                <div class="action-text">{action_text}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # AI Enhanced badge
    if report.ai_enhanced:
        st.markdown('''
        <div class="ai-badge">
            <span>🤖</span>
            <span>Enhanced with AI (Llama 3)</span>
        </div>
        ''', unsafe_allow_html=True)
    
    # Export option
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    gen = ReportGenerator()
    markdown_report = gen.format_markdown_report(report)
    
    st.download_button(
        label="📥 Export Report",
        data=markdown_report,
        file_name=f"reddit_analysis_{report.subreddit}_{report.timestamp[:10]}.md",
        mime="text/markdown"
    )
    



# ============================================================================
# RUN ANALYSIS ON BUTTON CLICK
# ============================================================================
if analyze_button:
    if not subreddit.strip():
        st.error("Please enter a subreddit name")
    else:
        report = run_analysis(
            subreddit=subreddit.strip(),
            post_limit=post_limit,
            comments_per_post=comments_per_post,
            mode=analysis_mode,
            use_ml=use_ml,
            use_ai=use_ai
        )
        
        if report:
            display_report(report)
else:
    # Premium empty state
    st.markdown('''
    <div class="empty-state">
        <h2>Ready to Analyze</h2>
        <p>Configure your analysis settings in the sidebar and click "Analyze Subreddit" to transform Reddit discussions into actionable insights.</p>
    </div>
    ''', unsafe_allow_html=True)

    # Show example
    with st.expander("💡 Example Output"):
        st.markdown("""
        **Subreddit:** r/GooglePixel  
        **Posts Analyzed:** 25  
        
        **Executive Summary:**  
        Analysis of 25 recent posts reveals mixed sentiment (Positive: 32%, Neutral: 28%, Negative: 40%). 
        Key discussion areas include Hardware Quality, Camera & Photography, Customer Support. 
        Notable concerns include hardware defects and warranty issues.
        
        **Themes Detected:**
        - 🔴 Hardware Quality — Predominantly negative (65%). Users expressing frustration.
        - 🟢 Camera & Photography — Mostly positive (72%). Users are satisfied.
        - ⚪ Customer Support — Mixed feedback with 15 mentions.
        """)


# Footer (always visible)
st.markdown('''
<div class="footer">
    <div class="footer-text">
        Built with <span class="footer-accent">♥</span> by <span class="footer-accent">Atharv Tripathi</span>
    </div>
</div>
''', unsafe_allow_html=True)
