"""
DeepSight Pro - Product Intelligence Platform
Version: 8.0.0
"""

import os
import re
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import streamlit as st

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_DAYS_BACK = 30
DEFAULT_MIN_COMMENTS = 5

@dataclass
class Evidence:
    direct_quote: str
    user: str
    source_url: str

@dataclass
class PMAnalysis:
    user_pain: str
    technical_hypothesis: str
    strategic_recommendation: str

@dataclass
class StatusBadge:
    severity: str
    frequency_label: str

@dataclass
class ProductTicket:
    ticket_id: str
    title: str
    category: str
    status_badge: StatusBadge
    pm_analysis: PMAnalysis
    evidence: Evidence

@dataclass
class DifferentiationItem:
    feature: str
    winner: str
    insight: str
    evidence_quote: str

@dataclass
class CompetitorIntelligence:
    active: bool
    differentiation_matrix: List[DifferentiationItem]

@dataclass
class DashboardMeta:
    analysis_period: str
    battleground_detected: str
    compatibility_warning: Optional[str]

@dataclass
class StrategyDashboard:
    dashboard_meta: DashboardMeta
    product_tickets: List[ProductTicket]
    competitor_intelligence: Optional[CompetitorIntelligence]

@dataclass
class CleanPost:
    id: str
    text: str
    url: str
    author: str
    comments: int
    date: str

STRATEGY_PROMPT = '''You are the **Head of Product Strategy** for a major technology firm. You are analyzing raw user feedback to create a **Engineering & Strategy Dashboard**.

**YOUR GOAL:**
Process the provided clusters of user feedback and output a list of distinct, actionable Product Tickets.

**INPUT DATA CONTEXT:**
- You will receive a list of "Clusters" (groups of similar posts).
- Each cluster contains: `cluster_size` (Frequency), `sample_posts` (Evidence).
- `source_brand`: The main brand being analyzed.
- `competitor_brand`: The brand for comparison.

---

### STEP 1: NICHE & CONTEXT REASONING ("The Smart Filter")
Before analyzing, determine the **Shared Battleground**.
- IF Source="Samsung" and Competitor="Apple":
  - Battleground = **"Premium Smartphones & Ecosystem"**.
  - Action: IGNORE complaints about Samsung Fridges or Apple TV unless they relate to the phone ecosystem.
  - Action: COMPARE only overlapping features (Camera, Battery, UI, Support).
- IF no overlap exists (e.g. Education vs Tech), mark as "Niche Mismatch" and skip comparison.

### STEP 2: CLUSTER-TO-TICKET CONVERSION
**CRITICAL INSTRUCTION:** Do not summarize multiple issues into one. If users report "Camera Crash" and "Camera Lens Flare", these are **TWO** separate tickets.

For **EVERY** cluster provided:
1.  **Validate:** Is this a genuine product issue/feature request? (Ignore "Shipping delays" or "Fanboy wars").
2.  **Triangulate Root Cause:**
    - *Symptom:* "FaceID fails in dark."
    - *Inference:* "IR Sensor sensitivity calibration issue."
3.  **Assign Severity:**
    - **P0 (Critical):** Data loss, Security, App Crash, inability to use core feature.
    - **P1 (High):** Major broken feature, significant friction.
    - **P2 (Medium):** UI annoyance, minor bug.
4.  **Extract Evidence:** You **MUST** find the exact quote and URL from the provided samples.

---

### STEP 3: OUTPUT JSON SCHEMA

Output ONLY valid JSON. No markdown, no explanation.

{
  "dashboard_meta": {
    "analysis_period": "Last 30 Days",
    "battleground_detected": "String (e.g. 'Flagship Mobile Computing')",
    "compatibility_warning": "String (or null)"
  },
  "product_tickets": [
    {
      "ticket_id": "TKT-101",
      "title": "String (Engineering style, e.g., 'Optimization failure in 120Hz Refresh Rate')",
      "category": "Bug | Feature Gap | UX Debt | Performance",
      "status_badge": {
        "severity": "P0 | P1 | P2",
        "frequency_label": "String (e.g. '🔥 Impacting 140+ Users')"
      },
      "pm_analysis": {
        "user_pain": "String (The human experience)",
        "technical_hypothesis": "String (The engineering reason)",
        "strategic_recommendation": "String (Specific action, e.g. 'Rollback Driver v2.4')"
      },
      "evidence": {
        "direct_quote": "String (Verbatim text)",
        "user": "String",
        "source_url": "String (Must match input URL exactly)"
      }
    }
  ],
  "competitor_intelligence": {
    "active": true,
    "differentiation_matrix": [
      {
        "feature": "String (e.g. 'Battery Management')",
        "winner": "Source | Competitor",
        "insight": "String (e.g. 'Competitor users praise standby time; Source users report 10% drain overnight.')",
        "evidence_quote": "String"
      }
    ]
  }
}
'''

class HighSignalFetcher:
    PULLPUSH_URL = "https://api.pullpush.io/reddit/search/submission/"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def fetch_high_signal_posts(self, subreddit: str, days_back: int = DEFAULT_DAYS_BACK,
                                 min_comments: int = DEFAULT_MIN_COMMENTS, max_posts: int = 50,
                                 progress_callback=None) -> List[CleanPost]:
        try:
            if progress_callback:
                progress_callback(f"📡 Connecting to PullPush for r/{subreddit}...")
            
            params = {"subreddit": subreddit, "sort": "desc", "sort_type": "score", "size": 500}
            response = self.session.get(self.PULLPUSH_URL, params=params, timeout=15)
            
            if response.status_code != 200:
                if progress_callback:
                    progress_callback(f"❌ API Error: {response.status_code}")
                return []
            
            raw_data = response.json().get('data', [])
            if not raw_data:
                if progress_callback:
                    progress_callback("❌ No data from API")
                return []
            
            if progress_callback:
                progress_callback(f"📥 Downloaded {len(raw_data)} posts")
            
            clean_data = []
            for post in raw_data:
                comment_count = post.get('num_comments', 0)
                if comment_count < min_comments:
                    continue
                
                body = post.get('selftext', '') or ''
                title = post.get('title', '') or ''
                
                if body in ["[removed]", "[deleted]"]:
                    continue
                if (len(body) + len(title)) < 15:
                    continue
                
                post_id = post.get('id', '')
                full_link = post.get('full_link') or f"https://reddit.com/r/{subreddit}/comments/{post_id}"
                
                clean_data.append(CleanPost(
                    id=post_id,
                    text=f"TITLE: {title}\nBODY: {body}",
                    url=full_link,
                    author=f"u/{post.get('author', 'unknown')}",
                    comments=comment_count,
                    date=datetime.utcfromtimestamp(post.get('created_utc', 0)).strftime('%Y-%m-%d')
                ))
                
                if len(clean_data) >= max_posts:
                    break
            
            if progress_callback:
                if clean_data:
                    progress_callback(f"🛡️ Filtered to {len(clean_data)} High-Signal Posts")
                else:
                    progress_callback(f"⚠️ Found {len(raw_data)} posts, but 0 passed filters")
            
            return clean_data
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error: {e}")
            return []

class StrategyIntelligenceEngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if GROQ_AVAILABLE and self.api_key else None
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def prepare_clusters(self, posts: List[CleanPost]) -> List[Dict]:
        clusters = []
        for i, post in enumerate(posts):
            clusters.append({
                "cluster_id": i + 1,
                "cluster_size": post.comments,
                "sample_posts": [{"text": post.text, "user": post.author, "url": post.url, "date": post.date, "comments": post.comments}]
            })
        return clusters
    
    def analyze(self, source_posts: List[CleanPost], source_name: str, 
                competitor_posts: List[CleanPost] = None, competitor_name: str = None,
                days_back: int = DEFAULT_DAYS_BACK, progress_callback=None) -> Optional[StrategyDashboard]:
        
        if not self.is_available():
            return None
        
        if progress_callback:
            progress_callback("🧠 Building strategy payload...")
        
        source_clusters = self.prepare_clusters(source_posts)
        input_payload = {
            "source_brand": source_name,
            "competitor_brand": competitor_name,
            "analysis_period": f"Last {days_back} Days",
            "clusters": source_clusters
        }
        
        if competitor_posts:
            input_payload["competitor_clusters"] = self.prepare_clusters(competitor_posts)
        
        if progress_callback:
            progress_callback(f"📦 Sending {len(source_clusters)} clusters to Strategy AI...")
        
        try:
            completion = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": STRATEGY_PROMPT},
                    {"role": "user", "content": json.dumps(input_payload)}
                ],
                temperature=0.2,
                max_tokens=8000
            )
            return self._parse(completion.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ AI error: {e}")
            return None
    
    def _parse(self, response: str) -> Optional[StrategyDashboard]:
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if not match:
                return None
            data = json.loads(match.group())
            
            dm = data.get('dashboard_meta', {})
            meta = DashboardMeta(
                analysis_period=dm.get('analysis_period', 'Last 30 Days'),
                battleground_detected=dm.get('battleground_detected', ''),
                compatibility_warning=dm.get('compatibility_warning')
            )
            
            tickets = []
            for item in data.get('product_tickets', []):
                sb = item.get('status_badge', {})
                pm = item.get('pm_analysis', {})
                ev = item.get('evidence', {})
                
                tickets.append(ProductTicket(
                    ticket_id=item.get('ticket_id', ''),
                    title=item.get('title', ''),
                    category=item.get('category', 'Bug'),
                    status_badge=StatusBadge(severity=sb.get('severity', 'P2'), frequency_label=sb.get('frequency_label', '')),
                    pm_analysis=PMAnalysis(user_pain=pm.get('user_pain', ''), technical_hypothesis=pm.get('technical_hypothesis', ''), strategic_recommendation=pm.get('strategic_recommendation', '')),
                    evidence=Evidence(direct_quote=ev.get('direct_quote', ''), user=ev.get('user', 'Anonymous'), source_url=ev.get('source_url', ''))
                ))
            
            comp = None
            ci = data.get('competitor_intelligence', {})
            if ci.get('active'):
                matrix = [DifferentiationItem(feature=item.get('feature', ''), winner=item.get('winner', ''), insight=item.get('insight', ''), evidence_quote=item.get('evidence_quote', '')) for item in ci.get('differentiation_matrix', [])]
                comp = CompetitorIntelligence(active=True, differentiation_matrix=matrix)
            
            return StrategyDashboard(dashboard_meta=meta, product_tickets=tickets, competitor_intelligence=comp)
        except Exception as e:
            st.error(f"Parse error: {e}")
            return None

def setup_page():
    st.set_page_config(page_title="DeepSight Pro | Strategy Dashboard", page_icon="🔬", layout="wide")

def get_api_key():
    try:
        return st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    except:
        return os.environ.get("GROQ_API_KEY")

def get_severity_display(severity: str):
    s = severity.upper()
    if "P0" in s:
        return "🔴 P0 (Critical)", "error"
    elif "P1" in s:
        return "🟠 P1 (High)", "warning"
    return "🟡 P2 (Medium)", "info"

def render_sidebar():
    with st.sidebar:
        st.title("🔬 DeepSight Pro")
        st.caption("v8.0 • Strategy Dashboard")
        st.divider()
        
        if get_api_key():
            st.success("🔑 API Active")
        else:
            st.error("🔑 API Missing")
        
        st.divider()
        
        st.subheader("🎯 Primary Brand")
        source = st.text_input("Subreddit", value="iphone", label_visibility="collapsed")
        
        st.subheader("⚔️ Competitor")
        competitor = st.text_input("Competitor", value="", placeholder="e.g., samsung", label_visibility="collapsed")
        enable_comp = st.checkbox("Enable Comparison", value=bool(competitor))
        
        st.subheader("⏰ Time Range")
        days_back = st.slider("Days to analyze", 7, 90, 30, 7)
        
        st.subheader("💬 Engagement Filter")
        min_comments = st.slider("Min comments", 1, 20, 5, 1)
        
        st.subheader("📊 Depth")
        max_posts = st.slider("Max posts", 10, 100, 30, 10)
        
        return source, competitor if enable_comp else "", days_back, min_comments, max_posts

def render_header():
    st.title("🔬 DeepSight Pro")
    st.caption("Strategy Dashboard • High-Signal Analysis • Differentiation Matrix")
    st.divider()

def render_metrics(dashboard: StrategyDashboard, post_count: int):
    tickets = dashboard.product_tickets
    p0 = sum(1 for t in tickets if 'P0' in t.status_badge.severity.upper())
    p1 = sum(1 for t in tickets if 'P1' in t.status_badge.severity.upper())
    bugs = sum(1 for t in tickets if 'bug' in t.category.lower())
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Posts Analyzed", post_count)
    with col2:
        st.metric("🎫 Tickets Generated", len(tickets))
    with col3:
        st.metric("🚨 P0/P1 Issues", f"{p0 + p1}")
    with col4:
        st.metric("🐛 Bugs", bugs)

def render_ticket(ticket: ProductTicket):
    severity_text, severity_type = get_severity_display(ticket.status_badge.severity)
    
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"🎫 {ticket.ticket_id}: {ticket.title}")
        with col2:
            st.caption(f"**{ticket.category}**")
        
        col1, col2 = st.columns(2)
        with col1:
            if severity_type == "error":
                st.error(severity_text)
            elif severity_type == "warning":
                st.warning(severity_text)
            else:
                st.info(severity_text)
        with col2:
            st.info(ticket.status_badge.frequency_label)
        
        st.divider()
        
        st.markdown("**📋 User Pain**")
        st.info(ticket.pm_analysis.user_pain)
        
        st.markdown("**🔍 Technical Hypothesis**")
        st.warning(ticket.pm_analysis.technical_hypothesis)
        
        st.markdown("**✅ Strategic Recommendation**")
        st.success(ticket.pm_analysis.strategic_recommendation)
        
        with st.expander("📎 Evidence"):
            st.markdown(f"> *\"{ticket.evidence.direct_quote}\"*")
            st.markdown(f"— **{ticket.evidence.user}**")
            if ticket.evidence.source_url:
                st.markdown(f"[🔗 View Original Post]({ticket.evidence.source_url})")

def render_differentiation_matrix(comp: CompetitorIntelligence):
    st.subheader("⚔️ Differentiation Matrix")
    
    for item in comp.differentiation_matrix:
        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 1, 3])
            
            with col1:
                st.markdown(f"**{item.feature}**")
            with col2:
                if item.winner.lower() == "source":
                    st.success("✅ You Win")
                else:
                    st.error("❌ They Win")
            with col3:
                st.caption(item.insight)
            
            if item.evidence_quote:
                st.markdown(f"> *\"{item.evidence_quote}\"*")

def render_results(dashboard: StrategyDashboard, post_count: int):
    render_metrics(dashboard, post_count)
    st.divider()
    
    meta = dashboard.dashboard_meta
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📅 **Period:** {meta.analysis_period}")
    with col2:
        if meta.battleground_detected:
            st.success(f"🎯 **Battleground:** {meta.battleground_detected}")
    
    if meta.compatibility_warning:
        st.warning(f"⚠️ {meta.compatibility_warning}")
    
    st.divider()
    
    if dashboard.competitor_intelligence and dashboard.competitor_intelligence.active:
        render_differentiation_matrix(dashboard.competitor_intelligence)
        st.divider()
    
    tickets = dashboard.product_tickets
    
    p0_tickets = [t for t in tickets if 'P0' in t.status_badge.severity.upper()]
    if p0_tickets:
        st.subheader(f"🔴 P0 - Critical ({len(p0_tickets)})")
        for ticket in p0_tickets:
            render_ticket(ticket)
    
    p1_tickets = [t for t in tickets if 'P1' in t.status_badge.severity.upper()]
    if p1_tickets:
        st.subheader(f"🟠 P1 - High ({len(p1_tickets)})")
        for ticket in p1_tickets:
            render_ticket(ticket)
    
    p2_tickets = [t for t in tickets if 'P2' in t.status_badge.severity.upper()]
    if p2_tickets:
        with st.expander(f"🟡 P2 - Medium ({len(p2_tickets)})"):
            for ticket in p2_tickets:
                render_ticket(ticket)

def main():
    setup_page()
    source, competitor, days_back, min_comments, max_posts = render_sidebar()
    api_key = get_api_key()
    
    render_header()
    
    if not api_key:
        st.warning("⚠️ Add GROQ_API_KEY to `.streamlit/secrets.toml`")
        st.code('GROQ_API_KEY = "your-api-key-here"', language="toml")
        st.stop()
    
    fetcher = HighSignalFetcher()
    engine = StrategyIntelligenceEngine(api_key)
    
    btn_text = f"🚀 Analyze r/{source}" + (f" vs r/{competitor}" if competitor else "")
    
    if st.button(btn_text, type="primary", use_container_width=True):
        with st.status("🔍 Running Strategy Analysis...", expanded=True) as status:
            def log(m): st.write(m)
            
            source_posts = fetcher.fetch_high_signal_posts(source, days_back, min_comments, max_posts, log)
            if not source_posts:
                status.update(label="❌ No high-signal data found", state="error")
                st.stop()
            
            comp_posts = None
            if competitor:
                log(f"📡 Scanning r/{competitor}...")
                comp_posts = fetcher.fetch_high_signal_posts(competitor, days_back, min_comments, max_posts, log)
            
            dashboard = engine.analyze(source_posts, source, comp_posts, competitor, days_back, log)
            
            if not dashboard:
                status.update(label="❌ Analysis failed", state="error")
                st.stop()
            
            st.session_state['dashboard'] = dashboard
            st.session_state['post_count'] = len(source_posts)
            
            status.update(label="✅ Strategy Analysis Complete!", state="complete")
        
        st.balloons()
    
    if 'dashboard' in st.session_state:
        render_results(st.session_state['dashboard'], st.session_state['post_count'])

if __name__ == "__main__":
    main()
