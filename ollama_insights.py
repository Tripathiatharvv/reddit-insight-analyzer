"""
AI Integration Module
Provides deep contextual AI insights using local LLM inference (Ollama)
or Cloud API (Groq) for deployment.
"""

import requests
import json
import os
import streamlit as st
from typing import List, Dict, Optional

# Try to import Groq (ignore if not installed in local env without it)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class AIInsights:
    """
    Generate deep AI insights using either Local Ollama or Groq Cloud API.
    Provider is selected based on configuration and API key availability.
    """
    
    # Provider Constants
    PROVIDER_OLLAMA = "ollama"
    PROVIDER_GROQ = "groq"
    
    # Configuration
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3.2"
    GROQ_MODEL = "llama-3.1-8b-instant"  # Current model on Groq (Dec 2024)
    
    def __init__(self):
        self.provider = self._determine_provider()
        self.groq_client = None
        
        if self.provider == self.PROVIDER_GROQ:
            try:
                # Get API key - env var first, then Streamlit secrets
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    try:
                        api_key = st.secrets["GROQ_API_KEY"]
                    except (KeyError, FileNotFoundError):
                        pass
                
                if api_key:
                    self.groq_client = Groq(api_key=api_key)
                    print(f"✅ Groq client initialized successfully")
                else:
                    print("⚠️ GROQ_API_KEY not found")
                    self.provider = self.PROVIDER_OLLAMA
            except Exception as e:
                print(f"⚠️ Failed to initialize Groq client: {e}")
                self.provider = self.PROVIDER_OLLAMA
                
        # Check local availability if using Ollama
        self.local_available = self._check_local_availability()
        
        print(f"🤖 AI Provider initialized: {self.provider.upper()}")
    
    def _determine_provider(self) -> str:
        """Determine which AI provider to use"""
        if not GROQ_AVAILABLE:
            return self.PROVIDER_OLLAMA
            
        # Check environment variable first
        if os.environ.get("GROQ_API_KEY"):
            return self.PROVIDER_GROQ
        
        # Check Streamlit secrets
        try:
            if "GROQ_API_KEY" in st.secrets:
                return self.PROVIDER_GROQ
        except (FileNotFoundError, AttributeError, KeyError):
            pass
            
        return self.PROVIDER_OLLAMA
    
    def _check_local_availability(self) -> bool:
        """Check if local Ollama is running"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using the active provider"""
        
        # 1. Try Groq (Cloud)
        if self.provider == self.PROVIDER_GROQ and self.groq_client:
            try:
                completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful product analyst AI."},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.GROQ_MODEL,
                    temperature=0.7,
                    max_tokens=max_tokens,
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"❌ Groq API Error: {e}")
                # Don't fallback silently to avoid confusion, just return empty
                return ""
        
        # 2. Try Ollama (Local)
        if self.local_available:
            try:
                response = requests.post(
                    self.OLLAMA_URL,
                    json={
                        "model": self.OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": max_tokens
                        }
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    return response.json().get("response", "")
            except Exception as e:
                print(f"❌ Ollama Error: {e}")
                
        return ""
    
    def generate_executive_summary(
        self, 
        subreddit: str, 
        posts: List[Dict],
        sentiment_dist: Dict[str, float]
    ) -> str:
        """Generate a deep, contextual executive summary with comment analysis"""
        
        # Prepare detailed post data including comments
        post_details = []
        for i, post in enumerate(posts[:12], 1):  # Top 12 posts
            sentiment = post.get('sentiment', {})
            label = sentiment.get('label', 'unknown') if isinstance(sentiment, dict) else getattr(sentiment, 'label', 'unknown')
            
            title = post.get('title', '')[:100]
            body = post.get('body', '')[:150] if post.get('body') else ''
            comments = post.get('comments', '')[:200] if post.get('comments') else ''
            
            detail = f"{i}. [{label.upper()}] {title}"
            if body:
                detail += f"\n   Body: {body}"
            if comments:
                detail += f"\n   User comments: {comments}"
            post_details.append(detail)
        
        posts_text = "\n".join(post_details)
        
        prompt = f"""You are a senior product analyst reviewing Reddit discussions from r/{subreddit}.
        
POSTS WITH USER DISCUSSIONS:
{posts_text}

SENTIMENT BREAKDOWN:
- Positive: {sentiment_dist.get('positive', 0):.1f}%
- Neutral: {sentiment_dist.get('neutral', 0):.1f}%
- Negative: {sentiment_dist.get('negative', 0):.1f}%

Based on the posts AND the user comments, write an insightful executive summary (4-5 sentences) that:
1. Identifies the SPECIFIC issues/topics users are actually discussing (not generic themes)
2. Explains WHY users feel the way they do based on their comments
3. Highlights the most critical pain points that need attention
4. Notes any positive aspects users appreciate
5. Provides ONE specific, actionable recommendation

Be very specific - cite actual topics from the posts. Write like a product manager reporting to executives."""

        return self._generate(prompt, max_tokens=400)
    
    def generate_deep_insights(
        self, 
        posts: List[Dict],
        themes: List[Dict]
    ) -> Dict[str, List[str]]:
        """Generate deep product insights from posts"""
        
        # Prepare data
        negative_posts = [p for p in posts if getattr(p.get('sentiment', {}), 'label', '') == 'negative' or 
                         (isinstance(p.get('sentiment'), dict) and p.get('sentiment', {}).get('label') == 'negative')]
        positive_posts = [p for p in posts if getattr(p.get('sentiment', {}), 'label', '') == 'positive' or 
                         (isinstance(p.get('sentiment'), dict) and p.get('sentiment', {}).get('label') == 'positive')]
        
        neg_titles = [p.get('title', '')[:80] for p in negative_posts[:8]]
        pos_titles = [p.get('title', '')[:80] for p in positive_posts[:8]]
        
        theme_names = [t.get('name', '') for t in themes[:5]]
        
        prompt = f"""Analyze these Reddit discussions for product insights:

NEGATIVE/FRUSTRATED POSTS:
{chr(10).join(f"- {t}" for t in neg_titles) if neg_titles else "- None"}

POSITIVE/SATISFIED POSTS:
{chr(10).join(f"- {t}" for t in pos_titles) if pos_titles else "- None"}

TOP THEMES: {', '.join(theme_names)}

Provide analysis in this exact JSON format:
{{
  "likes": ["specific thing users like 1", "specific thing users like 2", "specific thing users like 3"],
  "frustrations": ["specific frustration 1", "specific frustration 2", "specific frustration 3"],
  "improving": ["thing getting better based on posts"],
  "worsening": ["thing getting worse based on posts"],
  "opportunities": ["product improvement opportunity 1", "product improvement opportunity 2"]
}}

Be specific - reference actual topics from the posts. Return ONLY valid JSON."""

        response = self._generate(prompt, max_tokens=400)
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Return empty structure if parsing fails
        return {
            "likes": [],
            "frustrations": [],
            "improving": [],
            "worsening": [],
            "opportunities": []
        }
    
    def analyze_post_context(self, title: str, body: str) -> Dict:
        """Deep analysis of a single post for context"""
        
        prompt = f"""Analyze this Reddit post:

Title: {title}
Content: {body[:500]}

Provide a brief analysis:
1. Main topic/concern (1 sentence)
2. User emotion (frustrated, happy, confused, seeking help, etc.)
3. Actionability (can product team act on this? yes/no + why)

Keep response under 100 words."""

        response = self._generate(prompt, max_tokens=150)
        
        return {
            "analysis": response,
            "has_ai_insight": bool(response)
        }
    
    def generate_action_items(
        self, 
        high_impact_issues: List[Dict],
        themes: List[Dict]
    ) -> List[Dict]:
        """Generate specific action items based on issues"""
        
        issues_text = "\n".join([f"- {i.get('title', '')} (Impact: {i.get('impact_score', 0):.1f})" for i in high_impact_issues[:5]])
        themes_text = ", ".join([t.get('name', '') for t in themes[:3]])
        
        prompt = f"""Based on these high-impact user issues and themes from Reddit:

ISSUES:
{issues_text}

THEMES: {themes_text}

Generate 3-5 specific, actionable tasks for the product team.
For each task, provide:
1. Action: The specific thing to do (start with verb)
2. Priority: High/Medium/Low
3. Team: Engineering/Design/Product/Marketing

Provide response in this JSON format:
{{
    "items": [
        {{"action": "Fix camera crashing bug on startup", "priority": "High", "team": "Engineering"}},
        {{"action": "Update return policy documentation", "priority": "Medium", "team": "cx"}}
    ]
}}
Return ONLY valid JSON."""

        response = self._generate(prompt, max_tokens=300)
        
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('items', [])
        except:
            pass
            
        return []
