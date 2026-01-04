"""
Contextual Sentiment Analyzer
LLM-powered "Batch & Judge" classification using Groq API.
Understands sarcasm, nuance, and context - not just keywords.
"""

import os
import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# Fallback to VADER if Groq unavailable
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ContextualResult:
    """Result from contextual analysis of a single post"""
    id: str
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str    # Positive, Negative, Neutral
    category: str           # Bug, Feature Request, Praise, Question, News, Junk
    feature: str            # Detected subject/feature (Battery, Login, etc.)
    confidence: float = 0.8


# =============================================================================
# SYSTEM PROMPT FOR "THE JUDGE"
# =============================================================================

CONTEXTUAL_SYSTEM_PROMPT = """You are a Contextual Sentiment Analyst for product feedback.
I will give you a list of Reddit posts about a product/company.
Classify each post based on its MEANING, not just keywords.

CRITICAL RULES:
1. DETECT SARCASM: "Great job breaking it again" = NEGATIVE, not positive
2. DETECT QUESTIONS: "Is it worth buying?" = Category "Question", ignore for product insights
3. DETECT NEWS: "Apple announces..." or "Reportedly..." = Category "News", filter out
4. UNDERSTAND CONTEXT: "Battery is insane!" in positive context = POSITIVE
5. IGNORE NEUTRAL CHATTER: Memes, jokes, off-topic = Category "Junk"

CATEGORIES:
- "Bug" = User reporting a problem, crash, error, malfunction
- "Feature Request" = User wishing for new functionality
- "Praise" = Genuine positive feedback about product/feature
- "Question" = User asking for help or opinions (filter these out)  
- "News" = Announcements, rumors, links to articles (filter these out)
- "Junk" = Memes, jokes, unrelated discussion (filter these out)

For each post, determine:
- sentiment_score: Float from -1.0 (very negative) to 1.0 (very positive). Use 0.0 for neutral.
- category: One of the categories above
- feature: What specific feature/aspect is discussed? (e.g., "Battery", "Camera", "Login", "Update", "Display", "Performance")

Return ONLY a valid JSON array, no other text:
[
  {"id": "...", "sentiment_score": 0.0, "category": "...", "feature": "..."},
  ...
]"""


# =============================================================================
# CONTEXTUAL ANALYZER CLASS
# =============================================================================

class ContextualAnalyzer:
    """
    LLM-powered sentiment analyzer that understands context.
    Uses Groq API with Llama-3 for intelligent classification.
    Falls back to VADER if Groq unavailable.
    """
    
    def __init__(self, batch_size: int = 10, model: str = "llama-3.1-8b-instant"):
        """
        Initialize the contextual analyzer.
        
        Args:
            batch_size: Number of posts to send per API call (max 10 recommended)
            model: Groq model to use. Options:
                   - "llama-3.1-8b-instant" (faster, cheaper)
                   - "llama-3.1-70b-versatile" (more accurate)
        """
        self.batch_size = batch_size
        self.model = model
        self.api_key = os.environ.get("GROQ_API_KEY")
        self._vader = None
        
        if VADER_AVAILABLE:
            self._vader = SentimentIntensityAnalyzer()
    
    def _get_groq_client(self) -> Optional[Groq]:
        """Get Groq client if API key available"""
        if not self.api_key or not GROQ_AVAILABLE:
            return None
        return Groq(api_key=self.api_key)
    
    def _build_batch_prompt(self, posts: List[Dict]) -> str:
        """Build the user prompt with post data"""
        lines = []
        for p in posts:
            post_id = p.get('id', 'unknown')
            title = p.get('title', '')[:100]
            body = p.get('body', '')[:200]
            content = f"{title} - {body}".strip() if body else title
            lines.append(f"ID {post_id}: {content}")
        
        return "Analyze these posts:\n\n" + "\n".join(lines)
    
    def analyze_batch(self, posts: List[Dict]) -> List[Dict]:
        """
        Analyze a batch of posts using Groq LLM.
        
        Args:
            posts: List of dicts with 'id', 'title', 'body' keys
            
        Returns:
            List of result dicts with 'id', 'sentiment_score', 'category', 'feature'
        """
        client = self._get_groq_client()
        
        if not client:
            # Fallback to VADER
            return self._analyze_batch_vader(posts)
        
        try:
            user_prompt = self._build_batch_prompt(posts)
            
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CONTEXTUAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temp for consistent analysis
                max_tokens=2000
            )
            
            response_text = completion.choices[0].message.content
            
            # Parse JSON from response
            results = self._parse_json_response(response_text, posts)
            return results
            
        except Exception as e:
            print(f"⚠️ Groq API error: {e}. Falling back to VADER.")
            return self._analyze_batch_vader(posts)
    
    def _parse_json_response(self, response: str, original_posts: List[Dict]) -> List[Dict]:
        """Parse LLM JSON response, with fallback handling"""
        try:
            # Try to extract JSON array from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                results = json.loads(json_match.group())
                
                # Validate and normalize results
                normalized = []
                for r in results:
                    normalized.append({
                        'id': str(r.get('id', '')),
                        'sentiment_score': float(r.get('sentiment_score', 0.0)),
                        'category': r.get('category', 'Unknown'),
                        'feature': r.get('feature', 'General')
                    })
                return normalized
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️ JSON parse error: {e}")
        
        # Fallback: return basic results for all posts
        return self._analyze_batch_vader(original_posts)
    
    def _analyze_batch_vader(self, posts: List[Dict]) -> List[Dict]:
        """Fallback: Analyze using VADER (keyword-based)"""
        results = []
        
        for p in posts:
            post_id = p.get('id', 'unknown')
            text = f"{p.get('title', '')} {p.get('body', '')}"
            
            # VADER analysis
            if self._vader:
                scores = self._vader.polarity_scores(text)
                sentiment = scores['compound']
            else:
                sentiment = 0.0
            
            # Basic category detection (keyword-based fallback)
            category = self._detect_category_keywords(text)
            feature = self._detect_feature_keywords(text)
            
            results.append({
                'id': post_id,
                'sentiment_score': sentiment,
                'category': category,
                'feature': feature
            })
        
        return results
    
    def _detect_category_keywords(self, text: str) -> str:
        """Fallback keyword-based category detection"""
        text_lower = text.lower()
        
        # Question detection
        if '?' in text and any(w in text_lower for w in ['how', 'what', 'why', 'is it', 'should i', 'worth']):
            return 'Question'
        
        # News detection
        if any(w in text_lower for w in ['announces', 'reportedly', 'rumor', 'leaked', 'official']):
            return 'News'
        
        # Bug detection
        if any(w in text_lower for w in ['crash', 'bug', 'broken', 'error', 'fail', 'not working', 'issue']):
            return 'Bug'
        
        # Feature request
        if any(w in text_lower for w in ['wish', 'would be nice', 'feature request', 'should add', 'need']):
            return 'Feature Request'
        
        # Praise detection
        if any(w in text_lower for w in ['love', 'amazing', 'great', 'awesome', 'best', 'perfect']):
            return 'Praise'
        
        return 'Unknown'
    
    def _detect_feature_keywords(self, text: str) -> str:
        """Fallback keyword-based feature detection"""
        text_lower = text.lower()
        
        features = {
            'Battery': ['battery', 'charge', 'charging', 'drain', 'power', 'dead'],
            'Camera': ['camera', 'photo', 'picture', 'lens', 'zoom', 'video', 'selfie'],
            'Display': ['screen', 'display', 'oled', 'brightness', 'pixel', 'resolution'],
            'Performance': ['slow', 'lag', 'fast', 'performance', 'speed', 'ram', 'memory'],
            'Software': ['update', 'software', 'app', 'os', 'ios', 'android', 'version'],
            'Audio': ['speaker', 'sound', 'audio', 'volume', 'music', 'call quality'],
            'Connectivity': ['wifi', 'bluetooth', 'signal', '5g', 'network', 'connection'],
            'Login': ['login', 'sign in', 'account', 'password', 'authentication', '2fa'],
        }
        
        for feature, keywords in features.items():
            if any(kw in text_lower for kw in keywords):
                return feature
        
        return 'General'


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_posts_contextually(
    posts: List[Dict], 
    batch_size: int = 10,
    filter_noise: bool = True
) -> tuple[List[Dict], List[Dict]]:
    """
    Analyze posts using contextual LLM and optionally filter noise.
    
    Args:
        posts: List of post dicts with 'id', 'title', 'body'
        batch_size: Posts per API call
        filter_noise: If True, separate actionable vs noise posts
        
    Returns:
        Tuple of (actionable_posts, filtered_noise_posts) if filter_noise=True
        Otherwise (all_analyzed_posts, [])
    """
    analyzer = ContextualAnalyzer(batch_size=batch_size)
    all_results = []
    
    # Process in batches
    for i in range(0, len(posts), batch_size):
        batch = posts[i:i + batch_size]
        results = analyzer.analyze_batch(batch)
        all_results.extend(results)
    
    # Merge results back into posts
    results_map = {r['id']: r for r in all_results}
    
    for post in posts:
        if post['id'] in results_map:
            result = results_map[post['id']]
            post['sentiment_score'] = result['sentiment_score']
            post['category'] = result['category']
            post['feature'] = result['feature']
            
            # Set sentiment label
            score = result['sentiment_score']
            if score >= 0.15:
                post['sentiment_label'] = 'Positive'
            elif score <= -0.15:
                post['sentiment_label'] = 'Negative'
            else:
                post['sentiment_label'] = 'Neutral'
    
    if filter_noise:
        # Separate actionable feedback from noise
        noise_categories = {'Question', 'News', 'Junk'}
        actionable = [p for p in posts if p.get('category') not in noise_categories]
        noise = [p for p in posts if p.get('category') in noise_categories]
        return actionable, noise
    
    return posts, []


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    # Test cases including sarcasm
    test_posts = [
        {"id": "1", "title": "Great job breaking the volume button updates!", "body": "Every single update breaks something."},
        {"id": "2", "title": "Is the iPhone 15 worth it?", "body": "Thinking of upgrading from 12"},
        {"id": "3", "title": "The battery lasts literally forever", "body": "Impressed with the new chip efficiency"},
        {"id": "4", "title": "Apple announces new MacBook Pro", "body": "Official release coming next month"},
        {"id": "5", "title": "Login keeps failing after update", "body": "2FA not working, can't access my account"},
    ]
    
    print("🧪 Testing Contextual Analyzer...\n")
    
    actionable, noise = analyze_posts_contextually(test_posts, batch_size=5)
    
    print("✅ ACTIONABLE FEEDBACK:")
    for p in actionable:
        print(f"  [{p.get('category', '?'):15}] [{p.get('feature', '?'):12}] {p.get('sentiment_score', 0):+.2f} | {p['title'][:50]}")
    
    print("\n🗑️ FILTERED NOISE:")
    for p in noise:
        print(f"  [{p.get('category', '?'):15}] {p['title'][:50]}")
