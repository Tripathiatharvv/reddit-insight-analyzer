"""
NLP Analyzer Module
Handles sentiment analysis and summarization using Hugging Face transformers
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter


@dataclass
class SentimentResult:
    """Sentiment analysis result for a single post"""
    label: str  # 'positive', 'neutral', 'negative'
    score: float  # Confidence score
    raw_score: float  # Raw model score


@dataclass
class AnalyzedPost:
    """Post with sentiment analysis and comments"""
    id: str
    title: str
    body: str
    cleaned_comments: str  # Comments for context
    score: int
    num_comments: int
    sentiment: SentimentResult
    impact_score: float


class NLPAnalyzer:
    """
    NLP analysis using Hugging Face transformers.
    Uses lightweight models optimized for CPU inference.
    """
    
    # Model names - Using smaller, faster models
    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # 265MB
    SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"  # 305MB - much faster than BART-large (1.5GB)
    
    # Theme keywords for extraction
    THEME_KEYWORDS = {
        'Battery & Power': [
            'battery', 'charge', 'charging', 'drain', 'power', 'sot',
            'fast charging', 'wireless charging', 'dead', 'percentage'
        ],
        'Camera & Photography': [
            'camera', 'photo', 'picture', 'lens', 'zoom', 'portrait',
            'night sight', 'video', 'recording', 'selfie', 'megapixel'
        ],
        'Software & Updates': [
            'update', 'android', 'software', 'bug', 'patch', 'fix',
            'feature', 'app', 'version', 'beta', 'stable'
        ],
        'Hardware Quality': [
            'screen', 'display', 'build', 'quality', 'glass', 'speaker',
            'fingerprint', 'crack', 'scratch', 'durable', 'design'
        ],
        'Performance': [
            'performance', 'fast', 'slow', 'lag', 'smooth', 'heat',
            'overheating', 'hot', 'throttle', 'ram', 'memory', 'speed'
        ],
        'Customer Support': [
            'support', 'warranty', 'rma', 'repair', 'replacement',
            'refund', 'return', 'google', 'response', 'shipping'
        ],
        'Connectivity': [
            'wifi', 'bluetooth', 'signal', 'network', '5g', 'lte',
            'connection', 'drop', 'esim', 'carrier', 'cellular'
        ]
    }
    
    # Sentiment lexicon for lightweight analysis (no ML needed)
    SENTIMENT_LEXICON = {
        # Very positive
        'amazing': 5, 'excellent': 5, 'perfect': 5, 'love': 4, 'great': 4,
        'awesome': 4, 'fantastic': 5, 'wonderful': 4, 'best': 4, 'impressed': 4,
        
        # Positive
        'good': 3, 'nice': 3, 'happy': 3, 'helpful': 3, 'smooth': 3,
        'fast': 3, 'reliable': 3, 'solid': 3, 'better': 3, 'improved': 3,
        
        # Slightly positive
        'okay': 1, 'fine': 1, 'decent': 2, 'works': 2, 'useful': 2,
        
        # Slightly negative
        'issue': -1, 'concern': -1, 'confusing': -1, 'mediocre': -1,
        
        # Negative
        'bad': -2, 'poor': -2, 'problem': -2, 'annoying': -2, 'disappointed': -3,
        'frustrating': -2, 'slow': -2, 'laggy': -2, 'buggy': -2, 'broken': -2,
        'crash': -2, 'error': -2, 'fail': -2, 'failed': -2, 'freezing': -2,
        
        # Very negative
        'terrible': -4, 'awful': -4, 'horrible': -4, 'worst': -4, 'hate': -3,
        'useless': -3, 'defective': -3, 'garbage': -3, 'waste': -3, 'scam': -4,
    }
    
    def __init__(self, use_ml_models: bool = False):
        """
        Initialize analyzer.
        
        Args:
            use_ml_models: If True, use TextBlob + VADER for ML-enhanced analysis
                          If False, use lexicon-based analysis (faster)
        """
        self.use_ml_models = use_ml_models
        self._vader_analyzer = None
        self._textblob_ready = False
        
        if use_ml_models:
            self._load_models()
    
    def _load_models(self):
        """Load ML models - using VADER and TextBlob (stable on macOS Python 3.9)"""
        try:
            # Try VADER sentiment analyzer (from NLTK)
            import nltk
            try:
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                self._vader_analyzer = SentimentIntensityAnalyzer()
                print("✅ VADER sentiment analyzer loaded!")
            except LookupError:
                print("📥 Downloading VADER lexicon...")
                nltk.download('vader_lexicon', quiet=True)
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                self._vader_analyzer = SentimentIntensityAnalyzer()
                print("✅ VADER sentiment analyzer loaded!")
            
            # Try TextBlob for additional analysis
            try:
                from textblob import TextBlob
                # Test it works
                _ = TextBlob("test").sentiment
                self._textblob_ready = True
                print("✅ TextBlob analyzer loaded!")
            except ImportError:
                print("⚠️ TextBlob not installed. Using VADER only.")
                self._textblob_ready = False
            
            print("🎉 ML models ready! (VADER + TextBlob)")
            
        except Exception as e:
            print(f"⚠️ Error loading ML models: {e}")
            print("Falling back to lexicon-based analysis.")
            self.use_ml_models = False
    
    def analyze_sentiment_lexicon(self, text: str) -> SentimentResult:
        """
        Analyze sentiment using lexicon-based approach.
        Fast and doesn't require ML models.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        score = 0
        word_count = 0
        
        for word in words:
            if word in self.SENTIMENT_LEXICON:
                score += self.SENTIMENT_LEXICON[word]
                word_count += 1
        
        # Normalize score
        if word_count > 0:
            normalized = score / (word_count ** 0.5)
        else:
            normalized = 0
        
        # Classify
        if normalized > 0.5:
            label = 'positive'
        elif normalized < -0.5:
            label = 'negative'
        else:
            label = 'neutral'
        
        return SentimentResult(
            label=label,
            score=abs(normalized) / 5,  # Normalize to 0-1
            raw_score=normalized
        )
    
    def analyze_sentiment_ml(self, text: str) -> SentimentResult:
        """Analyze sentiment using VADER (social media optimized)"""
        if not self._vader_analyzer:
            return self.analyze_sentiment_lexicon(text)
        
        # VADER analysis
        scores = self._vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # Classify based on compound score
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        # If TextBlob is available, combine scores
        if self._textblob_ready:
            try:
                from textblob import TextBlob
                blob = TextBlob(text)
                tb_polarity = blob.sentiment.polarity
                # Weighted average: VADER (60%) + TextBlob (40%)
                combined = (compound * 0.6) + (tb_polarity * 0.4)
                if combined >= 0.05:
                    label = 'positive'
                elif combined <= -0.05:
                    label = 'negative'
                else:
                    label = 'neutral'
                compound = combined
            except:
                pass
        
        return SentimentResult(
            label=label,
            score=abs(compound),
            raw_score=compound
        )
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using configured method"""
        if self.use_ml_models and self._vader_analyzer:
            return self.analyze_sentiment_ml(text)
        return self.analyze_sentiment_lexicon(text)
    
    def analyze_posts(self, processed_posts: List) -> List[AnalyzedPost]:
        """
        Analyze sentiment for all posts and calculate impact scores.
        Includes comments for deeper context analysis.
        """
        analyzed = []
        
        for post in processed_posts:
            sentiment = self.analyze_sentiment(post.combined_text)
            
            # Calculate impact score
            # Higher for negative posts (they need attention)
            negative_weight = 10 if sentiment.label == 'negative' else 0
            impact_score = post.score + post.num_comments + negative_weight
            
            # Get cleaned comments if available
            cleaned_comments = getattr(post, 'cleaned_comments', '')
            
            analyzed.append(AnalyzedPost(
                id=post.id,
                title=post.cleaned_title,
                body=post.cleaned_body,
                cleaned_comments=cleaned_comments,
                score=post.score,
                num_comments=post.num_comments,
                sentiment=sentiment,
                impact_score=impact_score
            ))
        
        return analyzed
    
    def calculate_sentiment_distribution(
        self, 
        analyzed_posts: List[AnalyzedPost]
    ) -> Dict[str, float]:
        """Calculate percentage distribution of sentiments"""
        if not analyzed_posts:
            return {'positive': 0, 'neutral': 0, 'negative': 0}
        
        counts = Counter(p.sentiment.label for p in analyzed_posts)
        total = len(analyzed_posts)
        
        return {
            'positive': round((counts.get('positive', 0) / total) * 100, 1),
            'neutral': round((counts.get('neutral', 0) / total) * 100, 1),
            'negative': round((counts.get('negative', 0) / total) * 100, 1)
        }
    
    def extract_themes(
        self, 
        analyzed_posts: List[AnalyzedPost]
    ) -> List[Dict]:
        """
        Extract themes using keyword matching.
        Returns top themes sorted by frequency.
        """
        theme_counts = {theme: 0 for theme in self.THEME_KEYWORDS}
        theme_sentiment = {theme: {'positive': 0, 'neutral': 0, 'negative': 0} 
                          for theme in self.THEME_KEYWORDS}
        theme_examples = {theme: [] for theme in self.THEME_KEYWORDS}
        
        for post in analyzed_posts:
            text = f"{post.title} {post.body}".lower()
            
            for theme, keywords in self.THEME_KEYWORDS.items():
                matches = sum(1 for kw in keywords if kw in text)
                
                if matches > 0:
                    theme_counts[theme] += matches
                    theme_sentiment[theme][post.sentiment.label] += 1
                    
                    if len(theme_examples[theme]) < 2:
                        theme_examples[theme].append(post.title[:70])
        
        # Build results
        results = []
        for theme, count in theme_counts.items():
            if count > 0:
                sentiment = theme_sentiment[theme]
                total = sum(sentiment.values())
                
                # Determine predominant sentiment
                if total > 0:
                    neg_pct = sentiment['negative'] / total
                    pos_pct = sentiment['positive'] / total
                    
                    if neg_pct > 0.5:
                        mood = 'negative'
                        explanation = f"Predominantly negative ({int(neg_pct*100)}%). Users expressing frustration."
                    elif pos_pct > 0.5:
                        mood = 'positive'
                        explanation = f"Mostly positive ({int(pos_pct*100)}%). Users are satisfied."
                    else:
                        mood = 'mixed'
                        explanation = f"Mixed feedback with {count} mentions."
                else:
                    mood = 'neutral'
                    explanation = f"{count} mentions in discussions."
                
                results.append({
                    'name': theme,
                    'count': count,
                    'mood': mood,
                    'explanation': explanation,
                    'examples': theme_examples[theme]
                })
        
        # Sort by count descending
        results.sort(key=lambda x: x['count'], reverse=True)
        
        return results[:6]  # Top 6 themes
    
    def generate_summary(self, aggregated_text: str) -> str:
        """
        Generate executive summary.
        Uses ML if available, otherwise creates extractive summary.
        """
        if self.use_ml_models and self._summarizer_pipeline:
            return self._generate_summary_ml(aggregated_text)
        return self._generate_summary_extractive(aggregated_text)
    
    def _generate_summary_ml(self, text: str) -> str:
        """Generate summary using BART model"""
        # Truncate for model limits
        text = text[:1024]
        
        result = self._summarizer_pipeline(
            text,
            max_length=150,
            min_length=50,
            do_sample=False
        )
        
        return result[0]['summary_text']
    
    def _generate_summary_extractive(self, text: str) -> str:
        """
        Generate summary by extracting key sentences.
        Simple but effective for when ML isn't available.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return "Insufficient data for summary."
        
        # Score sentences by keyword importance
        important_words = [
            'issue', 'problem', 'love', 'hate', 'bug', 'feature',
            'update', 'camera', 'battery', 'support', 'quality'
        ]
        
        scored = []
        for sentence in sentences[:20]:  # Limit processing
            score = sum(1 for word in important_words if word in sentence.lower())
            scored.append((score, sentence))
        
        # Get top sentences
        scored.sort(reverse=True)
        top_sentences = [s[1] for s in scored[:4]]
        
        return ". ".join(top_sentences) + "."
    
    def get_high_impact_issues(
        self, 
        analyzed_posts: List[AnalyzedPost],
        top_n: int = 3
    ) -> List[Dict]:
        """Get top N high-impact issues for product team attention"""
        sorted_posts = sorted(
            analyzed_posts, 
            key=lambda p: p.impact_score, 
            reverse=True
        )
        
        return [
            {
                'title': p.title[:100],
                'impact_score': p.impact_score,
                'score': p.score,
                'comments': p.num_comments,
                'sentiment': p.sentiment.label
            }
            for p in sorted_posts[:top_n]
        ]


# Quick test
if __name__ == "__main__":
    analyzer = NLPAnalyzer(use_ml_models=False)
    
    # Test sentiment
    tests = [
        "This phone is amazing! Best camera ever!",
        "Terrible battery life, crashes constantly",
        "Just got my new phone, setting it up now"
    ]
    
    for text in tests:
        result = analyzer.analyze_sentiment(text)
        print(f"{result.label.upper()}: {text[:50]}...")
