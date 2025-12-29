"""
Report Generator Module
Generates structured product insight reports with optional AI enhancement
"""

from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class InsightReport:
    """Complete analysis report"""
    subreddit: str
    posts_analyzed: int
    analysis_mode: str
    timestamp: str
    processing_time_ms: float
    
    executive_summary: str
    sentiment_distribution: Dict[str, float]
    themes: List[Dict]
    product_insights: Dict
    high_impact_issues: List[Dict]
    
    # AI-enhanced fields
    ai_enhanced: bool = False
    action_items: List[str] = None


class ReportGenerator:
    """
    Generates product insight reports from analyzed data.
    Formats output suitable for product managers.
    Optionally uses AI (Local Ollama or Cloud Groq) for deep contextual insights.
    """
    
    def __init__(self, use_ai: bool = False):
        self.use_ai = use_ai
        self.ai = None
        
        if use_ai:
            try:
                from ollama_insights import AIInsights
                self.ai = AIInsights()
                
                # Check availability based on provider
                is_available = False
                if self.ai.provider == "groq" and self.ai.groq_client:
                    is_available = True
                elif self.ai.provider == "ollama" and self.ai.local_available:
                    is_available = True
                    
                if not is_available:
                    print(f"⚠️ AI provider ({self.ai.provider}) not available. Using rule-based insights.")
                    self.ai = None
                else:
                    print(f"🤖 AI insights enabled ({self.ai.provider.upper()})")
            except ImportError:
                print("⚠️ AI module error. Using rule-based insights.")
                self.ai = None
    
    def generate_executive_summary(
        self,
        subreddit: str,
        analyzed_posts: List,
        themes: List[Dict],
        sentiment_dist: Dict[str, float]
    ) -> str:
        """
        Generate a contextual executive summary.
        Uses AI for deep insights if available, otherwise rule-based.
        """
        total = len(analyzed_posts)
        
        if total == 0:
            return "No posts available for analysis."
        
        # Try AI-powered summary if available
        if self.ai:
            try:
                posts_data = [
                    {
                        'title': p.title,
                        'body': p.body[:200] if hasattr(p, 'body') else '',
                        'comments': getattr(p, 'cleaned_comments', '')[:300] if hasattr(p, 'cleaned_comments') else '',
                        'sentiment': {'label': p.sentiment.label, 'score': p.sentiment.score}
                    }
                    for p in analyzed_posts[:20]
                ]
                ai_summary = self.ai.generate_executive_summary(
                    subreddit, posts_data, sentiment_dist
                )
                if ai_summary and len(ai_summary) > 50:
                    return ai_summary
            except Exception as e:
                print(f"AI summary failed, using rule-based: {e}")
        
        # Fallback to rule-based summary
        summary_parts = []
        
        # Sentiment overview
        neg = sentiment_dist.get('negative', 0)
        pos = sentiment_dist.get('positive', 0)
        neu = sentiment_dist.get('neutral', 0)
        
        if neg > 50:
            summary_parts.append(
                f"Analysis of {total} recent posts reveals predominantly negative sentiment "
                f"({neg:.0f}%), indicating widespread user concerns."
            )
        elif pos > 50:
            summary_parts.append(
                f"Analysis of {total} recent posts shows generally positive sentiment "
                f"({pos:.0f}%), reflecting user satisfaction."
            )
        else:
            summary_parts.append(
                f"Analysis of {total} recent posts reveals mixed sentiment "
                f"(Positive: {pos:.0f}%, Neutral: {neu:.0f}%, Negative: {neg:.0f}%)."
            )
        
        # Top themes
        if themes:
            top_themes = [t['name'] for t in themes[:3]]
            summary_parts.append(
                f"Key discussion areas include {', '.join(top_themes)}."
            )
        
        # Negative highlights
        negative_posts = [p for p in analyzed_posts if p.sentiment.label == 'negative']
        if negative_posts:
            top_issue = max(negative_posts, key=lambda p: p.impact_score)
            summary_parts.append(
                f"Notable concerns include: \"{top_issue.title[:60]}...\""
            )
        
        # Positive highlights
        positive_posts = [p for p in analyzed_posts if p.sentiment.label == 'positive']
        if positive_posts:
            top_praise = max(positive_posts, key=lambda p: p.score)
            summary_parts.append(
                f"Users appreciate certain aspects, as shown in: \"{top_praise.title[:50]}...\""
            )
        
        # Engagement insight
        high_engagement = sorted(analyzed_posts, key=lambda p: p.num_comments, reverse=True)
        if high_engagement:
            summary_parts.append(
                f"The most engaged discussions have {high_engagement[0].num_comments}+ comments, "
                f"indicating high community interest in these topics."
            )
        
        return " ".join(summary_parts)
    
    def generate_product_insights(
        self,
        analyzed_posts: List,
        themes: List[Dict]
    ) -> Dict:
        """
        Generate structured product insights.
        Categories: likes, frustrations, trends
        """
        positive_posts = [p for p in analyzed_posts if p.sentiment.label == 'positive']
        negative_posts = [p for p in analyzed_posts if p.sentiment.label == 'negative']
        
        # What users like (top positive posts)
        likes = []
        for p in sorted(positive_posts, key=lambda x: x.score, reverse=True)[:3]:
            likes.append(p.title[:100])
        
        if not likes:
            likes = ["No strongly positive posts in this sample"]
        
        # What frustrates users (top negative by impact)
        frustrations = []
        for p in sorted(negative_posts, key=lambda x: x.impact_score, reverse=True)[:3]:
            frustrations.append(p.title[:100])
        
        if not frustrations:
            frustrations = ["No significant frustrations detected"]
        
        # Trends based on theme sentiment
        worsening = []
        improving = []
        
        for theme in themes:
            if theme.get('mood') == 'negative':
                worsening.append(theme['name'])
            elif theme.get('mood') == 'positive':
                improving.append(theme['name'])
        
        if not worsening:
            worsening = ["No clear worsening trends detected"]
        if not improving:
            improving = ["No clear improving trends detected"]
        
        # Try AI enhancement for deeper insights
        if self.ai:
            try:
                posts_data = [
                    {
                        'title': p.title,
                        'sentiment': {'label': p.sentiment.label}
                    }
                    for p in analyzed_posts[:30]
                ]
                themes_data = [{'name': t['name']} for t in themes[:5]]
                
                ai_insights = self.ai.generate_deep_insights(posts_data, themes_data)
                
                if ai_insights:
                    # Merge with rule-based, prioritizing AI
                    return ai_insights
            except Exception as e:
                print(f"AI insights failed: {e}")

        return {
            'likes': likes,
            'frustrations': frustrations,
            'worsening': worsening,
            'improving': improving,
            'opportunities': ["Increase sample size for better opportunity detection"]
        }
    
    def generate_interpretation(self, sentiment_dist: Dict[str, float]) -> str:
        """Generate 1-line sentiment interpretation"""
        neg = sentiment_dist.get('negative', 0)
        pos = sentiment_dist.get('positive', 0)
        
        if neg > 60:
            return "⚠️ Critical: Community sentiment is heavily negative. Immediate attention required."
        elif neg > 40:
            return "⚠️ Warning: Significant negative sentiment. Product team should investigate issues."
        elif pos > 60:
            return "✅ Positive: Users are generally satisfied. Continue current direction."
        elif pos > 40:
            return "✅ Healthy: Mostly positive feedback with some areas for improvement."
        else:
            return "ℹ️ Mixed: Varied user experiences. Focus on reducing friction points."
    
    def generate_full_report(
        self,
        subreddit: str,
        analyzed_posts: List,
        themes: List[Dict],
        sentiment_dist: Dict[str, float],
        high_impact: List[Dict],
        analysis_mode: str,
        processing_time_ms: float
    ) -> InsightReport:
        """Generate complete InsightReport object with optional AI enhancement"""
        
        executive_summary = self.generate_executive_summary(
            subreddit, analyzed_posts, themes, sentiment_dist
        )
        
        product_insights = self.generate_product_insights(
            analyzed_posts, themes
        )
        
        # Generate AI action items if available
        action_items = None
        ai_enhanced = False
        
        if self.ai:
            try:
                # Need to implement generate_action_items in AIInsights
                # Assuming it exists or fallback
                if hasattr(self.ai, 'generate_action_items'):
                    action_items = self.ai.generate_action_items(high_impact, themes)
                    ai_enhanced = True
            except Exception as e:
                print(f"AI action items failed: {e}")
        
        return InsightReport(
            subreddit=subreddit,
            posts_analyzed=len(analyzed_posts),
            analysis_mode=analysis_mode,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time_ms,
            executive_summary=executive_summary,
            sentiment_distribution=sentiment_dist,
            themes=themes,
            product_insights=product_insights,
            high_impact_issues=high_impact,
            ai_enhanced=ai_enhanced,
            action_items=action_items
        )
    
    def format_markdown_report(self, report: InsightReport) -> str:
        """Format report as Markdown for display"""
        lines = []
        
        # Header
        lines.append(f"# 📊 Reddit Product Insight Report")
        lines.append(f"**Subreddit:** r/{report.subreddit}")
        lines.append(f"**Posts Analyzed:** {report.posts_analyzed}")
        lines.append(f"**Mode:** {report.analysis_mode}")
        lines.append(f"**Generated:** {report.timestamp[:10]}")
        lines.append(f"**Processing Time:** {report.processing_time_ms:.0f}ms")
        lines.append("")
        
        # Executive Summary
        lines.append("---")
        lines.append("## 🔹 Executive Summary")
        lines.append(report.executive_summary)
        lines.append("")
        
        # Sentiment (if available)
        if report.sentiment_distribution:
            lines.append("---")
            lines.append("## 🔹 Sentiment Snapshot")
            lines.append(f"- 🟢 Positive: **{report.sentiment_distribution['positive']:.1f}%**")
            lines.append(f"- ⚪ Neutral: **{report.sentiment_distribution['neutral']:.1f}%**")
            lines.append(f"- 🔴 Negative: **{report.sentiment_distribution['negative']:.1f}%**")
            lines.append("")
            lines.append(f"*{self.generate_interpretation(report.sentiment_distribution)}*")
            lines.append("")
        
        # Themes (if available)
        if report.themes:
            lines.append("---")
            lines.append("## 🔹 Key Themes")
            for theme in report.themes:
                lines.append(f"- **{theme['name']}** — {theme['explanation']}")
            lines.append("")
        
        # Product Insights
        lines.append("---")
        lines.append("## 🔹 Product Insights")
        lines.append("")
        lines.append("### ✅ What Users Like")
        for item in report.product_insights['likes']:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("### ❌ What Frustrates Users")
        for item in report.product_insights['frustrations']:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("### 📈 Trends")
        lines.append(f"- **Worsening:** {', '.join(report.product_insights['worsening'])}")
        lines.append(f"- **Improving:** {', '.join(report.product_insights['improving'])}")
        lines.append("")
        
        # High Impact Issues
        if report.high_impact_issues:
            lines.append("---")
            lines.append("## 🔹 High-Impact Issues (Priority)")
            for i, issue in enumerate(report.high_impact_issues, 1):
                sentiment_emoji = "🔴" if issue['sentiment'] == 'negative' else "🟢" if issue['sentiment'] == 'positive' else "⚪"
                lines.append(f"**#{i}** {sentiment_emoji} {issue['title']}")
                lines.append(f"   - Score: {issue['score']} | Comments: {issue['comments']} | Impact: {issue['impact_score']}")
            lines.append("")
        
        return "\n".join(lines)
