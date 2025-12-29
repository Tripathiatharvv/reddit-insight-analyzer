"""
Text Processing Module
Cleans and prepares Reddit post text for NLP analysis
"""

import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ProcessedPost:
    """Container for processed post data with comments"""
    id: str
    original_title: str
    original_body: str
    cleaned_title: str
    cleaned_body: str
    cleaned_comments: str  # All comments cleaned and joined
    combined_text: str  # Title weighted 2x + body + comments
    score: int
    num_comments: int
    created_utc: float
    has_content: bool
    comment_count: int = 0  # Actual fetched comments


class TextProcessor:
    """
    Cleans Reddit post text for NLP analysis.
    Implements the weighting logic: Title = 2x importance
    Comments are included for deeper context.
    """
    
    # Regex patterns for cleaning
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    MARKDOWN_PATTERN = re.compile(r'[*_~`#\[\]()>|]')
    HTML_ENTITIES = re.compile(r'&[a-z]+;|&#\d+;', re.IGNORECASE)
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002600-\U000026FF"  # misc symbols
        "\U00002700-\U000027BF"  # dingbats
        "]+", 
        flags=re.UNICODE
    )
    
    # Reddit-specific noise patterns
    REDDIT_NOISE = [
        re.compile(r'\bedit\s*\d*\s*:', re.IGNORECASE),
        re.compile(r'\btl;?dr:?', re.IGNORECASE),
        re.compile(r'thanks for (reading|coming to my ted talk)', re.IGNORECASE),
        re.compile(r'obligatory .* disclaimer', re.IGNORECASE),
        re.compile(r'^(source|sauce):?\s*', re.IGNORECASE | re.MULTILINE),
        re.compile(r'\[removed\]|\[deleted\]', re.IGNORECASE),
    ]
    
    # Maximum body length to keep (characters)
    MAX_BODY_LENGTH = 1000
    MAX_COMMENTS_LENGTH = 2000  # Max total comment characters
    
    def __init__(self):
        pass
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Steps:
        1. Remove URLs
        2. Remove markdown syntax
        3. Decode HTML entities
        4. Remove emojis
        5. Remove Reddit-specific noise
        6. Normalize whitespace
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Remove URLs
        cleaned = self.URL_PATTERN.sub('', cleaned)
        
        # Remove markdown
        cleaned = self.MARKDOWN_PATTERN.sub(' ', cleaned)
        
        # Replace HTML entities with spaces
        cleaned = self.HTML_ENTITIES.sub(' ', cleaned)
        
        # Remove emojis
        cleaned = self.EMOJI_PATTERN.sub('', cleaned)
        
        # Remove Reddit noise
        for pattern in self.REDDIT_NOISE:
            cleaned = pattern.sub('', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def process_post(self, post) -> ProcessedPost:
        """
        Process a single Reddit post with its comments.
        
        Args:
            post: RedditPost object (with optional comments)
            
        Returns:
            ProcessedPost with cleaned text, comments, and weighting applied
        """
        cleaned_title = self.clean_text(post.title)
        cleaned_body = self.clean_text(post.selftext)
        
        # Truncate very long bodies
        if len(cleaned_body) > self.MAX_BODY_LENGTH:
            cleaned_body = cleaned_body[:self.MAX_BODY_LENGTH] + "..."
        
        # Process comments if available
        cleaned_comments = ""
        comment_count = 0
        if hasattr(post, 'comments') and post.comments:
            comment_texts = []
            total_len = 0
            for comment in post.comments:
                cleaned = self.clean_text(comment.body)
                if cleaned and total_len + len(cleaned) < self.MAX_COMMENTS_LENGTH:
                    comment_texts.append(cleaned)
                    total_len += len(cleaned)
                    comment_count += 1
            cleaned_comments = " | ".join(comment_texts)
        
        # Apply 2x title weighting + include comments for context
        combined_text = f"{cleaned_title} {cleaned_title} {cleaned_body} {cleaned_comments}".strip()
        
        # Determine if post has meaningful content
        has_content = len(cleaned_title) > 5
        
        return ProcessedPost(
            id=post.id,
            original_title=post.title,
            original_body=post.selftext,
            cleaned_title=cleaned_title,
            cleaned_body=cleaned_body,
            cleaned_comments=cleaned_comments,
            combined_text=combined_text,
            score=post.score,
            num_comments=post.num_comments,
            created_utc=post.created_utc,
            has_content=has_content,
            comment_count=comment_count
        )
    
    def process_posts(self, posts: List) -> List[ProcessedPost]:
        """
        Process a list of Reddit posts.
        Filters out posts with no meaningful content.
        """
        processed = []
        
        for post in posts:
            proc = self.process_post(post)
            if proc.has_content:
                processed.append(proc)
        
        return processed
    
    def aggregate_text(
        self, 
        processed_posts: List[ProcessedPost],
        max_chars: int = 10000
    ) -> Tuple[str, int]:
        """
        Aggregate all posts into a single document for summarization.
        
        Args:
            processed_posts: List of ProcessedPost objects
            max_chars: Maximum total characters to include
            
        Returns:
            Tuple of (aggregated_text, post_count_included)
        """
        texts = []
        total_chars = 0
        posts_included = 0
        
        for post in processed_posts:
            post_text = f"POST: {post.combined_text}"
            
            if total_chars + len(post_text) > max_chars:
                break
            
            texts.append(post_text)
            total_chars += len(post_text)
            posts_included += 1
        
        aggregated = "\n\n".join(texts)
        
        return aggregated, posts_included


# Quick test
if __name__ == "__main__":
    from reddit_fetcher import RedditFetcher
    
    fetcher = RedditFetcher()
    processor = TextProcessor()
    
    posts = fetcher.fetch_posts("GooglePixel", limit=3)
    processed = processor.process_posts(posts)
    
    for p in processed:
        print(f"Title: {p.cleaned_title[:60]}...")
        print(f"Body: {p.cleaned_body[:100]}...")
        print("---")
