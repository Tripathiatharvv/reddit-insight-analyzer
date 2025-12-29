"""
Reddit Data Fetcher Module
Uses PullPush.io API (Pushshift successor) which doesn't block cloud IPs.
Falls back to Reddit public API for local development.
"""

import requests
import time
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


@dataclass
class RedditComment:
    """Data class for a Reddit comment"""
    id: str
    body: str
    score: int
    author: str
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'body': self.body,
            'score': self.score,
            'author': self.author
        }


@dataclass
class RedditPost:
    """Data class for a Reddit post with comments"""
    id: str
    title: str
    selftext: str
    score: int
    num_comments: int
    created_utc: float
    comments: List[RedditComment] = field(default_factory=list)
    
    @property
    def created_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.created_utc)
    
    @property
    def all_text(self) -> str:
        """Get all text content: title + body + comments"""
        parts = [self.title, self.selftext]
        for comment in self.comments:
            parts.append(comment.body)
        return " ".join(part for part in parts if part)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'selftext': self.selftext,
            'score': self.score,
            'num_comments': self.num_comments,
            'created_utc': self.created_utc,
            'comments': [c.to_dict() for c in self.comments]
        }


class RedditFetcher:
    """
    Fetches Reddit posts using PullPush.io API (works from cloud servers).
    Falls back to Reddit public API if PullPush fails.
    """
    
    # PullPush.io API - doesn't block cloud IPs
    PULLPUSH_URL = "https://api.pullpush.io/reddit"
    # Fallback to Reddit
    REDDIT_URL = "https://old.reddit.com"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.use_pullpush = True  # Default to PullPush for cloud compatibility
        print("✅ Using PullPush.io API (Cloud-compatible)")
    
    def fetch_posts(
        self, 
        subreddit: str, 
        limit: int = 10,
        sort: str = 'new',
        comments_per_post: int = 0
    ) -> List[RedditPost]:
        """
        Fetch posts from a subreddit.
        
        Args:
            subreddit: Name of the subreddit (without r/)
            limit: Number of posts to fetch (1-100)
            sort: Sort order ('new', 'hot', 'top')
            comments_per_post: Number of comments to fetch per post
            
        Returns:
            List of RedditPost objects
        """
        if not subreddit or not subreddit.strip():
            raise ValueError("Subreddit name cannot be empty")
        
        limit = max(1, min(100, limit))
        comments_per_post = max(0, min(50, comments_per_post))
        subreddit = subreddit.strip()
        
        # Try PullPush first (cloud-compatible)
        try:
            posts = self._fetch_from_pullpush(subreddit, limit, sort)
            if posts:
                # Fetch comments if requested
                if comments_per_post > 0:
                    for post in posts[:10]:  # Limit comment fetching
                        if post.num_comments > 0:
                            post.comments = self._fetch_comments_pullpush(post.id, comments_per_post)
                            time.sleep(0.3)  # Rate limiting
                return posts
        except Exception as e:
            print(f"⚠️ PullPush failed: {e}, trying Reddit...")
        
        # Fallback to Reddit public API
        return self._fetch_from_reddit(subreddit, limit, sort, comments_per_post)
    
    def _fetch_from_pullpush(self, subreddit: str, limit: int, sort: str) -> List[RedditPost]:
        """Fetch posts from PullPush.io API"""
        
        # PullPush uses 'size' instead of 'limit'
        url = f"{self.PULLPUSH_URL}/search/submission/"
        params = {
            'subreddit': subreddit,
            'size': limit,
            'sort': 'desc',
            'sort_type': 'created_utc'
        }
        
        response = self.session.get(url, params=params, timeout=15)
        
        if response.status_code == 404:
            raise ValueError(f"Subreddit r/{subreddit} not found")
        elif not response.ok:
            raise ConnectionError(f"PullPush API error: {response.status_code}")
        
        data = response.json()
        posts = []
        
        for item in data.get('data', []):
            post = RedditPost(
                id=item.get('id', ''),
                title=item.get('title', ''),
                selftext=item.get('selftext', '') or '',
                score=item.get('score', 0),
                num_comments=item.get('num_comments', 0),
                created_utc=item.get('created_utc', 0)
            )
            posts.append(post)
        
        return posts
    
    def _fetch_comments_pullpush(self, post_id: str, limit: int) -> List[RedditComment]:
        """Fetch comments from PullPush.io API"""
        
        url = f"{self.PULLPUSH_URL}/search/comment/"
        params = {
            'link_id': f't3_{post_id}',
            'size': limit,
            'sort': 'desc',
            'sort_type': 'score'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            if not response.ok:
                return []
            
            data = response.json()
            comments = []
            
            for item in data.get('data', []):
                body = item.get('body', '')
                if body in ['[deleted]', '[removed]', '']:
                    continue
                    
                comment = RedditComment(
                    id=item.get('id', ''),
                    body=body,
                    score=item.get('score', 0),
                    author=item.get('author', '[deleted]')
                )
                comments.append(comment)
                
                if len(comments) >= limit:
                    break
            
            return comments
        except:
            return []
    
    def _fetch_from_reddit(
        self, 
        subreddit: str, 
        limit: int, 
        sort: str,
        comments_per_post: int
    ) -> List[RedditPost]:
        """Fallback: Fetch from Reddit public API"""
        
        url = f"{self.REDDIT_URL}/r/{subreddit}/{sort}.json"
        params = {'limit': limit}
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 404:
                raise ValueError(f"Subreddit r/{subreddit} not found")
            elif response.status_code == 403:
                raise ValueError(f"Subreddit r/{subreddit} not accessible")
            elif not response.ok:
                raise ConnectionError(f"Reddit API error: {response.status_code}")
            
            data = response.json()
            posts = []
            
            for item in data.get('data', {}).get('children', []):
                post_data = item.get('data', {})
                
                if post_data.get('stickied', False):
                    continue
                
                post = RedditPost(
                    id=post_data.get('id', ''),
                    title=post_data.get('title', ''),
                    selftext=post_data.get('selftext', ''),
                    score=post_data.get('score', 0),
                    num_comments=post_data.get('num_comments', 0),
                    created_utc=post_data.get('created_utc', 0)
                )
                
                if comments_per_post > 0 and post.num_comments > 0:
                    post.comments = self._fetch_comments_reddit(subreddit, post.id, comments_per_post)
                    time.sleep(0.2)
                
                posts.append(post)
            
            return posts
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Network error: {str(e)}")
    
    def _fetch_comments_reddit(self, subreddit: str, post_id: str, limit: int) -> List[RedditComment]:
        """Fetch comments from Reddit"""
        
        url = f"{self.REDDIT_URL}/r/{subreddit}/comments/{post_id}.json"
        params = {'limit': limit, 'depth': 1}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            if not response.ok:
                return []
            
            data = response.json()
            comments = []
            
            if len(data) < 2:
                return []
            
            for item in data[1].get('data', {}).get('children', []):
                if item.get('kind') != 't1':
                    continue
                
                comment_data = item.get('data', {})
                body = comment_data.get('body', '')
                
                if body in ['[deleted]', '[removed]', '']:
                    continue
                
                comment = RedditComment(
                    id=comment_data.get('id', ''),
                    body=body,
                    score=comment_data.get('score', 0),
                    author=comment_data.get('author', '[deleted]')
                )
                comments.append(comment)
                
                if len(comments) >= limit:
                    break
            
            return comments
        except:
            return []


# Quick test
if __name__ == "__main__":
    fetcher = RedditFetcher()
    
    try:
        posts = fetcher.fetch_posts("technology", limit=5, comments_per_post=3)
        print(f"\n✅ Fetched {len(posts)} posts:")
        for post in posts:
            print(f"  - {post.title[:60]}... ({len(post.comments)} comments)")
    except Exception as e:
        print(f"❌ Error: {e}")
