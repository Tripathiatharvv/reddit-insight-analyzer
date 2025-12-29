# Reddit Product Insight Analyzer

A sentiment analysis tool that transforms Reddit discussions into actionable product insights using NLP and AI.

## Overview

This application analyzes Reddit posts and comments to extract meaningful product intelligence including sentiment trends, key themes, user frustrations, and AI-generated recommendations.

**Live Demo:** [reddit-insight-analyzer.streamlit.app](https://reddit-insight-analyzer.streamlit.app)

---

## Features

- **Sentiment Analysis** - VADER and TextBlob ensemble for accurate sentiment scoring
- **Theme Detection** - Automatic extraction of trending discussion topics  
- **Comment Analysis** - Deep analysis including post comments for richer context
- **AI Insights** - LLM-powered executive summaries and action items (via Groq API)
- **High-Impact Detection** - Surfaces critical issues based on engagement and sentiment
- **Export** - Download analysis reports in Markdown format

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| NLP | NLTK (VADER), TextBlob |
| AI/LLM | Llama 3 via Groq API |
| Data Source | PullPush.io API |
| Styling | Custom CSS |

---

## Installation

```bash
git clone https://github.com/Tripathiatharvv/reddit-insight-analyzer.git
cd reddit-insight-analyzer
pip install -r requirements.txt
streamlit run app.py
```

---

## Configuration

### AI Features (Optional)
To enable AI-powered insights, set the `GROQ_API_KEY` environment variable:

```bash
export GROQ_API_KEY="your_groq_api_key"
streamlit run app.py
```

Get a free API key at [console.groq.com](https://console.groq.com)

---

## Project Structure

```
├── app.py                 # Main Streamlit application
├── reddit_fetcher.py      # Reddit data retrieval
├── text_processor.py      # Text cleaning and preprocessing
├── nlp_analyzer.py        # Sentiment and theme analysis
├── ollama_insights.py     # AI integration (Groq/Ollama)
├── report_generator.py    # Report generation
└── requirements.txt       # Dependencies
```

---

## Usage

1. Enter a subreddit name (e.g., `apple`, `iphone`, `android`)
2. Adjust the number of posts and comment depth
3. Enable AI insights for enhanced analysis
4. Click "Analyze Subreddit"
5. Review the generated report and export if needed

---

## License

MIT License

---

Built by [Atharv Tripathi](https://github.com/Tripathiatharvv)
