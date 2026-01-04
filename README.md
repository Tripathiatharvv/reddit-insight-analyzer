# 🔬 DeepSight Pro

**AI-Powered Product Intelligence Platform** — Transform Reddit feedback into actionable engineering tickets with competitive analysis.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-00A67E?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyeiIvPjwvc3ZnPg==&logoColor=white)](https://groq.com)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Smart Analysis** | AI-powered product issue detection using Groq's LLaMA 3.3 70B |
| 🎫 **JIRA-Style Tickets** | Auto-generates P0/P1/P2 severity engineering tickets |
| ⚔️ **Competitor Intel** | Side-by-side differentiation matrix (You Win / They Win) |
| 📊 **Evidence-Based** | Every insight linked to original Reddit posts |
| 🔥 **High-Signal Filter** | Filters noise, keeps only engaged discussions (5+ comments) |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/deepsight-pro.git
cd deepsight-pro
pip install -r requirements.txt
```

### 2. Configure API Key

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Get your free API key at [console.groq.com](https://console.groq.com)

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 📖 How It Works

```mermaid
graph LR
    A[Reddit Subreddit] --> B[PullPush API]
    B --> C[High-Signal Filter]
    C --> D[Groq LLaMA 3.3]
    D --> E[Product Tickets]
    E --> F[Strategy Dashboard]
```

1. **Fetch** — Pulls top posts from any subreddit via PullPush API
2. **Filter** — Removes noise (deleted posts, low engagement, spam)
3. **Analyze** — AI generates structured product insights with root cause analysis
4. **Display** — Triage dashboard with P0/P1/P2 severity grouping

---

## 🖥️ Dashboard Preview

| Metric | Description |
|--------|-------------|
| 📊 Posts Analyzed | Total high-signal posts processed |
| 🎫 Tickets Generated | AI-generated product issues |
| 🚨 P0/P1 Issues | Critical + High severity count |
| 🐛 Bugs | Bug category tickets |

### Ticket Structure

Each ticket includes:
- **User Pain** — What users experience
- **Technical Hypothesis** — Engineering root cause
- **Strategic Recommendation** — Actionable fix
- **Evidence** — Verbatim quote with source link

---

## ⚙️ Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Days to analyze | 30 | Historical data range |
| Min comments | 5 | Engagement threshold |
| Max posts | 30 | Analysis depth limit |

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI**: Groq API (LLaMA 3.3 70B Versatile)
- **Data**: PullPush.io Reddit API
- **Language**: Python 3.9+

---

## 📦 Project Structure

```
deepsight-pro/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── .streamlit/
│   └── secrets.toml    # API keys (gitignored)
└── README.md
```

---

## 🔐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Groq API authentication |

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

<p align="center">
  <strong>Built with ❤️ using Streamlit & Groq</strong>
</p>
