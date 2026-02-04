import re
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from googleapiclient.discovery import build

# 🔐 API KEY من Secrets
API_KEY = st.secrets["YOUTUBE_API_KEY"]

nltk.download("vader_lexicon")
analyzer = SentimentIntensityAnalyzer()

def extract_video_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s

    patterns = [
        r"v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(1)

    raise ValueError("Lien/ID invalide")

def fetch_comments(video_id: str, max_comments: int = 100):
    youtube = build("youtube", "v3", developerKey=API_KEY)

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText",
            order="relevance",
        )
        res = req.execute()

        for item in res.get("items", []):
            comments.append(
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            )

        next_page_token = res.get("nextPageToken")
        if not next_page_token:
            break

    return comments

def label(compound):
    if compound >= 0.05:
        return "positif"
    if compound <= -0.05:
        return "negatif"
    return "neutre"

st.title("📺 Analyse sentimentale YouTube")

url_or_id = st.text_input("🔗 Lien YouTube ou videoId")
max_comments = st.slider("💬 Nombre de commentaires", 20, 300, 100, 10)

if st.button("🚀 Analyser"):
    try:
        video_id = extract_video_id(url_or_id)
        comments = fetch_comments(video_id, max_comments)
    except Exception as e:
        st.error(e)
        st.stop()

    results = []
    for c in comments:
        s = analyzer.polarity_scores(c)
        results.append((label(s["compound"]), s["compound"], c))

    pos = sum(1 for r in results if r[0] == "positif")
    neu = sum(1 for r in results if r[0] == "neutre")
    neg = sum(1 for r in results if r[0] == "negatif")

    st.metric("😊 Positifs", pos)
    st.metric("😐 Neutres", neu)
    st.metric("😡 Négatifs", neg)

    st.dataframe(
        [{"label": r[0], "score": r[1], "commentaire": r[2]} for r in results]
    )
