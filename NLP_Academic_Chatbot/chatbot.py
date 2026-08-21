#chatbot.py

import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Downloads run automatically on first launch — no manual setup needed
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)

# ---------- Load data ----------
data      = pd.read_csv("dataset.csv")
questions = data["Question"]
answers   = data["Answer"]

# ---------- Preprocessing ----------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text   = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# Vectorize all questions once at startup
processed_questions = [preprocess(q) for q in questions]
vectorizer          = TfidfVectorizer()
X                   = vectorizer.fit_transform(processed_questions)

# ---------- Confidence thresholds ----------
# These define how we interpret cosine similarity scores:
#   High   ≥ 0.55  → confident answer
#   Medium ≥ 0.30  → partial / likely match, answer shown with caveat
#   Low    ≥ 0.10  → weak signal, best-guess answer shown with warning
#   None   < 0.10  → completely unrecognised
THRESHOLD_HIGH   = 0.55
THRESHOLD_MEDIUM = 0.30
THRESHOLD_LOW    = 0.10

def _confidence_tier(score):
    if score >= THRESHOLD_HIGH:
        return "High"
    elif score >= THRESHOLD_MEDIUM:
        return "Medium"
    elif score >= THRESHOLD_LOW:
        return "Low"
    else:
        return "None"

# ---------- Simple conversational replies ----------
SMALL_TALK = {
    ("hello", "hi", "hey"):
        "Hello! 👋 I'm your Academic NLP Chatbot. Ask me anything about NLP or Machine Learning.",
    ("how are you",):
        "I'm doing great! How can I help you with NLP or Machine Learning?",
    ("thank you", "thanks"):
        "You're welcome! 😊",
    ("bye", "goodbye"):
        "Goodbye! Have a great day! 👋",
    ("who are you",):
        "I'm an Academic Chatbot built with TF-IDF and Cosine Similarity.",
}

def _small_talk_reply(text):
    """Return a canned reply if the input is small talk, else None."""
    for triggers, reply in SMALL_TALK.items():
        if text in triggers:
            return reply
    return None

# ---------- Top-N matches ----------
def _top_matches(similarity_row, n=3):
    """Return the top-n (index, score) pairs sorted by score descending."""
    scores  = similarity_row.flatten()
    indices = scores.argsort()[::-1][:n]
    return [(int(i), float(scores[i])) for i in indices]

# ---------- Main response function ----------
def get_response(user_input):
    """
    Returns a dict:
        answer           – text to show the user
        score            – best cosine similarity score
        confidence       – 'High' | 'Medium' | 'Low' | 'None'
        matched_question – best-matching dataset question (or '—')
        answered         – True when a real answer is returned (High or Medium)
        top_matches      – list of dicts [{question, score}, ...] for top-3
    """
    reply = _small_talk_reply(user_input.lower().strip())
    if reply:
        return {
            "answer":           reply,
            "score":            0.0,
            "confidence":       "None",
            "matched_question": "—",
            "answered":         False,
            "top_matches":      [],
        }

    processed  = preprocess(user_input)
    user_vec   = vectorizer.transform([processed])
    similarity = cosine_similarity(user_vec, X)
    score      = float(similarity.max())
    index      = int(similarity.argmax())
    confidence = _confidence_tier(score)
    top        = _top_matches(similarity[0])

    top_matches = [
        {"question": questions.iloc[i], "score": round(s, 4)}
        for i, s in top
        if s > 0.01          # skip zero-similarity noise
    ]

    # ── Build answer based on confidence tier ──────────────────────────────
    if confidence == "High":
        answer   = answers.iloc[index]
        answered = True

    elif confidence == "Medium":
        answer   = (
            f"🟡 **Partial match** — I'm moderately confident this is relevant:\n\n"
            f"{answers.iloc[index]}\n\n"
            f"*(Matched: \"{questions.iloc[index]}\" — score {score:.2f})*"
        )
        answered = True

    elif confidence == "Low":
        answer   = (
            f"🟠 **Low confidence** — This is my best guess, but it may not be accurate:\n\n"
            f"{answers.iloc[index]}\n\n"
            f"*(Matched: \"{questions.iloc[index]}\" — score {score:.2f})*\n\n"
            f"Try rephrasing your question for a better result."
        )
        answered = False   # low confidence doesn't count as truly answered

    else:  # None
        answer   = (
            "❌ **No match found.** Your question didn't match anything in my knowledge base.\n\n"
            "Try asking about NLP concepts like tokenization, stemming, TF-IDF, or machine learning basics."
        )
        answered = False

    return {
        "answer":           answer,
        "score":            score,
        "confidence":       confidence,
        "matched_question": questions.iloc[index] if confidence != "None" else "—",
        "answered":         answered,
        "top_matches":      top_matches,
    }
