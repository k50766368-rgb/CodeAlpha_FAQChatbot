from __future__ import annotations

import os
from typing import TypedDict

from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FAQ(TypedDict):
    question: str
    answer: str


FAQS: list[FAQ] = [
    {
        "question": "What is Python?",
        "answer": (
            "Python is a general-purpose programming language known for its "
            "readable syntax and large ecosystem. It is commonly used for web "
            "apps, automation, data science, and machine learning."
        ),
    },
    {
        "question": "What is Flask?",
        "answer": (
            "Flask is a lightweight Python web framework. It gives you routing, "
            "request handling, templates, and a development server while letting "
            "you choose the rest of your application stack."
        ),
    },
    {
        "question": "How do I install Flask?",
        "answer": (
            "Create or activate a virtual environment, then run "
            "`pip install Flask`. Add Flask to requirements.txt so the "
            "dependency can be installed again when the app is deployed."
        ),
    },
    {
        "question": "How do I create a route in Flask?",
        "answer": (
            "Use Flask's route decorator. For example: "
            "`@app.route('/about')` followed by a function that returns the "
            "response for that URL."
        ),
    },
    {
        "question": "What is a Flask template?",
        "answer": (
            "A Flask template is usually an HTML file rendered with Jinja. "
            "Templates can include dynamic values, loops, conditionals, and "
            "template inheritance."
        ),
    },
    {
        "question": "How do I read JSON data in Flask?",
        "answer": (
            "For a JSON request, call `request.get_json()` inside the route. "
            "You can then read fields from the returned Python dictionary and "
            "send a JSON response with `jsonify()`."
        ),
    },
    {
        "question": "What is a virtual environment in Python?",
        "answer": (
            "A virtual environment is an isolated Python installation for one "
            "project. It keeps that project's dependencies separate from other "
            "projects and from your system Python."
        ),
    },
    {
        "question": "How do I run a Flask app?",
        "answer": (
            "Set the Flask app entry point and run `flask run` during "
            "development. In this project, you can also run `python main.py`; "
            "the app listens on the configured PORT or port 5000 by default."
        ),
    },
]

SIMILARITY_THRESHOLD = 0.22
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
faq_matrix = vectorizer.fit_transform([faq["question"] for faq in FAQS])

app = Flask(__name__)


def find_answer(question: str) -> tuple[str, FAQ | None, float]:
    """Return the closest FAQ answer when it clears the confidence threshold."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return "Sorry, I don't have an answer for that.", None, 0.0

    question_vector = vectorizer.transform([cleaned_question])
    scores = cosine_similarity(question_vector, faq_matrix)[0]
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])
    best_faq = FAQS[best_index]

    if best_score < SIMILARITY_THRESHOLD:
        return "Sorry, I don't have an answer for that.", None, best_score

    return best_faq["answer"], best_faq, best_score


@app.get("/")
def index():
    return render_template(
        "index.html",
        faqs=FAQS,
        threshold=SIMILARITY_THRESHOLD,
    )


@app.post("/api/ask")
def ask():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "")

    if not isinstance(question, str):
        return jsonify({"error": "Question must be text."}), 400

    answer, matched_faq, score = find_answer(question)
    return jsonify(
        {
            "answer": answer,
            "matched_question": matched_faq["question"] if matched_faq else None,
            "similarity": round(score, 3),
            "matched": matched_faq is not None,
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
