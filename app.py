from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
import hashlib
import re

from sentence_transformers import SentenceTransformer, util
import language_tool_python

app = Flask(__name__)

# Load models
encoder = SentenceTransformer("all-MiniLM-L6-v2")
tool = language_tool_python.LanguageTool("en-US")

dataset = pd.read_csv("dataset/ielts_writing_dataset.csv")

MIN_SCORE = dataset["Overall"].min()
MAX_SCORE = dataset["Overall"].max()

# Embedding cache
EMBEDDING_DIR = "embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)


def get_embedding(text):
    h = hashlib.md5(text.encode()).hexdigest()
    path = os.path.join(EMBEDDING_DIR, f"{h}.npy")

    if os.path.exists(path):
        return np.load(path)

    emb = encoder.encode(text)
    np.save(path, emb)
    return emb


# ---------- FEATURES ----------

def compute_content_score(essay, prompt):
    e = get_embedding(essay)
    p = get_embedding(prompt)
    return util.cos_sim(e, p).item()


def compute_grammar_score(text):
    matches = tool.check(text)
    error_rate = len(matches) / (len(text.split()) + 1e-6)
    return max(0, 1 - error_rate * 5)


def compute_length_score(text):
    wc = len(text.split())
    if wc < 150:
        return wc / 150
    elif wc > 350:
        return 350 / wc
    return 1.0


def plagiarism_score(essay):
    sample = dataset["Essay"].sample(5).tolist()
    essay_emb = get_embedding(essay)

    sims = []
    for ref in sample:
        emb = get_embedding(ref)
        sims.append(util.cos_sim(essay_emb, emb).item())

    return max(sims)


# ---------- SIMPLE AI DETECTION ----------
def detect_ai_score(text):
    # simple heuristic (replace later)
    if len(text.split()) > 200 and text.count(".") < 3:
        return 0.6
    return 0.2


# ---------- FINAL SCORE ----------
def compute_final_score(content, grammar, length, plagiarism):
    return (
        0.5 * content +
        0.2 * grammar +
        0.15 * length +
        0.15 * (1 - plagiarism)
    )


def generate_feedback():
    return """Strengths:
- Good structure
- Clear ideas

Weaknesses:
- Limited vocabulary

Improvements:
- Improve grammar
- Add examples
"""


# ---------- ROUTES ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/evaluate", methods=["POST"])
def evaluate():
    essay = request.form["essay"]
    prompt = request.form["prompt"]

    content = compute_content_score(essay, prompt)
    grammar = compute_grammar_score(essay)
    length = compute_length_score(essay)
    plagiarism = plagiarism_score(essay)
    ai = detect_ai_score(essay)

    if ai > 0.4:
        final_score = 0
        classification = "AI GENERATED — REJECTED"
    else:
        raw = compute_final_score(content, grammar, length, plagiarism)
        final_score = MIN_SCORE + raw * (MAX_SCORE - MIN_SCORE)
        classification = "Valid Essay"

    return render_template(
        "result.html",
        final_score=round(final_score, 2),
        content_score=round(content, 2),
        grammar_score=round(grammar, 2),
        length_score=round(length, 2),
        ai_score=round(ai, 2),
        plagiarism_score=round(plagiarism, 2),
        feedback=generate_feedback(),
        classification=classification
    )


if __name__ == "__main__":
    app.run(debug=True)
