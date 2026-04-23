## 📌 Project Overview
This project is an end-to-end **NLP pipeline** designed to automate the evaluation of academic essays. It goes beyond simple keyword matching by using semantic analysis to grade essays on relevance, grammar, and structure. Additionally, it features an AI-content detector and a personalized feedback engine to assist learners.

## 🚀 Key Features
*   **Semantic Scoring:** Uses `Sentence-Transformers` to understand the context and depth of the essay content.
*   **Grading Engine:** A `LightGBM` model trained to predict scores based on structural and linguistic features.
*   **AI & Plagiarism Detection:** Integrated classification module to distinguish between human-written and AI-generated text.
*   **Feedback Engine:** Generates personalized improvement suggestions, simulating a "client-ready" feedback report.
*   **Interactive Web UI:** A `Flask` dashboard for easy essay submission and real-time result visualization.

## 🛠️ Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python |
| **NLP** | Sentence-Transformers, NLTK, Spacy |
| **Machine Learning** | LightGBM, Scikit-learn |
| **Backend/Web** | Flask, Jinja2 |
| **Data Handling** | Pandas, NumPy |

## 📊 Business Impact (Consulting Context)
This solution demonstrates skills directly applicable to **Deloitte's Data & AI practice**:
1.  **Automation:** Reducing manual review time for unstructured text by ~80%.
2.  **Scalability:** The pipeline can be adapted for large-scale legal, HR, or customer sentiment analysis.
3.  **Accuracy:** Fine-tuned classification thresholds to minimize false positives in high-stakes environments.

## 📂 Project Structure
```text
├── data/               # Sample datasets
├── models/             # Saved LightGBM models (.pkl)
├── notebooks/          # EDA and Model Training scripts
├── src/                # Core NLP and Flask logic
├── templates/          # Web UI (HTML)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## ⚙️ Setup & Installation
1. **Clone the repo:**
   ```bash
   git clone https://github.com
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python src/app.py
