# 🤖 Academic NLP Chatbot

<p align="center">
  A lightweight academic question-answering chatbot built using <b>Natural Language Processing</b>, <b>TF-IDF</b>, and <b>Cosine Similarity</b>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/NLP-NLTK-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML-Scikit--learn-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/UI-Streamlit-red?style=for-the-badge&logo=streamlit" />
</p>

---

## 📌 Overview

This project is an **academic question-answering chatbot** built using **Natural Language Processing (NLP)** and **Machine Learning techniques**.

The chatbot compares a user's question against a predefined knowledge base using **TF-IDF vectorization** and **Cosine Similarity**, then returns the answer associated with the most relevant question.

The project also includes an interactive **Streamlit dashboard** for visualizing:

* Similarity scores
* Confidence levels
* Query accuracy
* Matching results
* Query statistics

---

## ✨ Features

| 🧠 NLP Processing   | 🎯 Matching System        | 📊 Analytics            |
| ------------------- | ------------------------- | ----------------------- |
| Tokenization        | TF-IDF Vectorization      | Query Statistics        |
| Stopword Removal    | Cosine Similarity         | Confidence Distribution |
| Punctuation Removal | Top-3 Matching            | Similarity Scores       |
| Lemmatization       | Confidence Classification | Cumulative Accuracy     |
| Text Normalization  | Best Match Selection      | CSV Export              |

### Additional Features

* 💬 **Conversational Responses** — Handles greetings, thanks, and goodbye messages separately from the NLP matching pipeline.
* 🥇 **Top-3 Matches** — Displays the three highest-scoring questions from the dataset.
* 📥 **CSV Export** — Allows query results to be downloaded for further analysis.
* 📊 **Interactive Dashboard** — Provides detailed analytics for queries submitted during the current Streamlit session.

---

# 🏗️ System Architecture

<p align="center">
  <img src="NLP_Academic_Chatbot/System_Architecture.png" alt="System Architecture" width="100%">
</p>

---

# 🔄 How It Works

The chatbot processes both dataset questions and user queries through the following pipeline:

```text
User Question
      │
      ▼
┌─────────────────────┐
│   Small-Talk Check  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Lowercasing     │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│    Tokenization     │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Stopword Removal    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Punctuation Removal │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│    Lemmatization    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ TF-IDF Vectorization│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Cosine Similarity  │
└──────────┬──────────┘
           ▼
      Best Match
           │
           ▼
┌─────────────────────┐
│ Confidence Check    │
└──────────┬──────────┘
           ▼
     Chatbot Response
```

---

## 🧠 NLP Pipeline

| Step                        | Process                                                 | Example                                 |
| --------------------------- | ------------------------------------------------------- | --------------------------------------- |
| **1️⃣ Lowercasing**         | Converts all text to lowercase for consistent matching. | `Machine Learning` → `machine learning` |
| **2️⃣ Tokenization**        | Splits a sentence into individual tokens.               | `"What is machine learning?"`           |
| **3️⃣ Stopword Removal**    | Removes common English words with low matching value.   | `is`, `the`, `a`, `of`                  |
| **4️⃣ Punctuation Removal** | Removes punctuation before vectorization.               | `?`, `!`, `,`                           |
| **5️⃣ Lemmatization**       | Converts words into their dictionary base form.         | `cars` → `car`                          |

### Example

```text
Input:
"What is machine learning?"

↓

Tokens:
["what", "is", "machine", "learning", "?"]
```

The processed text is then passed to the **TF-IDF vectorizer**.

---

# 📐 TF-IDF Vectorization

The project uses `TfidfVectorizer` from **Scikit-learn**.

TF-IDF represents each question as a numerical vector based on the importance of its words within the dataset.

```text
TF-IDF = Term Frequency × Inverse Document Frequency
```

The dataset is vectorized once when the chatbot starts:

```python
X = vectorizer.fit_transform(processed_questions)
```

When a user asks a question:

```python
user_vec = vectorizer.transform([processed])
```

This allows the user query to be compared directly against every question in the dataset.

---

# 🔍 Cosine Similarity

Cosine Similarity measures how similar the user's question is to questions stored in the knowledge base.

```text
                    A · B
Cosine Similarity = ───────
                     |A||B|
```

The score generally ranges from:

```text
0 → Completely Different
1 → Highly Similar
```

The chatbot selects the highest-scoring question:

```python
score = float(similarity.max())
index = int(similarity.argmax())
```

The corresponding answer is then retrieved from the dataset.

---

# 🎯 Confidence System

The similarity score is converted into a confidence tier.

|             Score | Confidence | Behaviour                                    |
| ----------------: | ---------- | -------------------------------------------- |
|        **≥ 0.55** | 🟢 High    | Answer returned directly                     |
| **0.30 – 0.5499** | 🟡 Medium  | Answer returned with a partial-match warning |
| **0.10 – 0.2999** | 🟠 Low     | Best guess shown with a warning              |
|        **< 0.10** | 🔴 None    | No answer returned                           |

> ⚠️ **Note:** These confidence levels are heuristic thresholds, not probabilities.

For example, a similarity score of `0.80` does **not** mean that the chatbot is 80% certain that the answer is correct.

---

# 💬 Streamlit Interface

The application contains two main sections:

| 💬 Chat                    | 📊 Result Analysis           |
| -------------------------- | ---------------------------- |
| Ask questions              | View query statistics        |
| View chatbot responses     | Analyze similarity scores    |
| See confidence warnings    | View confidence distribution |
| See best-matching question | Track cumulative accuracy    |
| View Top-3 matches         | Export results to CSV        |

---

## 📊 Dashboard Metrics

The Result Analysis dashboard tracks:

| Metric                      | Description                              |
| --------------------------- | ---------------------------------------- |
| **Total Queries**           | Total number of submitted queries        |
| **Answered Queries**        | Queries for which an answer was returned |
| **Average Match Score**     | Average cosine similarity score          |
| **Accuracy**                | High + Medium confidence queries         |
| **High Confidence Queries** | Queries classified as High confidence    |

### Additional Visualizations

* 📈 **Score Distribution**
* 🎯 **Confidence Tier Breakdown**
* 📊 **Match Score per Query**
* 📉 **Cumulative Accuracy**
* 📋 **Detailed Query Results**

Results can also be exported as:

```text
nlp_results.csv
```

---

# 🗂️ Dataset

The chatbot uses a CSV-based knowledge base:

```text
dataset.csv
```

The dataset must contain at least two columns:

```text
Question,Answer
```

### Example

```csv
Question,Answer
"What is NLP?","Natural Language Processing is a field of AI..."
"What is TF-IDF?","TF-IDF is a statistical method..."
"What is machine learning?","Machine learning is a branch of AI..."
```

| Column     | Purpose                                    |
| ---------- | ------------------------------------------ |
| `Question` | Used for similarity matching               |
| `Answer`   | Returned when a relevant question is found |

---

# 📁 Project Structure

```text
academic-nlp-chatbot/
│
├── chatbot.py
├── app.py
├── dataset.csv
├── requirements.txt
└── README.md
```

| File               | Description                                                                                                     |
| ------------------ | --------------------------------------------------------------------------------------------------------------- |
| `chatbot.py`       | Handles NLP processing, vectorization, similarity matching, confidence classification, and response generation. |
| `app.py`           | Contains the Streamlit interface, session handling, dashboard, charts, and CSV export.                          |
| `dataset.csv`      | Stores the chatbot's question-answer knowledge base.                                                            |
| `requirements.txt` | Contains project dependencies.                                                                                  |

---

# ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd academic-nlp-chatbot
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```text
pandas
nltk
scikit-learn
streamlit
matplotlib
numpy
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📚 NLTK Resources

The application automatically downloads the required NLTK resources during its first launch:

```python
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
```

No separate manual NLTK corpus setup is required.

---

# 🧪 Example

Suppose the dataset contains:

```text
Question:
What is TF-IDF?

Answer:
TF-IDF is a numerical representation that measures the importance
of a word in a document relative to a collection of documents.
```

The user asks:

```text
Can you explain TF IDF?
```

The processing flow becomes:

```text
User Query
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Cosine Similarity
    ↓
Similarity Score: 0.78
    ↓
🟢 High Confidence
    ↓
Return Matching Answer
```

---

# ⚠️ Limitations

| Limitation                    | Description                                                       |
| ----------------------------- | ----------------------------------------------------------------- |
| **Retrieval-Based System**    | The chatbot does not generate new answers using a language model. |
| **Dataset Dependency**        | Response quality depends heavily on the knowledge base.           |
| **Semantic Limitations**      | Synonyms and paraphrases may receive lower similarity scores.     |
| **Manual Thresholds**         | Confidence thresholds are heuristic and manually selected.        |
| **No Conversational Context** | Each question is processed independently.                         |
| **Exact Small-Talk Matching** | Small-talk detection relies on exact trigger strings.             |

For example:

```text
hello
Hello!
hey there
good morning
```

may not all behave identically because the current small-talk system uses exact matching rather than semantic understanding.

---

# 🛠️ Technologies Used

| Technology          | Purpose                                           |
| ------------------- | ------------------------------------------------- |
| 🐍 **Python**       | Core programming language                         |
| 📝 **NLTK**         | Tokenization, stopword removal, and lemmatization |
| 🐼 **Pandas**       | Dataset loading and result analysis               |
| 🤖 **Scikit-learn** | TF-IDF vectorization and cosine similarity        |
| 🔢 **NumPy**        | Numerical operations                              |
| 📊 **Matplotlib**   | Data visualization                                |
| 🚀 **Streamlit**    | Interactive web interface and analytics dashboard |

---

# 🔮 Future Improvements

* 🧠 Replace TF-IDF with **semantic embeddings** such as Sentence-BERT.
* 💬 Add fuzzy or semantic small-talk detection.
* 🗣️ Support conversational context.
* 🎯 Improve confidence calibration using a labelled evaluation dataset.
* 📊 Add a proper train/test evaluation pipeline.
* 📈 Introduce **Precision, Recall, and F1-score** evaluation.
* 🛠️ Add an administrator interface for updating the knowledge base.
* 💾 Persist query logs instead of storing them only in Streamlit session state.
* 🌍 Add multilingual NLP support.
* 🔀 Build a hybrid retrieval system combining lexical and semantic similarity.

---

# 🎓 Conclusion

The **Academic NLP Chatbot** demonstrates how a lightweight question-answering system can be built without relying on a large language model.

The project combines:

> **Text Preprocessing → TF-IDF → Cosine Similarity → Confidence Classification → Interactive Analytics**

This creates an interpretable academic question-answering system while providing visibility into **similarity scores, confidence levels, matching results, and query performance**.

---

<p align="center">
  Built with ❤️ using <b>Python · NLTK · Scikit-learn · Streamlit</b>
</p>
