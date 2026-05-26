# ⚖️ Legal Bail Prediction — Grounded Legal Reasoning System

> An end-to-end AI system for predicting bail outcomes from Indian court case texts.  
> Combines classical ML, LoRA-fine-tuned Legal-BERT, RAG-based precedent retrieval, and LLM-generated explanations — all grounded in real case law.

---

## 🧠 What This System Does

Given the **facts of a bail case** and the **IPC sections** involved, this system:

1. **Predicts** whether bail will be **Granted or Rejected** using an ensemble of trained models
2. **Retrieves** the most similar past cases from a FAISS vector database (legal precedents)
3. **Re-ranks** retrieved cases using a cross-encoder for precision
4. **Generates** a legally grounded natural language explanation using `flan-t5-base` + RAG
5. **Logs** every prediction to a SQLite database for history and audit

The key design decision: **RAG is used only for explanation quality, not for the prediction itself.** Prediction is handled by the ensemble model for higher accuracy.

---

## 🗂️ Dataset

- **1,200 Indian Bail Judgments** (structured CSV)
- Fields: `case_facts`, `ipc_sections`, `crime_type`, `decision_label` (Granted/Rejected), `reasoning`
- Split: Train / Validation / Test

---

## 🏗️ System Architecture

```
Input: Case Facts + IPC Sections
         │
         ▼
┌─────────────────────────────────────────────┐
│           PREDICTION LAYER                  │
│                                             │
│  TF-IDF + Logistic Regression  (60% weight) │
│         +                                   │
│  LoRA-tuned legal-bert-base-uncased (40%)   │
│         =                                   │
│       Ensemble (Best Accuracy)              │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│           EXPLANATION LAYER (RAG)           │
│                                             │
│  1. Encode query → all-MiniLM-L6-v2         │
│  2. FAISS vector search → Top-K precedents  │
│  3. Cross-encoder re-ranking (ms-marco)     │
│  4. Build prompt with retrieved context     │
│  5. flan-t5-base generates explanation      │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│           STORAGE LAYER (Dual DB)           │
│                                             │
│  FAISS  — vector index (precedent search)   │
│  SQLite — structured prediction history     │
└─────────────────────────────────────────────┘
         │
         ▼
      Streamlit Dashboard (Dark UI)
```

---

## 🤖 Models Used

### Prediction Models
| Model | Role | Notes |
|-------|------|-------|
| `TF-IDF + Logistic Regression` | Baseline classifier | 60% ensemble weight |
| `Linear SVM` | Baseline classifier | Evaluated independently |
| `nlpaueb/legal-bert-base-uncased` + LoRA (PEFT) | Fine-tuned classifier | 40% ensemble weight |
| **Ensemble (LR + LoRA-BERT)** | **Final prediction** | **Best accuracy** |

### Explanation / Generation
| Model | Role |
|-------|------|
| `google/flan-t5-base` | Seq2Seq LLM for explanation generation |
| `sentence-transformers/all-MiniLM-L6-v2` | Query & document embeddings for RAG |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranking retrieved precedents |

### Prompting Strategies Evaluated
- **Zero-Shot** — Direct prompt to flan-t5-base
- **Few-Shot** — 4 hand-picked examples (2 Granted, 2 Rejected)
- **Chain-of-Thought (CoT)** — Step-by-step legal reasoning prompt

---

## 🗄️ Dual Database Architecture

This system uses **two databases** for different purposes:

### 1. FAISS (Vector Database)
- Stores dense embeddings of all training case facts
- Used for **semantic similarity search** to find precedent cases
- `faiss.IndexFlatIP` with L2-normalized vectors
- File: `results/models/faiss_index.bin`

### 2. SQLite (Structured Database)
- Stores every prediction made through the Streamlit UI
- Schema: `input_facts`, `ipc_sections`, `predicted_decision`, `confidence`, `explanation`, `model_used`, `timestamp`
- File: `results/predictions.db`
- Enables prediction history, auditing, and analytics

---

## 📊 Evaluation Metrics

### Prediction Quality
- **Accuracy** and **Macro F1-Score** across all models

### Explanation Quality
| Metric | What it measures |
|--------|-----------------|
| **ROUGE-L** | Lexical overlap with ground-truth reasoning |
| **BERTScore-F1** | Semantic similarity to reference explanations |
| **NLI Entailment Score** | Factual consistency / hallucination detection |

---

## 🛠️ Full Tech Stack

| Category | Technology |
|----------|-----------|
| Classical ML | Scikit-learn — TF-IDF, Logistic Regression, Linear SVM |
| Fine-tuning | HuggingFace `transformers` + `peft` (LoRA on Legal-BERT) |
| Training acceleration | `accelerate`, `bitsandbytes` |
| Generation LLM | `google/flan-t5-base` (Seq2Seq) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector DB | FAISS (`faiss-cpu`) |
| Structured DB | SQLite (`sqlite3`) |
| Evaluation | `rouge-score`, `bert-score`, NLI-based consistency |
| Visualization | `plotly` (radar charts, comparison plots), `matplotlib`, `seaborn` |
| Frontend | Streamlit (dark glassmorphism UI) |
| Language | Python 3.10+ |

---

## 🗂️ Project Structure

```
├── notebooks/
│   └── legal_bail_prediction_rag_system.ipynb   # Full pipeline (training + evaluation)
├── app/
│   └── streamlit_app.py                          # Interactive Streamlit dashboard
├── REPORT.docx                                   # Project report
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Usage

```bash
# 1. Clone the repo
git clone https://github.com/Kushagra-Pachauri/Legal-Bail-Prediction-Grounded-Legal-Reasoning-System-.git
cd Legal-Bail-Prediction-Grounded-Legal-Reasoning-System-

# 2. Install dependencies
pip install transformers>=4.40.0 peft>=0.10.0 accelerate>=0.30.0 bitsandbytes>=0.43.0 \
            torch>=2.2.0 sentence-transformers>=2.7.0 faiss-cpu>=1.8.0 \
            scikit-learn>=1.4.0 pandas>=2.2.0 numpy>=1.26.0 nltk>=3.8.0 \
            rouge-score>=0.1.2 bert-score>=0.3.13 plotly>=5.22.0 streamlit>=1.35.0

# 3. Run the full training pipeline (Kaggle recommended — GPU T4)
jupyter notebook notebooks/legal_bail_prediction_rag_system.ipynb

# 4. Launch the Streamlit dashboard
streamlit run app/streamlit_app.py
```

> **Recommended:** Run the notebook on Kaggle with a GPU T4 accelerator for LoRA fine-tuning.

---

## 🔑 Key Design Decisions

- **LoRA over full fine-tuning** — Parameter-efficient, avoids catastrophic forgetting. Training loss dropped from >80 (seq2seq) to <1 (classification head).
- **Ensemble over single model** — LR (60%) + LoRA-BERT (40%) outperforms either model alone.
- **RAG for explanation only** — Keeps prediction fast and accurate; RAG adds grounding to the generated text, not noise to the prediction.
- **Cross-encoder re-ranking** — Bi-encoder FAISS retrieval is fast but imprecise; cross-encoder re-scores top-K results for better precedent quality.
- **NLI hallucination check** — Measures whether the generated explanation is factually entailed by the retrieved context.

---

## 📄 License

This project is for academic/research purposes.
