# ⚖️ Legal Bail Prediction — Grounded Legal Reasoning System

A hybrid AI system for predicting bail outcomes from Indian court case texts, combining classical ML, LoRA-fine-tuned BERT, and a Retrieval-Augmented Generation (RAG) pipeline for explainable, grounded legal reasoning.

---

## 🚀 Features

- **Bail Outcome Prediction** — Ensemble of TF-IDF + Logistic Regression and LoRA-tuned BERT
- **IPC Section Classification** — Predicts relevant IPC sections from case text
- **RAG-Based Explanation** — Retrieves similar precedents and generates grounded legal explanations
- **Evaluation Metrics** — BLEU, ROUGE, BERTScore for generation quality
- **Interactive Streamlit UI** — User-friendly dashboard for case input and result visualization

---

## 🗂️ Project Structure

```
├── notebooks/
│   └── legal_bail_prediction_rag_system.ipynb   # Full pipeline notebook
├── app/
│   └── streamlit_app.py                          # Streamlit dashboard
├── REPORT.docx                                   # Project report
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Classical ML | Scikit-learn (TF-IDF + LogReg) |
| Deep Learning | HuggingFace Transformers + PEFT (LoRA) |
| RAG Retrieval | FAISS / Sentence Transformers |
| Generation | LLM-based explanation module |
| Frontend | Streamlit |
| Evaluation | BLEU, ROUGE, BERTScore |

---

## ⚙️ Setup & Usage

```bash
# 1. Clone the repo
git clone https://github.com/Kushagra-Pachauri/Legal-Bail-Prediction-Grounded-Legal-Reasoning-System-.git
cd Legal-Bail-Prediction-Grounded-Legal-Reasoning-System-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app/streamlit_app.py
```

---

## 📓 Notebook

Open `notebooks/legal_bail_prediction_rag_system.ipynb` to explore the full training, evaluation, and RAG pipeline.

---

## 📄 License

This project is for academic purposes.
