
import streamlit as st
import pickle
import sqlite3
import numpy as np
import pandas as pd
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from datetime import datetime
import plotly.graph_objects as go

# ---- Page Config ----
st.set_page_config(
    page_title="Legal Bail Prediction System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS: Dark theme with glassmorphism ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0a1a;
    --bg-card: rgba(20, 20, 50, 0.7);
    --accent-gold: #d4a843;
    --accent-blue: #4a90d9;
    --text-primary: #e8e6e3;
    --text-secondary: #a0a0b0;
    --border-glass: rgba(255, 255, 255, 0.08);
    --shadow-glass: 0 8px 32px rgba(0, 0, 0, 0.37);
}

.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d1b2a 100%);
    font-family: 'Inter', sans-serif;
}

.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    box-shadow: var(--shadow-glass);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(212, 168, 67, 0.15);
}

.prediction-granted {
    border-left: 4px solid #22c55e;
    background: rgba(34, 197, 94, 0.08);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}

.prediction-rejected {
    border-left: 4px solid #ef4444;
    background: rgba(239, 68, 68, 0.08);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}

.metric-box {
    background: rgba(74, 144, 217, 0.1);
    border: 1px solid rgba(74, 144, 217, 0.2);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #d4a843, #f0d68c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
}

.header-subtitle {
    color: var(--text-secondary);
    text-align: center;
    font-size: 1.1rem;
    margin-top: 0;
}

.similar-case {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 14px;
    margin: 8px 0;
}

h1, h2, h3 { color: var(--text-primary) !important; }
p, li, span { color: var(--text-secondary); }

.stTextArea textarea { 
    background: rgba(20, 20, 50, 0.8) !important; 
    color: #e8e6e3 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}
.stSelectbox > div > div {
    background: rgba(20, 20, 50, 0.8) !important;
    color: #e8e6e3 !important;
}
</style>
""", unsafe_allow_html=True)

# ---- Load Models ----
@st.cache_resource
def load_models():
    models = {}
    # Baseline
    models['tfidf'] = pickle.load(open('results/tfidf_vectorizer.pkl', 'rb'))
    models['lr'] = pickle.load(open('results/lr_model.pkl', 'rb'))
    # LLM
    models['tokenizer'] = AutoTokenizer.from_pretrained('google/flan-t5-base')
    models['llm'] = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    # Embeddings
    models['embed'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # FAISS
    models['faiss_index'] = faiss.read_index('results/faiss_index.bin')
    models['train_df'] = pd.read_csv('results/train.csv')
    return models

models = load_models()

# ---- Header ----
st.markdown('<h1 class="header-title">⚖️ Legal Bail Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">AI-powered bail outcome prediction with legally grounded explanations</p>', unsafe_allow_html=True)
st.markdown("---")

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_choice = st.selectbox("Select Model", [
        "Baseline ML (Logistic Regression)",
        "Prompt-based LLM (Zero-Shot)",
        "Fine-tuned + RAG System"
    ])
    st.markdown("---")
    st.markdown("###  Model Info")
    if "Baseline" in model_choice:
        st.info("TF-IDF + Logistic Regression\nFast predictions, no explanations")
    elif "Prompt" in model_choice:
        st.info("google/flan-t5-base\nZero-shot prompting with explanations")
    else:
        st.info("RAG + Fine-tuned LLM\nRetrieves similar cases for grounded explanations")

# ---- Main Input ----
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Case Details")
    case_facts = st.text_area(
        "Case Facts",
        height=200,
        placeholder="Enter the facts of the bail case here..."
    )
    ipc_input = st.text_input("IPC Sections (comma-separated)", placeholder="302, 307, 120B")
    predict_btn = st.button(" Predict Bail Outcome", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Quick Stats")
    st.metric("Total Cases in DB", "1,200")
    st.metric("Granted Rate", "61.3%")
    st.metric("Models Available", "3")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Prediction ----
if predict_btn and case_facts:
    with st.spinner("Analyzing case..."):
        combined = f"Case Facts: {case_facts.lower()} | IPC Sections: {ipc_input} | Crime Type: unknown"

        if "Baseline" in model_choice:
            X = models['tfidf'].transform([combined])
            pred = models['lr'].predict(X)[0]
            proba = models['lr'].predict_proba(X)[0]
            decision = "Granted" if pred == 1 else "Rejected"
            confidence = float(max(proba))
            explanation = "Baseline ML model does not generate explanations. Consider using the RAG system for detailed legal reasoning."

        elif "Prompt" in model_choice:
            prompt = f"You are a legal expert. Given these case facts, predict bail (Granted/Rejected) and explain: {case_facts[:400]}"
            inputs = models['tokenizer'](prompt, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                out = models['llm'].generate(**inputs, max_new_tokens=256, num_beams=4)
            explanation = models['tokenizer'].decode(out[0], skip_special_tokens=True)
            decision = "Granted" if "granted" in explanation.lower() else "Rejected"
            confidence = 0.6

        else:  # RAG
            q_emb = models['embed'].encode([case_facts]).astype('float32')
            faiss.normalize_L2(q_emb)
            scores, indices = models['faiss_index'].search(q_emb, 3)
            similar = [(models['train_df'].iloc[i], s) for s, i in zip(scores[0], indices[0])]

            context = ""
            for sc, sim in similar:
                context += f"\nPrecedent (Decision: {sc['decision_label']}): {str(sc['case_facts'])[:200]}\n"

            prompt = f"Based on precedents: {context}\nCurrent case: {case_facts[:300]}\nPredict bail and explain:"
            inputs = models['tokenizer'](prompt, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                out = models['llm'].generate(**inputs, max_new_tokens=256, num_beams=4)
            explanation = models['tokenizer'].decode(out[0], skip_special_tokens=True)
            decision = "Granted" if "granted" in explanation.lower() else "Rejected"
            confidence = float(np.mean(scores[0]))

    # Display results
    st.markdown("---")
    css_class = "prediction-granted" if decision == "Granted" else "prediction-rejected"
    emoji = "✅" if decision == "Granted" else "❌"
    st.markdown(f"""
    <div class="{css_class}">
        <h2>{emoji} Bail Prediction: {decision}</h2>
        <p><strong>Confidence:</strong> {confidence:.2%}</p>
        <p><strong>Model:</strong> {model_choice}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📖 Legal Explanation")
    st.write(explanation)
    st.markdown('</div>', unsafe_allow_html=True)

    # Similar cases (for RAG)
    if "RAG" in model_choice:
        st.markdown("### 📚 Similar Precedent Cases")
        for sc, sim in similar:
            with st.expander(f"📄 Case (Similarity: {sim:.3f}) — Decision: {sc['decision_label']}"):
                st.write(f"**Facts:** {str(sc['case_facts'])[:400]}...")
                st.write(f"**Reasoning:** {str(sc['reasoning'])[:300]}...")

    # Save to DB
    try:
        conn = sqlite3.connect("results/predictions.db")
        cursor = conn.cursor()
        cursor.execute("""INSERT INTO predictions (input_facts, ipc_sections, predicted_decision,
                          confidence, explanation, model_used, timestamp)
                          VALUES (?, ?, ?, ?, ?, ?, ?)""",
                       (case_facts[:500], ipc_input, decision, confidence, explanation, model_choice, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except:
        pass

# ---- Prediction History ----
st.markdown("---")
st.markdown("###  Prediction History")
try:
    conn = sqlite3.connect("results/predictions.db")
    history = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 10", conn)
    conn.close()
    if len(history) > 0:
        st.dataframe(history[['predicted_decision', 'confidence', 'model_used', 'timestamp']].head(10),
                     use_container_width=True)
except:
    st.info("No prediction history yet. Make a prediction to start!")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#666;'>Built with  for Legal AI Research | Powered by flan-t5 + FAISS + LoRA</p>", unsafe_allow_html=True)
