"""
SpamGuard - Email / SMS Spam Classification dashboard
-----------------------------------------------------
Uses exactly the artefacts produced by the notebook (sms-spam-detection.ipynb):
    model.pkl       -> trained Multinomial Naive Bayes
    vectorizer.pkl  -> TF-IDF vectorizer (max_features=3000)
    spam.csv        -> dataset (used for the analytics / evaluation tabs)

The evaluation uses the SAME train/test split as the notebook
(test_size=0.2, random_state=2), so the numbers match the notebook.
"""

from collections import Counter
from pathlib import Path
import pickle
import re
import string

import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score,
    precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------ CONFIG
st.set_page_config(
    page_title="SpamGuard | Message Intelligence",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
DATA_PATH = BASE_DIR / "spam.csv"

SPAM_COLOR = "#ef4444"
HAM_COLOR = "#22c55e"
ACCENT = "#818cf8"

# --------------------------------------------------------------------- CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #080d1b 0%, #101a30 55%, #101827 100%); color: #f1f5f9; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1300px; }
.hero { padding: 28px 30px; border: 1px solid rgba(99,102,241,.35); border-radius: 22px;
        background: linear-gradient(120deg,#172554,#312e81,#172554); margin-bottom: 18px; }
.hero h1 { font-size: 36px; font-weight: 800; color: white; margin: 0 0 6px 0; }
.hero p { color: #c7d2fe; font-size: 15px; margin: 2px 0; }
.section-title { font-size: 21px; font-weight: 700; margin: 18px 0 8px 0; color: #e2e8f0; }
div[data-testid="stMetric"] { background: rgba(30,41,59,.75); border: 1px solid #334155;
        padding: 14px 16px; border-radius: 16px; }
div[data-testid="stMetricLabel"] { color: #94a3b8; }
div[data-testid="stMetricValue"] { color: #f8fafc; }
textarea { background-color: #111c32 !important; color: white !important; border-radius: 14px !important; }
.stButton > button { width: 100%; border-radius: 12px; height: 46px; font-weight: 700;
        background: linear-gradient(90deg,#6366f1,#8b5cf6); color: white; border: none; }
.stButton > button:hover { border: 1px solid #a5b4fc; color: white; }
.result-card { padding: 22px; border-radius: 18px; margin-top: 14px;
        background: rgba(30,41,59,.8); border: 1px solid #475569; }
.result-card.spam { border-left: 6px solid #ef4444; }
.result-card.ham { border-left: 6px solid #22c55e; }
.result-card h2 { margin: 0 0 6px 0; }
.small-note { color: #94a3b8; font-size: 13px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------- RESOURCE LOADING
@st.cache_resource
def load_nlp():
    """Stop-words + stemmer (downloads NLTK data on first run if missing)."""
    for pkg, path in [
        ("stopwords", "corpora/stopwords"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), PorterStemmer()


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def preprocess_steps(text, stop_words, stemmer):
    """Same logic as transform_text() in the notebook, returning every stage."""
    lowered = text.lower()
    tokens = word_tokenize(lowered)
    alnum = [t for t in tokens if t.isalnum()]
    no_stop = [t for t in alnum if t not in stop_words and t not in string.punctuation]
    stemmed = [stemmer.stem(t) for t in no_stop]
    return {
        "1. Lowercase": lowered,
        "2. Tokenize": tokens,
        "3. Keep alphanumeric only": alnum,
        "4. Remove stop-words": no_stop,
        "5. Stemming (final)": stemmed,
    }


def transform_text(text, stop_words, stemmer):
    return " ".join(preprocess_steps(text, stop_words, stemmer)["5. Stemming (final)"])


# ------------------------------------------------------------ DATA & STATS
@st.cache_data(show_spinner=False)
def load_data():
    """Reproduces the notebook's cleaning: drop extra cols, encode, drop duplicates."""
    stop_words, stemmer = load_nlp()
    df = pd.read_csv(DATA_PATH, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["target"] = (df["label"] == "spam").astype(int)  # ham=0, spam=1 (as LabelEncoder)
    df = df[["target", "text"]].drop_duplicates(keep="first").reset_index(drop=True)
    df["Class"] = df["target"].map({0: "Ham", 1: "Spam"})
    df["chars"] = df["text"].apply(len)
    df["words"] = df["text"].str.split().str.len()
    df["processed"] = df["text"].apply(lambda t: transform_text(t, stop_words, stemmer))
    return df


@st.cache_data(show_spinner=False)
def evaluate():
    """Evaluates the SAVED model on the notebook's test split (random_state=2)."""
    model, vec = load_model()
    df = load_data()
    X = vec.transform(df["processed"])
    y = df["target"].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "cm": confusion_matrix(y_test, pred),
        "fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr),
        "proba": proba, "y_test": y_test,
        "n_train": X.shape[0] - X_test.shape[0], "n_test": X_test.shape[0],
    }


@st.cache_data(show_spinner=False)
def compare_models():
    """Quick comparison of some of the notebook's classifiers on the same split."""
    _, vec = load_model()
    df = load_data()
    X = vec.transform(df["processed"])
    y = df["target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=2)
    models = {
        "Multinomial NB (chosen)": MultinomialNB(),
        "Bernoulli NB": BernoulliNB(),
        "Logistic Regression": LogisticRegression(solver="liblinear"),
        "SVC (sigmoid)": SVC(kernel="sigmoid", gamma=1.0),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=2),
        "Extra Trees": ExtraTreesClassifier(n_estimators=50, random_state=2),
    }
    rows = []
    for name, clf in models.items():
        clf.fit(X_tr, y_tr)
        p = clf.predict(X_te)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, p),
            "Precision": precision_score(y_te, p, zero_division=0),
            "Recall": recall_score(y_te, p, zero_division=0),
            "F1-score": f1_score(y_te, p, zero_division=0),
        })
    return pd.DataFrame(rows)


def top_words(df, target, n=15):
    counter = Counter(w for msg in df.loc[df["target"] == target, "processed"] for w in msg.split())
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])


def explain_prediction(model, vec, processed, top_n=10):
    """Naive Bayes log-odds contribution of each word: tfidf * (log P(w|spam) - log P(w|ham))."""
    X = vec.transform([processed]).toarray()[0]
    idx = np.nonzero(X)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["word", "contribution"])
    names = vec.get_feature_names_out()
    diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    out = pd.DataFrame({"word": names[idx], "contribution": X[idx] * diff[idx]})
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index)
    return out.head(top_n).sort_values("contribution")


def style(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# --------------------------------------------------------------- HEADER
st.markdown(
    """
<div class="hero">
    <h1>📧 SpamGuard</h1>
    <p>AI-powered message intelligence to detect spam and protect your inbox.</p>
    <p>NLP preprocessing &bull; TF-IDF features &bull; Multinomial Naive Bayes &bull; live analytics</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------- FILE CHECKS
missing = [p.name for p in (MODEL_PATH, VECTORIZER_PATH, DATA_PATH) if not p.exists()]
if missing:
    st.error(f"Missing file(s): {', '.join(missing)}. Keep them in the same folder as app.py.")
    st.stop()

try:
    stop_words, stemmer = load_nlp()
    model, vectorizer = load_model()
except Exception as e:
    st.error(f"Unable to load model resources: {e}")
    st.stop()

if not hasattr(model, "classes_"):
    st.error(
        "model.pkl contains an UNTRAINED model. Open the notebook, run the last cell "
        "(mnb.fit(...) then pickle.dump) after 'Restart & Run All', then reload this page."
    )
    st.stop()

with st.spinner("Preparing dataset and evaluating the model (first run only)..."):
    data = load_data()
    ev = evaluate()

# -------------------------------------------------------------- SIDEBAR
with st.sidebar:
    st.markdown("## 📧 SpamGuard")
    st.caption("Message Intelligence System")
    st.divider()
    st.markdown("### Model Information")
    st.write("**Algorithm:** Multinomial Naive Bayes")
    st.write("**Vectorization:** TF-IDF")
    st.write(f"**Features:** {len(vectorizer.vocabulary_):,}")
    st.write("**Dataset:** SMS Spam Collection")
    st.divider()
    st.markdown("### Dataset")
    st.write(f"**Messages (after dedupe):** {len(data):,}")
    st.write(f"**Train / Test:** {ev['n_train']:,} / {ev['n_test']:,}")
    st.write(f"**Spam share:** {data['target'].mean():.1%}")
    st.divider()
    st.markdown("### Test performance")
    st.write(f"**Accuracy:** {ev['accuracy']:.2%}")
    st.write(f"**Precision:** {ev['precision']:.2%}")
    st.caption("Academic Project | Email Spam Classification")

# ----------------------------------------------------------------- TABS
tab_an, tab_perf, tab_data, tab_cmp, tab_about = st.tabs(
    ["🔍 Analyzer", "📈 Model Performance", "📊 Dataset Insights", "⚖️ Model Comparison", "ℹ️ How it works"]
)

# =========================================================== TAB 1: ANALYZER
with tab_an:
    st.session_state.setdefault("history", [])

    def set_sample(text):
        st.session_state["message"] = text

    c1, c2 = st.columns([1.6, 1])
    with c1:
        message = st.text_area(
            "Enter your message",
            key="message",
            height=200,
            placeholder="Paste your email or message here...\n\nExample: Congratulations! You have won a free prize. Click here now!",
        )
        analyze = st.button("🚀 Analyze Message")
    with c2:
        st.markdown("### 💡 Try a sample")
        st.button(
            "🚨 Spam Example", on_click=set_sample,
            args=("Congratulations! You have won a free prize. Click here to claim your reward now!",),
        )
        st.button(
            "✅ Normal Example", on_click=set_sample,
            args=("Hey, are we still meeting at the library at 4 pm?",),
        )
        st.info("Tip: try prize claims, urgent requests, suspicious links or promotional offers.")

    if analyze:
        if not message.strip():
            st.warning("Please enter a message before analyzing.")
        else:
            processed = transform_text(message, stop_words, stemmer)
            if not processed.strip():
                st.warning("The message contains no usable words after preprocessing.")
            else:
                vec_in = vectorizer.transform([processed])
                prediction = int(model.predict(vec_in)[0])
                probs = model.predict_proba(vec_in)[0]
                p_spam = float(probs[list(model.classes_).index(1)])
                p_ham = float(probs[list(model.classes_).index(0)])
                confidence = max(p_spam, p_ham)
                risk = "High" if p_spam >= 0.7 else "Medium" if p_spam >= 0.3 else "Low"

                st.session_state["history"].append({
                    "Message": (message[:70] + "...") if len(message) > 70 else message,
                    "Verdict": "SPAM" if prediction == 1 else "HAM",
                    "Spam probability": f"{p_spam:.1%}",
                })

                st.markdown('<div class="section-title">📊 Analysis Results</div>', unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown(
                        '<div class="result-card spam"><h2 style="color:#f87171;">🚨 SPAM DETECTED</h2>'
                        "<p>This message has been classified as potentially unwanted or spam.</p></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-card ham"><h2 style="color:#4ade80;">✅ NOT SPAM</h2>'
                        "<p>This message has been classified as a normal message (Ham).</p></div>",
                        unsafe_allow_html=True,
                    )
                st.write("")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Prediction", "SPAM" if prediction == 1 else "HAM")
                m2.metric("Confidence", f"{confidence:.1%}")
                m3.metric("Risk level", risk)
                m4.metric("Characters", len(message))
                m5.metric("Words", len(message.split()))

                g1, g2 = st.columns(2)
                with g1:
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=p_spam * 100,
                        number={"suffix": "%"},
                        title={"text": "Spam probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": SPAM_COLOR if prediction == 1 else HAM_COLOR},
                            "steps": [
                                {"range": [0, 30], "color": "rgba(34,197,94,.20)"},
                                {"range": [30, 70], "color": "rgba(250,204,21,.20)"},
                                {"range": [70, 100], "color": "rgba(239,68,68,.20)"},
                            ],
                        },
                    ))
                    st.plotly_chart(style(gauge, 330), key="gauge")
                with g2:
                    bar = px.bar(
                        pd.DataFrame({"Category": ["Not Spam (Ham)", "Spam"], "Probability": [p_ham, p_spam]}),
                        x="Category", y="Probability", color="Category",
                        color_discrete_map={"Not Spam (Ham)": HAM_COLOR, "Spam": SPAM_COLOR},
                        text_auto=".1%", title="Class probabilities",
                    )
                    bar.update_layout(yaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False)
                    st.plotly_chart(style(bar, 330), key="probbar")

                st.markdown('<div class="section-title">🧠 Why this prediction?</div>', unsafe_allow_html=True)
                expl = explain_prediction(model, vectorizer, processed)
                if expl.empty:
                    st.info("None of the message's words are in the model vocabulary.")
                else:
                    expl["direction"] = np.where(expl["contribution"] > 0, "pushes towards SPAM", "pushes towards HAM")
                    why = px.bar(
                        expl, x="contribution", y="word", orientation="h", color="direction",
                        color_discrete_map={"pushes towards SPAM": SPAM_COLOR, "pushes towards HAM": HAM_COLOR},
                        title="Most influential words (Naive Bayes log-odds contribution)",
                    )
                    why.update_layout(legend_title_text="")
                    st.plotly_chart(style(why, 380), key="why")
                    st.caption("Contribution = TF-IDF weight x (log P(word | spam) - log P(word | ham)). "
                               "Positive values push the message towards spam.")

                with st.expander("🔬 View preprocessing pipeline (step by step)"):
                    for step, value in preprocess_steps(message, stop_words, stemmer).items():
                        st.markdown(f"**{step}**")
                        st.code(value if isinstance(value, str) else " | ".join(value) or "(empty)")

    if st.session_state["history"]:
        st.markdown('<div class="section-title">🕘 Session history</div>', unsafe_allow_html=True)
        hist = pd.DataFrame(st.session_state["history"][::-1])
        st.dataframe(hist, hide_index=True)
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.rerun()

# ================================================= TAB 2: MODEL PERFORMANCE
with tab_perf:
    st.markdown('<div class="section-title">Performance on the held-out test set</div>', unsafe_allow_html=True)
    st.caption(f"Same split as the notebook: {ev['n_train']:,} training / {ev['n_test']:,} test messages "
               "(test_size=0.2, random_state=2).")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accuracy", f"{ev['accuracy']:.2%}")
    k2.metric("Precision", f"{ev['precision']:.2%}")
    k3.metric("Recall", f"{ev['recall']:.2%}")
    k4.metric("F1-score", f"{ev['f1']:.2%}")
    k5.metric("ROC AUC", f"{ev['auc']:.3f}")

    p1, p2 = st.columns(2)
    with p1:
        cm = ev["cm"]
        heat = go.Figure(go.Heatmap(
            z=cm, x=["Predicted Ham", "Predicted Spam"], y=["Actual Ham", "Actual Spam"],
            colorscale=[[0, "#1e1b4b"], [1, "#6366f1"]], texttemplate="%{z}",
            textfont={"size": 22}, showscale=False,
        ))
        heat.update_layout(title="Confusion matrix", yaxis=dict(autorange="reversed"))
        st.plotly_chart(style(heat, 380), key="cm")
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"✔ {tn} ham correctly kept  |  ✘ {fp} ham wrongly flagged as spam  |  "
                   f"✘ {fn} spam missed  |  ✔ {tp} spam caught")
    with p2:
        roc = go.Figure()
        roc.add_trace(go.Scatter(x=ev["fpr"], y=ev["tpr"], mode="lines", fill="tozeroy",
                                 name=f"Naive Bayes (AUC = {ev['auc']:.3f})", line=dict(color=ACCENT, width=3)))
        roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                 line=dict(color="#64748b", dash="dash")))
        roc.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate",
                          legend=dict(x=0.45, y=0.08))
        st.plotly_chart(style(roc, 380), key="roc")

    dist = pd.DataFrame({
        "Spam probability": ev["proba"],
        "Actual class": np.where(ev["y_test"] == 1, "Spam", "Ham"),
    })
    hist_fig = px.histogram(
        dist, x="Spam probability", color="Actual class", nbins=30, barmode="overlay", opacity=0.8,
        color_discrete_map={"Ham": HAM_COLOR, "Spam": SPAM_COLOR}, log_y=True,
        title="How confident is the model? (distribution of predicted spam probability, log scale)",
    )
    st.plotly_chart(style(hist_fig, 340), key="probdist")

    rep = pd.DataFrame({
        "Class": ["Ham (0)", "Spam (1)"],
        "Precision": [tn / (tn + fn) if (tn + fn) else 0, ev["precision"]],
        "Recall": [tn / (tn + fp) if (tn + fp) else 0, ev["recall"]],
        "Support": [int(tn + fp), int(fn + tp)],
    })
    rep["F1-score"] = 2 * rep["Precision"] * rep["Recall"] / (rep["Precision"] + rep["Recall"]).replace(0, np.nan)
    st.markdown("**Per-class report**")
    st.dataframe(rep.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}), hide_index=True)

# ==================================================== TAB 3: DATASET INSIGHTS
with tab_data:
    st.markdown('<div class="section-title">Exploring the SMS Spam Collection</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Messages", f"{len(data):,}")
    d2.metric("Ham", f"{(data['target'] == 0).sum():,}")
    d3.metric("Spam", f"{(data['target'] == 1).sum():,}")
    d4.metric("Avg. length (chars)", f"{data['chars'].mean():.0f}")

    e1, e2 = st.columns(2)
    with e1:
        counts = data["Class"].value_counts()
        donut = go.Figure(go.Pie(
            labels=counts.index, values=counts.values, hole=0.55,
            marker=dict(colors=[HAM_COLOR if l == "Ham" else SPAM_COLOR for l in counts.index]),
        ))
        donut.update_layout(title="Class distribution (imbalanced data)")
        st.plotly_chart(style(donut, 340), key="donut")
    with e2:
        lenfig = px.histogram(
            data, x="chars", color="Class", nbins=60, barmode="overlay", opacity=0.75,
            color_discrete_map={"Ham": HAM_COLOR, "Spam": SPAM_COLOR},
            title="Message length (characters): spam messages are longer",
        )
        st.plotly_chart(style(lenfig, 340), key="lenhist")

    e3, e4 = st.columns(2)
    with e3:
        box = px.box(
            data, x="Class", y="words", color="Class",
            color_discrete_map={"Ham": HAM_COLOR, "Spam": SPAM_COLOR}, title="Words per message",
        )
        box.update_layout(showlegend=False)
        st.plotly_chart(style(box, 340), key="wordbox")
    with e4:
        stats = data.groupby("Class")[["chars", "words"]].mean().round(1).reset_index()
        stats.columns = ["Class", "Avg characters", "Avg words"]
        st.markdown("**Average length by class**")
        st.dataframe(stats, hide_index=True)
        st.caption("These length differences are why message length is a useful signal in the notebook's EDA.")

    w1, w2 = st.columns(2)
    for col, target, color, name in ((w1, 1, SPAM_COLOR, "spam"), (w2, 0, HAM_COLOR, "ham")):
        tw = top_words(data, target).sort_values("count")
        fig = px.bar(tw, x="count", y="word", orientation="h", title=f"Top 15 words in {name} messages (stemmed)")
        fig.update_traces(marker_color=color)
        with col:
            st.plotly_chart(style(fig, 460), key=f"top_{name}")

# ==================================================== TAB 4: MODEL COMPARISON
with tab_cmp:
    st.markdown('<div class="section-title">Why Multinomial Naive Bayes?</div>', unsafe_allow_html=True)
    st.caption("Several of the notebook's classifiers, trained live on the same TF-IDF features and the same "
               "train/test split. (The notebook compares 11 models; these are a representative subset.)")
    with st.spinner("Training comparison models (first run only)..."):
        comp = compare_models()
    long = comp.melt(id_vars="Model", var_name="Metric", value_name="Score")
    cmp_fig = px.bar(
        long, x="Model", y="Score", color="Metric", barmode="group",
        title="Accuracy / Precision / Recall / F1 by model",
        color_discrete_sequence=["#6366f1", "#22c55e", "#f59e0b", "#ef4444"],
    )
    cmp_fig.update_layout(yaxis=dict(range=[0.4, 1.02]))
    st.plotly_chart(style(cmp_fig, 430), key="cmp")

    radar = go.Figure()
    for _, r in comp.iterrows():
        vals = [r["Accuracy"], r["Precision"], r["Recall"], r["F1-score"]]
        radar.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=["Accuracy", "Precision", "Recall", "F1-score", "Accuracy"],
                                        name=r["Model"], fill="toself", opacity=0.35))
    radar.update_layout(title="Radar comparison", polar=dict(radialaxis=dict(range=[0.4, 1])))
    st.plotly_chart(style(radar, 460), key="radar")

    st.dataframe(
        comp.sort_values("Precision", ascending=False).style.format(
            {c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1-score"]}),
        hide_index=True,
    )
    st.info("In a spam filter, **precision** is critical: a false positive means a real message is lost. "
            "Naive Bayes reaches (near) perfect precision with high accuracy, which is why it was selected.")

# ========================================================== TAB 5: HOW IT WORKS
with tab_about:
    st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.info("**1. Clean**\n\nlowercase, tokenize, keep alphanumeric")
    a2.info("**2. Filter**\n\nremove stop-words & punctuation")
    a3.info("**3. Stem**\n\nPorter stemmer (loving → love)")
    a4.info("**4. Vectorize**\n\nTF-IDF, 3000 features")
    a5.info("**5. Classify**\n\nMultinomial Naive Bayes")

    st.markdown('<div class="section-title">Project mapping</div>', unsafe_allow_html=True)
    st.table(pd.DataFrame({
        "File": ["sms-spam-detection.ipynb", "model.pkl", "vectorizer.pkl", "spam.csv", "app.py"],
        "Role": [
            "Cleaning, EDA, preprocessing, model comparison, training and saving",
            "Trained Multinomial Naive Bayes classifier",
            "Fitted TF-IDF vectorizer (3000 features)",
            "SMS Spam Collection dataset (5,572 rows, 5,169 after removing duplicates)",
            "This dashboard: prediction, explanation and live analytics",
        ],
    }))
    st.markdown(
        "**Limitations:** the model is trained on short SMS text, so long emails or phishing messages "
        "may be classified less reliably; the dataset is imbalanced (about 87% ham / 13% spam)."
    )

# ----------------------------------------------------------------- FOOTER
st.divider()
st.markdown(
    """
<div style="text-align:center; padding:12px;">
    <p style="color:#94a3b8;">Built with Python &bull; Streamlit &bull; NLTK &bull; Scikit-learn &bull; Plotly</p>
    <p style="color:#64748b; font-size:12px;">Academic Project | Email Spam Classification</p>
</div>
""",
    unsafe_allow_html=True,
)