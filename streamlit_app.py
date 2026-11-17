"""
Assignment 5: Integration & Packaging - Customer Insight Dashboard

Tab 1  Customer Churn     (Assignment 1: logistic regression, tabular form)
Tab 2  Text Sentiment     (VADER lexicon, free-text input)
Tab 3  Audio Sentiment    (Assignment 4: MFCC + delta -> neural network, WAV upload)
Tab 4  Business Summary   (how the three pieces work together)

Dashboard structure follows the Week 11 tutorial (cached model loading,
sidebar-free tabs, matplotlib figures rendered with st.pyplot).
Run:  streamlit run streamlit_app.py
"""
import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Insight Dashboard", page_icon="📊", layout="wide")

CHURN_PATH = Path("churn_artifacts.joblib")
AUDIO_PATH = Path("audio_model.pt")
SAMPLE_DIR = Path("sample_audio")


# --------------------------------------------------------------------------
# Cached loaders: each returns None if the file is missing so the app never crashes
# --------------------------------------------------------------------------
@st.cache_resource
def load_churn():
    if not CHURN_PATH.exists():
        return None
    return joblib.load(CHURN_PATH)


@st.cache_resource
def load_audio_bundle():
    if not AUDIO_PATH.exists():
        return None
    try:
        import torch
        from model_utils import AudioSentimentNet

        ckpt = torch.load(AUDIO_PATH, map_location="cpu", weights_only=True)
        model = AudioSentimentNet(input_dim=ckpt["input_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return {
            "model": model,
            "mean": np.array(ckpt["mean"], dtype=np.float32),
            "std": np.array(ckpt["std"], dtype=np.float32),
            "labels": ckpt["label_names"],
            "metrics": ckpt.get("metrics", {}),
        }
    except Exception as exc:  # missing torch / python_speech_features, bad file...
        st.session_state["audio_load_error"] = str(exc)
        return None


@st.cache_resource
def load_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def show_metrics(metrics: dict):
    cols = st.columns(4)
    for col, key, name in zip(cols, ["accuracy", "precision", "recall", "f1"],
                              ["Accuracy", "Precision", "Recall", "F1"]):
        col.metric(name, f"{metrics.get(key, 0):.1%}")


# --------------------------------------------------------------------------
# Prediction functions (one per model)
# --------------------------------------------------------------------------
def predict_churn(art, values):
    """Run the Assignment 1 logistic regression. Returns (churn probability, scaled input row)."""
    row = pd.DataFrame([[values[f] for f in art["features"]]], columns=art["features"])
    scaled = art["scaler"].transform(row)
    prob = float(art["model"].predict_proba(scaled)[0, 1])
    return prob, scaled


def predict_text(analyzer, text):
    """Run VADER sentiment on free text. Returns the score dictionary."""
    return analyzer.polarity_scores(text)


def predict_audio(bundle, feats):
    """Run the Assignment 4 audio network. Returns (class probabilities, predicted class, feature importance)."""
    import torch

    x = torch.tensor((feats - bundle["mean"]) / bundle["std"], dtype=torch.float32).unsqueeze(0)
    x.requires_grad_(True)
    logits = bundle["model"](x)
    probs = torch.softmax(logits, dim=1)[0].detach().numpy()
    pred = int(np.argmax(probs))
    # Explainability: gradient x input for the predicted class
    logits[0, pred].backward()
    saliency = (x.grad * x).detach().numpy()[0]
    return probs, pred, saliency


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("📊 Customer Insight Dashboard")
st.markdown(
    "One place for a customer-success team to answer three questions: "
    "**Who is likely to leave? How does this customer feel (in writing)? "
    "How did this customer sound (on the phone)?**  "
    "Pick a tab below, enter data, and read the plain-English explanation next to each prediction."
)

tab_churn, tab_text, tab_audio, tab_about = st.tabs(
    ["1️⃣ Customer Churn", "2️⃣ Text Sentiment", "3️⃣ Audio Sentiment", "📘 Business Summary"]
)

# ==========================================================================
# TAB 1: CHURN
# ==========================================================================
with tab_churn:
    st.header("Customer Churn Risk")
    st.markdown(
        "Enter a customer's account details to estimate the probability they will leave. "
        "Use it to decide **who gets a retention offer first**."
    )
    art = load_churn()
    if art is None:
        st.warning("`churn_artifacts.joblib` not found. Run `python train_models.py` first (see README).")
    else:
        model, scaler, features, spec = art["model"], art["scaler"], art["features"], art["spec"]

        threshold = st.slider(
            "Risk threshold (flag a customer as 'high risk' above this probability)",
            0.1, 0.9, 0.5, 0.05,
            help="Lower = catch more churners but contact more customers who would have stayed.",
        )

        with st.form("churn_form"):
            st.subheader("Customer details")
            cols = st.columns(3)
            values = {}
            for i, feat in enumerate(features):
                s = spec[feat]
                with cols[i % 3]:
                    if s["type"] == "numeric":
                        if s["is_int"]:
                            values[feat] = st.number_input(
                                feat, min_value=int(s["min"]), max_value=int(s["max"]), value=int(s["median"]))
                        else:
                            values[feat] = st.number_input(
                                feat, min_value=float(s["min"]), max_value=float(s["max"]), value=float(s["median"]))
                    elif s["type"] == "binary":
                        values[feat] = 1 if st.selectbox(feat, ["No", "Yes"]) == "Yes" else 0
                    else:
                        choice = st.selectbox(feat, s["options"])
                        values[feat] = s["options"].index(choice)
            submitted = st.form_submit_button("Predict churn risk")

        if submitted:
            prob, scaled = predict_churn(art, values)

            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Churn probability", f"{prob:.1%}")
                if prob >= threshold:
                    st.error("🔴 HIGH RISK")
                    st.markdown("**Suggested action:** personal outreach with a retention offer "
                                "(discount, contract upgrade, service check-in).")
                elif prob >= threshold * 0.6:
                    st.warning("🟠 WATCH")
                    st.markdown("**Suggested action:** low-cost nudge, e.g. satisfaction survey or loyalty email.")
                else:
                    st.success("🟢 LOW RISK")
                    st.markdown("**Suggested action:** no intervention needed.")

            # Explainability: contribution of each feature to the log-odds of churn
            contrib = model.coef_[0] * scaled[0]
            order = np.argsort(np.abs(contrib))[::-1][:8][::-1]
            with c2:
                st.subheader("Why this prediction?")
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh([features[i] for i in order], contrib[order],
                        color=["#d62728" if contrib[i] > 0 else "#2ca02c" for i in order])
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("Effect on churn risk  (← lowers risk | raises risk →)")
                ax.set_title("Top drivers for this customer")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            up = [features[i] for i in np.argsort(contrib)[::-1][:3] if contrib[i] > 0]
            down = [features[i] for i in np.argsort(contrib)[:3] if contrib[i] < 0]
            if up:
                st.markdown("**Pushing risk up:** " + ", ".join(f"`{f}`" for f in up))
            if down:
                st.markdown("**Pulling risk down:** " + ", ".join(f"`{f}`" for f in down))
            st.caption("Each bar = model coefficient × the customer's standardized value. "
                       "Red bars raise the churn estimate, green bars lower it.")

        with st.expander("How reliable is this model? (held-out test set)"):
            show_metrics(art["metrics"])
            st.markdown(
                "Logistic regression trained on historical customers (80/20 split). "
                "Use the score to **prioritize** outreach, not as a final decision; "
                "precision tells you how many flagged customers really would leave, "
                "recall tells you how many leavers we catch."
            )

# ==========================================================================
# TAB 2: TEXT SENTIMENT
# ==========================================================================
with tab_text:
    st.header("Text Sentiment")
    st.markdown("Paste a customer review, survey comment, or support email to see its tone.")
    analyzer = load_vader()
    if analyzer is None:
        st.warning("`vaderSentiment` is not installed. Run `pip install -r requirements.txt`.")
    else:
        text = st.text_area("Customer text", height=140,
                            placeholder="e.g. The agent was friendly but my internet is still slow and I'm frustrated.")
        if st.button("Analyze text"):
            if not text.strip():
                st.info("Please type or paste some text first.")
            else:
                scores = predict_text(analyzer, text)
                compound = scores["compound"]
                label = "Positive 😊" if compound >= 0.05 else "Negative 😠" if compound <= -0.05 else "Neutral 😐"

                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("Overall sentiment", label)
                    st.metric("Score (-1 to +1)", f"{compound:+.2f}")
                    st.progress((compound + 1) / 2)
                    if compound <= -0.05:
                        st.markdown("**Suggested action:** route to a human agent / escalate.")
                    elif compound >= 0.05:
                        st.markdown("**Suggested action:** candidate for a testimonial or referral ask.")
                with c2:
                    # Explainability: which words carried the sentiment
                    words = [w.strip(".,!?;:\"'()").lower() for w in text.split()]
                    hits = [(w, analyzer.lexicon[w]) for w in words if w in analyzer.lexicon]
                    st.subheader("Which words drove the score?")
                    if hits:
                        hits_df = pd.DataFrame(hits, columns=["word", "valence"]).groupby("word", as_index=False).sum()
                        hits_df = hits_df.sort_values("valence")
                        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(hits_df))))
                        ax.barh(hits_df["word"], hits_df["valence"],
                                color=["#d62728" if v < 0 else "#2ca02c" for v in hits_df["valence"]])
                        ax.axvline(0, color="black", linewidth=0.8)
                        ax.set_xlabel("Word sentiment strength")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.write("No strongly emotional words were found, so the text reads as neutral.")
                st.caption("Method: VADER, a rule-based sentiment lexicon (not a trained neural network). "
                           "It handles negation, capitalization and punctuation emphasis, but can miss sarcasm.")

# ==========================================================================
# TAB 3: AUDIO SENTIMENT
# ==========================================================================
with tab_audio:
    st.header("Call-Center Audio Sentiment")
    st.markdown("Upload a short **.wav** recording of a customer call. The model flags negative-sounding calls "
                "so supervisors can review them first.")
    bundle = load_audio_bundle()
    if bundle is None:
        err = st.session_state.get("audio_load_error")
        st.warning("Audio model unavailable. Make sure `audio_model.pt` is present and the packages in "
                   "`requirements.txt` are installed." + (f"  \nDetails: {err}" if err else ""))
    else:
        uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
        samples = sorted(SAMPLE_DIR.glob("*.wav")) if SAMPLE_DIR.exists() else []
        sample_choice = None
        if samples:
            sample_choice = st.selectbox("…or try a bundled sample clip", ["(none)"] + [p.name for p in samples])

        audio_bytes = None
        if uploaded is not None:
            audio_bytes = uploaded.getvalue()
        elif sample_choice and sample_choice != "(none)":
            audio_bytes = (SAMPLE_DIR / sample_choice).read_bytes()

        if audio_bytes is not None:
            try:
                from model_utils import FEATURE_NAMES, load_audio, mfcc_delta_matrix

                st.audio(audio_bytes, format="audio/wav")
                sr, signal = load_audio(io.BytesIO(audio_bytes))
                mfcc_mat, combined = mfcc_delta_matrix(signal, sr)
                feats = combined.mean(axis=0).astype(np.float32)
                probs, pred, saliency = predict_audio(bundle, feats)
                names = bundle["labels"]

                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("Predicted sentiment", names[pred])
                    st.metric("Confidence", f"{probs[pred]:.1%}")
                    for n, p in zip(names, probs):
                        st.progress(float(p), text=f"{n}: {p:.1%}")
                    if names[pred].lower().startswith("neg"):
                        st.error("Flag for supervisor review.")
                    else:
                        st.success("No escalation needed.")
                    st.caption(f"Clip length: {len(signal) / sr:.1f} s at {sr} Hz")
                with c2:
                    fig, axes = plt.subplots(2, 1, figsize=(8, 5))
                    t = np.arange(len(signal)) / sr
                    axes[0].plot(t, signal, linewidth=0.5)
                    axes[0].set_title("Waveform")
                    axes[0].set_xlabel("Seconds")
                    im = axes[1].imshow(mfcc_mat.T, aspect="auto", origin="lower", cmap="viridis")
                    axes[1].set_title("MFCC features over time (what the model 'hears')")
                    axes[1].set_xlabel("Frame")
                    axes[1].set_ylabel("Coefficient")
                    fig.colorbar(im, ax=axes[1])
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                st.subheader("Which audio features drove this prediction?")
                top = np.argsort(np.abs(saliency))[::-1][:10][::-1]
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.barh([FEATURE_NAMES[i] for i in top], saliency[top],
                         color=["#2ca02c" if saliency[i] > 0 else "#d62728" for i in top])
                ax2.axvline(0, color="black", linewidth=0.8)
                ax2.set_xlabel(f"Push toward '{names[pred]}'  (green) / away from it (red)")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
                st.caption("MFCCs summarize the tone/timbre of the voice; Delta features capture how quickly it changes "
                           "(e.g., agitation). Importance = gradient × input for the predicted class.")
            except Exception as exc:
                st.error(f"Could not process this file. Please upload a valid, non-empty WAV. ({exc})")

        with st.expander("How reliable is this model? (held-out test set)"):
            if bundle["metrics"]:
                show_metrics(bundle["metrics"])
                st.caption(f"Based on {bundle['metrics'].get('n_test', '?')} test clips.")
            st.markdown("Trained on a **small** mini dataset of short clips, so treat results as a screening aid, "
                        "not a verdict. Real calls have noise, accents and overlapping speakers.")

# ==========================================================================
# TAB 4: BUSINESS SUMMARY
# ==========================================================================
with tab_about:
    st.header("Business Summary")
    st.markdown(
        '''
**Purpose** A company can't contact every customer, so this dashboard helps decide who to reach out to first.

**Tabs**

| Tab | You give it | What you get back |
|---|---|---|
| Customer Churn | Customer's Account Details | How likely they are to leave, and which details pushed that up or down |
| Text Sentiment | Review or Message | Whether it sounds positive, neutral, or negative and which words caused that |
| Audio Sentiment | Short .wav Recording | Whether the call sounds positive or negative and which parts of the sound matter |

**Use** If I get a high churn score and a negative review contact or ask for feedback. If the score is high their feedback is positive, give a small offer.

**Charts** The chart shows which account details raise or lower the risk for a customer.

**Limits** With a 66% accuracy, it should only help us pick who to contact, not to make the decision
''')
    