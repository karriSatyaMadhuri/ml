import streamlit as st
import pickle

# Load model and vectorizer
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="YouTube Toxic Comment Detector", layout="centered")

st.markdown("<h1 style='text-align: center;'>🛡️ YouTube Toxic Comment Classifier</h1>", unsafe_allow_html=True)
st.markdown("### Paste a comment below to check if it's offensive:", unsafe_allow_html=True)

# Input box
comment = st.text_area("", height=150, placeholder="Type a comment here...")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🧠 Classify"):
        if comment.strip() == "":
            st.warning("Please enter a comment.")
        else:
            # Transform and predict
            vec = vectorizer.transform([comment])
            pred = clf.predict(vec)[0]
            label = "🔥 Offensive" if pred == 1 else "✅ Not Offensive"
            color = "red" if pred == 1 else "green"
            st.markdown(f"<h2 style='text-align: center; color: {color};'>{label}</h2>", unsafe_allow_html=True)
