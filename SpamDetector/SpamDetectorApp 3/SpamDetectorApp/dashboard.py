import streamlit as st
import pickle
import re
import pandas as pd
import os

# Load ML model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

HISTORY_FILE = "history.csv"

# -------------------------
# Spam Category Classifier
# -------------------------
def categorize_spam(text):
    text = text.lower()
    if any(word in text for word in ["win", "lottery", "prize", "congratulations"]):
        return "🎁 Lottery/Prize Scam"
    elif any(word in text for word in ["bank", "account", "verify", "password"]):
        return "💳 Banking/Phishing"
    elif any(word in text for word in ["delivery", "courier", "fedex", "dhl", "bluedart"]):
        return "📦 Courier/Delivery Scam"
    elif any(word in text for word in ["pill", "medicine", "weight", "health"]):
        return "💊 Health/Medicine Scam"
    elif any(word in text for word in ["image", "attachment", "click picture"]):
        return "🖼 Image Spam"
    else:
        return "❓ General Spam"

# -------------------------
# Suspicious Link Scanner
# -------------------------
def scan_links(text):
    url_pattern = r'(https?://\S+|www\.\S+)'
    links = re.findall(url_pattern, text)
    suspicious = []

    for link in links:
        if any(ext in link for ext in [".xyz", ".shop", ".net", ".online"]) or \
           any(word in link for word in ["verify", "update", "secure", "free", "gift"]):
            suspicious.append((link, "⚠️ Suspicious"))
        else:
            suspicious.append((link, "✔️ Safe"))
    return suspicious

# -------------------------
# Load/Save History
# -------------------------
def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE).to_dict("records")
    return []

def save_history(history):
    df = pd.DataFrame(history)
    df.to_csv(HISTORY_FILE, index=False)

# -------------------------
# Dashboard Function
# -------------------------
def dashboard():
    st.title("📧 Spam Detector Dashboard")

    # Load history at start
    if "history" not in st.session_state:
        st.session_state.history = load_history()

    input_sms = st.text_area("✉️ Enter the email / message text here:")

    if st.button("Predict"):
        if input_sms.strip() == "":
            st.warning("Please enter some text first.")
            return

        # Transform and predict
        processed_sms = vectorizer.transform([input_sms])
        prediction = model.predict(processed_sms)[0]
        prediction_proba = model.predict_proba(processed_sms)[0]

        # Spam Score
        spam_score = round(prediction_proba[1] * 100, 2)

        # Show result
        if prediction == 1:
            st.error("🚨 Spam Detected!")
            st.write(f"📊 **Spam Confidence Score:** {spam_score}%")
            spam_type = categorize_spam(input_sms)
            st.write(f"📂 **Spam Category:** {spam_type}")
        else:
            st.success("✅ Not Spam (Safe Message)")
            st.write(f"📊 **Spam Confidence Score:** {spam_score}%")

        # 🔗 Show link analysis
        links = scan_links(input_sms)
        if links:
            st.write("🔗 **Links found in message:**")
            for l, status in links:
                st.write(f"{status} → {l}")

        # Save to history (memory + file)
        st.session_state.history.append({
            "text": input_sms,
            "prediction": "Spam" if prediction == 1 else "Ham",
            "score": spam_score
        })
        save_history(st.session_state.history)

    # -------------------------
    # Sidebar History Button
    # -------------------------
    if st.sidebar.button("📂 View History"):
        if st.session_state.history:
            st.subheader("📈 Analysis History")
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df)

            spam_count = (df["prediction"] == "Spam").sum()
            ham_count = (df["prediction"] == "Ham").sum()
            st.write(f"✅ Ham: {ham_count} | 🚨 Spam: {spam_count} | 📊 Total: {len(df)}")

            # 🧹 Clear History button
            if st.button("🧹 Clear History"):
                st.session_state.history = []
                save_history([])  # clear file too
                st.success("History cleared successfully!")
        else:
            st.info("No history yet. Run some predictions first.")
