# main.py
import streamlit as st
from login import login_page
from dashboard import dashboard
from ai_image_check import ai_image_checker

st.set_page_config(page_title="Email Spam Classifier", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if st.session_state.logged_in:
    st.sidebar.title("📂 Navigation")
    choice = st.sidebar.radio("Go to", ["📧 Spam Classifier", "🧠 AI Image Checker", "🔒 Logout"])

    if choice == "📧 Spam Classifier":
        dashboard()
    elif choice == "🧠 AI Image Checker":
        ai_image_checker()
    elif choice == "🔒 Logout":
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
else:
    login_page()
