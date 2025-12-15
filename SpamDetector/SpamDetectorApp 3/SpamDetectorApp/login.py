# login.py
import streamlit as st
import json
import bcrypt
import os

USER_FILE = "users.json"

# Load users from file
def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r") as f:
        return json.load(f)

# Save users to file
def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

# Signup function
def signup():
    st.markdown("""
        <style>
        .form-box {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 15px 30px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease;
        }
        .form-title {
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            color: #4facfe;
            margin-bottom: 20px;
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(-10px);}
            to {opacity: 1; transform: translateY(0);}
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="form-box">', unsafe_allow_html=True)
    st.markdown('<div class="form-title">📝 Create an Account</div>', unsafe_allow_html=True)

    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    confirm_pass = st.text_input("Confirm Password", type="password")

    if st.button("Signup"):
        users = load_users()
        if new_user in users:
            st.error("Username already exists!")
        elif new_pass != confirm_pass:
            st.error("Passwords do not match.")
        else:
            hashed = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt()).decode()
            users[new_user] = {"password": hashed, "role": "user"}
            save_users(users)
            st.success("Account created! Please login.")
            st.session_state.mode = "login"

    if st.button("Back to Login"):
        st.session_state.mode = "login"

    st.markdown('</div>', unsafe_allow_html=True)

# Login function
def login():
    st.markdown("""
        <style>
        .form-box {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 15px 30px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease;
        }
        .form-title {
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            color: #4facfe;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="form-box">', unsafe_allow_html=True)
    st.markdown('<div class="form-title">🔐 Login to Continue</div>', unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()
        if username in users and bcrypt.checkpw(password.encode(), users[username]["password"].encode()):
            st.success(f"Welcome, {username} 👋")
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid credentials.")

    if st.button("Create new account"):
        st.session_state.mode = "signup"                                    

    st.markdown('</div>', unsafe_allow_html=True)

# Login Page Controller
def login_page():
    if "mode" not in st.session_state:
        st.session_state.mode = "login"

    if st.session_state.mode == "login":
        login()
    elif st.session_state.mode == "signup":
        signup()
