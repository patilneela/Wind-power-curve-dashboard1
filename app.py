import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from datetime import timedelta
import os
import io

from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(layout="wide")

# =========================
# LOGIN
# =========================
def login_gate():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return

    st.title("Login Required")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login")

    if submitted:
        cfg = st.secrets.get("auth", {})
        ok = (username == cfg.get("username")) and (password == cfg.get("password"))
        if ok:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()


login_gate()

if st.sidebar.button("Logout", key="logout_btn"):
    st.session_state.authenticated = False
    st.rerun()

# =========================
# SAFE KALEIDO CHECK
# =========================
try:
    import kaleido  # noqa: F401
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(__file__)
logo_path = os.path.join(BASE_DIR, "Envision.png")
BIN_SIZE = 0.5

# =========================
# HELPERS
# =========================
def normalize_text(s):
    if pd.isna(s):
        return ""
    return (
        str(s).strip().lower()
        .replace("-", "")
        .replace("_", "")
        .replace(",", "")
        .replace(" ", "")
        .replace("\n", "")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )


def detect_column(columns, candidates):
    normalized_map = {c: normalize_text(c) for c in columns}

    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        exact = [col for col, norm in normalized_map.items() if norm == candidate_norm]
        if exact:
            return exact[0]

    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
