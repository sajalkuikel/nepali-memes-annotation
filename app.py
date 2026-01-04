import streamlit as st
import pandas as pd
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import requests
from PIL import Image
import io

# ---------------- CONFIG ----------------
SHEET_NAME = "annotation_db"

GITHUB_OWNER = "sajalkuikel"
GITHUB_REPO = "nepali_memes"
GITHUB_BRANCH = "main"

LABEL_MAP = {
    "1️⃣ Positive": "Positive",
    "2️⃣ Negative": "Negative",
    "3️⃣ Neutral": "Neutral",
    "4️⃣ Not a meme": "Not a meme"
}

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Nepali Meme Annotation", layout="wide")

# ---------------- CSS ----------------
st.markdown(
    """
    <style>
    .meme-container {
        height: 100vh;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 16px;
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- GOOGLE SHEETS ----------------
@st.cache_resource
def get_sheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )
    gc = gspread.authorize(creds)
    return gc.open(SHEET_NAME).sheet1


sheet = get_sheet()

# ---------------- GITHUB HELPERS ----------------
@st.cache_data(show_spinner=False)
def github_list_folders(owner, repo, path=""):
    """List top-level folders in a GitHub repo"""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={GITHUB_BRANCH}"
    headers = {
        "Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json"
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    items = r.json()
    return [i["name"] for i in items if i["type"] == "dir"]


@st.cache_data(show_spinner=False)
def load_page_jsonl(owner, repo, page_name):
    path = f"{page_name}/facebook_posts.jsonl"
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={GITHUB_BRANCH}"

    headers = {
        "Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github.raw"
    }

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    df = pd.read_json(io.BytesIO(r.content), lines=True)

    # Ensure post_id exists and is string
    if "post_id" not in df.columns:
        st.error(" post_id column missing in dataset")
        st.stop()

    df["post_id"] = df["post_id"].astype(str)
    return df



@st.cache_data(show_spinner=False)
def load_private_github_image(owner, repo, path):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={GITHUB_BRANCH}"
    headers = {
        "Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github.raw"
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


# ===================== LAYOUT =====================
col_meme, col_ui = st.columns([4, 6])

# ---------------- RIGHT UI ----------------
with col_ui:
    st.title("📝 Meme Annotation Tool")

    annotator = st.text_input("Annotator ID")
    if not annotator:
        st.stop()

    # -------- PAGE SELECTION --------
    pages = github_list_folders(GITHUB_OWNER, GITHUB_REPO)

    page_name = st.selectbox("Select Page / Dataset", pages)

    # -------- LOAD PAGE DATA --------
    data = load_page_jsonl(GITHUB_OWNER, GITHUB_REPO, page_name)

    # -------- LOAD ANNOTATIONS --------
    records = sheet.get_all_records()
    ann_df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["page_name", "post_id", "annotator", "label", "timestamp"]
    )

    ann_df["post_id"] = ann_df["post_id"].astype(str)

    done_ids = ann_df[
        (ann_df["annotator"] == annotator) &
        (ann_df["page_name"] == page_name)
    ]["post_id"].tolist()

    remaining = data[~data["post_id"].isin(done_ids)]

    if remaining.empty:
        st.success(f"🎉 All annotations completed for **{page_name}**")
        st.stop()

    row = remaining.iloc[0]

    st.markdown(f"🔗 **[View original Facebook post]({row['post_url']})**")
    st.markdown("---")

    # -------- LABEL FORM --------
    with st.form("annotation_form"):
        choice = st.radio(
            "Label (1–4)",
            list(LABEL_MAP.keys()),
            index=None
        )

        submitted = st.form_submit_button("➡️ Submit & Next")

        if submitted:
            if choice is None:
                st.warning("Select a label first.")
            else:
                sheet.append_row([
                    page_name,
                    row["post_id"],
                    annotator,
                    LABEL_MAP[choice],
                    datetime.now().isoformat()
                ])
                st.rerun()

    # -------- PROGRESS --------
    progress = len(done_ids) / len(data)
    st.progress(progress)
    st.caption(f"{len(done_ids)} / {len(data)} annotated for {page_name}")

# ---------------- LEFT MEME ----------------
with col_meme:
    # st.markdown('<div class="meme-container">', unsafe_allow_html=True)

    if row.get("post_text"):
        st.markdown(row["post_text"])

    img = load_private_github_image(
        GITHUB_OWNER,
        GITHUB_REPO,
        f"{page_name}/{row['image_file']}"
    )

    st.image(img, use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
