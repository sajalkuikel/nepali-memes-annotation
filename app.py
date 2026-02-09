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

# ======================================================
# 🔐 AUTHENTICATION
# ======================================================
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = st.secrets["auth_users"]

        if username in users and password == users[username]:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("❌ Invalid username or password")


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

annotator = st.session_state["username"]

# ======================================================
# GOOGLE SHEETS
# ======================================================
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

# ======================================================
# GITHUB HELPERS
# ======================================================
@st.cache_data(show_spinner=False)
def github_list_folders(owner, repo, path=""):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={GITHUB_BRANCH}"
    headers = {
        "Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json"
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return [i["name"] for i in r.json() if i["type"] == "dir"]


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

# ======================================================
# LAYOUT
# ======================================================
col_meme, col_ui = st.columns([4, 6])

# ======================================================
# RIGHT UI
# ======================================================
with col_ui:

    st.title("Meme Annotation Tool")
    st.markdown("👤 Logged in as: **" + annotator + "**")

    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    pages = github_list_folders(GITHUB_OWNER, GITHUB_REPO)
    page_name = st.selectbox("Select Page / Dataset", pages)

    data = load_page_jsonl(GITHUB_OWNER, GITHUB_REPO, page_name)

    records = sheet.get_all_records()
    ann_df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["page_name", "post_id", "annotator", "meme", "sentiment", "sarcasm", "emotion", "timestamp"]
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

    st.markdown("---")

    # ======================================================
    # LABEL FORM — fully inside col_ui (RIGHT SIDE)
    # ======================================================
    with st.form("annotation_form"):

        meme_label = st.radio(
            "Is this a meme?",
            ["Yes", "No"],
            horizontal=True,
            key=f"meme_label_{row['post_id']}"
        )

        sentiment = ""
        sarcasm = ""
        emotion = ""

        if meme_label == "Yes":  
            st.markdown("### 📌 Meme Attributes")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                sentiment = st.radio(
                    "Sentiment",
                    ["Positive", "Negative", "Neutral"],
                    index=None,
                    key=f"sentiment_{row['post_id']}"
                )
                sarcasm = st.radio(
                    "Type of Sarcasm",
                    ["Benign / Playful", "Mocking/Sarcasm", "Critical / Satirical", "Malicious", "Deceptive"],
                    index=None,
                    key=f"sarcasm_{row['post_id']}"
                )

            with col2:
                 cyberbullying = st.radio(
                    "Presence of Hate / Cyber Bullying",
                    ["Yes", "No"],
                    index=None,
                    key=f"cyberbullying_{row['post_id']}"
                )
                 
                 target = st.radio(
                    "Target of the meme",
                    ["Individual", "Organizatrion", 'Community', "None"],
                    index=None,
                    key=f"target{row['post_id']}"
                )
                 
                 protected_group = st.radio(
                    "Is target a protected group?",
                    ["Yes", "No"],
                    index=None,
                    key=f"protected_group{row['post_id']}"
                )
                 
            with col3:
                harm = st.radio(
                    "How does this meme harm the target?",
                    ["Psychological/Emotional", "Social/Reputational", "Financial or Material",  "No Harm"],
                    index=None,
                    key=f"harm{row['post_id']}"
                )
                
                harmfulness = ""
                st.write('')
                harmfulness = st.radio(
                    "If 'Harmfull,', please label Harmfulness Score",
                    ["(1) Offensive", "(2) Partially harmful", "(3) Very harmful" ],
                    index=None,
                    key=f"harmfulness_{row['post_id']}",
                    horizontal=True
                )

            with col4:
                emotion = st.radio(
                    "Emotion",
                    [
                        "Joy",
                        "Sadness",
                        "Fear",
                        "Anger",
                        "Disgust",
                        "Surprise",
                        "Trust",
                        "Anticipation",
                        "Ridicule",
                        "Other"
                    ],
                    index=None,
                    key=f"emotion_{row['post_id']}",
                    horizontal=True
                )
                
                # age = st.slider('How old are you?', 0, 130, 25)
                contribution = st.radio(
                    "Modality Contributing to Harm",
                    [
                        "Image",
                        "Text",
                        "Image + text combined",
                        "None",
                    ],
                    index=None,
                    key=f"contribution_{row['post_id']}",
                    horizontal=True
                )



        submitted = st.form_submit_button("➡️ Submit & Next")

        if submitted:

            # VALIDATION ONLY IF MEME = YES
            if meme_label == "Yes":
                if (sentiment is None) or (sarcasm is None) or (emotion is None):
                    st.error("⚠️ Please label sentiment, sarcasm, and emotion before submitting.")
                    st.stop()

            # SAVE THE DATA
            sheet.append_row([
                page_name,
                row["post_id"],
                annotator,
                meme_label,
                sentiment if sentiment else "",
                sarcasm if sarcasm else "",
                emotion if emotion else "",
                datetime.now().isoformat()
            ])

            st.rerun()

    progress = len(done_ids) / len(data)
    st.progress(progress)
    st.caption(f"{len(done_ids)} / {len(data)} annotated for {page_name}")

# ======================================================
# LEFT MEME DISPLAY
# ======================================================
with col_meme:

    if row.get("post_text"):
        st.markdown(f"🔗 **[View Original Post]({row['post_url']})**")
        st.markdown("---")
        st.markdown(row["post_text"])

    try:
        img = load_private_github_image(GITHUB_OWNER, GITHUB_REPO, f"{page_name}/{row['image_file']}")
        st.image(img, use_column_width=True)
    except:
        st.error("No image available for this post.")
