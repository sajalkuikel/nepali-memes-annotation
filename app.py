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
        columns=["page_name", "post_id", "annotator", "meme", "sentiment", "intent", "cyberbullying", "target", "protected_group", "harm", "harmfulness", "emotion", "modality", "timestamp"]
    )

    ann_df["post_id"] = ann_df["post_id"].astype(str)

    done_ids = ann_df[
        ann_df["page_name"] == page_name
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

        sentiment = None
        intent = None
        cyberbullying = None
        target = None
        protected_group = None
        harm = None
        harmfulness = None
        emotion = None
        modality = None


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
                intent = st.radio(
                    "Intent of Meme",
                    ["Benign / Playful - (हानिरहित / रमाइलो उद्देश्य)", "Mocking/Sarcasm (उडाउने / व्यंग्यात्मक)", "Critical / Satirical (आलोचनात्मक/ व्यंग्यसहितको)", "Malicious (हानि पुर्‍याउने नियत)", "Deceptive (भ्रामक / गलत धारणा फैलाउने)"],
                    index=None,
                    key=f"intent_{row['post_id']}"
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
                    ["Individual", "Organization", 'Community', "None"],
                    index=None,
                    key=f"target_{row['post_id']}"
                )
                 
                 protected_group = st.radio(
                    "Is target a protected group?",
                    ["Yes", "No"],
                    index=None,
                    key=f"protected_group_{row['post_id']}",
                    help=(
                        "**Nepal Context:** Select 'Yes' if the target belongs to a group eligible for "
                        "reservation/protection under Nepal's Civil Service Act or Constitution.\n\n"
                        "**Includes:**\n"
                        "- **Women**\n"
                        "- **Adibasi / Janajati** (Indigenous Nationalities)\n"
                        "- **Madhesi / Tharu / Muslim** \n"
                        "- **Dalit**\n"
                        "- **Persons with Disabilities**\n"
                        "- **Residents of Backward Areas** (Karnali zone/remote districts)\n"
                        "- **Gender & Sexual Minorities** (LGBTQ+)"
                    )
                )
                 st.caption("""
                    Includes: Caste/ Religion/ Gender & Sexual Minorities/ Disability/ Region/ Language/ Economic Class/ Ideology  
                    *Eg. Dalits, Madhesis, Muslims, LGBTQ+, disabled, etc.*
                    """)
            with col3:
                harm = st.radio(
                    "How does this meme harm the target?",
                    ["Psychological/Emotional (मानसिक / भावनात्मक)", "Social/Reputational (सामाजिक / प्रतिष्ठासम्बन्धी)", "Financial or Material (आर्थिक वा भौतिक हानि)",  "No Harm"],
                    index=None,
                    key=f"harm_{row['post_id']}"
                )
                
                harmfulness = ""
                st.write('')
                harmfulness = st.radio(
                    "If 'Harmful' , please label Harmfulness Score",
                    ["(1) Offensive", "(2) Partially harmful", "(3) Very harmful" ],
                    index=None,
                    key=f"harmfulness_{row['post_id']}",
                    horizontal=True
                )

            with col4:
                emotion = st.radio(
                    "Emotion",
                    [
                        "Joy (खुशी)",
                        "Sadness (दुःख)",
                        "Fear (डर)",
                        "Anger (रिस)",
                        "Disgust (घृणा)",
                        "Surprise (आश्चर्य)",
                        "Trust (विश्वास)",
                        "Anticipation (अपेक्षा)",
                        "Ridicule (उपहास / खिल्ली उडाउने)",
                        "Other"
                    ],
                    index=None,
                    key=f"emotion_{row['post_id']}",
                    horizontal=True
                )
                
                # age = st.slider('How old are you?', 0, 130, 25)
                modality = st.radio(
                    "Modality.\n (Select how the meme conveys meaning) ",
                    [
                        "Image",
                        "Text",
                        "Image + text combined",
                        "None",
                    ],
                    index=None,
                    key=f"modality_{row['post_id']}",
                    horizontal=True
                )

        submitted = st.form_submit_button("➡️ Submit & Next")

        if submitted:
            # ==============================
            # VALIDATION ONLY IF MEME = YES
            # ==============================
            if meme_label == "Yes":

                required_fields = {
                    "Sentiment": sentiment,
                    "Intent": intent,
                    "Cyberbullying": cyberbullying,
                    "Target": target,
                    "Harm Type": harm,
                    "Emotion": emotion,
                    "Modality": modality
                }

                missing = [k for k, v in required_fields.items() if v is None]

                if missing:
                    st.error(f"⚠️ Please label: {', '.join(missing)}")
                elif harm != "No Harm" and harmfulness is None:
                    st.error("⚠️ Please provide a Harmfulness score for harmful content.")
                else:
                    save_and_next = True
            else:
                # Meme = No → always valid
                save_and_next = True


            if submitted and 'save_and_next' in locals() and save_and_next:

                # SAVE THE DATA
                sheet.append_row([
                    page_name,
                    row["post_id"],
                    annotator,
                    meme_label,
                    sentiment if sentiment else "",
                    intent if intent else "",
                    cyberbullying if cyberbullying else "",
                    target if target else "",
                    protected_group if protected_group else "",
                    harm if harm else "",
                    harmfulness if harmfulness else "",
                    emotion if emotion else "",
                    modality if modality else "",

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
