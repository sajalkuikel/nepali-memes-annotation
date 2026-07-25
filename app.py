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

ANN_COLUMNS = [
    "page_name", "post_id", "annotator", "meme", "sentiment", "intent",
    "cyberbullying", "target", "protected_group", "harm", "harmfulness",
    "emotion", "modality", "timestamp"
]


MAX_ANNOTATIONS_PER_MEME = 3

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
#  AUTHENTICATION
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
# SMALL HELPERS
# ======================================================
def load_annotations():
    records = sheet.get_all_records()
    df = pd.DataFrame(records) if records else pd.DataFrame(columns=ANN_COLUMNS)
    if "post_id" in df.columns:
        df["post_id"] = df["post_id"].astype(str)
    return df


# Fields compared to decide whether two annotators "matched" on a post.
LABEL_COLUMNS = [
    "meme", "sentiment", "intent", "cyberbullying", "target",
    "protected_group", "harm", "harmfulness", "emotion", "modality"
]


def compute_agreement(page_ann_df, current_annotator):
    """For each other annotator who shares at least one annotated post with
    `current_annotator` on this page, return (other_annotator, matched, total),
    sorted by total shared posts descending."""
    if page_ann_df.empty:
        return []

    my_df = page_ann_df[page_ann_df["annotator"] == current_annotator].drop_duplicates("post_id").set_index("post_id")
    others = sorted(
        page_ann_df.loc[page_ann_df["annotator"] != current_annotator, "annotator"].dropna().unique().tolist()
    )

    results = []
    for other in others:
        other_df = page_ann_df[page_ann_df["annotator"] == other].drop_duplicates("post_id").set_index("post_id")
        shared_ids = my_df.index.intersection(other_df.index)
        total = len(shared_ids)
        if total == 0:
            continue
        matched = 0
        for pid in shared_ids:
            mine = my_df.loc[pid, LABEL_COLUMNS].fillna("")
            theirs = other_df.loc[pid, LABEL_COLUMNS].fillna("")
            if (mine == theirs).all():
                matched += 1
        results.append((other, matched, total))

    results.sort(key=lambda r: r[2], reverse=True)
    return results


def show_agreement_summary(page_ann_df, current_annotator):
    results = compute_agreement(page_ann_df, current_annotator)
    if not results:
        return
    st.markdown("#### 🤝 Agreement with other annotators")
    for other, matched, total in results:
        st.write(f"**{matched}/{total}** matched with **{other}**")


# ======================================================
# LAYOUT
# ======================================================
col_meme, col_ui = st.columns([4, 6])

# The mode toggle lives on the meme/image side so it stays in the same
# screen fold as the image while you're deciding what to work on next.
with col_meme:
    st.markdown("### Nepali Meme Annotation Dashboard")
    mode = st.radio(
        "📝 Annotation Mode",
        ["🆕 Fresh Annotation", "🔁 Re-annotate (Majority Voting)"],
        key="annotation_mode"
    )
    is_reannotation = mode.startswith("🔁")

# ======================================================
# RIGHT UI
# ======================================================
with col_ui:
    # same row: logout + dataset
    c1, c2 = st.columns([1, 4])

    with c1:
        st.markdown("👤 Logged in as: **" + annotator + "**")
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

    with c2:
        pages = github_list_folders(GITHUB_OWNER, GITHUB_REPO)
        page_name = st.selectbox("Select Page / Dataset (also selects the page to re-annotate)", pages, key="page_select")

    data = load_page_jsonl(GITHUB_OWNER, GITHUB_REPO, page_name)

    ann_df = load_annotations()
    page_ann_df = ann_df[ann_df["page_name"] == page_name] if not ann_df.empty else ann_df

    # A post counts as "annotated" the moment ANY annotator has labeled it —
    # not just the current user. That's what makes a post ineligible for
    # "Fresh Annotation" and eligible for "Re-annotate".
    all_done_ids = page_ann_df["post_id"].unique().tolist() if not page_ann_df.empty else []
    my_done_ids = (
        page_ann_df.loc[page_ann_df["annotator"] == annotator, "post_id"].tolist()
        if not page_ann_df.empty else []
    )
    # Posts an earlier annotator marked as "not a meme" are excluded from
    # re-annotation entirely (nothing to build majority-vote consensus on).
    non_meme_ids = (
        page_ann_df.loc[page_ann_df["meme"] == "No", "post_id"].unique().tolist()
        if not page_ann_df.empty else []
    )

    # --------------------------------------------------
    # BUILD THE QUEUE DEPENDING ON MODE
    # --------------------------------------------------
    ann_counts = page_ann_df["post_id"].value_counts() if not page_ann_df.empty else pd.Series(dtype=int)

    if is_reannotation:
        # A meme that already has MAX_ANNOTATIONS_PER_MEME annotations is
        # considered complete and drops out of the re-annotation pool.
        maxed_out_ids = ann_counts[ann_counts >= MAX_ANNOTATIONS_PER_MEME].index.tolist()

        eligible_pool_ids = [
            pid for pid in all_done_ids
            if pid not in non_meme_ids and pid not in maxed_out_ids
        ]
        review_ids = [pid for pid in eligible_pool_ids if pid not in my_done_ids]
        remaining = data[data["post_id"].isin(review_ids)].copy()

        # Prioritize memes that already have MORE annotations, so a meme with
        # 2 annotations reaches a 3rd annotator before a meme with only 1
        # annotation reaches its 2nd — this closes out majority-vote sets faster.
        remaining["_ann_count"] = remaining["post_id"].map(ann_counts).fillna(0)
        remaining = remaining.sort_values("_ann_count", ascending=False).drop(columns="_ann_count")
    else:
        # Fresh = nobody has annotated it yet, regardless of who you are.
        remaining = data[~data["post_id"].isin(all_done_ids)]

    if remaining.empty:
        if is_reannotation:
            st.success(f"🎉 You've re-annotated everything eligible in **{page_name}**")
        else:
            st.success(f"🎉 No fresh (unannotated) memes left in **{page_name}**")
        show_agreement_summary(page_ann_df, annotator)
        st.stop()

    row = remaining.iloc[0]

    # Coverage info for THIS post, regardless of mode
    post_ann = page_ann_df[page_ann_df["post_id"] == row["post_id"]] if not page_ann_df.empty else page_ann_df
    annotators_so_far = post_ann["annotator"].unique().tolist() if not post_ann.empty else []

    if annotators_so_far:
        st.info(f"📊 This meme has been annotated **{len(post_ann)}** time(s) — by: {', '.join(annotators_so_far)}")
    else:
        st.info("📊 This meme has not been annotated by anyone yet.")

    # ======================================================
    # LABEL FORM — fully inside col_ui (RIGHT SIDE)
    # ======================================================
    with st.form("annotation_form"):

        meme_label = st.radio(
            "Is this a meme?",
            ["Yes", "No"],
            horizontal=True,
            key=f"meme_label_{row['post_id']}_{mode}"
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
                modality = st.radio(
                    "Modality.\n (Select how the meme conveys meaning) ",
                    [
                        "Image",
                        "Text",
                        "Image + text combined",
                        "None",
                    ],
                    index=None,
                    key=f"modality_{row['post_id']}_{mode}",
                    horizontal=True,
                    help="""
                        Select how the meme mainly delivers its meaning.

                        Image — The picture alone gives the message. (केवल तस्बिरले बुझिन्छ)
                        Text — Only the words/caption give the message. (केवल शब्द/क्याप्शनले बुझिन्छ)
                        Image + text combined — Both picture and text together are needed. (तस्बिर र शब्द दुवैले मिलेर मात्र बुझिन्छ)
                        None — No clear meaning or not intended to convey a message. (स्पष्ट अर्थ छैन)

                        Check image and text together before choosing.
                        """

                )
                intent = st.radio(
                    "Intent of Meme",
                    [
                        "Benign / Playful - (हानिरहित / रमाइलो उद्देश्य)",
                        "Mocking/Sarcasm (उडाउने / व्यंग्यात्मक)",
                        "Critical / Satirical (आलोचनात्मक/ व्यंग्यसहितको)",
                        "Malicious (हानि पुर्‍याउने नियत)",
                        "Deceptive (भ्रामक / गलत धारणा फैलाउने)"
                    ],
                    index=None,
                    key=f"intent_{row['post_id']}_{mode}",
                    help="""
                        Select the PRIMARY intent behind the meme (choose the dominant intent).

                        Benign / Playful (हानिरहित / रमाइलो)
                        - Lighthearted, friendly, or purely humorous with no intent to harm.
                        - For fun, jokes, casual entertainment.
                        - हल्का रमाइलो, कसैलाई हानि पुर्‍याउने उद्देश्य नभएको।

                        Mocking / Sarcasm (उडाउने / व्यंग्यात्मक)
                        - Ridicules, taunts, or belittles a person/group.
                        - Uses irony or sarcastic tone to mock.
                        - जिस्क्याउने, होच्याउने वा व्यंग्य गरेर उडाउन खोजिएको।

                        Critical / Satirical (आलोचनात्मक / व्यंग्यसहित)
                        - Criticizes people, systems, or situations.
                        - Uses humor/satire to highlight issues or opinions/ expose flaws, criticize politics/society
                        - समाज, राजनीति वा अवस्थाको आलोचना/व्यंग्य गरिएको।

                        Malicious (हानि पुर्‍याउने नियत)
                        - Intends to harm, threaten, harass, or spread hate.
                        - Encourages abuse or hostility toward target.
                        - हानि, घृणा वा आक्रमण फैलाउने उद्देश्य।

                        Deceptive (भ्रामक / गलत धारणा फैलाउने)
                        - Intends to mislead or spread false information.
                        - Uses edited visuals/text or wrong context.
                        - गलत सूचना वा भ्रम सिर्जना गर्ने उद्देश्य रहेको।

                        Always consider image + text + caption together before choosing.
                        """

                )

            with col2:
                cyberbullying = st.radio(
                    "Presence of Hate / Cyber Bullying",
                    ["Yes", "No"],
                    index=None,
                    key=f"cyberbullying_{row['post_id']}_{mode}",
                    help="""
                        Does this meme contain hate or cyber-bullying?

                        Yes
                        - Uses abusive, insulting, or hateful language.
                        - Threatens or encourages harm/harassment.
                        - Targets a person or group in a mean or harmful way.
                        - कसैलाई गाली, घृणा, धम्की वा जानाजानी होच्याउने सामग्री छ।

                        No
                        - No hate, threats, or serious insults.
                        - Only normal humor, satire, or neutral content.
                        - घृणा वा साइबर बुलिङ छैन, सामान्य रमाइलो वा तटस्थ सामग्री मात्र।

                        Read image + text together before choosing.
                        """
                )

                target = st.radio(
                    "Target of the meme",
                    ["Individual", "Organization", "Community", "None"],
                    index=None,
                    key=f"target_{row['post_id']}_{mode}",
                    help="""
                        Select the PRIMARY target of the meme (choose one).

                        Individual (व्यक्ति)
                        - A single named or clearly identifiable person (public figure or private individual).
                        - एक जना व्यक्तिलाई लक्षित

                        Organization (संस्था)
                        - A company, government body, political party, NGO, school, or other formal group.
                        - कम्पनी, सरकारी निकाय, पार्टी, संस्था आदिलाई लक्षित।

                        Community (समुदाय)
                        - A social group defined by identity (ethnicity, religion, caste, gender), region, profession, or an online community.
                        - जाति/धर्म/लैङ्गिक/क्षेत्र/व्यवसाय/अनलाइन समूह जस्ता समूहहरूलाई लक्षित।
                        - Even if the target is a nation as a whole, select 'Community' as the label.

                        None 
                        - No specific target (absurdist, template meme, object, situation, or purely contextual humor).
                        
                        Notes:
                        - If ambiguous, choose the closest category.
                        - If the target is a protected community, mark `protected_group = Yes` separately.
                        - Read image + overlaid text + caption together before deciding.
                        """
                )

                protected_group = st.radio(
                    "Is target a protected group?",
                    ["Yes", "No"],
                    index=None,
                    key=f"protected_group_{row['post_id']}_{mode}",
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
                    [
                        "Psychological/Emotional (मानसिक / भावनात्मक)",
                        "Social/Reputational (सामाजिक / प्रतिष्ठासम्बन्धी)",
                        "Financial or Material (आर्थिक वा भौतिक हानि)",
                        "No Harm"
                    ],
                    index=None,
                    key=f"harm_{row['post_id']}_{mode}",
                    help="""
                        Select the PRIMARY way this meme harms the target (choose one).

                        Psychological/Emotional (मानसिक / भावनात्मक)
                        - Causes distress, fear, humiliation, or emotional harm.
                        - मानसिक पीडा, डर, अपमान, आत्मसम्मान घटाउनु।

                        Social/Reputational (सामाजिक / प्रतिष्ठासम्बन्धी)
                        - Damages reputation, social standing, or relationships.
                        - सामाजिक प्रतिष्ठा, विश्वास, सम्बन्धमा नोक्सान पुर्याउने।

                        Financial or Material (आर्थिक वा भौतिक हानि)
                        - Leads to economic loss, property damage, job risk, or doxxing.
                        - आर्थिक घाटा, सम्पत्ति नोक्सान, रोजगारी/आयमा असर पुर्याउने।

                        No Harm (हानि नभएको)
                        - No identifiable harm; neutral or purely humorous without negative effects.
                        - स्पष्ट हानि/नोक्सान नभएको, केवल रमाइलो/तटस्थ।

                        Notes:
                        - If multiple harms appear, choose the most severe or dominant harm.
                        - If malicious intent targets a protected group, mark protected_group = Yes 
                        - Always read image + overlaid text + caption together before deciding.
                        """

                )

                st.write('')
                harmfulness = st.radio(
                    "If 'Harmful' , please label Harmfulness Score",
                    ["(1) Offensive", "(2) Partially harmful", "(3) Very harmful"],
                    index=None,
                    key=f"harmfulness_{row['post_id']}_{mode}",
                    horizontal=True,
                    help="""
                        If the meme is harmful, choose how harmful it is.

                        (1) Offensive
                        - Mild insult, rude joke, or low-level negativity.
                        - हल्का गाली वा अपमान, कम स्तरको हानि।

                        (2) Partially harmful
                        - Clear harassment, humiliation, or reputational damage.
                        - मानसिक वा सामाजिक रूपमा नोक्सान पुर्‍याउने सामग्री।

                        (3) Very harmful
                        - Serious hate, threats, violence, or strong harmful intent.
                        - गम्भीर घृणा, धम्की, हिंसा वा ठूलो हानि पुर्‍याउने सामग्री।

                        Choose the highest level of harm shown in the meme.
                        """

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
                    key=f"emotion_{row['post_id']}_{mode}",
                    horizontal=True,
                    help="""
                        Select the PRIMARY emotion the meme conveys toward its target.

                        Joy (खुशी)
                        - Positive, happy, celebratory feeling. — सकारात्मक, खुशी।

                        Sadness (दुःख)
                        - Sorrowful or upset tone. — उदास, निराश।

                        Fear (डर)
                        - Shows worry, threat, or anxiety. — डर, चिन्ता।

                        Anger (रिस)
                        - Angry, hostile, or furious tone. — रिस, क्रोध।

                        Disgust (घृणा)
                        - Shows revulsion or strong dislike. — घिन, अरूचि।

                        Surprise (आश्चर्य)
                        - Shocked or amazed reaction. — आश्चर्य, अचम्म।

                        Trust (विश्वास)
                        - Shows confidence, faith, or support. — भरोसा, समर्थन।

                        Anticipation (अपेक्षा)
                        - Expectation or looking forward. — प्रतिक्षा, अपेक्षा।

                        Ridicule (उपहास / खिल्ली उडाउने)
                        - Mocking, making fun, or derisive tone. — जिस्क्याउने, उपहास गर्ने।

                        Other
                        - Any other clear emotion not listed above.

                        Read the image + overlaid text + caption together. If multiple emotions appear, pick the dominant one.
                        """

                )

                sentiment = st.radio(
                    "Sentiment",
                    ["Positive", "Negative", "Neutral"],
                    index=None,
                    key=f"sentiment_{row['post_id']}_{mode}",
                    help="""
                        Choose the sentiment expressed toward the target in the meme.

                        Positive -> Praise, support, admiration, celebration, or clearly pleasant emotion toward the target.

                        Negative -> Insult, mockery, criticism, hate, anger, or clearly unpleasant emotion toward the target. 
                        (Sarcastic praise used to mock = Negative.)

                        Neutral -> Informational, descriptive, absurd/non-targeted humor, or sentiment unclear/mixed.

                        Always judge using BOTH image and text together, and consider context if available.
                        """
                )

        submitted = st.form_submit_button("➡️ Submit & Next")

        if submitted:
            save_and_next = False
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

            if save_and_next:

                # SAVE THE DATA (each annotator's row is appended independently,
                # so the same post_id can accumulate multiple annotator rows
                # for majority voting)
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

    # --------------------------------------------------
    # PROGRESS (depends on mode)
    # --------------------------------------------------
    if is_reannotation:
        total_eligible = len(eligible_pool_ids)
        my_reannotated = len([pid for pid in eligible_pool_ids if pid in my_done_ids])
        st.progress(my_reannotated / total_eligible if total_eligible else 0)
        st.caption(
            f"🗳️ Eligible for re-annotation in **{page_name}**: **{total_eligible}** — "
            f"you've re-annotated **{my_reannotated}** of them"
        )
    else:
        total = len(data)
        done = len(all_done_ids)
        st.progress(done / total if total else 0)
        st.caption(f"{done} / {total} memes have at least one annotation in {page_name}")

# ======================================================
# LEFT MEME DISPLAY (continues the col_meme container opened above)
# ======================================================
with col_meme:
    if row.get("post_text"):
        st.markdown(f"🔗 **[Click here to view original post]({row['post_url']})**")
        st.info(row["post_text"])

    try:
        img = load_private_github_image(GITHUB_OWNER, GITHUB_REPO, f"{page_name}/{row['image_file']}")
        st.image(img, use_column_width=True)
    except Exception:
        st.error("No image available for this post.")