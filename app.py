import streamlit as st
import joblib
import re
from typing import List, Tuple

# ---------------------------------------------------------
# 0) Safe model loading (show clear message if missing)
# ---------------------------------------------------------
try:
    toxicity_model = joblib.load("model/toxic_model.pkl")
except Exception as e:
    st.error("Failed to load model file 'model/toxic_model.pkl'. Make sure the file exists in the repo.")
    st.stop()

try:
    base_bad_words = joblib.load("model/final_bad_words.pkl")
except Exception:
    # If the PKL isn't available we continue with an empty base set.
    base_bad_words = []

# ---------------------------------------------------------
# 1) Manual English + Telugu Offensive Word Lists
# ---------------------------------------------------------
english_offensive_words = [
    "fuck","fucking","fucker","fucked","mf","motherfucker","bitch","bitches",
    "bich","biatch","slut","whore","hoe","cunt","pussy","dick","cock","prick",
    "asshole","asshat","asswipe","dumbass","jackass","shit","shitty","bullshit",
    "crap","bastard","jerk","retard","fucktard","douche","douchebag","moron",
    "idiot","stupid","dumb","loser","scumbag","piss off","pisshead","pissbrain",
    "pissface","twat","skank","tramp","pig","donkey","airhead","brainless",
    "jerkoff","cocksucker","dickhead","wtf","hell","damn","f@ck","a$$hole",
    "sl*t","wh0re","c*nt","p*ssy","di*k","co*k","fuk","fck","fucc","fvk",
    "fukr","harami","gandu","kutti","kutta","rand","randi"
]

telugu_offensive_words = [
    "వెధవ","మూర్ఖుడు","దద్దమ్మ","చెత్తోడు","నాయాలా","దుర్మార్గుడు",
    "నీచుడు","పిచ్చోడు","బుద్ధిలేని","పిచ్చి","పిచ్చివాడు","చెత్త",
    "లంజ","రాండీ","దెంగు","దెంగు కొడుకు","దెంగినోడు","దొంగ","దొంగమూత",
    "మోసగాడు","అసహ్యం","దరిద్రుడు","మూత్రపు ముఖం","తిక్క","తిక్కోడు",
    "పిరికివాడు","వాజ","పంది","పందికొక్క","రాక్షసుడు","అసభ్యుడు","బూతు",
    "బూతులాడటం","బూతు మాటలు","దుష్టుడు","సిగ్గులేని","నోరు సిగ్గులేని",
    "గుణం లేని","యవ్వారం","దుర్మార్గం","దరిద్రపు","రాక్షసపు"
]

# ---------------------------------------------------------
# 2) Merge + filter safe words (avoid matching 'you', 'me', etc.)
# ---------------------------------------------------------
SAFE_WORDS = {
    "you", "me", "we", "they", "he", "she", "it", "your", "our", "my", "her", "him", "them",
    "is", "are", "the", "a", "an", "and", "or", "if", "in", "on", "at"
}

merged_set = set()

# add base PKL words
for w in base_bad_words:
    if isinstance(w, str) and w.strip():
        merged_set.add(w.strip().lower())

# add manual lists
for w in english_offensive_words + telugu_offensive_words:
    if isinstance(w, str) and w.strip():
        merged_set.add(w.strip().lower())

# remove safe/common words & very short tokens
final_offensive_word_list = {w for w in merged_set if len(w) > 1 and w not in SAFE_WORDS}
final_offensive_word_list = sorted(final_offensive_word_list)

# ---------------------------------------------------------
# 3) Preprocessing + normalization
# ---------------------------------------------------------
COMMON_MISSPELLINGS = {
    "idoit": "idiot", "stupit": "stupid", "fuk": "fuck",
    "fck": "fuck", "bich": "bitch", "asshloe": "asshole",
    "mottherfucker": "motherfucker"
}

def preprocess_comment(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    # keep Telugu unicode block \u0C00-\u0C7F and English letters and spaces
    text = re.sub(r"[^\u0C00-\u0C7Fa-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_slang_words(text: str) -> str:
    return " ".join([COMMON_MISSPELLINGS.get(tok, tok) for tok in text.split()])

# ---------------------------------------------------------
# 4) Compile robust regex patterns (Unicode-aware boundaries)
# Use character class that considers English letters/digits and Telugu block,
# so lookbehind/lookahead match properly for multilingual text.
# ---------------------------------------------------------
WORD_CHAR_CLASS = r"A-Za-z0-9\u0C00-\u0C7F"  # english letters/digits + telugu block

mask_patterns: List[Tuple[str, re.Pattern]] = []
for word in final_offensive_word_list:
    # build a regex with left/right boundaries not being word or telugu chars
    # (?<![A-Za-z0-9\u0C00-\u0C7F])WORD(?![A-Za-z0-9\u0C00-\u0C7F])
    escaped = re.escape(word)
    pattern = re.compile(rf"(?<![{WORD_CHAR_CLASS}]){escaped}(?![{WORD_CHAR_CLASS}])", flags=re.IGNORECASE)
    mask_patterns.append((word, pattern))

# ---------------------------------------------------------
# 5) Masking + highlighting functions (preserve length with stars)
# ---------------------------------------------------------
def mask_offensive_words(original_text: str) -> str:
    """Mask any offensive token occurrences in the original_text preserving length with '*'."""
    masked = original_text
    # iterate patterns — replace using function to preserve matched length
    for _, pattern in mask_patterns:
        masked = pattern.sub(lambda m: "*" * len(m.group(0)), masked)
    return masked

def highlight_offensive_words(original_text: str) -> Tuple[str, List[str]]:
    """
    Return highlighted markdown text and list of unique detected words in original casing.
    We perform pattern.finditer on the current string and do safe left-to-right replacement.
    """
    highlighted = original_text
    detected_matches = []

    # To avoid nested replacements interfering with indexes, we run pattern-by-pattern,
    # rebuilding string on each pattern using finditer spans.
    for _, pattern in mask_patterns:
        matches = list(pattern.finditer(highlighted))
        if not matches:
            continue

        new_parts = []
        last_index = 0
        for m in matches:
            s, e = m.span()
            new_parts.append(highlighted[last_index:s])
            matched_text = highlighted[s:e]
            new_parts.append(f"**:red[{matched_text}]**")
            detected_matches.append(matched_text)
            last_index = e
        new_parts.append(highlighted[last_index:])
        highlighted = "".join(new_parts)

    # deduplicate while preserving order (case-insensitive key)
    seen = set()
    unique_detected = []
    for w in detected_matches:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            unique_detected.append(w)
    return highlighted, unique_detected

# ---------------------------------------------------------
# 6) Detection pipeline
# ---------------------------------------------------------
def detect_and_mask_pipeline(text: str):
    cleaned_text = normalize_slang_words(preprocess_comment(text))
    prediction_label = toxicity_model.predict([cleaned_text])[0]
    probability = round(float(toxicity_model.predict_proba([cleaned_text])[0][1]) * 100, 2)

    highlighted_text, detected_words = highlight_offensive_words(text)
    masked_text = mask_offensive_words(text) if prediction_label == 1 else text

    label = "Toxic ⚠️" if prediction_label == 1 else "Non-Toxic ✅"

    return {
        "label": label,
        "probability": probability,
        "masked_text": masked_text,
        "highlighted_text": highlighted_text,
        "detected_words": detected_words,
        "cleaned": cleaned_text
    }

# ---------------------------------------------------------
# 7) Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Advanced Toxic Comment Detector", page_icon="🛡️", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📘 Project Guide")
    st.write("Detects toxic/offensive language (English + Telugu), highlights harmful words and masks them for safe display.")
    st.subheader("💡 Why this project?")
    st.write("- Reduce cyberbullying\n- Auto-protect audiences\n- Demonstrate multilingual NLP (Telugu + English)")
    st.subheader("📝 Examples (click to copy then press Analyze):")
    if st.button("Example 1: rafi you idiot"):
        st.session_state['sample'] = "rafi you idiot"
    if st.button("Example 2: rafi బుద్ధిలేని వెధవ"):
        st.session_state['sample'] = "rafi బుద్ధిలేని వెధవ"
    if st.button("Example 3: fuck you asshole"):
        st.session_state['sample'] = "fuck you asshole"

    st.subheader("⚙️ Settings")
    threshold = st.slider("Toxicity threshold (%)", 10, 90, 50, 5)
    st.write("You can change threshold to make detection stricter/looser.")
    st.markdown("---")
    st.write(f"Dictionary entries used: **{len(final_offensive_word_list)}**")

# Main layout
st.title("🛡️ Advanced Toxic Comment Detection & Masking System")
st.write("Enter a comment (Telugu / English / mixed) to analyze:")

# Use session_state sample autofill
default_text = st.session_state.get('sample', "")
input_comment = st.text_area("Enter your comment here:", value=default_text, height=150)

if st.button("Analyze Comment"):
    if not input_comment.strip():
        st.warning("Please enter a comment to analyze.")
    else:
        result = detect_and_mask_pipeline(input_comment)
        prob = result["probability"]
        is_toxic_by_threshold = prob >= threshold

        st.markdown("## 🔍 Analysis Result")
        display_label = result["label"]
        # If model predicted toxic but user changed threshold, show adjusted label
        if (result["label"] == "Toxic ⚠️") != is_toxic_by_threshold:
            display_label = "Toxic ⚠️" if is_toxic_by_threshold else "Non-Toxic ✅"

        st.write("**Prediction:**", display_label)
        st.write("**Confidence Score:**", f"{prob}%")

        # Show model input (cleaned) for transparency
        st.markdown("### 🔎 Model Input (cleaned & normalized):")
        st.code(result["cleaned"], language="text")

        if result["detected_words"]:
            st.markdown("### 🚨 Detected Offensive Words")
            st.error(", ".join(result["detected_words"]))

            st.markdown("### 🟥 Highlighted Text")
            st.markdown(result["highlighted_text"])
        else:
            st.info("No offensive words found by dictionary matching.")

        st.markdown("### 🛡️ Masked Output")
        # show masked result in code block for fixed width
        final_masked = result["masked_text"] if is_toxic_by_threshold else input_comment
        st.code(final_masked, language="text")

        # Downloadable report
        report = (
            f"Original: {input_comment}\n"
            f"Cleaned: {result['cleaned']}\n"
            f"Prediction: {display_label}\n"
            f"Confidence: {prob}%\n"
            f"Detected: {', '.join(result['detected_words'])}\n"
            f"Masked: {final_masked}\n"
        )
        st.download_button("Download report (txt)", report, file_name="toxicity_report.txt")
