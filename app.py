import streamlit as st
import joblib
import re

# ---------------------------------------------------------
# 1) LOAD TRAINED MODEL + BASE OFFENSIVE WORD LIST (PKL)
# ---------------------------------------------------------
toxicity_model = joblib.load("model/toxic_model.pkl")
base_bad_words = joblib.load("model/final_bad_words.pkl")  # extracted from dataset

# ---------------------------------------------------------
# 2) ADD ENGLISH + TELUGU OFFENSIVE WORD LISTS (Industry-grade)
#    (You already provided these lists; they are extended here)
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
# 3) MERGE + FILTER (remove common SAFE words to avoid false positives)
# ---------------------------------------------------------
SAFE_WORDS = {
    "you", "me", "we", "they", "he", "she", "it", "your", "our", "my", "her", "him", "them",
    "is", "are", "the", "a", "an", "and", "or", "if", "in", "on", "at"
}

# Build set from PKL and manual lists (lowercased)
merged_set = set()

# Add PKL words (if any)
for w in base_bad_words:
    if isinstance(w, str) and w.strip():
        merged_set.add(w.strip().lower())

# Add manual English + Telugu lists
for w in english_offensive_words + telugu_offensive_words:
    if isinstance(w, str) and w.strip():
        merged_set.add(w.strip().lower())

# Remove safe/common words & very short tokens
final_offensive_word_list = {
    w for w in merged_set
    if len(w) > 2 and w not in SAFE_WORDS
}

# Convert to sorted list for deterministic order (nice for debugging/UI)
final_offensive_word_list = sorted(final_offensive_word_list)

# ---------------------------------------------------------
# 4) TEXT PREPROCESSING + NORMALIZATION
# ---------------------------------------------------------
COMMON_MISSPELLINGS = {
    "idoit": "idiot", "stupit": "stupid", "fuk": "fuck",
    "fck": "fuck", "bich": "bitch", "asshloe": "asshole",
    "mottherfucker": "motherfucker"
}

def preprocess_comment(text: str) -> str:
    """Lowercase, remove urls/handles/hashtags, remove punctuation (keep Telugu + English letters)."""
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    # Keep Telugu Unicode block and English letters and spaces
    text = re.sub(r"[^\u0C00-\u0C7Fa-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_slang_words(text: str) -> str:
    """Fix common misspellings/slang."""
    return " ".join([COMMON_MISSPELLINGS.get(tok, tok) for tok in text.split()])

# ---------------------------------------------------------
# 5) COMPILE REGEX PATTERNS (word-boundary safe)
# Note: \b might behave differently for Telugu; we use lookarounds to be robust.
# ---------------------------------------------------------
mask_patterns = []
for word in final_offensive_word_list:
    # Use Unicode-aware boundaries: (?<!\w) and (?!\w) — this avoids matching inside words
    pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", flags=re.IGNORECASE)
    mask_patterns.append((word, pattern))  # keep the base word too

# ---------------------------------------------------------
# 6) MASKING + HIGHLIGHTING FUNCTIONS
# ---------------------------------------------------------
def mask_offensive_words(original_text: str) -> str:
    """Return masked version of original_text by replacing offensive tokens with asterisks."""
    masked = original_text
    for _, pattern in mask_patterns:
        # Replace using the match length to keep same number of '*' characters
        masked = pattern.sub(lambda m: "*" * len(m.group(0)), masked)
    return masked

def highlight_offensive_words(original_text: str):
    """
    Highlight offensive words (markdown) and return:
      - highlighted_text: markdown friendly text with red highlights
      - detected_words: list of unique matched words (as matched in the text)
    """
    highlighted = original_text
    detected = []
    # To avoid repeatedly replacing the same area and breaking indices, we perform replacements
    # by iterating patterns — each substitution uses the matched text itself.
    for base_word, pattern in mask_patterns:
        # finditer gives Match objects preserving original casing
        matches = list(pattern.finditer(highlighted))
        if not matches:
            continue
        # For replacements without interfering with previously added markdown, do a single pass:
        # Build new string by splitting and replacing matches left-to-right.
        new_parts = []
        last_idx = 0
        for m in matches:
            start, end = m.span()
            # Append text before match
            new_parts.append(highlighted[last_idx:start])
            matched_text = highlighted[start:end]
            # Add markdown red highlight preserving the original matched casing
            new_parts.append(f"**:red[{matched_text}]**")
            detected.append(matched_text)
            last_idx = end
        new_parts.append(highlighted[last_idx:])
        highlighted = "".join(new_parts)
    # Remove duplicates but preserve order
    seen = set()
    unique_detected = []
    for w in detected:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            unique_detected.append(w)
    return highlighted, unique_detected

# ---------------------------------------------------------
# 7) PREDICTION + PROCESSING PIPELINE
# ---------------------------------------------------------
def detect_and_mask_pipeline(text: str):
    """
    1) Clean & normalize for model
    2) Predict toxicity with the trained model
    3) Highlight and mask on ORIGINAL input
    """
    cleaned = normalize_slang_words(preprocess_comment(text))
    prediction_label = toxicity_model.predict([cleaned])[0]
    probability = round(float(toxicity_model.predict_proba([cleaned])[0][1]) * 100, 2)

    highlighted_text, detected_words = highlight_offensive_words(text)
    masked_text = mask_offensive_words(text) if prediction_label == 1 else text

    label = "Toxic ⚠️" if prediction_label == 1 else "Non-Toxic ✅"

    return {
        "label": label,
        "probability": probability,
        "masked_text": masked_text,
        "highlighted_text": highlighted_text,
        "detected_words": detected_words
    }

# ---------------------------------------------------------
# 8) STREAMLIT UI (sidebar, examples, result layout)
# ---------------------------------------------------------
st.set_page_config(page_title="Advanced Toxic Comment Detector", page_icon="🛡️", layout="wide")

# Sidebar - Project guide and examples
with st.sidebar:
    st.title("📘 Project Guide")
    st.write(
        "Detects toxic/offensive language (English + Telugu), highlights harmful words and masks them "
        "for safe display. Built with ML (TF-IDF + classifier) and a masking dictionary."
    )
    st.subheader("💡 Why this project?")
    st.write(
        "- Reduce cyberbullying & harassment\n"
        "- Auto-protect audiences from abusive language\n"
        "- Demonstrate multilingual NLP (Telugu + English)"
    )
    st.subheader("📝 Examples (copy to main box):")
    st.code("rafi you idiot")
    st.code("rafi బుద్ధిలేని వెధవ")
    st.code("fuck you asshole")
    st.subheader("✅ What to expect")
    st.write("Offensive words will be highlighted and the masked output will show asterisks.")

# Main UI
st.title("🛡️ Advanced Toxic Comment Detection & Masking System")
st.write("Enter a comment (Telugu / English / mixed) to analyze:")

input_comment = st.text_area("Enter your comment here:", height=140)

if st.button("Analyze Comment"):
    if not input_comment.strip():
        st.warning("Please enter a comment to analyze.")
    else:
        result = detect_and_mask_pipeline(input_comment)

        st.markdown("## 🔍 Analysis Result")
        st.write("**Prediction:**", result["label"])
        st.write("**Confidence Score:**", f"{result['probability']}%")

        if result["detected_words"]:
            st.markdown("### 🚨 Detected Offensive Words")
            # show detected words as red tags (error box)
            st.error(", ".join(result["detected_words"]))

            st.markdown("### 🟥 Highlighted Text")
            # highlighted_text contains markdown red highlights
            st.markdown(result["highlighted_text"])
        else:
            st.info("No offensive words found in the text (by dictionary match).")

        st.markdown("### 🛡️ Masked Output")
        st.success(result["masked_text"])

        # Optionally show the final_offensive_word_list count for demonstration
        st.markdown("---")
        st.write(f"Total dictionary entries used for masking: **{len(final_offensive_word_list)}**")
