import streamlit as st
import joblib
import re
import string
from typing import List, Tuple
# Safe model & PKL loading (friendly error messages)
try:
    toxicity_model = joblib.load("model/toxic_model.pkl")
except Exception as e:
    st.error("Failed to load model file 'model/toxic_model.pkl'. Make sure that file exists in the repository under /model/")
    st.stop()

try:
    base_bad_words = joblib.load("model/final_bad_words.pkl")
except Exception:
    base_bad_words = [] 
# Curated strong offensive word lists (English + Telugu)
ENGLISH_STRONG = [
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

TELUGU_STRONG = [
    "వెధవ","దెంగు","లంజ","రాండీ","చెత్తోడు","నీచుడు","దుర్మార్గుడు",
    "బుద్ధిలేని","పిచ్చి","బూతు","దొంగ","బూతులాడటం"
    "వెధవ","మూర్ఖుడు","దద్దమ్మ","చెత్తోడు","నాయాలా","దుర్మార్గుడు",
    "నీచుడు","పిచ్చోడు","బుద్ధిలేని","పిచ్చి","పిచ్చివాడు","చెత్త",
    "లంజ","రాండీ","దెంగు","దెంగు కొడుకు","దెంగినోడు","దొంగ","దొంగమూత",
    "మోసగాడు","అసహ్యం","దరిద్రుడు","మూత్రపు ముఖం","తిక్క","తిక్కోడు",
    "పిరికివాడు","వాజ","పంది","పందికొక్క","రాక్షసుడు","అసభ్యుడు","బూతు",
    "బూతులాడటం","బూతు మాటలు","దుష్టుడు","సిగ్గులేని","నోరు సిగ్గులేని",
    "గుణం లేని","యవ్వారం","దుర్మార్గం","దరిద్రపు","రాక్షసపు"
]

STRONG_WORDS = set([w.lower() for w in ENGLISH_STRONG + TELUGU_STRONG])
# 2) Safe / stop words we will remove from PKL dictionary
COMMON_SAFE_WORDS = {
    "you","me","we","they","he","she","it","your","our","my","her","him","them",
    "is","are","the","a","an","and","or","if","in","on","at","to","for","of",
    "good","person","humorous","nice","great","best","better","hello","hi",
    "thanks","thank","ok","okay","love","like","help","support","please"
}

# ---------------------------------------------------------
# Build merged set and clean PKL-derived words
# ---------------------------------------------------------
merged_set = set()

# Add base PKL words (if any)
for w in base_bad_words:
    if isinstance(w, str) and w.strip():
        merged_set.add(w.strip().lower())

# Add curated lists as well (to ensure they are included)
for w in STRONG_WORDS:
    merged_set.add(w)

# Remove safe / too-short tokens and punctuation-only entries
cleaned_pkl_words = set()
for w in merged_set:
    if not isinstance(w, str): 
        continue
    token = w.strip().lower()
    if len(token) <= 1:
        continue
    # drop tokens that are plain safe words
    if token in COMMON_SAFE_WORDS:
        continue
    # drop tokens made of punctuation
    if all(ch in string.punctuation for ch in token):
        continue
    cleaned_pkl_words.add(token)

# Build weak-word set by excluding strong words (strong words remain authoritative)
WEAK_WORDS = cleaned_pkl_words - STRONG_WORDS

# Final combined offensive set for display / counts (we keep types separate in logic)
FINAL_OFFENSIVE_WORD_LIST = sorted(list(STRONG_WORDS | WEAK_WORDS))
# 4) Preprocessing & normalization (misspellings map)
COMMON_MISSPELLINGS = {
    "idoit": "idiot", "stupit": "stupid", "fuk": "fuck", "fck": "fuck",
    "bich": "bitch", "asshloe": "asshole", "mottherfucker": "motherfucker"
}

def preprocess_comment(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    # keep Telugu block \u0C00-\u0C7F and English letters and spaces
    text = re.sub(r"[^\u0C00-\u0C7Fa-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_slang_words(text: str) -> str:
    return " ".join([COMMON_MISSPELLINGS.get(tok, tok) for tok in text.split()])

# Compile robust regex patterns (Unicode-aware boundaries)
# Use a char class for English letters/digits + Telugu block.
# Patterns use negative lookbehind/lookahead to create boundaries that work for Telugu.
WORD_CHAR_CLASS = r"A-Za-z0-9\u0C00-\u0C7F"

def compile_patterns_from_list(words_iterable):
    patterns = []
    for w in words_iterable:
        if not w or len(w.strip()) == 0:
            continue
        escaped = re.escape(w)
        # (?<![...])word(?![...]) ensures not in the middle of other word/telugu characters
        pattern = re.compile(rf"(?<![{WORD_CHAR_CLASS}]){escaped}(?![{WORD_CHAR_CLASS}])", flags=re.IGNORECASE)
        patterns.append((w, pattern))
    return patterns

PAT_STRONG = compile_patterns_from_list(sorted(STRONG_WORDS))
PAT_WEAK = compile_patterns_from_list(sorted(WEAK_WORDS))

# ---------------------------------------------------------
# 6) Masking & highlighting helpers
# ---------------------------------------------------------
def mask_offensive_words(original_text: str) -> str:
    masked = original_text
    # Replace strong first (so stronger terms get replaced even if overlapping)
    for _, pat in PAT_STRONG + PAT_WEAK:
        masked = pat.sub(lambda m: "*" * len(m.group(0)), masked)
    return masked

def highlight_offensive_words(original_text: str) -> Tuple[str, List[str]]:
    highlighted = original_text
    detected = []
    # perform pattern-by-pattern safe replacements (left-to-right rebuild)
    for _, pat in PAT_STRONG + PAT_WEAK:
        matches = list(pat.finditer(highlighted))
        if not matches:
            continue
        new_parts = []
        last = 0
        for m in matches:
            s, e = m.span()
            new_parts.append(highlighted[last:s])
            matched = highlighted[s:e]
            new_parts.append(f"**:red[{matched}]**")
            detected.append(matched)
            last = e
        new_parts.append(highlighted[last:])
        highlighted = "".join(new_parts)
    # dedupe preserving order (case-insensitive)
    seen = set()
    unique_detected = []
    for tok in detected:
        key = tok.lower()
        if key not in seen:
            seen.add(key)
            unique_detected.append(tok)
    return highlighted, unique_detected

# ---------------------------------------------------------
# 7) Hybrid detection pipeline (strong -> weak -> ML)
# Returns label, prob, masked_text, highlighted_text, detected_words, cleaned_input
# ---------------------------------------------------------
def detect_and_mask_pipeline(text: str, threshold: float = 50.0):
    cleaned = normalize_slang_words(preprocess_comment(text))
    # model prediction probability
    prob = round(float(toxicity_model.predict_proba([cleaned])[0][1]) * 100, 2)
    model_pred = 1 if prob >= threshold else 0

    # highlight and detect dictionary matches
    highlighted_text, detected_words = highlight_offensive_words(text)

    # strong-word check (instant toxic)
    strong_hits = []
    for word, pat in PAT_STRONG:
        if pat.search(text):
            strong_hits.append(word)

    if strong_hits:
        masked = mask_offensive_words(text)
        return {
            "label": "Toxic ⚠️ (strong-word)",
            "probability": prob,
            "masked_text": masked,
            "highlighted_text": highlighted_text,
            "detected_words": detected_words,
            "cleaned": cleaned,
            "source": "strong-dict"
        }

    # weak-word check
    weak_hits = []
    for word, pat in PAT_WEAK:
        if pat.search(text):
            weak_hits.append(word)

    # final decision logic:
    # if any weak hit → toxic (dictionary-driven)
    # else if model probability >= threshold → toxic
    # otherwise non-toxic
    if weak_hits:
        masked = mask_offensive_words(text)
        return {
            "label": "Toxic ⚠️ (weak-dict)",
            "probability": prob,
            "masked_text": masked,
            "highlighted_text": highlighted_text,
            "detected_words": detected_words,
            "cleaned": cleaned,
            "source": "weak-dict"
        }

    if model_pred == 1:
        masked = mask_offensive_words(text)
        return {
            "label": "Toxic ⚠️ (model)",
            "probability": prob,
            "masked_text": masked,
            "highlighted_text": highlighted_text,
            "detected_words": detected_words,
            "cleaned": cleaned,
            "source": "model"
        }

    # non-toxic
    return {
        "label": "Non-Toxic ✅",
        "probability": prob,
        "masked_text": text,
        "highlighted_text": highlighted_text,
        "detected_words": detected_words,
        "cleaned": cleaned,
        "source": "none"
    }

# ---------------------------------------------------------
# 8) Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Advanced Toxicity Detector", page_icon="🛡️", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📘 Project Guide")
    st.write("Detects toxic/offensive language (English + Telugu), highlights harmful words and masks them for safe display.")
    st.subheader("💡 Why this project?")
    st.write("- Reduce cyberbullying & harassment\n- Auto-protect audiences\n- Demonstrate multilingual NLP (Telugu + English)")
    st.subheader("📝 Examples (click to copy then press Analyze):")
    if st.button("Example 1: rafi you idiot"):
        st.session_state['sample'] = "rafi you idiot"
    if st.button("Example 2: rafi you బుద్ధిలేని వెధవ"):
        st.session_state['sample'] = "rafi you బుద్ధిలేని వెధవ"
    if st.button("Example 3: fuck you asshole"):
        st.session_state['sample'] = "fuck you asshole"

    st.markdown("---")
    st.subheader("⚙️ Settings")
    threshold = st.slider("Toxicity threshold (%)", 10, 90, 60, 5)
    st.write("Threshold controls the minimum model probability (in %) to consider model-based toxic predictions.")
    st.markdown("---")
    st.write(f"Curated strong entries: **{len(STRONG_WORDS)}**")
    st.write(f"Weak PKL-derived entries (filtered): **{len(WEAK_WORDS)}**")
    st.write(f"Total dictionary entries available: **{len(FINAL_OFFENSIVE_WORD_LIST)}**")

# Main area
st.title("🛡️ Advanced Toxic Comment Detection & Masking System")
st.write("Enter a comment (Telugu / English / mixed) to analyze:")

default_text = st.session_state.get('sample', "")
input_comment = st.text_area("Enter your comment here:", value=default_text, height=160)

if st.button("Analyze Comment"):
    if not input_comment.strip():
        st.warning("Please enter a comment to analyze.")
    else:
        result = detect_and_mask_pipeline(input_comment, threshold=threshold)
        # Display results
        st.markdown("## 🔍 Analysis Result")
        st.write("**Prediction:**", result["label"])
        st.write("**Confidence Score:**", f"{result['probability']}%")
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
        st.code(result["masked_text"], language="text")

        # download report
        report = (
            f"Original: {input_comment}\n"
            f"Cleaned: {result['cleaned']}\n"
            f"Prediction: {result['label']}\n"
            f"Confidence: {result['probability']}%\n"
            f"Detected: {', '.join(result['detected_words'])}\n"
            f"Masked: {result['masked_text']}\n"
            f"Decision source: {result['source']}\n"
        )
        st.download_button("Download report (txt)", report, file_name="toxicity_report.txt")
