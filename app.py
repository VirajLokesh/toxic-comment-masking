import streamlit as st
import joblib
import re

#LOAD TRAINED MODEL + BASE OFFENSIVE WORD LIST

toxicity_model = joblib.load("model/toxic_model.pkl")
base_bad_words = joblib.load("model/final_bad_words.pkl")  # extracted from dataset
# ADD ENGLISH + TELUGU OFFENSIVE WORD LISTS (Industry-grade)
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
# 3) MERGE EVERYTHING INTO FINAL OFFENSIVE WORD SET

SAFE_WORDS = {
    "you","me","we","they","he","she","it","your","our","my","her","him","them"
}

final_offensive_word_list = set()

# Add dataset extracted bad words
for word in base_bad_words:
    if isinstance(word, str):
        final_offensive_word_list.add(word.lower())

# Add English list
for word in english_offensive_words:
    final_offensive_word_list.add(word.lower())

# Add Telugu list
for word in telugu_offensive_words:
    final_offensive_word_list.add(word.lower())

# Remove safe/common words
final_offensive_word_list = {w for w in final_offensive_word_list if w not in SAFE_WORDS
                             
# TEXT PREPROCESSING + NORMALIZATION
                             
COMMON_MISSPELLINGS = {
    "idoit": "idiot", "stupit": "stupid", "fuk": "fuck",
    "fck": "fuck", "bich": "bitch", "asshloe": "asshole"
}

def preprocess_comment(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\u0C00-\u0C7Fa-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_slang_words(text):
    return " ".join([COMMON_MISSPELLINGS.get(w, w) for w in text.split()])

# 5) COMPILE REGEX PATTERNS FOR MASKING + HIGHLIGHTING

mask_patterns = []

for word in final_offensive_word_list:
    if len(word) > 1:
        mask_patterns.append(
            re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
        )

# 6) MASKING + HIGHLIGHTING FUNCTIONS

def mask_offensive_words(text):
    masked = text
    for pattern in mask_patterns:
        masked = pattern.sub(lambda m: "*" * len(m.group()), masked)
    return masked

def highlight_offensive_words(text):
    highlighted = text
    found_words = set()

    for pattern in mask_patterns:
        matches = pattern.findall(text)
        for w in matches:
            found_words.add(w)
            highlighted = pattern.sub(f"**:red[{w}]**", highlighted)

    return highlighted, list(found_words)

# FINAL PREDICTION + PROCESSING PIPELINE
def detect_and_mask(text):
    cleaned_text = normalize_slang_words(preprocess_comment(text))

    prediction_label = toxicity_model.predict([cleaned_text])[0]
    probability = round(toxicity_model.predict_proba([cleaned_text])[0][1] * 100, 2)

    highlighted, found_words = highlight_offensive_words(text)

    if prediction_label == 1:
        masked = mask_offensive_words(text)
        final_label = "Toxic ⚠️"
    else:
        masked = text
        final_label = "Non-Toxic ✅"

    return final_label, probability, masked, highlighted, found_words

# STREAMLIT UI (Professional Industry-Grade)

st.set_page_config(page_title="Advanced Toxicity Detector", page_icon="🛡️", layout="wide")

st.title("🛡️ Advanced Toxic Comment Detection & Masking System")
st.write("Detects, highlights, and masks abusive language in **English & Telugu** using ML & NLP.")

# Comment input
user_input = st.text_area("Enter your comment:", height=150)

if st.button("Analyze Comment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a comment.")
    else:
        label, score, masked_output, highlighted_text, found_words = detect_and_mask(user_input)

        st.subheader("🔍 Analysis Result")
        st.write("**Prediction:**", label)
        st.write("**Confidence Score:**", score, "%")

        # Offensive words found
        if found_words:
            st.markdown("### 🚨 Detected Offensive Words:")
            st.error(", ".join(found_words))

            st.markdown("### 🟥 Highlighted Text:")
            st.markdown(highlighted_text)
        else:
            st.info("No offensive words found.")

        st.markdown("### 🛡️ Masked Output:")
        st.success(masked_output)
