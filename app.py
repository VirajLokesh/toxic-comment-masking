import streamlit as st
import joblib
import re
# Load Saved ML Model & Base Word List
toxicity_model = joblib.load("model/toxic_model.pkl")
base_bad_words = joblib.load("model/final_bad_words.pkl")
# Massive English + Telugu Offensive Dictionary
english_offensive_words = [
    "fuck","fucking","fucker","fucked","mf","motherfucker","bitch","bitches",
    "bich","biatch","slut","whore","hoe","cunt","pussy","dick","cock","prick",
    "asshole","asshat","asswipe","dumbass","jackass","shit","shitty","bullshit",
    "crap","bastard","jerk","retard","fucktard","douche","douchebag","moron",
    "idiot","stupid","dumb","loser","scumbag","piss off","pisshead","pissbrain",
    "pissface","twat","skank","tramp","pig","donkey","airhead","brainless",
    "jerkoff","cocksucker","dickhead","wtf","hell","damn","f@ck","a$$hole",
    "sl*t","wh0re","c*nt","p*ssy","di*k","co*k","fuk","fck","fucc","fvk","fukr",
    "harami","gandu","kutti","kutta","rand","randi"
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
# Merge PKL + manual lists
final_offensive_word_list = set(base_bad_words)
final_offensive_word_list.update(english_offensive_words)
final_offensive_word_list.update(telugu_offensive_words)
final_offensive_word_list = list(final_offensive_word_list)
# Misspellings Normalize
COMMON_MISSPELLINGS = {
    "idoit": "idiot", "stupit": "stupid", "fuk": "fuck", "fck": "fuck",
    "bich": "bitch", "asshloe": "asshole"
}
def preprocess_comment(comment_text):
    comment_text = comment_text.lower()
    comment_text = re.sub(r"http\S+|@\w+|#\w+", "", comment_text)
    comment_text = re.sub(r"[^\u0C00-\u0C7Fa-zA-Z\s]", "", comment_text)
    comment_text = re.sub(r"\s+", " ", comment_text).strip()
    return comment_text

def normalize_slang_words(clean_text):
    words = clean_text.split()
    fixed = [COMMON_MISSPELLINGS.get(w, w) for w in words]
    return " ".join(fixed)
# Compile Regex Patterns (HIGH ACCURACY)
mask_patterns = []

for word in final_offensive_word_list:
    if isinstance(word, str) and len(word.strip()) > 1:
        pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
        mask_patterns.append(pattern)
# Masking Function
def mask_offensive_words(original):
    masked = original
    for pattern in mask_patterns:
        masked = pattern.sub(lambda m: "*" * len(m.group()), masked)
    return masked
# Final Toxicity Pipeline
def detect_and_mask(text):
    cleaned = normalize_slang_words(preprocess_comment(text))
    label = toxicity_model.predict([cleaned])[0]
    score = toxicity_model.predict_proba([cleaned])[0][1]

    if label == 1:
        masked = mask_offensive_words(text)
        return "Toxic ⚠️", round(score * 100, 2), masked
    else:
        return "Non-Toxic ✅", round(score * 100, 2), text
# Streamlit UI
st.set_page_config(page_title="Toxic Comment Detector", page_icon="🛡️")

st.title("🛡️ Toxic Comment Detection & Masking System")
st.write("Enter a comment below to check if it is toxic and automatically mask offensive words.")

user_input = st.text_area("Enter your comment:")

if st.button("Check Toxicity"):
    if not user_input.strip():
        st.warning("Please enter a comment.")
    else:
        label, score, masked = detect_and_mask(user_input)
        st.subheader("Result:")
        st.write("Prediction:", label)
        st.write("Confidence Score:", str(score) + "%")
        st.success(masked)
