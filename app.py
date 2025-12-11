import streamlit as st
import joblib
import re
import string
from typing import List, Tuple


# Load ML model and dictionary
try:
    toxicity_model = joblib.load("model/toxic_model.pkl")
except Exception:
    st.error("Model file missing: model/toxic_model.pkl")
    st.stop()

try:
    base_bad_words = joblib.load("model/final_bad_words.pkl")
except Exception:
    base_bad_words = []


# English abusive dictionary (industry-grade)
ENGLISH_STRONG = [
    "fuck","fucking","fucker","fucked","motherfucker","mf","mfker","mfer",
    "shit","bullshit","shitty","crap",
    "dick","dickhead","cock","cocksucker","pussy","cunt",
    "slut","whore","hoe","bitch","bitches","biatch",
    "asshole","asshat","asswipe","dumbass","jackass",
    "retard","fucktard","scumbag","douche","douchebag",
    "moron","idiot","stupid","dumb","loser","jerk","prick",
    "pig","dog","snake","rat","donkey","ape","cockroach","worm",
    "garbage","trash","filth","gutter rat","dirtbag",
    "harami","gandu","chutiya","chutya","madarchod","behenchod","bhenchod",
    "lavde","randi","rand","randwa","kamina","kaminey","kutti","kutta",
    "suck my dick","lick my dick","bend over","sit on my face",
    "fuck you","bang you","hump you",
    "kill yourself","go die","i will kill you","i will beat you",
    "beat your ass","break your face","rip your head off",
    "f@ck","f*ck","f**k","fucc","fvk","phuck","phuk",
    "a$$hole","a55hole","wh0re","sl*t","c*nt","p*ssy",
    "di*k","co*k","fuk","fck","fakk","fuc","fk",
    "stupid fellow","idiot fellow","nonsense fellow","dirty fellow"
]


# Telugu abusive dictionary (clean + expanded)
TELUGU_STRONG = [
    "వెధవ","మూర్ఖుడు","దద్దమ్మ","పిచ్చోడు","పిచ్చివాడు","పిచ్చి",
    "దుర్మార్గుడు","నీచుడు","చెత్తోడు","చెత్త","అసహ్యం","దరిద్రుడు",
    "పిరికివాడు","తిక్కోడు","రాక్షసుడు","గాడిద","లోఫర్",
    "లంజ","రాండీ","దెంగు","దెంగినోడు","దెంగు కొడుకు","పాండీ",
    "అశ్లీలం","బూతు","బూతులాడటం","బూతు మాటలు","యవ్వారం",
    "దొంగమూత","మూత్రపు ముఖం","తల్లి దెంగుడు","నాయాలా",
    "కొడుకు లంజ","నాన్న దెంగుడు","మూతి దాని",
    "మోసగాడు","నోరు సిగ్గులేని","సిగ్గులేని","గుణం లేని",
    "చెత్తబుద్ధి","గొర్రెబుద్ధి","పందికొక్క","కుక్క","పిచ్చికుక్క",
    "చంపేస్తా","పగలగొడతా","తల పగలగొడతా","కాళ్లు విరుస్తా",
    "ముక్కు పగలగొడతా","నిన్ను చంపేస్తా","తల కొడతా","దెబ్బ కొడతా"
]


# Mixed Telugu-English abusive slang (Instagram/WhatsApp real usage)
TELUGU_ENGLISH_MIXED = [
    "vedhava","wedhava","vedava","vedhava fellow",
    "lanja","lanja koduka","lanjakoduka","lanchakoduka",
    "randi pilla","randipilla","pandi","paandi",
    "dengu","denguthaa","dengudu","dengipoya",
    "pichhi fellow","pichi fellow","pichodi","pichodu",
    "budhdhi leni","buddhi leni","buddhilenodu","budhileni",
    "chetta fellow","chetta manishi","neechoduu","neechodu","neechi fellow",
    "stupid panni","dirty lanja","rascal fellow","rowdy fellow",
    "chutiya thopu","pichhi gaadu","tikkafellow","lofer fellow",
    "road rowdy","setting gaadu","waste fellow","mental gaadu"
]


# Merge all offensive dictionaries into STRONG_WORDS
STRONG_WORDS = {w.lower() for w in (
    ENGLISH_STRONG + TELUGU_STRONG + TELUGU_ENGLISH_MIXED
)}


# Safe words (to reduce false positives)
COMMON_SAFE_WORDS = {
    "you","me","we","they","he","she","it","your","our","my","her","him","them",
    "is","are","the","a","an","and","or","if","in","on","at","to","for","of",
    "good","person","nice","great","hello","hi","ok","okay","love","like",
    "help","support","please","bro","anna","dude","friend","buddy"
}


# Add PKL words → filter + classify as weak words
merged_set = set(base_bad_words)
cleaned_pkl_words = set()

for w in merged_set:
    if isinstance(w, str):
        w = w.strip().lower()
        if len(w) > 1 and w not in COMMON_SAFE_WORDS and not all(ch in string.punctuation for ch in w):
            cleaned_pkl_words.add(w)

WEAK_WORDS = cleaned_pkl_words - STRONG_WORDS


# Normalization for slang
COMMON_MISSPELLINGS = {
    "idoit": "idiot","stupit": "stupid","fuk": "fuck","fck": "fuck",
    "bich": "bitch","asshloe": "asshole","mottherfucker": "motherfucker",
    "vedhava": "వెధవ","lanja": "లంజ","dengu": "దెంగు"
}


def preprocess_comment(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\u0C00-\u0C7Fa-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_slang_words(text):
    return " ".join([COMMON_MISSPELLINGS.get(w, w) for w in text.split()])


# Build regex patterns
WORD_CHAR_CLASS = r"A-Za-z0-9\u0C00-\u0C7F"


def build_patterns(words):
    out = []
    for w in words:
        escaped = re.escape(w)
        pat = re.compile(
            rf"(?<![{WORD_CHAR_CLASS}]){escaped}(?![{WORD_CHAR_CLASS}])",
            flags=re.IGNORECASE
        )
        out.append((w, pat))
    return out


PAT_STRONG = build_patterns(sorted(STRONG_WORDS))
PAT_WEAK = build_patterns(sorted(WEAK_WORDS))


def mask_offensive_words(text):
    masked = text
    for _, p in PAT_STRONG + PAT_WEAK:
        masked = p.sub(lambda m: "*" * len(m.group()), masked)
    return masked


def highlight_offensive_words(text):
    detected = []
    highlighted = text

    for _, p in PAT_STRONG + PAT_WEAK:
        matches = list(p.finditer(highlighted))
        if not matches:
            continue

        new_parts, last = [], 0
        for m in matches:
            s, e = m.span()
            new_parts.append(highlighted[last:s])
            word = highlighted[s:e]
            new_parts.append(f"**:red[{word}]**")
            detected.append(word)
            last = e
        new_parts.append(highlighted[last:])
        highlighted = "".join(new_parts)

    # Deduplicate
    seen, unique = set(), []
    for w in detected:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            unique.append(w)

    return highlighted, unique


# Toxicity decision logic
def detect_and_mask_pipeline(text, threshold=50):
    cleaned = normalize_slang_words(preprocess_comment(text))
    prob = round(float(toxicity_model.predict_proba([cleaned])[0][1]) * 100, 2)

    highlighted, detected = highlight_offensive_words(text)

    # Strong dictionary → instant toxic
    for _, p in PAT_STRONG:
        if p.search(text):
            return {
                "label": "Toxic ⚠️ (dictionary)",
                "probability": prob,
                "masked": mask_offensive_words(text),
                "highlighted": highlighted,
                "detected": detected,
                "cleaned": cleaned
            }

    # Weak dictionary
    for _, p in PAT_WEAK:
        if p.search(text):
            return {
                "label": "Toxic ⚠️ (dictionary)",
                "probability": prob,
                "masked": mask_offensive_words(text),
                "highlighted": highlighted,
                "detected": detected,
                "cleaned": cleaned
            }

    # ML model decision
    if prob >= threshold:
        return {
            "label": "Toxic ⚠️ (model)",
            "probability": prob,
            "masked": mask_offensive_words(text),
            "highlighted": highlighted,
            "detected": detected,
            "cleaned": cleaned
        }

    # Non-toxic
    return {
        "label": "Non-Toxic ✅",
        "probability": prob,
        "masked": text,
        "highlighted": highlighted,
        "detected": detected,
        "cleaned": cleaned
    }


# Streamlit UI
st.set_page_config(page_title="Advanced Toxic Comment Detector", page_icon="🛡️", layout="wide")

with st.sidebar:
    st.title("Project Guide")
    st.write("Multilingual toxic comment detector with advanced masking & highlighting.")
    st.subheader("Examples")
    if st.button("Example: rafi you idiot"):
        st.session_state['sample'] = "rafi you idiot"
    if st.button("Example: నువ్వు వెధవ పిచ్చోడు"):
        st.session_state['sample'] = "నువ్వు వెధవ పిచ్చోడు"
    if st.button("Example: fuck you asshole"):
        st.session_state['sample'] = "fuck you asshole"
    if st.button("Example: lokesh you are  good humorous person"):
        st.session_state['sample'] = "lokesh you are  good humorous person"

    st.subheader("Settings")
    threshold = st.slider("Model toxicity threshold", 10, 90, 60)


st.title("🛡️ Toxic Comment Detection & Masking System")

default = st.session_state.get("sample", "")
user_input = st.text_area("Enter comment:", value=default, height=160)

if st.button("Analyze Comment"):
    if not user_input.strip():
        st.warning("Enter some text first.")
    else:
        result = detect_and_mask_pipeline(user_input, threshold)

        st.markdown("## Analysis Result")
        st.write("Prediction:", result["label"])
        st.write("Confidence Score:", f"{result['probability']}%")

        st.markdown("### Cleaned Input")
        st.code(result["cleaned"])

        if result["detected"]:
            st.markdown("### Detected Offensive Words")
            st.error(", ".join(result["detected"]))

            st.markdown("### Highlighted Text")
            st.markdown(result["highlighted"])

        st.markdown("### Masked Output")
        st.code(result["masked"])
