import streamlit as st
import joblib
import re

# ----------------------------------
# Load saved model & offensive words
# ----------------------------------
toxicity_model = joblib.load("model/toxic_model.pkl")
final_offensive_word_list = joblib.load("model/final_bad_words.pkl")

# ----------------------------------
# Text Preprocessing Functions
# ----------------------------------
COMMON_MISSPELLING_MAP = {
    "idoit": "idiot", "stupit": "stupid", "fuk": "fuck",
    "fck": "fuck", "bich": "bitch", "asshloe": "asshole"
}

def preprocess_comment(comment_text):
    comment_text = comment_text.lower()
    comment_text = re.sub(r"http\S+|@\w+|#\w+", "", comment_text)
    comment_text = re.sub(r"[^\u0C00-\u0C7Fa-zA-Z\s]", "", comment_text)
    comment_text = re.sub(r"\s+", " ", comment_text).strip()
    return comment_text

def normalize_slang_words(clean_text):
    words = clean_text.split()
    corrected_words = [COMMON_MISSPELLING_MAP.get(word, word) for word in words]
    return " ".join(corrected_words)

# ----------------------------------
# Compile Regex for Fast Masking
# ----------------------------------
compiled_masking_patterns = []
for word in final_offensive_word_list:
    if len(word) > 2:
        compiled_masking_patterns.append(re.compile(re.escape(word), flags=re.IGNORECASE))

# ----------------------------------
# Masking Function
# ----------------------------------
def advanced_toxic_word_masking(original_text):
    masked_text = original_text
    for pattern in compiled_masking_patterns:
        masked_text = pattern.sub(lambda match: "*" * len(match.group(0)), masked_text)
    return masked_text

# ----------------------------------
# Final Detection + Masking Function
# ----------------------------------
def detect_and_mask_toxicity(input_text):
    cleaned_text = normalize_slang_words(preprocess_comment(input_text))

    prediction_label = toxicity_model.predict([cleaned_text])[0]
    prediction_probability = toxicity_model.predict_proba([cleaned_text])[0][1]

    if prediction_label == 1:
        masked_output_text = advanced_toxic_word_masking(input_text)
        final_label = "Toxic ⚠️"
    else:
        masked_output_text = input_text
        final_label = "Non-Toxic ✅"

    return final_label, round(prediction_probability * 100, 2), masked_output_text

# ----------------------------------
# Streamlit UI Design
# ----------------------------------
st.set_page_config(page_title="Toxic Comment Detector", page_icon="🛡️")

st.title("🛡️ Toxic Comment Detection & Masking System")
st.write("Enter a comment below to check if it is toxic and automatically mask offensive words.")

user_input = st.text_area("Enter your comment here:")

if st.button("Check Toxicity"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        label, score, masked_output = detect_and_mask_toxicity(user_input)

        st.subheader("Result:")
        st.write("Prediction:", label)
        st.write("Confidence Score:", str(score) + "%")
        st.write("Masked Output:")
        st.success(masked_output)
