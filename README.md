# 🛡️ Toxic Comment Detection & Automatic Masking System  
### Machine Learning & Natural Language Processing Project

---

## 📌 Project Overview

Social media platforms such as Instagram, YouTube, and Twitter receive millions of comments every day. While many comments are positive and supportive, a large number of them are abusive, hateful, and toxic. These toxic comments negatively affect mental health and create unsafe online environments.

This project presents a **Machine Learning–based Toxic Comment Detection and Automatic Masking System**. The system first detects whether a comment is toxic or non-toxic using a trained machine learning model. If the comment is found to be toxic, the system automatically masks the offensive words using asterisks (`****`) so that harmful content is hidden.

This project supports:
- ✅ English language  
- ✅ Telugu language  
- ✅ Mixed (English + Telugu) code-mixed comments  

It demonstrates a real-world application of **Machine Learning + Natural Language Processing (NLP)** for online content safety.

---

## 🎯 Objectives of the Project

- To automatically detect toxic and non-toxic comments  
- To build an ML-based classification system  
- To support English and Telugu mixed text  
- To correct misspelled abusive words like `idoit → idiot`  
- To detect disguised abusive words like `f*ck`, `idi0t`  
- To automatically mask abusive words  
- To display prediction with confidence score  

---

## 🧠 How the System Works (Simple Explanation)

The project works in two main stages:

### ✅ 1. Toxicity Detection using Machine Learning
- A real-world dataset is used to train the model.
- Text is converted into numerical features using **TF-IDF Vectorizer**.
- A **Logistic Regression** model is trained for classification.
- For every new comment, the model predicts:
  - Toxic or Non-Toxic
  - Confidence probability

### ✅ 2. Automatic Toxic Word Masking
- Offensive words are:
  - Automatically extracted from toxic comments
  - Manually added in English and Telugu
  - Expanded using disguised and misspelled patterns
- If the comment is toxic:
  - Offensive words are replaced with `****`
- If the comment is non-toxic:
  - The original comment is displayed

---

## ⚙️ Technologies & Tools Used

- Python  
- Pandas & NumPy – Data processing  
- Scikit-learn – Machine Learning  
- TF-IDF Vectorizer – Feature extraction  
- Logistic Regression – Classification  
- Regular Expressions (Regex) – Word masking  
- Jupyter Notebook – Model development  
- Streamlit – Web application  
- GitHub – Version control & deployment  

---

## 📂 Dataset Description

This project uses the **Jigsaw Toxic Comment Classification Dataset**.

The dataset contains the following columns:
- `comment_text`
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

All these labels are combined into one binary label:
- `0 → Non-Toxic`
- `1 → Toxic`

The dataset is **balanced before training** to improve model performance.

---

## 🔍 Key Features of the Project

- ✅ Machine Learning based toxic comment detection  
- ✅ Supports English and Telugu language  
- ✅ Handles misspelled abusive words  
- ✅ Handles disguised abusive word patterns  
- ✅ Automatic toxic word masking  
- ✅ Displays prediction with confidence score  
- ✅ Works in real-time as a web application  
- ✅ Industry-style detection + censorship pipeline  

---

## 🧪 Sample Input & Output

### Input:
fuck you idoit, నువ్వు వెధవ


### Output:

Prediction: Toxic ⚠️
Confidence: 67.84 %
Masked Output: **** you *****, నువ్వు ****



---

## 📁 Project Folder Structure


toxic-comment-masking/
├── app.py

├── requirements.txt

├── README.md

├── train.csv

├──README.md

└── model/

  ├── toxic_model.pkl
  
  └── final_bad_words.pkl



---

## ▶️ How to Run the Project Locally

### ✅ Step 1: Install Required Libraries
pip install -r requirements.txt


### ✅ Step 2: Run the Web App
streamlit run app.py


### ✅ Step 3: Open the Browser
A local link will open automatically:
http://localhost:8501


---

## 🌐 Live Deployment (If Deployed on Streamlit Cloud)

Once deployed, the project will be available at:
[https://your-project-name.streamlit.app](https://toxic-comment-masking.streamlit.app/)


This link can be shared with:
- Professors
- Friends
- Recruiters

---

## 🎓 Academic Importance of This Project

This project:
- Uses a real-world dataset  
- Implements Machine Learning classification  
- Demonstrates NLP preprocessing  
- Combines ML with rule-based masking  
- Solves a real social media safety problem  

It is suitable for:
- ✅ Final Year Project  
- ✅ Mini Project  
- ✅ Machine Learning Lab  
- ✅ NLP Project Demonstration  

---

## 🚀 Future Enhancements

- Add Deep Learning models like **LSTM / BERT**
- Support more Indian languages  
- Integrate with live social media APIs  
- Deploy as a mobile application  
- Add image and video toxic content detection  

---

## ✅ Final Conclusion

This project successfully demonstrates how **Machine Learning and Natural Language Processing can be used together to automatically detect and mask toxic comments**. It provides a complete end-to-end solution from data preprocessing to live web deployment.

---

## 👤 Author

**Name:** Viraj  
**Project Type:** Machine Learning & NLP  
**Year:** 2025
---
