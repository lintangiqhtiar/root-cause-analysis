# Application Service Improvement via Root Cause Analysis of User Review Sentiment: A Case Study on Digital Korlantas Polri 👮‍♂️

## 💡 Business Understanding
Digital Korlantas POLRI is a mobile application that helps users renew their driving licenses in Indonesia.
The app has received 148K+ reviews on the Google Play Store with an average rating of 3.8 stars.

At first glance, this rating suggests acceptable performance. However, many users still report negative experiences, even when leaving 5-star reviews. This inconsistency highlights the importance of conducting a Root Cause Analysis (RCA) to identify underlying issues.

This project leverages LDA and Guided LDA to extract hidden topics from user comments and uses a fine-tuned GPT-2 model to generate automated recommendations for users experiencing problems.

## 🎯 Adjective
1. Perform Root Cause Analysis using topic modeling on user review datasets.
2. Develop a recommendation chatbot powered by GPT-2 to suggest actions for user-reported issues.

## 📑 Project Scope
1. Dataset obtained by scraping reviews from the Digital Korlantas POLRI app on Google Play Store.
2. Reviews undergo sentiment analysis using a pre-trained BERT model.
3. Only negative-labeled reviews are included for RCA.
4. Perform topic modeling using LDA and Guided LDA.
5. Integrate a fine-tuned GPT-2 model to deliver automated recommendations for user issues.

## 💻 Setup & Preparation
You can download this repository and instal the requirement by using this code below
Setup environment:
Make enviroment
```
python -m venv .env
```
Activate the enviroment
```
.env\Scripts\Activate
```
Install library
```
pip install -r requirements.txt
```

## 👩‍💻 Running the Prototype
make sure you have intall streamlit on your environment by
```
pip install streamlit
```
If all set you can run this code below on your terminal
```
streamlit run app.py
```

## 💎 Conslusion
This project develops a prototype platform to compare the performance of LDA and GuidedLDA for topic modeling in Root Cause Analysis. It also includes a fine-tuned GPT-2 model to serve as a recommendation chatbot for users who submit complaints.

## ⚠ Notes
Ensure all required libraries are installed before running the project.
The Guided LDA model may require additional setup. Reference implementation: [GuidedLDA_WorkAround.](https://github.com/dex314/GuidedLDA_WorkAround)
