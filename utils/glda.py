import os
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser
from lda import guidedlda
import os


import pickle

try:
    with open("models/glda_model-fix2.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    
# Load Stopwords
stopword_path = "indonesian_stopword_1.txt"
if os.path.exists(stopword_path):
    with open(stopword_path, 'r') as file:
        indonesian_stop_words = [line.strip() for line in file.readlines()]
else:
    raise FileNotFoundError(f"Stopword list tidak ditemukan di {stopword_path}")

# Fungsi untuk menghapus stopwords
def remove_stopwords(text, stopwords):
    words = text.split()
    words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(words)

# Fungsi preprocessing data
def preprocessing(df):
    df = df.copy()
    df['clean_text'] = df['Review'].apply(lambda x: remove_stopwords(x, indonesian_stop_words))
    return df[['clean_text']]

# Fungsi untuk membuat bigram & trigram
def bigram_trigram(df):
    data = preprocessing(df)
    sentences = [text.split() for text in data['clean_text']]
    
    bigram = Phrases(sentences, min_count=2, threshold=5)
    trigram = Phrases(bigram[sentences], threshold=5)
    bigram_model = Phraser(bigram)
    trigram_model = Phraser(trigram)

    data['clean_text_trigram'] = [' '.join(trigram_model[bigram_model[sent]]) for sent in sentences]
    return data['clean_text_trigram']

# Fungsi untuk memproses LDA
def hasil_lda(df):
    data = bigram_trigram(df)
    corpus = data.tolist()
    
    with open("models/vectorizer-fix2.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    X = vectorizer.transform(corpus)
    # Mapping kata ke indeks
    word2id = vectorizer.vocabulary_
    vocab = list(vectorizer.get_feature_names_out())


    seed_topic_list = {
        "Login & Pendaftaran": ["login", "verifikasi", "daftar", "registrasi", "akun", "otp", "sms", "verifikasi_wajah", "KTP", "NIK"], #Content
        "Fitur Aplikasi": ["perpanjangan", "layanan", "SIM", "fitur", "pengajuan", "unduh", "update", "pengembangan"], #Content => mencakup layanan yang disediakan aplikasi setelah pengguna berhasil login
        "Kebijakan & Proses": ["biaya", "proses", "prosedur", "hasil", "persyaratan", "ketentuan", "pembayaran", "tolak", "ditolak", "validasi", "satpas"], #Accuracy
        "Tampilan & Desain": ["UI", "UX", "mudah", "susah", "tampilan", "bug", "layout", "warna", "design", "grafis", "tema", "interface", "desain"], #Format
        "Aksesibilitas": ["bingung", "digunakan", "akses", "tombol", "button", "navigasi", "sulit", "kesulitan"], #Easy of use
        "Kecepatan & Respons": ["cepat", "lambat", "loading", "respons", "lag", "timeout", "stuck", "hang", "lelet", "responsif", "crash", "timeout"] #Timeliness erorr
    }

    # Menyesuaikan dengan vocab
    for key in seed_topic_list.keys():
        seed_topic_list[key] = [word for word in seed_topic_list[key] if word in word2id]

    print(dir(loaded_model))  # Cek atribut yang tersedia dalam model

    # Prediksi topik
    topic_word = loaded_model.components_
    n_top_words = 6
    seed_topic_dict = {}

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        seed_topic_dict[list(seed_topic_list.keys())[i]] = list(topic_words)

    # Simpan hasil ke JSON
    json_path = "data/hasil_topik.json"
    with open(json_path, "w") as f:
        json.dump(seed_topic_dict, f, indent=4)

    return json_path
