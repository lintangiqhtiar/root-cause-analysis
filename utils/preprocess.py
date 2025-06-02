import pandas as pd
import re

# Fungsi membersihkan teks
def clean_review(text):
    if not text: return ""
    text = re.sub(r'<.*?>', '', text)  # Hapus HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Hapus simbol
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
    return text.lower()  # Ubah menjadi huruf kecil (case folding)

# Load kamus normalisasi
normalisasi_kata_df = pd.read_csv('data/normalisasi-new.csv')
normalisasi_kata_dict = dict(zip(normalisasi_kata_df['before'], normalisasi_kata_df['after']))

def normalisasi(text):
    words = text.split()  # Pisahkan teks menjadi kata-kata
    return " ".join([normalisasi_kata_dict.get(word, word) for word in words])  # Gabungkan kembali menjadi string

# Fungsi utama untuk preprocessing
def preprocess_data(df):
    df = df.dropna().copy()  # Hindari mengubah DataFrame asli
    df["Review"] = df["Review"].apply(clean_review)  # Membersihkan review
    df["Review"] = df["Review"].apply(normalisasi)  # Normalisasi kata

    #Menghapus kolom confidence
    df.drop(['confidence'], inplace=True, axis ='columns')

    # Mengambil hanya data dengan label 'Negative'
    df_negatif = df[df["label"] == "Negative"].reset_index(drop=True)
    return df_negatif
