import os
import torch
import gdown
import zipfile
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

@st.cache_resource
def load_model():
    model_dir = "models/gpt2-20test"
    zip_path = "models/gpt2-20test.zip"
    
    # File ID dari Google Drive
    file_id = "1nwYSGqTLr-H1nHGP54WgQ99db2dVkreU"  # Ganti dengan file ID kamu
    gdrive_url = f"https://drive.google.com/uc?id={file_id}"

    # Cek apakah model sudah ada
    if not os.path.exists(model_dir):
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(zip_path):
            with st.spinner("Mengunduh model dari Google Drive..."):
                gdown.download(gdrive_url, zip_path, quiet=False)

        with st.spinner("Mengekstrak model..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("models")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device

# Inisialisasi model
model, tokenizer, device = load_model()

def generate_recommendation(text):
    model.eval()
    prompt = f"{text.strip()}\nRekomendasi:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Rekomendasi:" in decoded:
        final_output = decoded.split("Rekomendasi:")[-1].strip()
    else:
        final_output = decoded.strip()
    
    return f"Rekomendasi: {final_output}"
