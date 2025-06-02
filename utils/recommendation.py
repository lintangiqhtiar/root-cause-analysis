import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from peft import PeftModel, PeftConfig, AutoPeftModelForSeq2SeqLM

# # Load model IndoBERT
# model_path = "models/indobert_model"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Load model Bloom
#@st.cache_resource
# def load_model():
#     config = PeftConfig.from_pretrained("models/bloomz-finetuned-model")
#     base_model = AutoPeftModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
#     model = PeftModel.from_pretrained(base_model, "models/bloomz-finetuned-model")
#     tokenizer= AutoTokenizer.from_pretrained(config.base_model_name_or_path)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     return model, tokenizer, device
# Load model Bloom dengan adapter lokal

#@st.cache_resource
# def load_model():
#     model_dir = "models/bloomz-finetuned-model"  # path ke adapter lokal
#     base_model_path = "models/base/bloomz-1b7"   # path ke model dasar lokal
    
#     config = PeftConfig.from_pretrained(model_dir)

#     # Load base model (dari path lokal)
#     base_model = AutoPeftModelForSeq2SeqLM.from_pretrained(base_model_path)
#     model = PeftModel.from_pretrained(base_model, model_dir)
#     tokenizer = AutoTokenizer.from_pretrained(base_model_path)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     return model, tokenizer, device

# def load_model():
#     config = PeftConfig.from_pretrained("models/bloomz-finetuned-model")
#     base_model = AutoPeftModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
#     model = PeftModel.from_pretrained(base_model, "models/bloomz-finetuned-model")
#     tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     return model, tokenizer, device

# model, tokenizer, device = load_model()
# # Load data training
# df = pd.read_csv("data/dataset_keluhan_rekomendasi.csv")  # Gantilah dengan path file dataset Anda

# # Ambil daftar unik rekomendasi
# label_mapping = df["Rekomendasi"].unique().tolist()

# # Buat dan latih LabelEncoder
# label_encoder = LabelEncoder()
# label_encoder.fit(label_mapping)


# # Fungsi untuk melakukan prediksi
# def predict(text):
#     model.eval()
#     encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)

#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask)
#         pred = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1).cpu().numpy()[0]

#     return label_encoder.inverse_transform([pred])[0]

#Fungsi untuk melakukan prediksi (Generatif) Bloom
# def generate_recommendation(text):
#     model.eval()
#     inputs = tokenizer(text, return_tensor = "pt",truncation = True, padding=True).to(device)

#     with torch.no_grad():
#         output = model.generate(**inputs, max_length=100)
#         decode = tokenizer.decode(output[0], skip_special_tokens = True)
    
#     return decode

"""
pakai gpt2


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model dan tokenizer sekali saja saat import
model_path = "models/gpt2-indonesian-finetuned"  # sesuaikan dengan path kamu

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate_recommendation(keluhan):
    prompt = f"Keluhan: {keluhan} Rekomendasi:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=10,
            do_sample=True,
            top_k=10,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Load model dan tokenizer
def load_model():
    # model_path = "models/gpt2-indonesian-finetuned"  # Path ke model yang sudah kamu fine-tune
    model_path = "models/gpt2-20test"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Inisialisasi model saat modul ini di-import
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

    # Ambil bagian setelah kata "Rekomendasi:"
    if "Rekomendasi:" in decoded:
        final_output = decoded.split("Rekomendasi:")[-1].strip()
    else:
        final_output = decoded.strip()
    
    return f"Rekomendasi: {final_output}"

