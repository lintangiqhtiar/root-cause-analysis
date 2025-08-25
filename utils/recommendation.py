import torch
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# Load model dan tokenizer
def load_model():
    # model_path = "models/gpt2-indonesian-finetuned"  # Path ke model yang sudah kamu fine-tune
    # model_path = "models/gpt2rev-aug20test"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("lintangiqhtiar/my-finetuned-model")
    model = AutoModelForCausalLM.from_pretrained("lintangiqhtiar/my-finetuned-model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# from huggingface_hub import hf_hub_download
# model_path = hf_hub_download(repo_id="username/my-finetuned-model", filename="pytorch_model.bin")


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

