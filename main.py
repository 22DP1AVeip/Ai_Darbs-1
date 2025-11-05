import os
import time
import json
import requests
from dotenv import load_dotenv

DEFAULT_MODEL = "HuggingFaceH4/zephyr-7b-beta"
HF_API_BASE = "https://router.huggingface.co/hf-inference/models"

def load_token():
    load_dotenv()
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

def query_model(model, prompt, token):
    url = f"{HF_API_BASE}/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt}
    for _ in range(3):
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            break
        time.sleep(1.5)
    if resp.status_code != 200:
        return f"Kļūda: {resp.status_code} - {resp.text}"
    try:
        data = resp.json()
    except:
        return resp.text
    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    if isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"].strip()
        if "error" in data:
            return f"Kļūda: {data['error']}"
        if "choices" in data:
            c = data["choices"][0]
            if "message" in c and "content" in c["message"]:
                return c["message"]["content"].strip()
    return str(data)

def summarize_text(text, token):
    prompt = f"Summarize this text briefly:\n{text}"
    return query_model(DEFAULT_MODEL, prompt, token)

def generate_keywords(text, n, token):
    prompt = f"Extract {n} descriptive keywords from this text:\n{text}\nKeywords:"
    return query_model(DEFAULT_MODEL, prompt, token)

def generate_quiz(text, token):
    prompt = f"Generate 3 multiple-choice quiz questions with 4 options each based on this text:\n{text}\nInclude the correct answer for each question."
    return query_model(DEFAULT_MODEL, prompt, token)

if __name__ == "__main__":
    token = load_token()
    if not token:
        raise ValueError("HF_TOKEN vai HUGGINGFACE_API_KEY nav atrasts .env failā")
    file = input("Ievadi .txt faila nosaukumu: ")
    if not file.endswith(".txt"):
        file += ".txt"
    if not os.path.exists(file):
        raise FileNotFoundError(f"Fails '{file}' netika atrasts!")
    with open(file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    summary = summarize_text(text, token)
    print("\nKopsavilkums:\n", summary)
    num_kw = int(input("\nCik atslēgvārdus ģenerēt?: "))
    keywords = generate_keywords(summary, num_kw, token)
    print("\nAtslēgvārdi:\n", keywords)
    quiz = generate_quiz(summary, token)
    print("\nĢenerētie jautājumi:\n", quiz)
