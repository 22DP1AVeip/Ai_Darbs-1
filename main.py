import os
import time
import json
import requests
from dotenv import load_dotenv

HF_API_BASE = "https://router.huggingface.co/hf-inference"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def load_token():
    load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        raise ValueError("HF_TOKEN vai HUGGINGFACE_API_KEY nav atrasts .env failā!")
    return token

def query_model(model, prompt, token):
    url = f"{HF_API_BASE}/{model}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}

    for attempt in range(3):
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            break
        elif resp.status_code == 503:
            print("Modelis tiek ielādēts... gaidām 5 sekundes.")
            time.sleep(5)
            continue
        else:
            return f"Kļūda: {resp.status_code} - {resp.text}"

    if resp.status_code != 200:
        return f"Kļūda: {resp.status_code} - {resp.text}"

    try:
        data = resp.json()
    except Exception:
        return resp.text

    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    elif isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"].strip()
        elif "error" in data:
            return f"Kļūda: {data['error']}"
        elif "choices" in data:
            c = data["choices"][0]
            if "message" in c and "content" in c["message"]:
                return c["message"]["content"].strip()

    return str(data)

def summarize_text(text, token):
    prompt = f"Kopsavilkums par šo tekstu īsi un skaidri latviski:\n{text}"
    return query_model(DEFAULT_MODEL, prompt, token)

def generate_keywords(text, n, token):
    prompt = f"Izraksti {n} atslēgvārdus no šī teksta (atdalītus ar komatiem):\n{text}"
    return query_model(DEFAULT_MODEL, prompt, token)

def generate_quiz(text, token):
    prompt = (
        f"Izveido 3 testa jautājumus ar 4 atbilžu variantiem (a, b, c, d), "
        f"balstoties uz šo tekstu, un norādi pareizās atbildes:\n{text}"
    )
    return query_model(DEFAULT_MODEL, prompt, token)

if __name__ == "__main__":
    token = load_token()
    file = input("Ievadi .txt faila nosaukumu: ")
    if not file.endswith(".txt"):
        file += ".txt"

    if not os.path.exists(file):
        raise FileNotFoundError(f"Fails '{file}' netika atrasts!")

    with open(file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print("\n🔹 Ģenerēju kopsavilkumu...")
    summary = summarize_text(text, token)
    print("\nKopsavilkums:\n", summary)

    num_kw = int(input("\nCik atslēgvārdus ģenerēt?: "))
    print("\n🔹 Ģenerēju atslēgvārdus...")
    keywords = generate_keywords(summary, num_kw, token)
    print("\nAtslēgvārdi:\n", keywords)

    print("\n🔹 Ģenerēju jautājumus...")
    quiz = generate_quiz(summary, token)
    print("\nĢenerētie jautājumi:\n", quiz)
