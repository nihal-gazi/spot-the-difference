from config import USE_OFFLINE_LLM, OFFLINE_LLM_PATH, MISTRAL_API_KEY
import requests
import json

def ask_llm(prompt):
    if USE_OFFLINE_LLM:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(OFFLINE_LLM_PATH)
        model = AutoModelForCausalLM.from_pretrained(OFFLINE_LLM_PATH)

        input_ids = tok(prompt, return_tensors="pt").input_ids
        out = model.generate(input_ids, max_new_tokens=256)
        return tok.decode(out[0], skip_special_tokens=True)

    else:
        r = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        return r.json()["choices"][0]["message"]["content"]
