# pip install google-genai
from google import genai
import os
import json
from dotenv import load_dotenv  
load_dotenv() 
API_KEY = os.environ.get("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)

def list_gemini_models(page_size=50):
    try:
        pager = client.models.list(config={"page_size": page_size})
        all_models = []
        for m in pager:
            try:
                info = json.loads(json.dumps(m, default=lambda o: getattr(o, "__dict__", str(o))))
            except Exception:
                info = str(m)
            all_models.append(info)
        return all_models
    except Exception as e:
        raise RuntimeError(f"List models failed: {e}")

if __name__ == "__main__":
    models = list_gemini_models()
    print(f"Found {len(models)} model(s).")
    for i, m in enumerate(models, 1):
        print(f"{i}. {m.get('name') if isinstance(m, dict) else m}")