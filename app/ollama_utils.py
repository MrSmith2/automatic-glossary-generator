import requests
import json
import re
from config import OLLAMA_API_URL, OLLAMA_TAGS_URL, TIMEOUT_CONNECTION, TIMEOUT_GENERATION

def check_ollama_connection():
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=TIMEOUT_CONNECTION)
        if response.status_code == 200:
            return True, [m['name'] for m in response.json().get('models', [])]
        return False, []
    except:
        return False, []

def call_ollama(prompt, model_name, context_size, timeout=TIMEOUT_GENERATION, seed=None):
    try:
        options = {"num_ctx": context_size, "num_gpu": -1}
        if seed is not None:
            options["seed"] = seed 
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": model_name, "prompt": prompt, "stream": False,
                  "options": options},
            timeout=timeout
        )
        if response.status_code == 200:
            return response.json().get('response', ''), None
        return None, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return None, "Таймаут"
    except Exception as e:
        return None, str(e)

def parse_json_response(text, key="definition"):
    try:
        return json.loads(text).get(key, text), None
    except json.JSONDecodeError:
        match = re.search(r'\{[^}]+\}', text)
        if match:
            try:
                return json.loads(match.group()).get(key, text), None
            except:
                pass
        return text.strip(), "Не JSON"

def generate_definition(term, context, model_name, context_size, prompt_template, seed=None): 
    prompt = prompt_template.format(term=term, context=context)
    response, error = call_ollama(prompt, model_name, context_size, seed=seed) 
    
    if error:
        return {"term": term, "definition": None, "error": error}
    
    definition, parse_error = parse_json_response(response)
    return {"term": term, "definition": definition, "error": parse_error}

def generate_definitions_batch(terms_with_context, model_name, context_size, prompt_template, seed=None):
    return [generate_definition(t["term"], t["context"], model_name, context_size, prompt_template, seed=seed) 
            for t in terms_with_context]
