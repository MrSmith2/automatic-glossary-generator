import re
import json
from collections import Counter
from config import TIMEOUT_EXTRACTION, DEFAULT_LLM_CHUNK_SIZE, DEFAULT_SPACY_BATCH_SIZE
from constants import STOPWORDS, TECH_TERMS, GARBAGE_PATTERNS
from prompts import TERM_EXTRACTION_PROMPT
from ollama_utils import call_ollama

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("ru_core_news_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "ru_core_news_sm"], check=True)
            _nlp = spacy.load("ru_core_news_sm")
    return _nlp

def is_garbage(term):
    term_lower = term.lower().strip()
    if any(re.search(p, term_lower) for p in GARBAGE_PATTERNS):
        return True
    if sum(c.isdigit() for c in term) / max(len(term), 1) > 0.5:
        return True
    return bool(re.search(r'[©®™§№]', term))

def is_likely_term(term, text_lower):
    term_lower = term.lower()
    words = term_lower.split()
    
    if re.match(r'^[a-z][a-z0-9\-_]*$', term_lower) or (term.isupper() and len(term) >= 2):
        return True
    if term_lower in TECH_TERMS or any(w in TECH_TERMS for w in words):
        return True
    if bool(re.search(r'[a-zA-Z]', term)) and bool(re.search(r'[а-яА-Я]', term)):
        return True
    if f'"{term}"' in text_lower or f'«{term}»' in text_lower:
        return True
    return text_lower.count(term_lower) >= 3

def extract_candidates_from_chunk(text_chunk):
    nlp = get_nlp()
    doc = nlp(text_chunk)
    candidates = []
    tokens = list(doc)
    
    for i, token in enumerate(tokens):
        if token.is_punct or token.is_space or token.is_digit or len(token.text.strip()) < 2:
            continue
        
        if token.pos_ == 'NOUN':
            candidates.append(token.lemma_.lower())
            if i > 0 and tokens[i-1].pos_ in ('ADJ', 'NOUN') and tokens[i-1].text.lower() not in STOPWORDS:
                phrase = f"{tokens[i-1].lemma_} {token.lemma_}".lower()
                if not is_garbage(phrase):
                    candidates.append(phrase)
        elif token.pos_ == 'PROPN':
            candidates.append(token.text.lower())
    
    for match in re.finditer(r'\b([A-Z]{2,}|[A-Za-z][a-z]*[A-Z][A-Za-z]*|[a-z]{3,})\b', text_chunk):
        word = match.group()
        if len(word) >= 2 and word.lower() not in {'the', 'and', 'for', 'with', 'from', 'that', 'this'}:
            candidates.append(word.lower())
    
    for match in re.finditer(r'[«"]([^»"]{2,50})[»"]', text_chunk):
        quoted = match.group(1).strip()
        if len(quoted) >= 2 and not is_garbage(quoted):
            candidates.append(quoted.lower())
    
    return candidates

def extract_terms_spacy(text, top_n=30, coverage_percent=80, batch_size=None, progress_callback=None):
    if batch_size is None:
        batch_size = DEFAULT_SPACY_BATCH_SIZE
    
    total_len = len(text)
    process_len = int(total_len * coverage_percent / 100)
    text_to_process = text[:process_len]
    
    num_batches = (process_len // batch_size) + 1
    
    if progress_callback: 
        progress_callback(0, num_batches + 1)
    
    all_candidates = []
    for i, start in enumerate(range(0, process_len, batch_size)):
        end = min(start + batch_size, process_len)
        chunk = text_to_process[start:end]
        
        if end < process_len:
            space_pos = chunk.rfind(' ')
            if space_pos > batch_size * 0.8:
                chunk = chunk[:space_pos]
        
        candidates = extract_candidates_from_chunk(chunk)
        all_candidates.extend(candidates)
        
        if progress_callback:
            progress_callback(i + 1, num_batches + 1)
    
    stats = {
        "method": "spacy",
        "total_chars": total_len,
        "processed_chars": process_len,
        "coverage_percent": round(process_len / total_len * 100, 1),
        "batch_size": batch_size,
        "batches_processed": num_batches,
        "candidates_found": len(all_candidates)
    }
    
    if not all_candidates:
        return [], stats
    
    freq = Counter(all_candidates)
    text_lower = text.lower()
    
    terms, seen = [], set()
    for term, count in freq.most_common(top_n * 5):
        if term in seen or len(term) < 2 or is_garbage(term):
            continue
        if all(w in STOPWORDS for w in term.split()):
            continue
        
        score = count / len(all_candidates)
        if is_likely_term(term, text_lower): score *= 2.0
        if re.match(r'^[a-z][a-z0-9\-_]*$', term): score *= 1.5
        if term.upper() == term and len(term) <= 6: score *= 1.5
        
        seen.add(term)
        terms.append({"term": term, "score": round(score, 4), "frequency": count})
    
    terms.sort(key=lambda x: x["score"], reverse=True)
    
    if progress_callback:
        progress_callback(num_batches + 1, num_batches + 1)
    
    stats["terms_extracted"] = len(terms[:top_n * 2])
    return terms[:top_n * 2], stats

def extract_terms_llm(text, model_name="qwen3:4b", top_n=30, coverage_percent=80, chunk_size=None, progress_callback=None, seed=None):
    if chunk_size is None:
        chunk_size = DEFAULT_LLM_CHUNK_SIZE
    
    total_len = len(text)
    process_len = int(total_len * coverage_percent / 100)
    text_to_process = text[:process_len]
    
    chunks = []
    for i in range(0, process_len, chunk_size):
        chunk = text_to_process[i:i + chunk_size]
        
        if i + chunk_size < process_len:
            space_pos = chunk.rfind(' ')
            if space_pos > chunk_size * 0.8:
                chunk = chunk[:space_pos]
        
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
    
    stats = {
        "method": "llm",
        "total_chars": total_len,
        "processed_chars": process_len,
        "coverage_percent": round(process_len / total_len * 100, 1),
        "chunk_size": chunk_size,
        "chunks_total": len(chunks),
        "chunks_successful": 0
    }
    
    if not chunks:
        return [], stats
    
    all_terms = []
    for i, chunk in enumerate(chunks):
        if progress_callback: 
            progress_callback(i + 1, len(chunks))
        
        response, error = call_ollama(
            TERM_EXTRACTION_PROMPT.format(text=chunk), model_name, 8192, TIMEOUT_EXTRACTION, seed=seed
        )
        if error:
            continue
        
        try:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                terms = json.loads(match.group())
                extracted = [t for t in terms if isinstance(t, str)]
                all_terms.extend(extracted)
                stats["chunks_successful"] += 1
        except json.JSONDecodeError:
            pass
    
    if not all_terms:
        stats["terms_extracted"] = 0
        return [], stats
    
    text_lower = text.lower()
    freq = {}
    for term in all_terms:
        term_clean = term.lower().strip()
        if len(term_clean) >= 2 and not is_garbage(term_clean):
            freq[term_clean] = freq.get(term_clean, 0) + 1
    
    terms = [{"term": t, "score": round(c / len(all_terms), 4), "frequency": text_lower.count(t)}
             for t, c in freq.items()]
    terms.sort(key=lambda x: (x["score"], x["frequency"]), reverse=True)
    
    stats["terms_extracted"] = len(terms[:top_n * 2])
    stats["raw_terms_from_llm"] = len(all_terms)
    return terms[:top_n * 2], stats

def extract_terms(text, top_n=30, min_length=2, method="spacy", model_name="qwen3:4b", 
                  coverage_percent=80, chunk_size=None, batch_size=None, progress_callback=None, seed=None):
    if method == "llm":
        try:
            return extract_terms_llm(text, model_name, top_n, coverage_percent, chunk_size, progress_callback, seed=seed)
        except Exception as e:
            terms, stats = extract_terms_spacy(text, top_n, coverage_percent, batch_size, progress_callback)
            stats["fallback_reason"] = str(e)
            return terms, stats
    return extract_terms_spacy(text, top_n, coverage_percent, batch_size, progress_callback)

def get_term_context(text, term, window=300):
    text_lower, term_lower = text.lower(), term.lower()
    pos = text_lower.find(term_lower)
    if pos == -1:
        return ""
    
    start, end = max(0, pos - window), min(len(text), pos + len(term) + window)
    
    while start > 0 and text[start] not in '.!?\n' and pos - start < window + 100:
        start -= 1
    while end < len(text) and text[end-1] not in '.!?\n' and end - pos < window + 100:
        end += 1
    
    context = text[start:end].strip()
    return ("..." if start > 0 else "") + context + ("..." if end < len(text) else "")

def filter_terms(terms):
    filtered, seen = [], set()
    
    for t in terms:
        term_lower = t["term"].lower().strip()
        if term_lower in seen or len(term_lower) < 2 or is_garbage(t["term"]):
            continue
        
        words = term_lower.split()
        while words and words[0] in STOPWORDS: words = words[1:]
        while words and words[-1] in STOPWORDS: words = words[:-1]
        if not words:
            continue
        
        clean_term = ' '.join(words)
        if len(clean_term) < 2 or clean_term in seen:
            continue
        
        seen.add(clean_term)
        seen.add(term_lower)
        filtered.append({"term": clean_term, "score": t["score"], "frequency": t["frequency"]})
    
    return filtered
