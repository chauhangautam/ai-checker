"""AI-content checker: metadata signals + optional ML models.
Har checker ek dict deta hai: {score: 0-100, reasons: [...], method: str}
Score sirf ESTIMATE hai, proof nahi."""
import io, re, zipfile

# AI tools ke naam jo metadata me mil sakte hain
AI_KEYWORDS = [
    "midjourney", "dall-e", "dalle", "openai", "chatgpt", "stable diffusion",
    "firefly", "adobe firefly", "imagen", "gemini", "sora", "runway", "leonardo.ai",
    "claude", "anthropic", "copilot", "bing image creator", "ideogram", "flux",
    "synthid", "trainedalgorithmicmedia", "c2pa", "contentcredentials",
]

def _find_keywords(text):
    t = text.lower()
    return sorted({k for k in AI_KEYWORDS if k in t})

def _raw_scan(data: bytes):
    """File ke raw bytes me C2PA / AI markers dhundo (images, video, pdf sab ke liye)."""
    hits = []
    blob = data[:5_000_000].lower()  # pehle 5MB kaafi
    if b"c2pa" in blob or b"jumb" in blob:
        hits.append("C2PA / Content Credentials marker mila")
    if b"trainedalgorithmicmedia" in blob:
        hits.append("IPTC tag 'trainedAlgorithmicMedia' mila (AI-generated ka standard tag)")
    for k in (b"midjourney", b"dall-e", b"stable diffusion", b"firefly", b"synthid"):
        if k in blob:
            hits.append(f"'{k.decode()}' text file me mila")
    return hits

def check_image(data: bytes):
    reasons, score = [], 10
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        info = {str(k): str(v) for k, v in (img.info or {}).items()}
        # PNG text chunks (Stable Diffusion 'parameters' yahin hota hai)
        if "parameters" in info or "prompt" in info:
            reasons.append("PNG me generation prompt/parameters mile (Stable Diffusion / ComfyUI jaisa)")
            score = 95
        kw = _find_keywords(" ".join(info.values()))
        if kw:
            reasons.append(f"Metadata me AI keywords: {', '.join(kw)}")
            score = max(score, 90)
        try:
            exif = img.getexif()
            sw = str(exif.get(305, ""))  # Software tag
            make = str(exif.get(271, ""))
            if sw and _find_keywords(sw):
                reasons.append(f"EXIF Software: {sw}")
                score = max(score, 85)
            if not make and not exif:
                reasons.append("Camera EXIF (make/model) nahi hai — AI ya screenshot/edited photo me aisa hota hai (kamzor signal)")
                score = max(score, 30)
            elif make:
                reasons.append(f"Camera EXIF mila: {make} {exif.get(272, '')} (asli camera ka signal)")
                score = min(score, 10)
        except Exception:
            pass
    except Exception as e:
        reasons.append(f"Image padh nahi paye: {e}")
    raw = _raw_scan(data)
    if raw:
        reasons += raw
        score = max(score, 90)
    reasons += _ml_image(data, lambda s: None) or []
    return {"score": score, "reasons": reasons or ["Koi AI signal nahi mila"], "method": "metadata + raw scan"}

_img_model = None
def _ml_image(data, _):
    """Optional: Hugging Face image detector. transformers install ho tabhi chalega."""
    global _img_model
    try:
        from transformers import pipeline
        from PIL import Image
    except ImportError:
        return ["(ML image detector off hai — `pip install transformers torch` se on karo)"]
    try:
        if _img_model is None:
            _img_model = pipeline("image-classification", model="umm-maybe/AI-image-detector")
        out = _img_model(Image.open(io.BytesIO(data)).convert("RGB"))
        ai = next((o["score"] for o in out if "artificial" in o["label"].lower() or "ai" in o["label"].lower()), None)
        if ai is not None:
            return [f"ML model ke hisab se AI hone ki sambhavna: {ai*100:.0f}%"]
    except Exception as e:
        return [f"ML model error: {e}"]
    return []

def check_video(data: bytes):
    reasons = _raw_scan(data)
    score = 90 if reasons else 15
    reasons.append("Video ka pixel-level deepfake detection is version me nahi hai — sirf metadata check hua.")
    return {"score": score, "reasons": reasons, "method": "metadata + raw scan"}

def _text_heuristic(text: str):
    """Halka heuristic (kamzor!). Zyada accuracy ke liye ML model use karo."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 80:
        return None, "Text bahut chhota hai (80+ words chahiye)"
    sents = [s for s in re.split(r"[.!?]+\s", text) if s.strip()]
    lens = [len(s.split()) for s in sents] or [0]
    mean = sum(lens) / len(lens)
    var = sum((l - mean) ** 2 for l in lens) / len(lens)
    uniform = var ** 0.5 / (mean or 1)          # kam = sab vakya ek jaise lambe
    phrases = ["delve", "tapestry", "in conclusion", "it is important to note",
               "furthermore", "moreover", "landscape of", "testament to", "as an ai"]
    ph = sum(text.lower().count(p) for p in phrases)
    score = 30
    if uniform < 0.45: score += 20
    score += min(ph * 8, 30)
    return min(score, 80), f"Vakya-lambai variation {uniform:.2f}, AI-style phrases: {ph}"

_txt_model = None
def check_text(text: str):
    reasons = []
    s, why = _text_heuristic(text)
    if s is None:
        return {"score": None, "reasons": [why], "method": "text"}
    reasons.append(why + " (heuristic — kamzor signal)")
    try:
        from transformers import pipeline
        global _txt_model
        if _txt_model is None:
            _txt_model = pipeline("text-classification", model="openai-community/roberta-base-openai-detector")
        r = _txt_model(text[:2000], truncation=True, max_length=512)[0]
        p = r["score"] if r["label"].lower() == "fake" else 1 - r["score"]
        reasons.append(f"ML model: AI hone ki sambhavna {p*100:.0f}%")
        s = int((s + p * 100) / 2)
    except ImportError:
        reasons.append("(ML text detector off hai — `pip install transformers torch` se on karo)")
    except Exception as e:
        reasons.append(f"ML model error: {e}")
    return {"score": s, "reasons": reasons, "method": "heuristic + optional ML"}

def check_pdf(data: bytes):
    from pypdf import PdfReader
    reasons, score = [], 10
    r = PdfReader(io.BytesIO(data))
    meta = r.metadata or {}
    meta_text = " ".join(str(v) for v in meta.values())
    kw = _find_keywords(meta_text)
    if kw:
        reasons.append(f"PDF metadata (Producer/Creator) me AI keywords: {', '.join(kw)}")
        score = 85
    if meta_text:
        reasons.append(f"Creator/Producer: {meta.get('/Creator', '-')} / {meta.get('/Producer', '-')}")
    raw = _raw_scan(data)
    if raw:
        reasons += raw; score = max(score, 85)
    text = "\n".join((p.extract_text() or "") for p in r.pages[:30])
    t = check_text(text)
    reasons += t["reasons"]
    if t["score"] is not None:
        score = max(score, t["score"]) if score < 85 else score
    return {"score": score, "reasons": reasons, "method": "metadata + text analysis"}

def check_docx(data: bytes):
    import docx
    reasons, score = [], 10
    d = docx.Document(io.BytesIO(data))
    cp = d.core_properties
    meta_text = f"{cp.author} {cp.last_modified_by} {cp.comments} {cp.title}"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        try: meta_text += " " + z.read("docProps/app.xml").decode("utf8", "ignore")
        except KeyError: pass
    kw = _find_keywords(meta_text)
    if kw:
        reasons.append(f"Document properties me AI keywords: {', '.join(kw)}")
        score = 85
    reasons.append(f"Author: {cp.author or '-'}, Last modified by: {cp.last_modified_by or '-'}")
    text = "\n".join(p.text for p in d.paragraphs)
    t = check_text(text)
    reasons += t["reasons"]
    if t["score"] is not None and score < 85:
        score = max(score, t["score"])
    return {"score": score, "reasons": reasons, "method": "metadata + text analysis"}

def analyze(filename: str, data: bytes):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in ("jpg", "jpeg", "png", "webp", "gif", "bmp", "tiff", "heic"): return check_image(data)
    if ext in ("mp4", "mov", "avi", "mkv", "webm"): return check_video(data)
    if ext == "pdf": return check_pdf(data)
    if ext == "docx": return check_docx(data)
    if ext in ("txt", "md"): return check_text(data.decode("utf8", "ignore"))
    raise ValueError(f"'.{ext}' support nahi hai. Photo, video, PDF, DOCX ya TXT upload karo.")
