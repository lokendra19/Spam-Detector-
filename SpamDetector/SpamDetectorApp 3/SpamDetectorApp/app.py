"""
SpamShield Flask backend
Spam: SVM Pipeline (TF-IDF + custom features + LinearSVC)
Image: EXIF metadata + pixel-level statistical analysis (entropy, noise, color)
"""
from flask import Flask, render_template, request, jsonify
import pickle, re, os, csv, io, math, warnings
import tldextract
import exifread
from PIL import Image, ImageFilter
from datetime import datetime
from features import TextFeatures  # noqa: F401 — needed for pickle to deserialise the model

warnings.filterwarnings("ignore")

app = Flask(__name__)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model.pkl")
HISTORY_FILE = os.path.join(BASE_DIR, "history.csv")

# model.pkl is now a full sklearn Pipeline — no separate vectorizer
pipeline = pickle.load(open(MODEL_PATH, "rb"))

SUSPICIOUS_PATTERNS = [
    "bit.ly","tinyurl.com","goo.gl","click","login","verify",
    "account","free","prize","secure","update","confirm","banking"
]

# ── Spam helpers ──────────────────────────────────────────────────────────────

def categorize_spam(text):
    t = text.lower()
    if any(w in t for w in ["win","lottery","prize","congratulations","winner","jackpot"]):
        return "Lottery / Prize Scam"
    if any(w in t for w in ["bank","account","verify","password","credential","phishing","suspended","blocked","paypal"]):
        return "Banking / Phishing"
    if any(w in t for w in ["delivery","courier","fedex","dhl","bluedart","parcel","usps","royal mail","ups"]):
        return "Courier / Delivery Scam"
    if any(w in t for w in ["pill","medicine","weight","health","diet","cure","treatment"]):
        return "Health / Medicine Scam"
    if any(w in t for w in ["earn","income","investment","crypto","forex","work from home","passive"]):
        return "Financial / Investment Scam"
    if any(w in t for w in ["image","attachment","click picture"]):
        return "Image Spam"
    return "General Spam"

def scan_links(text):
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    results = []
    for url in urls:
        domain = tldextract.extract(url).domain
        suspicious = any(p in url.lower() or p in domain for p in SUSPICIOUS_PATTERNS) or \
                     any(ext in url for ext in [".xyz",".shop",".online",".biz",".info",".tk",".ml"])
        results.append({"url": url, "suspicious": suspicious})
    return results

def get_spam_features(text):
    """Return the hand-crafted signals for the result card."""
    words = text.split()
    total = max(len(words), 1)
    return {
        "caps_ratio":    round(sum(1 for w in words if w.isupper() and len(w)>1) / total * 100),
        "url_count":     len(re.findall(r'https?://\S+|www\.\S+', text)),
        "money_signs":   len(re.findall(r'[\$£€₹]', text)),
        "exclamations":  text.count('!'),
        "urgent_words":  len(re.findall(r'\b(urgent|act now|limited|expires|immediately|verify|suspended|blocked|claim|winner|congratulations|free|prize)\b', text, re.I)),
    }

# ── History helpers ───────────────────────────────────────────────────────────

def load_history():
    rows = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows

def save_history_row(text, prediction, score):
    exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp","text","prediction","score"])
        if not exists:
            w.writeheader()
        w.writerow({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "text": text[:120], "prediction": prediction, "score": score})

# ── Image analysis ────────────────────────────────────────────────────────────

def extract_image_metadata(file_bytes):
    try:
        tags = exifread.process_file(io.BytesIO(file_bytes), details=True)
        return {t: str(tags[t]) for t in tags if t not in ("JPEGThumbnail","TIFFThumbnail")}
    except Exception:
        return {}

def summarize_metadata(meta):
    s = {}
    cam = meta.get("Image Model") or meta.get("Image Make")
    if cam: s["Camera / Device"] = cam
    sw  = meta.get("Image Software")
    if sw:  s["Software"] = sw
    dt  = meta.get("EXIF DateTimeOriginal") or meta.get("Image DateTime")
    if dt:  s["Date Taken"] = dt
    lens = meta.get("EXIF LensModel") or meta.get("EXIF LensMake")
    if lens: s["Lens"] = lens
    return s

def shannon_entropy(values):
    """Shannon entropy of a 1-D distribution (e.g. histogram bins)."""
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v/total for v in values if v > 0]
    return -sum(p * math.log2(p) for p in probs)

def pixel_analysis(img_bytes):
    """
    Statistical pixel-level signals.
    Real camera photos:  higher local noise, higher entropy, richer histograms.
    AI-generated images: smoother gradients, lower noise, sometimes clamped histograms.
    Returns a dict of signals + a 0–100 'ai_likelihood' score.
    """
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Resize for speed (analysis doesn't need full res)
        img_small = img.resize((256, 256), Image.LANCZOS)

        # ── 1. Entropy of each channel ──
        ch_entropy = []
        for band in img_small.split():
            hist = band.histogram()          # 256 bins
            ch_entropy.append(shannon_entropy(hist))
        avg_entropy = sum(ch_entropy) / 3    # max theoretical ≈ 8 bits

        # ── 2. Local noise (high-pass via edge filter) ──
        gray = img_small.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_pixels = list(edges.getdata())
        avg_noise = sum(edge_pixels) / len(edge_pixels)   # 0–255

        # ── 3. Color channel variance ──
        r, g, b = img_small.split()
        def var(ch):
            px = list(ch.getdata())
            mean = sum(px)/len(px)
            return sum((x-mean)**2 for x in px)/len(px)
        ch_var = (var(r) + var(g) + var(b)) / 3

        # ── 4. Histogram spread (how many bins are used) ──
        gray256 = img_small.convert("L")
        hist_gray = gray256.histogram()
        non_zero = sum(1 for h in hist_gray if h > 0)   # max 256
        histogram_spread = non_zero / 256 * 100          # %

        # ── Combine into ai_likelihood ──
        # Low entropy → more likely AI (smooth gradients)
        entropy_score = max(0, (7.5 - avg_entropy) / 7.5 * 100)
        # Low noise → more likely AI
        noise_score   = max(0, (30 - avg_noise) / 30 * 100)
        # Low spread → more likely AI (clamped / narrow histogram)
        spread_score  = max(0, (60 - histogram_spread) / 60 * 100)
        # Low variance → more likely AI (very uniform colors)
        var_score     = max(0, (1000 - ch_var) / 1000 * 100)

        ai_likelihood = round(entropy_score*0.35 + noise_score*0.35 + spread_score*0.2 + var_score*0.1)
        ai_likelihood = max(0, min(100, ai_likelihood))

        return {
            "entropy":      round(avg_entropy, 2),
            "noise_level":  round(avg_noise, 1),
            "ch_variance":  round(ch_var, 1),
            "hist_spread":  round(histogram_spread, 1),
            "ai_likelihood": ai_likelihood,
        }
    except Exception as e:
        return {"error": str(e), "ai_likelihood": 50}

def analyze_image(file_bytes):
    meta    = extract_image_metadata(file_bytes)
    summary = summarize_metadata(meta)
    stats   = pixel_analysis(file_bytes)

    issues = []
    ai_sw  = ["Stable Diffusion","DALL-E","MidJourney","Midjourney","AI","Generated","Firefly","Adobe Generative"]

    # Metadata checks
    has_camera   = bool(summary.get("Camera / Device"))
    has_date     = bool(summary.get("Date Taken"))
    has_software = bool(summary.get("Software"))

    if not has_camera:
        issues.append({"type":"warn","msg":"No camera/device model in metadata — common with AI images."})
    if not has_date:
        issues.append({"type":"warn","msg":"No capture date/time found — often absent in synthetic images."})
    for k, v in meta.items():
        if "Software" in k and any(s.lower() in v.lower() for s in ai_sw):
            issues.append({"type":"danger","msg":f"AI software detected in metadata: {v}"})

    # Pixel analysis interpretation
    ai_pct = stats.get("ai_likelihood", 50)
    entropy = stats.get("entropy", 0)
    noise   = stats.get("noise_level", 0)
    spread  = stats.get("hist_spread", 0)

    if entropy < 6.5:
        issues.append({"type":"warn","msg":f"Low pixel entropy ({entropy}) — real photos are usually richer in detail."})
    else:
        issues.append({"type":"ok","msg":f"Good pixel entropy ({entropy}/8.0) — typical of real camera output."})

    if noise < 12:
        issues.append({"type":"warn","msg":f"Very smooth image (noise={noise}/255) — AI images tend to be unrealistically smooth."})
    else:
        issues.append({"type":"ok","msg":f"Natural noise level detected ({noise}/255) — suggests real camera sensor."})

    if spread < 55:
        issues.append({"type":"warn","msg":f"Narrow histogram spread ({spread}%) — AI images sometimes clip or clamp pixel ranges."})
    else:
        issues.append({"type":"ok","msg":f"Wide histogram spread ({spread}%) — consistent with natural photographic range."})

    if has_camera and has_date:
        issues.append({"type":"ok","msg":"Camera model and capture date both present — strong indicator of a real photo."})

    # Final verdict: combine metadata + pixel signals
    # If metadata says real but pixels look AI → uncertain; trust both
    meta_score = (has_camera * 40) + (has_date * 30) + (has_software * 10 if not any(
        s.lower() in str(summary.get("Software","")).lower() for s in ai_sw) else -30)
    pixel_score = 100 - ai_pct    # higher = more real
    combined    = meta_score * 0.5 + pixel_score * 0.5

    if combined >= 55:
        verdict = "likely_real"
    elif combined <= 35:
        verdict = "likely_ai"
    else:
        verdict = "uncertain"

    return {
        "summary":      summary,
        "issues":       issues,
        "stats":        stats,
        "verdict":      verdict,
        "ai_pct":       ai_pct,
        "real_pct":     100 - ai_pct,
    }

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/history")
def history():
    rows = load_history()
    spam_count = sum(1 for r in rows if r.get("prediction") == "Spam")
    ham_count  = sum(1 for r in rows if r.get("prediction") == "Ham")
    return render_template("history.html", rows=rows,
                           spam_count=spam_count, ham_count=ham_count)

@app.route("/clear_history", methods=["POST"])
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return jsonify({"ok": True})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "empty"}), 400

    pred  = int(pipeline.predict([text])[0])
    proba = pipeline.predict_proba([text])[0]
    spam_score = round(float(proba[1]) * 100, 1)
    label      = "Spam" if pred == 1 else "Ham"

    return jsonify({
        "label":    label,
        "score":    spam_score,
        "category": categorize_spam(text) if pred == 1 else None,
        "links":    scan_links(text),
        "signals":  get_spam_features(text),
    })

@app.route("/api/image_check", methods=["POST"])
def api_image_check():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "no file"}), 400
    return jsonify(analyze_image(f.read()))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
