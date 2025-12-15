import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import exifread

@st.cache_resource
def load_model():
    model_name = "Ateeqq/ai-vs-human-image-detector"
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor

def extract_metadata(uploaded_file):
    """Extract EXIF/metadata using exifread"""
    try:
        tags = exifread.process_file(uploaded_file, details=True)
        metadata = {tag: str(tags[tag]) for tag in tags.keys() if tag not in ("JPEGThumbnail", "TIFFThumbnail")}
        return metadata
    except Exception as e:
        return {"Error": str(e)}

def summarize_metadata(metadata: dict):
    """Return a clean summary of useful metadata"""
    summary = {}
    if not metadata or "Error" in metadata:
        return {"Summary": "❌ No metadata found or unreadable."}

    # Camera/Device
    camera = metadata.get("Image Model", metadata.get("Image Make", None))
    if camera:
        summary["Camera/Device"] = camera

    # Software
    software = metadata.get("Image Software", None)
    if software:
        summary["Software"] = software

    # Date/Time
    datetime = metadata.get("EXIF DateTimeOriginal", metadata.get("Image DateTime", None))
    if datetime:
        summary["Date Taken"] = datetime

    if not summary:
        summary["Summary"] = "⚠️ Metadata found but no key details (Camera/Date/Software) present."
    return summary

def analyze_metadata(metadata: dict):
    """Check metadata consistency and flag suspicious/missing fields"""
    if not metadata:
        return ["❌ No metadata found (could mean stripped or edited)."]

    issues = []
    if not any(k for k in metadata.keys() if "Model" in k or "Make" in k):
        issues.append("⚠️ Missing Camera/Device Model info (common in AI or edited images).")

    if not any("DateTime" in k for k in metadata.keys()):
        issues.append("⚠️ Missing Date/Time info (may indicate synthetic image).")

    suspicious_software = ["Stable Diffusion", "DALL-E", "MidJourney", "AI", "Generated"]
    for k, v in metadata.items():
        if "Software" in k and any(s in v for s in suspicious_software):
            issues.append(f"⚠️ Suspicious Software tag detected: `{v}`")

    if not issues:
        issues.append("✅ Metadata looks normal and consistent with real camera output.")

    return issues

def ai_image_checker():
    st.title("🖼️ AI Image Checker")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Metadata summary
        st.subheader("📑 Image Metadata Summary")
        uploaded_file.seek(0)  # Reset pointer
        metadata = extract_metadata(uploaded_file)
        summary = summarize_metadata(metadata)
        st.json(summary)

        st.write("🔍 Running AI vs Human Detector...")

        #AI/Human classifier
        model, processor = load_model()
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

        labels = list(model.config.id2label.values())
        prob_dict = {labels[i].lower(): float(probs[i]) * 100 for i in range(len(labels))}

        real_prob = prob_dict.get("real", prob_dict.get("human", 0.0))
        ai_prob = prob_dict.get("fake", prob_dict.get("ai", 0.0))

        # Detection Result (Metadata + Model Combined)
        st.subheader("📌 Detection Result")

        has_meta = any(k in summary for k in ["Camera/Device", "Date Taken", "Software"])

        if has_meta:
            st.success("✅ Metadata found → Likely Human Image")
        elif not metadata or "Error" in metadata or not has_meta:
            st.error("⚠️ No metadata found → Possibly AI-generated")
        else:
            # fallback to model-based prediction
            if ai_prob >= 70:
                st.error(f"⚠️ Likely AI-generated Image (Confidence: {ai_prob:.1f}%)")
            elif real_prob >= 70:
                st.success(f"✅ Likely Real Image (Confidence: {real_prob:.1f}%)")
            else:
                st.warning(f"🤔 Uncertain Prediction\n\nAI: {ai_prob:.1f}% | Real: {real_prob:.1f}%")

        # ✅ Probability Breakdown
        st.subheader("📊 Probability Breakdown")
        st.write(f"**Real/Human:** {real_prob:.1f}%")
        st.write(f"**AI/Fake:** {ai_prob:.1f}%")

        # ✅ Metadata consistency analysis
        st.subheader("🧩 Metadata Consistency Check")
        analysis = analyze_metadata(metadata)
        for issue in analysis:
            st.write(issue)

        st.info(
            "ℹ️ AI detection + metadata analysis together improves reliability. "
            "If metadata is present, it's usually a strong indicator of a real human-taken photo."
        )
