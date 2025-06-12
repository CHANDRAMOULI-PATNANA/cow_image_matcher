import streamlit as st
import os
import json
import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

# Setup paths
KNOWN_COWS_DIR = "known_cows"
EMBEDDINGS_FILE = "cow_embeddings.json"

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Get image embedding using CLIP
def get_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)  # Normalize
    return embedding.squeeze().numpy().tolist()

# Load JSON embeddings
def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}
    with open(EMBEDDINGS_FILE, "r") as f:
        return json.load(f)

# Save updated JSON embeddings
def save_embeddings(embeddings):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings, f)

# Match uploaded cow with saved embeddings
def match_cow(embedding, db, threshold=0.85):
    for name, emb_list in db.items():
        for emb in emb_list:
            # Validate shape: must match CLIP embedding size
            if isinstance(emb, list) and len(emb) == len(embedding):
                score = cosine_similarity([embedding], [emb])[0][0]
                if score > threshold:
                    return name, score
    return None, None

# Streamlit UI
st.set_page_config(page_title="🐄 Cow Matcher", layout="centered")
st.title("🐄 Cow Identity Matcher")

menu = st.sidebar.selectbox("Choose Option", ["Upload & Match Cow", "Register New Cow"])

embeddings = load_embeddings()

# 🐄 Registering a new cow
if menu == "Register New Cow":
    st.header("📸 Register Your Cow")
    name = st.text_input("Enter Cow Name")
    image_file = st.file_uploader("Upload Cow Image", type=["jpg", "jpeg", "png"])

    if name and image_file:
        image = Image.open(image_file).convert("RGB")
        embedding = get_embedding(image)

        # Store multiple embeddings per cow
        if name in embeddings:
            embeddings[name].append(embedding)
        else:
            embeddings[name] = [embedding]

        save_embeddings(embeddings)
        st.success(f"✅ Cow '{name}' registered successfully!")

# 🧠 Matching an uploaded cow image
elif menu == "Upload & Match Cow":
    st.header("🔍 Match Cow Image")
    test_file = st.file_uploader("Upload Cow Image to Match", type=["jpg", "jpeg", "png"])

    if test_file:
        image = Image.open(test_file).convert("RGB")
        test_embedding = get_embedding(image)

        matched_name, score = match_cow(test_embedding, embeddings)

        if matched_name:
            st.success(f"✅ Match Found: {matched_name} (Similarity Score: {score:.2f})")
        else:
            st.error("❌ No Match Found. Try registering multiple angles of the same cow.")
