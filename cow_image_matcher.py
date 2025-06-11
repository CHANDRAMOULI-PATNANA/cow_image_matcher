# cow_image_matcher.py

import streamlit as st
import os
import json
import torch
import timm
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms

# Set paths
KNOWN_COWS_DIR = "known_cows"
EMBEDDINGS_FILE = "cow_embeddings.json"

# Load model
model = timm.create_model("resnet50", pretrained=True, num_classes=0)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_embedding(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.squeeze().numpy().tolist()

def load_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        return {}
    with open(EMBEDDINGS_FILE, "r") as f:
        return json.load(f)

def save_embeddings(embeddings):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings, f)

def match_cow(embedding, db):
    for name, emb in db.items():
        score = cosine_similarity([embedding], [emb])[0][0]
        if score > 0.9:
            return name, score
    return None, None

# Streamlit UI
st.title("🐄 Cow Identity Matcher")

menu = st.sidebar.selectbox("Choose Option", ["Upload & Match Cow", "Register New Cow"])

embeddings = load_embeddings()

if menu == "Register New Cow":
    st.header("Register Your Cow")
    name = st.text_input("Cow Name")
    image_file = st.file_uploader("Upload Cow Image", type=["jpg", "jpeg", "png"])

    if name and image_file:
        image = Image.open(image_file).convert("RGB")
        embedding = get_embedding(image)
        embeddings[name] = embedding
        save_embeddings(embeddings)
        st.success(f"✅ Cow '{name}' registered successfully!")

elif menu == "Upload & Match Cow":
    st.header("Match Cow Image")
    test_file = st.file_uploader("Upload Cow Image to Match", type=["jpg", "jpeg", "png"])

    if test_file:
        image = Image.open(test_file).convert("RGB")
        test_embedding = get_embedding(image)
        matched_name, score = match_cow(test_embedding, embeddings)

        if matched_name:
            st.success(f"✅ Match Found: {matched_name} (Score: {score:.2f})")
        else:
            st.error("❌ No Match Found")
