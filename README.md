# 🐄 Cow Image Matcher (Computer Vision with Deep Learning)

This project allows users to **register** and **verify** cows using uploaded images. It uses **deep learning** and **computer vision** techniques to generate image embeddings and match cows based on visual similarity.

---

## 🚀 Features

- Register new cows with their images and names.
- Match uploaded cow images against registered entries.
- Uses **ResNet50** deep learning model for feature extraction.
- Real-time results using **Streamlit** web app.
- Stores and loads image embeddings in JSON format.

## 🧠 Tech Stack

| Area              | Tools / Libraries                         |
|-------------------|-------------------------------------------|
| Frontend          | Streamlit                                 |
| Deep Learning     | PyTorch, TIMM (for pretrained ResNet50)   |
| Image Processing  | Pillow (PIL), torchvision.transforms       |
| Similarity Check  | Scikit-learn (Cosine Similarity)           |
| File I/O          | JSON, OS                                   |

---

## 🖼️ Sample Workflow

1. Go to `Register New Cow` → upload an image and enter the cow's name.
2. The system saves a vector (embedding) representing the image.
3. Go to `Upload & Match Cow` → upload another image.
4. The app calculates its embedding and compares it with saved entries.
5. If similarity > 0.9, a match is shown; else, it reports no match.

---

## 🛠️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/cow-image-matcher.git
cd cow-image-matcher
