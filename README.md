# FreshScan AI 🍃

A Streamlit web app that tells you whether a fruit or vegetable is **fresh or rotten** from a photo, explains the decision with a **Grad-CAM** heatmap, and gives strawberries a **freshness score out of 10**. The interface is in Arabic.

## Features

- Upload an image or take a photo with your camera
- Classifies 10 fruits and vegetables as fresh or rotten (20 classes): apple, banana, bell pepper, carrot, cucumber, mango, orange, potato, strawberry and tomato
- Shows the prediction confidence
- Explainable AI: a Grad-CAM heatmap highlights the image regions that drove the prediction
- Freshness score (0 to 10) for strawberries, from a separate regression model

## How it works

| Component | Model | File |
|---|---|---|
| Fresh / rotten classifier | EfficientNet (Keras / TensorFlow) | `best_fruit_model.keras` |
| Strawberry freshness score | EfficientNet-B0 regressor (PyTorch) | `best_model-2.pth` |

Images are resized to 224 x 224 before prediction.

## Getting started

Requirements: Python 3.11.

```bash
git clone https://github.com/IMawaddah/freshscan-ai.git
cd freshscan-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL that Streamlit prints (usually `http://localhost:8501`).

## Project structure

```
.
├── app.py                    # Streamlit application
├── best_fruit_model.keras    # Fresh / rotten classifier
├── best_model-2.pth          # Strawberry freshness score model
├── requirements.txt
└── runtime.txt               # Python version for deployment
```

## Tech stack

Python, Streamlit, TensorFlow / Keras, PyTorch, torchvision, Pillow, NumPy, Matplotlib.
