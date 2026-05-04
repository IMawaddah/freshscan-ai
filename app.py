import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import matplotlib.cm as cm
import os

import torch
import torch.nn as nn
from torchvision import transforms, models

st.set_page_config(page_title="FreshScan AI", page_icon="🍃", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0a0f0d; color: #e8f5eb; }
[data-testid="stSidebar"] { background-color: #111a14; border-right: 1px solid #1e3023; }
[data-testid="stHeader"] { background: transparent; }
.result-card { background:#162019; border:1px solid #1e3023; border-radius:14px; padding:20px; margin-bottom:16px; }
.card-header { font-size:0.75rem; font-weight:700; color:#5a7a60; letter-spacing:2px; text-transform:uppercase; margin-bottom:14px; }
.badge-fresh { display:inline-block; background:rgba(45,255,110,0.15); color:#2dff6e; border:1px solid rgba(45,255,110,0.35); border-radius:100px; padding:6px 20px; font-size:1rem; font-weight:700; }
.badge-rotten { display:inline-block; background:rgba(255,77,77,0.15); color:#ff4d4d; border:1px solid rgba(255,77,77,0.35); border-radius:100px; padding:6px 20px; font-size:1rem; font-weight:700; }
.info-row { font-size:0.82rem; color:#5a7a60; margin-top:6px; }
.info-row span { color:#e8f5eb; font-weight:600; }
.stButton > button { background:#2dff6e !important; color:#0a0f0d !important; font-weight:800 !important; border-radius:10px !important; border:none !important; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = [
    'FreshApple','FreshBanana','FreshBellpepper','FreshCarrot',
    'FreshCucumber','FreshMango','FreshOrange','FreshPotato',
    'FreshStrawberry','FreshTomato',
    'RottenApple','RottenBanana','RottenBellpepper','RottenCarrot',
    'RottenCucumber','RottenMango','RottenOrange','RottenPotato',
    'RottenStrawberry','RottenTomato'
]

ARABIC_NAMES = {
    'Apple':      'تفاح',
    'Banana':     'موز',
    'Bellpepper': 'فلفل',
    'Carrot':     'جزر',
    'Cucumber':   'خيار',
    'Mango':      'مانجو',
    'Orange':     'برتقال',
    'Potato':     'بطاطس',
    'Strawberry': 'فراولة',
    'Tomato':     'طماطم',
}

EMOJIS = {
    'Apple':'🍎','Banana':'🍌','Bellpepper':'🫑','Carrot':'🥕',
    'Cucumber':'🥒','Mango':'🥭','Orange':'🍊','Potato':'🥔',
    'Strawberry':'🍓','Tomato':'🍅'
}

MODEL_PATH = "best_fruit_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess(pil_img):
    img = pil_img.resize((224, 224)).convert("RGB")
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return effnet_preprocess(arr)

def predict(model, pil_img):
    arr = preprocess(pil_img)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = CLASS_NAMES[idx]
    is_fresh = label.startswith("Fresh")
    item = label.replace("Fresh", "").replace("Rotten", "")
    return label, item, is_fresh, confidence, arr

def make_gradcam(model, arr, class_idx):
    backbone = model.layers[0]
    last_conv_layer = backbone.get_layer("top_activation")
    grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[last_conv_layer.output, backbone.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, backbone_output = grad_model(arr, training=False)
        x = backbone_output
        for layer in model.layers[1:]:
            x = layer(x)
        class_channel = x[:, class_idx]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(pil_img, heatmap, alpha=0.45):
    img = np.array(pil_img.resize((224, 224)).convert("RGB"))
    heatmap_resized = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = np.uint8(jet_colors[heatmap_resized] * 255)
    jet_img = Image.fromarray(jet_heatmap).resize((224, 224))
    overlay = Image.blend(Image.fromarray(img), jet_img, alpha=alpha)
    return overlay

SCORE_MODEL_PATH = "best_model.pth" 

# ── Freshness Score Model (Strawberry only) ─────────────
class EfficientNetRegressor(nn.Module):
    def __init__(self, dropout1=0.4, dropout2=0.2):
        super().__init__()
        # No download during deployment; weights come from best_model.pth
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        features = self.backbone(x)
        score = self.regressor(features).squeeze(1)
        score = torch.clamp(score, 0.0, 1.0)
        return score

@st.cache_resource
def load_score_model():
    if not os.path.exists(SCORE_MODEL_PATH):
        return None
    device = torch.device("cpu")
    score_model = EfficientNetRegressor().to(device)
    state_dict = torch.load(SCORE_MODEL_PATH, map_location=device)
    score_model.load_state_dict(state_dict)
    score_model.eval()
    return score_model

def predict_freshness_score(score_model, pil_img):
    if score_model is None:
        return None
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    score_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    x = score_transform(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        score_0_1 = float(score_model(x).item())
    return max(0.0, min(10.0, score_0_1 * 10.0))

# ── Sidebar ──────────────────────────────────────
with st.sidebar:
    st.markdown("### 🍃 FreshScan AI")
    st.markdown("---")
    st.markdown("**نبذة عن نموذج التصنيف**")
    st.markdown("""
    <div style='font-size:0.82rem;color:#5a7a60;line-height:2;'>
    • النموذج: <span style='color:#e8f5eb'>EfficientNetB0</span><br>
    • الفئات: <span style='color:#e8f5eb'>20 فئة</span><br>
    • دقة التحقق: <span style='color:#2dff6e'>~96.9%</span><br>
    • XAI: <span style='color:#e8f5eb'>Grad-CAM</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**الأصناف المدعومة**")
    for en, ar in ARABIC_NAMES.items():
        st.markdown(
            f"<div style='font-size:0.8rem;color:#5a7a60;'>{EMOJIS[en]} {ar}</div>",
            unsafe_allow_html=True
        )

# ── Main ─────────────────────────────────────────
st.markdown("## تحليل الطازجية 🔬")
st.markdown(
    '<p style="color:#5a7a60;margin-top:-8px;">ارفع صورة أو التقطها من الكاميرا</p>',
    unsafe_allow_html=True
)

model = load_model()
if model is None:
    st.error("لم يتم العثور على النموذج في:\n`/content/drive/MyDrive/best_fruit_model.keras`")
    st.stop()

score_model = load_score_model()

tab_upload, tab_camera = st.tabs(["📁  رفع صورة", "📷  كاميرا"])
pil_img = None

with tab_upload:
    uploaded = st.file_uploader("اختر صورة", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")

with tab_camera:
    cam = st.camera_input("التقط صورة")
    if cam:
        pil_img = Image.open(cam).convert("RGB")

if pil_img is not None:
    st.markdown("---")
    with st.spinner("جاري التحليل..."):
        label, item, is_fresh, confidence, arr = predict(model, pil_img)
        
        if item == "Strawberry":
            freshness_score = predict_freshness_score(score_model, pil_img)
        class_idx = CLASS_NAMES.index(label)
        try:
            heatmap     = make_gradcam(model, arr, class_idx)
            gradcam_img = overlay_gradcam(pil_img, heatmap)
            gradcam_ok  = True
        except Exception as e:
            gradcam_ok  = False
            gradcam_err = str(e)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="result-card"><div class="card-header">🖼️ الصورة الاصلية</div>',
                    unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-card"><div class="card-header">نتيجة التصنيف</div>',
                    unsafe_allow_html=True)
        emoji  = EMOJIS.get(item, "?")
        arabic = ARABIC_NAMES.get(item, item)
        if is_fresh:
            st.markdown('<div class="badge-fresh">طازج</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-rotten">فاسد</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:2.5rem;margin:10px 0;">{emoji}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="info-row">النوع: <span>{arabic} ({item})</span></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="info-row">نسبة الثقة: <span>{confidence*100:.1f}%</span></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="info-row">الفئة: <span>{label}</span></div>',
                    unsafe_allow_html=True)

        if item == "Strawberry":
                st.markdown(
                    f'<div class="info-row">درجة النضارة: <span>{freshness_score:.2f} / 10</span></div>',
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="result-card"><div class="card-header">🔥 خريطة Grad-CAM</div>',
                    unsafe_allow_html=True)
        if gradcam_ok:
            st.image(gradcam_img, use_container_width=True)
            st.markdown("""
            <div style='display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;'>
              <span style='font-size:0.75rem;color:#5a7a60;'>🔴 تاثير عالٍ</span>
              <span style='font-size:0.75rem;color:#5a7a60;'>🟡 تاثير متوسط</span>
              <span style='font-size:0.75rem;color:#5a7a60;'>🔵 تاثير منخفض</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.warning(f"تعذر توليد Grad-CAM: {gradcam_err}")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='background:#162019;border:2px dashed #1e3023;border-radius:16px;
                padding:60px;text-align:center;color:#5a7a60;margin-top:16px;'>
        <div style='font-size:3rem;margin-bottom:12px;'>📤</div>
        <div style='font-size:1.1rem;color:#e8f5eb;font-weight:600;'>
            ارفع صورة او استخدم الكاميرا للبدء
        </div>
    </div>""", unsafe_allow_html=True)
