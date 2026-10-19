import streamlit as st
import numpy as np
import cv2
import joblib

from PIL import Image
from sklearn.neighbors import NearestNeighbors
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

st.set_page_config(
    page_title="Classification Car vs Ship",
    page_icon="🚗",
    layout="centered"
)

CLASS_NAMES = ["Car", "Ship"]

# Seuil de confiance minimal (KNN k=5 : 0.6 = 3 voisins sur 5)
MIN_CONFIDENCE = 0.60


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    model_name = "Meilleur modèle"
    try:
        with open("model_name.txt", "r", encoding="utf-8") as f:
            model_name = f.read().strip()
    except FileNotFoundError:
        pass

    # Détection d'images « hors sujet » (facultatif mais recommandé)
    ood = None
    try:
        data = joblib.load("ood.pkl")
        nn = NearestNeighbors(n_neighbors=data["k"]).fit(data["X_ref"])
        ood = {"nn": nn, "threshold": float(data["threshold"])}
    except FileNotFoundError:
        pass

    return model, scaler, model_name, ood


def color_Moments(img):
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    features = np.array([
        np.mean(R), np.std(R),
        np.mean(G), np.std(G),
        np.mean(B), np.std(B)
    ], dtype=float)

    mean_value = np.mean(features)
    if mean_value != 0:
        features /= mean_value

    return features


def hsvHistogramFeatures(image):
    imageHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist(
        [imageHSV],
        [0, 1, 2],
        None,
        [8, 2, 2],
        [0, 180, 0, 256, 0, 256]
    )

    hist = hist.flatten()
    total = np.sum(hist)
    if total != 0:
        hist /= total

    return hist


def textureFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = np.array([
        graycoprops(glcm, "contrast")[0][0],
        graycoprops(glcm, "correlation")[0][0],
        graycoprops(glcm, "energy")[0][0],
        graycoprops(glcm, "homogeneity")[0][0]
    ], dtype=float)

    total = np.sum(features)
    if total != 0:
        features /= total

    return features


def lbpFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    P = 8
    R = 1

    lbp = local_binary_pattern(
        gray, P, R, method="uniform"
    )

    n_bins = P + 2
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    hist = hist.astype(float)
    total = np.sum(hist)
    if total != 0:
        hist /= total

    return hist


def shapeFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()

    mean_value = np.mean(hu)
    if mean_value != 0:
        hu /= mean_value

    return hu


def getFeatures(img):
    rgb = color_Moments(img)
    hsv = hsvHistogramFeatures(img)
    glcm = textureFeatures(img)
    lbp = lbpFeatures(img)
    hu = shapeFeatures(img)

    return np.concatenate([
        rgb, hsv, glcm, lbp, hu
    ])


st.title("🚗🚢 Classification d'images")
st.subheader("Car vs Ship")
st.write(
    "Importez une image et le modèle entraîné dans votre TP "
    "prédira automatiquement sa classe. "
    "Si l'image n'est ni une voiture ni un bateau, le résultat sera **Inconnu**."
)

try:
    model, scaler, model_name, ood = load_artifacts()
except Exception as e:
    st.error("Les fichiers model.pkl et scaler.pkl sont nécessaires.")
    st.code(str(e))
    st.stop()

uploaded_file = st.file_uploader(
    "📤 Choisissez une image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(
        image,
        caption="Image importée",
        use_container_width=True
    )

    with st.spinner("Analyse de l'image..."):
        features = getFeatures(image_np).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = int(model.predict(features_scaled)[0])
        predicted_class = CLASS_NAMES[prediction]

        # Confiance du modèle
        probabilities = None
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = float(np.max(probabilities))

        # Distance de l'image aux images d'apprentissage
        ood_score = None
        is_far = False
        if ood is not None:
            dist, _ = ood["nn"].kneighbors(features_scaled)
            ood_score = float(dist.mean())
            is_far = ood_score > ood["threshold"]

        # Décision finale
        is_unknown = is_far or (
            confidence is not None and confidence < MIN_CONFIDENCE
        )

    st.divider()
    st.subheader("🔎 Résultat")

    if is_unknown:
        st.warning("❓ Classe prédite : INCONNU (ni Car, ni Ship)")
    elif predicted_class == "Car":
        st.success("🚗 Classe prédite : CAR")
    else:
        st.info("🚢 Classe prédite : SHIP")

    st.write(f"**Modèle utilisé :** {model_name}")

    if confidence is not None:
        st.metric("Confiance de la prédiction", f"{confidence * 100:.2f} %")

        st.write("Probabilités :")
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"- {name} : {probabilities[i] * 100:.2f} %")

    if ood_score is not None:
        with st.expander("Détails de la détection « Inconnu »"):
            st.write(f"Distance aux images d'apprentissage : {ood_score:.3f}")
            st.write(f"Seuil maximal accepté : {ood['threshold']:.3f}")
            if is_far:
                st.write("➡️ L'image est trop différente des voitures et bateaux connus.")
else:
    st.info("👆 Importez une image pour commencer.")