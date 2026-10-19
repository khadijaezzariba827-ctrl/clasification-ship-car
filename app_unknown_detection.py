
import streamlit as st
import numpy as np
import cv2
import joblib

from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.neighbors import NearestNeighbors

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Classification Car vs Ship",
    page_icon="🚗🚢",
    layout="centered"
)

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
MODEL_NAME_FILE = "model_name.txt"

# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    try:
        with open(MODEL_NAME_FILE, "r", encoding="utf-8") as f:
            model_name = f.read().strip()
    except FileNotFoundError:
        model_name = type(model).__name__

    return model, scaler, model_name


model, scaler, model_name = load_model()

# ============================================================
# EXTRACTION DES CARACTERISTIQUES
# EXACTEMENT COMME DANS LE NOTEBOOK
# ============================================================

def color_Moments(img):
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    colorFeatures = [
        np.mean(R),
        np.std(R),
        np.mean(G),
        np.std(G),
        np.mean(B),
        np.std(B)
    ]

    colorFeatures = np.array(colorFeatures)

    mean_value = np.mean(colorFeatures)
    if mean_value != 0:
        colorFeatures /= mean_value

    return colorFeatures


def hsvHistogramFeatures(image):
    imageHSV = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

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
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0][0]
    correlation = graycoprops(glcm, 'correlation')[0][0]
    energy = graycoprops(glcm, 'energy')[0][0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0][0]

    features = np.array([
        contrast,
        correlation,
        energy,
        homogeneity
    ], dtype=float)

    total = np.sum(features)
    if total != 0:
        features /= total

    return features


def lbpFeatures(img):
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    P = 8
    R = 1

    lbp = local_binary_pattern(
        gray,
        P,
        R,
        method='uniform'
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
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    moments = cv2.moments(gray)

    shapeFeat = cv2.HuMoments(
        moments
    ).flatten()

    mean_value = np.mean(shapeFeat)
    if mean_value != 0:
        shapeFeat /= mean_value

    return shapeFeat


def getFeatures(img):
    rgb = color_Moments(img)
    hsv = hsvHistogramFeatures(img)
    glcm = textureFeatures(img)
    lbp = lbpFeatures(img)
    hu = shapeFeatures(img)

    features = np.concatenate([
        rgb,
        hsv,
        glcm,
        lbp,
        hu
    ])

    return features


# ============================================================
# DETECTION D'IMAGE INCONNUE
#
# Le modèle a seulement appris CAR et SHIP.
# Pour éviter qu'une personne, un animal, etc. soit
# automatiquement forcé dans CAR ou SHIP, on mesure sa
# distance par rapport aux données d'entraînement.
# ============================================================

@st.cache_resource
def build_unknown_detector(_model):
    """
    Construit un seuil à partir des données utilisées par KNN.

    Pour chaque image d'entraînement, on calcule la distance
    moyenne à ses k plus proches voisins (en excluant elle-même).
    Le 95e percentile sert de seuil pour les nouvelles images.
    """

    if not hasattr(_model, "_fit_X"):
        return None

    X_train_model = _model._fit_X

    # Pour KNN(k=5), on utilise 5 voisins.
    if hasattr(_model, "n_neighbors"):
        k = int(_model.n_neighbors)
    else:
        k = 5

    k = min(k, len(X_train_model) - 1)

    if k < 1:
        return None

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_train_model)

    distances, _ = nn.kneighbors(X_train_model)

    # La première distance est 0 : l'image elle-même.
    train_mean_distances = distances[:, 1:].mean(axis=1)

    # Seuil relativement permissif pour ne pas rejeter
    # trop facilement des vraies voitures/bateaux.
    threshold = float(np.percentile(train_mean_distances, 95))

    return {
        "nn": nn,
        "k": k,
        "threshold": threshold,
        "train_mean_distances": train_mean_distances
    }


unknown_detector = build_unknown_detector(model)


def unknown_check(features_scaled):
    """
    Retourne :
        is_unknown, distance, threshold
    """

    if unknown_detector is None:
        return False, None, None

    nn = unknown_detector["nn"]
    k = unknown_detector["k"]
    threshold = unknown_detector["threshold"]

    distances, _ = nn.kneighbors(features_scaled, n_neighbors=k)

    mean_distance = float(np.mean(distances[0]))

    is_unknown = mean_distance > threshold

    return is_unknown, mean_distance, threshold


# ============================================================
# INTERFACE
# ============================================================

st.title("🚗🚢 Classification d'images")
st.subheader("Car vs Ship")

st.write(
    "Importez une image pour déterminer si elle correspond "
    "à une voiture ou à un bateau."
)

st.info(
    "Le modèle a été entraîné uniquement sur deux classes : "
    "**Car** et **Ship**. Les images qui semblent trop éloignées "
    "des données d'entraînement sont maintenant signalées comme "
    "**Image inconnue**."
)

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

    st.divider()
    st.header("🔎 Résultat")

    try:
        # Extraction des caractéristiques
        features = getFeatures(image_np).reshape(1, -1)

        # Même normalisation que pendant l'entraînement
        features_scaled = scaler.transform(features)

        # Prédiction du modèle
        prediction = model.predict(features_scaled)[0]

        # Vérification image connue/inconnue
        is_unknown, mean_distance, threshold = unknown_check(
            features_scaled
        )

        if is_unknown:
            st.warning(
                "⚠️ **Image non reconnue**\n\n"
                "Cette image est trop éloignée des données "
                "d'entraînement du modèle. Elle ne semble pas "
                "correspondre suffisamment aux classes **Car** ou **Ship**."
            )

            st.write(f"**Modèle utilisé :** {model_name}")

            if mean_distance is not None:
                st.write(
                    f"Distance moyenne aux voisins : "
                    f"**{mean_distance:.3f}**"
                )
                st.write(
                    f"Seuil d'acceptation : "
                    f"**{threshold:.3f}**"
                )

            st.caption(
                f"Le modèle brut aurait prédit : {prediction}"
            )

        else:
            # ------------------------------------------------
            # IMAGE ACCEPTÉE : CAR OU SHIP
            # ------------------------------------------------

            if str(prediction).lower() == "car":
                st.success("🚗 **Classe prédite : CAR**")
            else:
                st.success("🚢 **Classe prédite : SHIP**")

            st.write(f"**Modèle utilisé :** {model_name}")

            if mean_distance is not None:
                st.write(
                    f"Distance aux données d'entraînement : "
                    f"**{mean_distance:.3f}**"
                )

            # Probabilités si disponibles
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features_scaled)[0]
                classes = model.classes_

                max_probability = float(np.max(probabilities))

                st.subheader("Confiance de la prédiction")
                st.metric(
                    "Confiance",
                    f"{max_probability * 100:.2f} %"
                )

                st.write("**Probabilités :**")

                for cls, prob in zip(classes, probabilities):
                    if str(cls).lower() == "car":
                        label = "🚗 Car"
                    else:
                        label = "🚢 Ship"

                    st.write(
                        f"- **{label} : {prob * 100:.2f} %**"
                    )

    except Exception as e:
        st.error(
            "Une erreur est survenue pendant la classification."
        )
        st.exception(e)

else:
    st.caption(
        "👆 Importez une image pour commencer."
    )
