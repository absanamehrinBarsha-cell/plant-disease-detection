import os

# ============================================================
# RENDER / TENSORFLOW CPU CONFIGURATION
# ============================================================

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import re
import time

import numpy as np
import tensorflow as tf
import joblib
import gradio as gr


# ============================================================
# EXPLICITLY DISABLE TENSORFLOW JIT / XLA
# ============================================================

tf.config.optimizer.set_jit(False)

print("==============================================")
print("TensorFlow configuration")
print("==============================================")
print("TensorFlow version:", tf.__version__)

try:
    print("JIT enabled:", tf.config.optimizer.get_jit())
except Exception:
    print("JIT status: unable to query")

print("==============================================")


# ============================================================
# RENDER DEPLOYMENT CONFIGURATION
# ============================================================

RENDER_HOST = "0.0.0.0"
RENDER_PORT = int(os.environ.get("PORT", "10000"))

os.environ["GRADIO_SERVER_NAME"] = RENDER_HOST
os.environ["GRADIO_SERVER_PORT"] = str(RENDER_PORT)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

vision_path = os.path.join(
    MODEL_DIR,
    "mobilenetv2.keras"
)

tfidf_path = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

svm_path = os.path.join(
    MODEL_DIR,
    "svm_classifier.pkl"
)

calibrated_svm_path = os.path.join(
    MODEL_DIR,
    "calibrated_svm_classifier.pkl"
)

label_encoder_path = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("==============================================")
print("Loading Plant Disease Detection models...")
print("==============================================")

load_start = time.perf_counter()

# ------------------------------------------------------------
# MobileNetV2
# ------------------------------------------------------------

vision_model = tf.keras.models.load_model(
    vision_path,
    compile=False
)

try:
    vision_model.jit_compile = False
except Exception:
    pass


# ------------------------------------------------------------
# Scikit-learn models
# ------------------------------------------------------------

tfidf_vectorizer = joblib.load(
    tfidf_path
)

svm_classifier = joblib.load(
    svm_path
)

calibrated_svm = joblib.load(
    calibrated_svm_path
)

label_encoder = joblib.load(
    label_encoder_path
)

classes = label_encoder.classes_

IMAGE_SIZE = (224, 224)

load_time = time.perf_counter() - load_start

print("Models loaded successfully.")
print("Number of classes:", len(classes))
print(f"Model loading time: {load_time:.2f} seconds")
print("==============================================")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = tf.convert_to_tensor(
        np.asarray(image)
    )

    # --------------------------------------------------------
    # Make sure image has three channels
    # --------------------------------------------------------

    if image.shape.rank == 2:

        image = tf.stack(
            [image, image, image],
            axis=-1
        )

    if image.shape[-1] == 4:

        image = image[..., :3]

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    image = tf.cast(
        image,
        tf.float32
    )

    # --------------------------------------------------------
    # MobileNetV2 preprocessing
    # --------------------------------------------------------

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# VISION PREDICTION
# ============================================================
#
# Direct eager inference is intentionally used instead of
# vision_model.predict() to avoid slow XLA compilation on Render.
# ============================================================

def vision_predict(image):

    total_start = time.perf_counter()

    print("")
    print("Image analysis: preprocessing image...")

    preprocess_start = time.perf_counter()

    processed_image = preprocess_image(
        image
    )

    preprocess_time = (
        time.perf_counter()
        -
        preprocess_start
    )

    print(
        f"Image preprocessing complete "
        f"({preprocess_time:.3f}s)"
    )

    # --------------------------------------------------------
    # Direct eager model inference
    # --------------------------------------------------------

    print(
        "Running MobileNetV2 using direct inference..."
    )

    inference_start = time.perf_counter()

    prediction = vision_model(
        processed_image,
        training=False
    )

    inference_time = (
        time.perf_counter()
        -
        inference_start
    )

    print(
        f"Image analysis complete "
        f"({inference_time:.3f}s)"
    )

    # --------------------------------------------------------
    # Convert output to NumPy
    # --------------------------------------------------------

    probabilities = np.asarray(
        prediction
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_disease = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(
        probabilities[predicted_index]
    )

    total_time = (
        time.perf_counter()
        -
        total_start
    )

    print(
        "Image prediction:",
        predicted_disease,
        f"({confidence * 100:.2f}%)"
    )

    print(
        f"Image analysis total time: {total_time:.3f}s"
    )

    return {
        "disease": predicted_disease,
        "confidence": confidence,
        "class_index": predicted_index,
        "probabilities": probabilities
    }


# ============================================================
# TEXT PREDICTION
# ============================================================

def text_predict(symptoms):

    total_start = time.perf_counter()

    print("Processing symptom description...")

    cleaned_text = clean_text(
        symptoms
    )

    # --------------------------------------------------------
    # Empty text
    # --------------------------------------------------------

    if not cleaned_text:

        probabilities = np.zeros(
            len(classes),
            dtype=np.float32
        )

        print(
            "No symptoms supplied."
        )

        return {
            "disease": "Not provided",
            "confidence": 0.0,
            "class_index": -1,
            "probabilities": probabilities
        }

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    tfidf_start = time.perf_counter()

    text_features = tfidf_vectorizer.transform(
        [cleaned_text]
    )

    tfidf_time = (
        time.perf_counter()
        -
        tfidf_start
    )

    print(
        f"Symptom processing complete "
        f"({tfidf_time:.3f}s)"
    )

    # --------------------------------------------------------
    # SVM probability prediction
    # --------------------------------------------------------

    svm_start = time.perf_counter()

    probabilities = calibrated_svm.predict_proba(
        text_features
    )[0]

    svm_time = (
        time.perf_counter()
        -
        svm_start
    )

    print(
        f"Symptom prediction complete "
        f"({svm_time:.3f}s)"
    )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_disease = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(
        probabilities[predicted_index]
    )

    total_time = (
        time.perf_counter()
        -
        total_start
    )

    print(
        "Symptom prediction:",
        predicted_disease,
        f"({confidence * 100:.2f}%)"
    )

    print(
        f"Symptom analysis total time: {total_time:.3f}s"
    )

    return {
        "disease": predicted_disease,
        "confidence": confidence,
        "class_index": predicted_index,
        "probabilities": probabilities
    }


# ============================================================
# TOP PREDICTIONS
# ============================================================

def get_top_predictions(
    probabilities,
    top_k=3
):

    probabilities = np.asarray(
        probabilities
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        disease = label_encoder.inverse_transform(
            [int(index)]
        )[0]

        confidence = float(
            probabilities[index]
        )

        results.append({
            "disease": disease,
            "confidence": confidence
        })

    return results


# ============================================================
# FUSION
# ============================================================

def fusion_predict(
    vision_result,
    text_result,
    vision_weight=0.5,
    text_weight=0.5
):

    print(
        "Combining image and symptom predictions..."
    )

    vision_probabilities = np.asarray(
        vision_result["probabilities"],
        dtype=np.float32
    )

    text_probabilities = np.asarray(
        text_result["probabilities"],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(vision_probabilities) != len(
        text_probabilities
    ):

        raise ValueError(
            "Image and symptom probability vectors "
            "have different lengths."
        )

    # --------------------------------------------------------
    # Combine probabilities
    # --------------------------------------------------------

    final_probabilities = (
        vision_weight * vision_probabilities
        +
        text_weight * text_probabilities
    )

    final_index = int(
        np.argmax(
            final_probabilities
        )
    )

    final_disease = label_encoder.inverse_transform(
        [final_index]
    )[0]

    final_confidence = float(
        final_probabilities[final_index]
    )

    agents_agree = (
        vision_result["disease"]
        ==
        text_result["disease"]
    )

    verification_needed = (
        not agents_agree
        or
        final_confidence < 0.70
    )

    if agents_agree:

        if final_confidence >= 0.70:

            decision = "Both predictions agree"

        else:

            decision = "Both predictions agree with lower certainty"

    else:

        if (
            vision_result["confidence"]
            >=
            text_result["confidence"]
        ):

            decision = "Image prediction preferred"

        else:

            decision = "Symptom prediction preferred"

    print(
        "Combined prediction:",
        final_disease,
        f"({final_confidence * 100:.2f}%)"
    )

    return {

        "vision_disease":
            vision_result["disease"],

        "vision_confidence":
            vision_result["confidence"],

        "text_disease":
            text_result["disease"],

        "text_confidence":
            text_result["confidence"],

        "final_disease":
            final_disease,

        "final_confidence":
            final_confidence,

        "agents_agree":
            agents_agree,

        "verification_needed":
            verification_needed,

        "decision":
            decision,

        "final_probabilities":
            final_probabilities
    }


# ============================================================
# USER-FRIENDLY DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "Bacterial Spot": {

        "what_is_it":
            "A bacterial disease that causes dark spots and damaged areas on leaves.",

        "why_happens":
            "It can spread through infected plant material, splashing water, and wet conditions.",

        "what_to_do": [
            "Remove badly affected leaves.",
            "Avoid splashing water onto the leaves.",
            "Keep good air circulation around the plant.",
            "Remove infected plant material from the growing area."
        ],

        "prevention": [
            "Keep leaves as dry as possible.",
            "Avoid working with plants when they are wet.",
            "Use clean gardening tools."
        ]
    },


    "Black Rot": {

        "what_is_it":
            "A plant disease that can cause dark, damaged areas on leaves and other plant parts.",

        "why_happens":
            "It can spread through infected plant material and is encouraged by warm, wet conditions.",

        "what_to_do": [
            "Remove affected leaves and plant material.",
            "Keep the plant area clean.",
            "Avoid unnecessary overhead watering.",
            "Monitor nearby plants for similar symptoms."
        ],

        "prevention": [
            "Remove infected plant debris.",
            "Improve air circulation.",
            "Avoid keeping foliage continuously wet."
        ]
    },


    "Early Blight": {

        "what_is_it":
            "A common fungal disease that causes dark spots and damaged areas on leaves.",

        "why_happens":
            "The disease is encouraged by moisture, warm conditions, and poor air circulation.",

        "what_to_do": [
            "Remove severely affected leaves.",
            "Water near the soil instead of wetting the leaves.",
            "Improve air circulation around plants.",
            "Remove infected plant debris."
        ],

        "prevention": [
            "Keep foliage dry when possible.",
            "Give plants enough space for air movement.",
            "Clean up fallen infected leaves."
        ]
    },


    "Esca Black Measles": {

        "what_is_it":
            "A serious disease associated with grapevines that can affect leaves and fruit.",

        "why_happens":
            "It is associated with fungal infections that can enter and spread through damaged plant tissues.",

        "what_to_do": [
            "Remove and manage severely affected plant material.",
            "Inspect the plant regularly for worsening symptoms.",
            "Avoid unnecessary injuries to the plant.",
            "Seek advice from a local agricultural specialist for severe cases."
        ],

        "prevention": [
            "Use healthy planting material.",
            "Protect pruning wounds where appropriate.",
            "Maintain good vineyard sanitation."
        ]
    },


    "Healthy": {

        "what_is_it":
            "The plant image does not show clear signs of the diseases included in this system.",

        "why_happens":
            "No obvious disease symptoms were identified from the submitted image.",

        "what_to_do": [
            "Continue normal plant care.",
            "Monitor the plant regularly.",
            "Check new leaves for changes in color, spots, or damage."
        ],

        "prevention": [
            "Provide suitable water and growing conditions.",
            "Keep the growing area clean.",
            "Regularly inspect plants for early symptoms."
        ]
    },


    "Huanglongbing": {

        "what_is_it":
            "Huanglongbing, also called citrus greening, is a serious disease affecting citrus plants.",

        "why_happens":
            "It is caused by bacteria associated with citrus greening and is spread by insect vectors.",

        "what_to_do": [
            "Inspect the plant for additional symptoms.",
            "Remove or manage affected plants according to local agricultural guidance.",
            "Control insect vectors using appropriate local recommendations.",
            "Consult an agricultural specialist for confirmation."
        ],

        "prevention": [
            "Use healthy planting material.",
            "Monitor for insect vectors.",
            "Follow local citrus disease-management recommendations."
        ]
    },


    "Late Blight": {

        "what_is_it":
            "A destructive disease that can cause dark, water-soaked-looking damage on leaves and other plant parts.",

        "why_happens":
            "It spreads more easily during cool, wet, and humid conditions.",

        "what_to_do": [
            "Remove severely affected plant material.",
            "Avoid wetting the leaves during watering.",
            "Improve air circulation.",
            "Act quickly if symptoms are spreading."
        ],

        "prevention": [
            "Keep foliage dry when possible.",
            "Avoid overcrowding plants.",
            "Regularly inspect plants during humid or wet weather."
        ]
    },


    "Leaf Blight": {

        "what_is_it":
            "A condition in which areas of the leaf become damaged, discolored, and eventually dry out.",

        "why_happens":
            "Leaf blight can be associated with infectious organisms and favorable environmental conditions such as prolonged moisture.",

        "what_to_do": [
            "Remove severely damaged leaves.",
            "Keep foliage dry when possible.",
            "Improve air circulation.",
            "Monitor the plant for spreading symptoms."
        ],

        "prevention": [
            "Avoid prolonged leaf wetness.",
            "Remove infected plant debris.",
            "Keep plants adequately spaced."
        ]
    },


    "Leaf Mold": {

        "what_is_it":
            "A fungal disease that can produce yellowing on the upper leaf surface and mold-like growth on the underside.",

        "why_happens":
            "It is encouraged by high humidity and prolonged moisture around the leaves.",

        "what_to_do": [
            "Remove badly affected leaves.",
            "Improve ventilation around the plant.",
            "Reduce excessive humidity where possible.",
            "Avoid unnecessary wetting of leaves."
        ],

        "prevention": [
            "Improve air circulation.",
            "Avoid excessive humidity.",
            "Keep foliage dry when possible."
        ]
    },


    "Leaf Scorch": {

        "what_is_it":
            "Leaf scorch appears as dry, brown, or damaged areas along leaf edges or surfaces.",

        "why_happens":
            "It can be associated with environmental stress, water imbalance, heat, or other growing conditions.",

        "what_to_do": [
            "Check whether the plant is receiving appropriate water.",
            "Protect plants from excessive environmental stress where possible.",
            "Inspect the plant for additional symptoms.",
            "Maintain suitable growing conditions."
        ],

        "prevention": [
            "Maintain consistent appropriate watering.",
            "Avoid severe environmental stress.",
            "Monitor plants during very hot or dry conditions."
        ]
    },


    "Powdery Mildew": {

        "what_is_it":
            "A fungal disease that often appears as a white, powder-like coating on leaves.",

        "why_happens":
            "It is favored by certain humid conditions and poor air circulation, although the leaf surface does not always need to remain wet.",

        "what_to_do": [
            "Remove severely affected leaves.",
            "Improve air circulation around the plant.",
            "Avoid overcrowding.",
            "Use an appropriate treatment recommended for the specific plant."
        ],

        "prevention": [
            "Provide good air circulation.",
            "Avoid overcrowding plants.",
            "Monitor new growth regularly."
        ]
    },


    "Septoria Leaf Spot": {

        "what_is_it":
            "A fungal leaf disease that produces small spots on leaves and can cause affected leaves to decline.",

        "why_happens":
            "It spreads more easily when leaves remain wet and infected plant debris is present.",

        "what_to_do": [
            "Remove severely affected leaves.",
            "Remove fallen infected leaves.",
            "Water near the soil instead of over the foliage.",
            "Improve air circulation."
        ],

        "prevention": [
            "Keep foliage dry when possible.",
            "Clean up infected plant debris.",
            "Avoid overcrowding plants."
        ]
    },


    "Spider Mite": {

        "what_is_it":
            "Spider mites are tiny pests that feed on plant tissues and can cause yellowing, speckling, and leaf damage.",

        "why_happens":
            "They can increase rapidly during hot and dry conditions.",

        "what_to_do": [
            "Inspect the undersides of leaves.",
            "Separate heavily affected plants when appropriate.",
            "Wash plant surfaces with suitable water pressure when appropriate.",
            "Use a pest-management treatment suitable for the plant if needed."
        ],

        "prevention": [
            "Regularly inspect leaves.",
            "Avoid severe plant stress.",
            "Maintain appropriate growing conditions."
        ]
    },


    "TYLCV": {

        "what_is_it":
            "TYLCV stands for Tomato Yellow Leaf Curl Virus, a viral disease that can cause leaf curling and yellowing.",

        "why_happens":
            "The virus is commonly spread by whiteflies feeding on infected plants.",

        "what_to_do": [
            "Remove severely affected plants when appropriate.",
            "Control whiteflies using suitable local pest-management practices.",
            "Remove nearby infected plant material.",
            "Consult an agricultural specialist for severe outbreaks."
        ],

        "prevention": [
            "Monitor plants for whiteflies.",
            "Use healthy planting material.",
            "Control insect vectors appropriately."
        ]
    },


    "Target Spot": {

        "what_is_it":
            "A fungal disease that can cause circular spots on leaves, sometimes with ring-like patterns.",

        "why_happens":
            "It is encouraged by warm, humid conditions and prolonged leaf wetness.",

        "what_to_do": [
            "Remove severely affected leaves.",
            "Avoid overhead watering.",
            "Improve air circulation.",
            "Remove infected plant debris."
        ],

        "prevention": [
            "Keep foliage dry when possible.",
            "Provide adequate spacing between plants.",
            "Clean up infected leaves and debris."
        ]
    },


    "Tomato Mosaic Virus": {

        "what_is_it":
            "A viral disease that can cause mottled or mosaic-like patterns, distorted growth, and reduced plant health.",

        "why_happens":
            "The virus can spread through infected plant material and contaminated hands or tools.",

        "what_to_do": [
            "Remove severely affected plants when appropriate.",
            "Clean tools after handling affected plants.",
            "Avoid handling healthy plants immediately after affected plants.",
            "Use healthy planting material."
        ],

        "prevention": [
            "Disinfect gardening tools.",
            "Use healthy seeds or planting material.",
            "Remove infected plant material promptly."
        ]
    }
}


# ============================================================
# GET DISEASE INFORMATION
# ============================================================

def get_disease_info(disease):

    return DISEASE_INFO.get(
        disease,
        {
            "what_is_it":
                "The system detected a possible plant health problem.",

            "why_happens":
                "The exact cause cannot be determined from the prediction alone.",

            "what_to_do": [
                "Inspect the plant carefully.",
                "Remove severely damaged parts where appropriate.",
                "Monitor the plant for changes.",
                "Consult a local agricultural specialist if the problem continues."
            ],

            "prevention": [
                "Keep the growing area clean.",
                "Provide suitable growing conditions.",
                "Monitor the plant regularly."
            ]
        }
    )


# ============================================================
# USER-FRIENDLY WEB PREDICTION
# ============================================================

def web_predict(
    image,
    symptoms
):

    total_start = time.perf_counter()

    print("")
    print("==============================================")
    print("STARTING PLANT ANALYSIS")
    print("==============================================")

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if image is None:

        return (
            "⚠️ **Please upload a clear leaf image.**",
            "",
            ""
        )

    # --------------------------------------------------------
    # CLEAN SYMPTOMS
    # --------------------------------------------------------

    if symptoms is None:
        symptoms = ""

    symptoms = str(symptoms).strip()

    # ========================================================
    # IMAGE PREDICTION
    # ========================================================

    print("")
    print("Analyzing leaf image...")

    vision_result = vision_predict(
        image
    )

    # ========================================================
    # COMBINE WITH SYMPTOMS
    # ========================================================

    if symptoms:

        print("")
        print("Analyzing described symptoms...")

        text_result = text_predict(
            symptoms
        )

        print("")
        print("Combining available information...")

        fusion_result = fusion_predict(
            vision_result,
            text_result
        )

        final_disease = fusion_result[
            "final_disease"
        ]

        final_probabilities = fusion_result[
            "final_probabilities"
        ]

    else:

        print("")
        print("No symptom description provided.")
        print("Using image result.")

        final_disease = vision_result[
            "disease"
        ]

        final_probabilities = vision_result[
            "probabilities"
        ]

    # ========================================================
    # GET DISEASE INFORMATION
    # ========================================================

    disease_info = get_disease_info(
        final_disease
    )

    # ========================================================
    # MAIN DIAGNOSIS CARD
    # ========================================================

    diagnosis_card = f"""
# 🌿 Diagnosis: {final_disease}

---

### ❓ What is it?

{disease_info["what_is_it"]}

---

### 🔎 Why did this happen?

{disease_info["why_happens"]}

---

### 🩺 What should you do?

"""

    for action in disease_info["what_to_do"]:

        diagnosis_card += (
            f"- {action}\n"
        )

    diagnosis_card += """

---

### 🛡️ How can you prevent it?

"""

    for prevention in disease_info["prevention"]:

        diagnosis_card += (
            f"- {prevention}\n"
        )

    diagnosis_card += """

---

### ⚠️ Important

This is an AI-assisted prediction based on the
submitted leaf image and symptoms. The result is
not guaranteed to be a confirmed diagnosis.

For serious, rapidly spreading, or uncertain
plant problems, confirmation from a local
agricultural specialist is recommended.
"""

    # ========================================================
    # OTHER POSSIBLE RESULTS
    # ========================================================

    top_predictions = get_top_predictions(
        final_probabilities,
        top_k=3
    )

    other_results = ""

    alternatives = []

    for item in top_predictions:

        if item["disease"] != final_disease:

            alternatives.append(
                item["disease"]
            )

    if alternatives:

        other_results = """
### 🔍 Other possible results

The system also considered:

"""

        for disease in alternatives:

            other_results += (
                f"- **{disease}**\n"
            )

        other_results += """

> These are alternative AI predictions and are
> not shown as confirmed diagnoses.
"""

    # ========================================================
    # COMPLETE
    # ========================================================

    total_time = (
        time.perf_counter()
        -
        total_start
    )

    print("")
    print("==============================================")
    print(
        f"ANALYSIS COMPLETE — {total_time:.3f} seconds"
    )
    print("==============================================")

    return (
        diagnosis_card,
        other_results,
        ""
    )


# ============================================================
# CLEAR
# ============================================================

def clear_interface():

    return (
        None,
        "",
        "",
        "",
        ""
    )


# ============================================================
# PROFESSIONAL WEBSITE CSS
# ============================================================

custom_css = """

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    padding: 0 25px 35px 25px !important;
}

.hero {
    text-align: center;
    padding: 48px 30px 42px 30px;
    margin: 10px 0 30px 0;
    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            #064e3b 0%,
            #047857 55%,
            #059669 100%
        );

    color: white;

    box-shadow:
        0 12px 30px rgba(0, 0, 0, 0.12);
}

.hero h1 {
    font-size: 44px !important;
    font-weight: 800 !important;
    margin: 0 0 12px 0 !important;
}

.hero p {
    font-size: 17px !important;
    margin: 7px 0 !important;
}

.hero .subtitle {
    font-size: 15px !important;
    opacity: 0.90;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin: 15px 0 12px 0;
}

.input-card {
    border-radius: 20px !important;
    padding: 18px !important;
    border: 1px solid #d1d5db !important;

    box-shadow:
        0 5px 18px rgba(0, 0, 0, 0.06);
}

.input-card textarea {
    border-radius: 12px !important;
}

.image-upload {
    border-radius: 18px !important;
    overflow: hidden !important;
}

.analyze-button {
    min-height: 56px !important;
    border-radius: 14px !important;
    font-size: 18px !important;
    font-weight: 750 !important;

    box-shadow:
        0 7px 18px rgba(4, 120, 87, 0.20);
}

.clear-button {
    min-height: 56px !important;
    border-radius: 14px !important;
    font-size: 16px !important;
}

.result-card {
    border-radius: 22px !important;
    padding: 30px !important;
    border: 1px solid #d1d5db !important;

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.07);

    margin-top: 12px;
}

.result-card h1 {
    font-size: 34px !important;
}

.result-card h3 {
    margin-top: 20px !important;
}

.result-card li {
    margin-bottom: 8px !important;
    line-height: 1.55 !important;
}

.top-card {
    border-radius: 18px !important;
    padding: 22px !important;
    border: 1px solid #d1d5db !important;

    box-shadow:
        0 5px 18px rgba(0, 0, 0, 0.05);

    margin-top: 18px;
}

.info-box {
    border-radius: 18px;
    padding: 22px;
    margin: 25px 0;
    border: 1px solid #d1d5db;
}

.footer {
    text-align: center;
    margin-top: 40px;
    padding: 25px 10px;
    font-size: 13px;
    opacity: 0.75;
    border-top: 1px solid #d1d5db;
}

@media (max-width: 700px) {

    .gradio-container {
        padding: 0 12px 25px 12px !important;
    }

    .hero {
        padding: 32px 18px;
    }

    .hero h1 {
        font-size: 30px !important;
    }

    .hero p {
        font-size: 14px !important;
    }

    .result-card h1 {
        font-size: 27px !important;
    }

}

"""


# ============================================================
# BUILD INTERFACE
# ============================================================

with gr.Blocks(
    title="Plant Disease Detection System"
) as demo:

    # ========================================================
    # HERO
    # ========================================================

    gr.HTML("""
    <div class="hero">

        <h1>🌿 Plant Disease Detection</h1>

        <p>
            Identify possible plant diseases and learn
            what you can do next.
        </p>

        <p class="subtitle">
            Upload a leaf photo and describe its symptoms
            to get a possible diagnosis and practical guidance.
        </p>

    </div>
    """)


    # ========================================================
    # INPUT SECTION
    # ========================================================

    gr.HTML("""
    <div class="section-title">
        🔎 Analyze Your Plant
    </div>
    """)

    with gr.Row():

        with gr.Column(
            scale=1,
            elem_classes=["input-card"]
        ):

            gr.Markdown(
                "### 📷 Upload Leaf Image"
            )

            image_input = gr.Image(
                type="pil",
                label="Leaf Image",
                height=360
            )

            gr.Markdown(
                "Upload a clear image of the affected leaf."
            )


        with gr.Column(
            scale=1,
            elem_classes=["input-card"]
        ):

            gr.Markdown(
                "### 📝 Describe Symptoms"
            )

            symptoms_input = gr.Textbox(
                label="Symptoms",
                placeholder=(
                    "Example:\n"
                    "The leaves are curling and turning yellow. "
                    "There are also small spots on the leaves."
                ),
                lines=9
            )

            gr.Markdown("""
            💡 **Tip:** Describe visible symptoms such as:

            • Leaf color changes  
            • Spots or lesions  
            • Curling  
            • Powdery coating  
            • Drying or discoloration
            """)


    # ========================================================
    # BUTTONS
    # ========================================================

    with gr.Row():

        predict_button = gr.Button(
            "🔍  Analyze Plant",
            variant="primary",
            elem_classes=["analyze-button"]
        )

        clear_button = gr.Button(
            "🗑️  Clear",
            variant="secondary",
            elem_classes=["clear-button"]
        )


    # ========================================================
    # RESULTS
    # ========================================================

    gr.HTML("""
    <div class="section-title">
        🎯 Plant Disease Result
    </div>
    """)

    diagnosis_output = gr.Markdown(
        visible=True,
        elem_classes=["result-card"]
    )


    # ========================================================
    # OTHER POSSIBLE RESULTS
    # ========================================================

    other_results_output = gr.Markdown(
        visible=True,
        elem_classes=["top-card"]
    )


    # ========================================================
    # HIDDEN TECHNICAL OUTPUT
    # ========================================================
    #
    # This output is intentionally hidden.
    # The user does not need to see internal model details.
    #

    technical_output = gr.Markdown(
        visible=False
    )


    # ========================================================
    # INFORMATION
    # ========================================================

    gr.HTML("""
    <div class="info-box">

        <h3>🌱 About This System</h3>

        <p>
        This system uses artificial intelligence to analyze
        plant leaf images and, when provided, the symptoms
        described by the user.
        </p>

        <p>
        It provides a possible disease identification together
        with information about why the problem may occur and
        what the user can do next.
        </p>

        <p>
        The result is an AI-assisted prediction and should be
        confirmed by an agricultural specialist when the plant
        problem is serious, spreading, or uncertain.
        </p>

    </div>
    """)


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML("""
    <div class="footer">

        <b>🌿 Plant Disease Detection System</b>

        <br><br>

        AI-assisted plant disease identification

        <br>

        16-class plant disease classification

        <br><br>

        Machine Learning Lab • CSE 0619 321L(1)

    </div>
    """)


    # ========================================================
    # ANALYZE EVENT
    # ========================================================

    predict_button.click(
        fn=web_predict,
        inputs=[
            image_input,
            symptoms_input
        ],
        outputs=[
            diagnosis_output,
            other_results_output,
            technical_output
        ]
    )


    # ========================================================
    # CLEAR EVENT
    # ========================================================

    clear_button.click(
        fn=clear_interface,
        inputs=[],
        outputs=[
            image_input,
            symptoms_input,
            diagnosis_output,
            other_results_output,
            technical_output
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    print("==============================================")
    print("STARTING PLANT DISEASE DETECTION APPLICATION")
    print("Host:", RENDER_HOST)
    print("Port:", RENDER_PORT)
    print("==============================================")

    demo.launch(
        server_name=RENDER_HOST,
        server_port=RENDER_PORT,
        css=custom_css,
        theme=gr.themes.Soft()
    )
