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
# DISABLE TENSORFLOW JIT / XLA
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

vision_model = tf.keras.models.load_model(
    vision_path,
    compile=False
)

try:
    vision_model.jit_compile = False
except Exception:
    pass


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

    if image.shape.rank == 2:

        image = tf.stack(
            [image, image, image],
            axis=-1
        )

    if image.shape[-1] == 4:

        image = image[..., :3]

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# VISION PREDICTION
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

    if not cleaned_text:

        probabilities = np.zeros(
            len(classes),
            dtype=np.float32
        )

        print("No symptoms supplied.")

        return {
            "disease": "Not provided",
            "confidence": 0.0,
            "class_index": -1,
            "probabilities": probabilities
        }

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

    if len(vision_probabilities) != len(
        text_probabilities
    ):

        raise ValueError(
            "Image and symptom probability vectors "
            "have different lengths."
        )

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
# DISEASE INFORMATION
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
# USER-FRIENDLY PREDICTION
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

    if image is None:

        return (
            "⚠️ **Please upload a clear leaf image before starting the analysis.**",
            "",
            ""
        )

    if symptoms is None:
        symptoms = ""

    symptoms = str(symptoms).strip()

    # ========================================================
    # IMAGE
    # ========================================================

    vision_result = vision_predict(
        image
    )

    # ========================================================
    # IMAGE + SYMPTOMS
    # ========================================================

    if symptoms:

        text_result = text_predict(
            symptoms
        )

        fusion_result = fusion_predict(
            vision_result,
            text_result
        )

        final_disease = fusion_result[
            "final_disease"
        ]

        final_confidence = fusion_result[
            "final_confidence"
        ]

        final_probabilities = fusion_result[
            "final_probabilities"
        ]

        using_both = True

    else:

        final_disease = vision_result[
            "disease"
        ]

        final_confidence = vision_result[
            "confidence"
        ]

        final_probabilities = vision_result[
            "probabilities"
        ]

        using_both = False

    # ========================================================
    # INFORMATION
    # ========================================================

    disease_info = get_disease_info(
        final_disease
    )

    confidence_percent = final_confidence * 100

    # ========================================================
    # CONFIDENCE MESSAGE
    # ========================================================

    if confidence_percent >= 85:

        confidence_message = (
            "The assessment shows strong confidence based on "
            "the information provided."
        )

        confidence_icon = "🟢"

    elif confidence_percent >= 70:

        confidence_message = (
            "The assessment shows moderate-to-high confidence. "
            "The plant should still be monitored."
        )

        confidence_icon = "🟡"

    else:

        confidence_message = (
            "The assessment has lower confidence. "
            "Consider providing a clearer image and more specific symptoms."
        )

        confidence_icon = "🟠"

    # ========================================================
    # MAIN RESULT
    # ========================================================

    diagnosis_card = f"""
<div class="diagnosis-card">

<div class="result-eyebrow">
🌿 AI-ASSISTED PLANT HEALTH ASSESSMENT
</div>

<h1>Possible Condition</h1>

<div class="diagnosis-name">
{final_disease}
</div>

<div class="confidence-pill">
{confidence_icon} Confidence: {confidence_percent:.1f}%
</div>

<div class="confidence-message">
{confidence_message}
</div>

</div>

<div class="guidance-card">

<h2>🌱 What does this mean?</h2>

<p>
{disease_info["what_is_it"]}
</p>

</div>

<div class="guidance-card">

<h2>🔎 Why might this happen?</h2>

<p>
{disease_info["why_happens"]}
</p>

</div>

<div class="guidance-card action-card">

<h2>🩺 What should you do now?</h2>

<ul>
"""

    for action in disease_info["what_to_do"]:

        diagnosis_card += f"""
<li>✓ {action}</li>
"""

    diagnosis_card += """
</ul>

</div>

<div class="guidance-card prevention-card">

<h2>🛡️ How can you prevent it?</h2>

<ul>
"""

    for prevention in disease_info["prevention"]:

        diagnosis_card += f"""
<li>✓ {prevention}</li>
"""

    diagnosis_card += """
</ul>

</div>

<div class="important-card">

<h3>⚠️ Important guidance</h3>

<p>
This result is an AI-assisted assessment, not a laboratory-confirmed diagnosis.
Different plant problems can sometimes produce similar visible symptoms.
</p>

<p>
If the condition is spreading quickly, severely damaging the plant,
or the result is uncertain, consider confirmation from a qualified
agricultural or plant-health specialist.
</p>

</div>
"""

    # ========================================================
    # OTHER POSSIBLE RESULTS
    # ========================================================

    top_predictions = get_top_predictions(
        final_probabilities,
        top_k=3
    )

    alternatives = []

    for item in top_predictions:

        if item["disease"] != final_disease:

            alternatives.append(item)

    other_results = ""

    if alternatives:

        other_results = """
<div class="alternatives-card">

<h3>🔍 Other possibilities considered</h3>

<p class="small-note">
The AI considered these additional possibilities.
They are alternatives, not confirmed diagnoses.
</p>

"""

        for item in alternatives:

            other_results += f"""
<div class="alternative-item">
<span>{item["disease"]}</span>
<span>{item["confidence"] * 100:.1f}%</span>
</div>
"""

        other_results += """
</div>
"""

    # ========================================================
    # COMPLETION
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
# CSS
# ============================================================

custom_css = """

/* ==========================================================
   GLOBAL
   ========================================================== */

* {
    box-sizing: border-box !important;
}

html,
body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
}

body {
    overflow-x: hidden !important;
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    padding: 0 25px 45px 25px !important;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero {
    text-align: center;
    padding: 48px 30px 42px 30px;
    margin: 10px 0 30px 0;
    border-radius: 26px;

    background:
        linear-gradient(
            135deg,
            #064e3b 0%,
            #047857 50%,
            #059669 100%
        );

    color: white;

    box-shadow:
        0 14px 35px rgba(0, 70, 45, 0.20);

    position: relative;
    overflow: hidden;
}

.hero::before {
    content: "";
    position: absolute;

    width: 300px;
    height: 300px;

    border-radius: 50%;

    border: 1px solid rgba(255,255,255,0.15);

    right: -100px;
    top: -130px;
}

.hero::after {
    content: "";
    position: absolute;

    width: 260px;
    height: 260px;

    border-radius: 50%;

    border: 1px solid rgba(255,255,255,0.10);

    left: -110px;
    bottom: -160px;
}

.hero h1 {
    font-size: 44px !important;
    font-weight: 800 !important;
    margin: 0 0 12px 0 !important;
}

.hero p {
    font-size: 18px !important;
    margin: 8px 0 !important;
    line-height: 1.6 !important;
}

.hero .subtitle {
    font-size: 15px !important;
    opacity: 0.92;
    max-width: 780px;
    margin-left: auto !important;
    margin-right: auto !important;
}


/* ==========================================================
   SECTION TITLES
   ========================================================== */

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #064e3b;
    margin: 24px 0 14px 0;
}


/* ==========================================================
   INPUT CARDS
   ========================================================== */

.input-card {
    border-radius: 22px !important;
    padding: 22px !important;

    border: 1px solid #cce9dc !important;

    background:
        linear-gradient(
            145deg,
            #ffffff 0%,
            #f0fdf4 100%
        ) !important;

    box-shadow:
        0 8px 25px rgba(6, 78, 59, 0.08);

    min-width: 0 !important;
}

.input-card h3 {
    color: #065f46 !important;
    font-weight: 800 !important;
}

.input-card textarea {
    border-radius: 14px !important;
}


/* ==========================================================
   IMAGE UPLOAD
   ========================================================== */

.image-upload {
    border-radius: 18px !important;
    overflow: hidden !important;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

.analyze-button {
    min-height: 58px !important;

    border-radius: 15px !important;

    font-size: 18px !important;
    font-weight: 800 !important;

    box-shadow:
        0 8px 20px rgba(4, 120, 87, 0.20);
}

.clear-button {
    min-height: 58px !important;

    border-radius: 15px !important;

    font-size: 16px !important;
}


/* ==========================================================
   DIAGNOSIS RESULT
   ========================================================== */

.result-card {
    border-radius: 24px !important;

    padding: 0 !important;

    border: none !important;

    background: transparent !important;

    box-shadow: none !important;

    margin-top: 12px;
}

.diagnosis-card {
    border-radius: 25px;

    padding: 35px;

    background:
        linear-gradient(
            135deg,
            #064e3b 0%,
            #047857 55%,
            #10b981 100%
        );

    color: white;

    box-shadow:
        0 15px 35px rgba(6, 78, 59, 0.20);

    margin-bottom: 18px;

    text-align: center;
}

.result-eyebrow {
    font-size: 13px;
    letter-spacing: 2px;
    font-weight: 800;
    opacity: 0.88;
    margin-bottom: 10px;
}

.diagnosis-card h1 {
    font-size: 30px;
    margin: 8px 0;
}

.diagnosis-name {
    font-size: 36px;
    font-weight: 850;
    margin: 12px 0 18px 0;
}

.confidence-pill {
    display: inline-block;

    padding: 10px 18px;

    border-radius: 999px;

    background: rgba(255,255,255,0.17);

    border: 1px solid rgba(255,255,255,0.22);

    font-weight: 750;
}

.confidence-message {
    margin-top: 16px;
    font-size: 15px;
    line-height: 1.6;
    opacity: 0.94;
}


/* ==========================================================
   GUIDANCE CARDS
   ========================================================== */

.guidance-card {
    border-radius: 22px;

    padding: 27px;

    margin: 17px 0;

    background:
        linear-gradient(
            145deg,
            #ffffff 0%,
            #f0fdf4 100%
        );

    border: 1px solid #cce9dc;

    box-shadow:
        0 7px 22px rgba(6, 78, 59, 0.07);
}

.guidance-card h2 {
    color: #065f46;
    font-size: 22px;
    margin-top: 0;
    margin-bottom: 13px;
}

.guidance-card p {
    color: #31564a;
    line-height: 1.75;
    font-size: 16px;
}

.guidance-card ul {
    padding-left: 20px;
}

.guidance-card li {
    color: #31564a;
    margin-bottom: 11px;
    line-height: 1.6;
}

.action-card {
    background:
        linear-gradient(
            145deg,
            #ecfdf5,
            #d1fae5
        );
}

.prevention-card {
    background:
        linear-gradient(
            145deg,
            #f0fdf4,
            #dcfce7
        );
}


/* ==========================================================
   IMPORTANT CARD
   ========================================================== */

.important-card {
    border-radius: 20px;

    padding: 24px;

    margin-top: 18px;

    background:
        linear-gradient(
            145deg,
            #f0fdf4,
            #ecfdf5
        );

    border-left: 5px solid #059669;

    box-shadow:
        0 6px 18px rgba(6, 78, 59, 0.06);
}

.important-card h3 {
    color: #065f46;
    margin-top: 0;
}

.important-card p {
    color: #31564a;
    line-height: 1.65;
}


/* ==========================================================
   ALTERNATIVE RESULTS
   ========================================================== */

.alternatives-card {
    border-radius: 21px;

    padding: 25px;

    margin-top: 20px;

    background:
        linear-gradient(
            145deg,
            #f8fffb,
            #ecfdf5
        );

    border: 1px solid #cce9dc;

    box-shadow:
        0 7px 20px rgba(6, 78, 59, 0.06);
}

.alternatives-card h3 {
    color: #065f46;
    margin-top: 0;
}

.small-note {
    color: #5d756c;
    line-height: 1.5;
}

.alternative-item {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 13px 15px;

    margin-top: 9px;

    border-radius: 12px;

    background: white;

    border: 1px solid #d7eee2;

    color: #174c3d;

    font-weight: 650;
}


/* ==========================================================
   INFORMATION SECTION
   ========================================================== */

.info-box {
    border-radius: 22px;

    padding: 27px;

    margin: 30px 0;

    background:
        linear-gradient(
            145deg,
            #064e3b,
            #047857
        );

    color: white;

    box-shadow:
        0 10px 28px rgba(6, 78, 59, 0.14);
}

.info-box h3 {
    margin-top: 0;
    font-size: 22px;
}

.info-box p {
    line-height: 1.7;
    opacity: 0.93;
}


/* ==========================================================
   AI ASSESSMENT EXPLAINER
   HIDDEN UNTIL USER CLICKS
   ========================================================== */

.ai-details {
    margin-top: 22px;

    border-radius: 22px;

    overflow: hidden;

    border: 1px solid #b7e4d0;

    background:
        linear-gradient(
            145deg,
            #f0fdf4,
            #ecfdf5
        );

    box-shadow:
        0 7px 22px rgba(6, 78, 59, 0.07);
}

.ai-details summary {
    cursor: pointer;

    list-style: none;

    padding: 20px 23px;

    color: #065f46;

    font-size: 17px;

    font-weight: 800;
}

.ai-details summary::-webkit-details-marker {
    display: none;
}

.ai-details summary::after {
    content: "＋";

    float: right;

    font-size: 20px;
}

.ai-details[open] summary::after {
    content: "−";
}

.ai-details-content {
    padding: 0 23px 25px 23px;

    color: #31564a;

    line-height: 1.65;
}

.ai-step {
    margin-top: 17px;

    padding: 18px;

    border-radius: 16px;

    background: white;

    border: 1px solid #d7eee2;
}

.ai-step strong {
    color: #047857;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {
    text-align: center;

    margin-top: 42px;

    padding: 28px 12px;

    font-size: 13px;

    color: #557168;

    border-top: 1px solid #cfe7db;
}

.creator-name {
    color: #047857;

    font-size: 16px;

    font-weight: 800;

    margin-top: 8px;
}


/* ==========================================================
   DESKTOP / TABLET
   ========================================================== */

@media (min-width: 701px) {

    .gradio-container {
        max-width: 1250px !important;
    }

}


/* ==========================================================
   MOBILE FIX
   IMPORTANT:
   ONLY APPLIES BELOW 700PX.
   DESKTOP IS NOT CHANGED.
   ========================================================== */

@media (max-width: 700px) {

    html,
    body {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;

        overflow-x: hidden !important;

        background: #ffffff !important;
    }

    body {
        margin: 0 !important;
        padding: 0 !important;
    }

    .gradio-container {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;

        margin: 0 !important;

        padding: 0 12px 28px 12px !important;

        overflow-x: hidden !important;

        background: #ffffff !important;
    }


    /* ------------------------------------------------------
       Prevent Gradio rows from becoming wider than phone
       ------------------------------------------------------ */

    .gradio-container .row {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;

        margin-left: 0 !important;
        margin-right: 0 !important;

        padding-left: 0 !important;
        padding-right: 0 !important;

        flex-wrap: wrap !important;
        overflow: visible !important;
    }


    /* ------------------------------------------------------
       Stack columns vertically
       ------------------------------------------------------ */

    .gradio-container .row > .column {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;

        flex: 1 1 100% !important;

        margin-left: 0 !important;
        margin-right: 0 !important;

        padding-left: 0 !important;
        padding-right: 0 !important;
    }


    /* ------------------------------------------------------
       Hero
       ------------------------------------------------------ */

    .hero {
        width: 100% !important;

        margin: 8px 0 20px 0 !important;

        padding: 30px 18px 32px 18px !important;

        border-radius: 22px !important;

        overflow: hidden !important;
    }

    .hero h1 {
        font-size: 30px !important;

        line-height: 1.15 !important;
    }

    .hero p {
        font-size: 15px !important;

        line-height: 1.55 !important;
    }

    .hero .subtitle {
        font-size: 13px !important;

        line-height: 1.55 !important;
    }


    /* ------------------------------------------------------
       Section titles
       ------------------------------------------------------ */

    .section-title {
        font-size: 22px !important;

        margin: 22px 0 12px 2px !important;

        line-height: 1.25 !important;
    }


    /* ------------------------------------------------------
       Input cards
       ------------------------------------------------------ */

    .input-card {
        width: 100% !important;
        max-width: 100% !important;

        margin: 0 0 14px 0 !important;

        padding: 17px !important;

        border-radius: 20px !important;

        overflow: hidden !important;
    }

    .input-card h3 {
        font-size: 21px !important;
        line-height: 1.25 !important;
    }


    /* ------------------------------------------------------
       Image component
       ------------------------------------------------------ */

    .input-card .image-upload,
    .input-card .gr-image {
        width: 100% !important;
        max-width: 100% !important;
    }


    /* ------------------------------------------------------
       Textbox
       ------------------------------------------------------ */

    .input-card textarea,
    .input-card input {
        width: 100% !important;
        max-width: 100% !important;

        font-size: 15px !important;
    }


    /* ------------------------------------------------------
       Buttons
       ------------------------------------------------------ */

    .analyze-button,
    .clear-button {
        width: 100% !important;

        min-width: 0 !important;

        margin-bottom: 8px !important;

        min-height: 54px !important;
    }


    /* ------------------------------------------------------
       Diagnosis
       ------------------------------------------------------ */

    .diagnosis-card {
        width: 100% !important;

        padding: 26px 17px !important;

        border-radius: 21px !important;

        margin-top: 4px !important;
    }

    .result-eyebrow {
        font-size: 10px !important;

        letter-spacing: 1.4px !important;

        line-height: 1.5 !important;
    }

    .diagnosis-card h1 {
        font-size: 25px !important;
    }

    .diagnosis-name {
        font-size: 29px !important;

        line-height: 1.2 !important;

        overflow-wrap: anywhere !important;
    }

    .confidence-pill {
        font-size: 13px !important;

        padding: 9px 13px !important;
    }

    .confidence-message {
        font-size: 13px !important;
    }


    /* ------------------------------------------------------
       Guidance
       ------------------------------------------------------ */

    .guidance-card {
        width: 100% !important;

        padding: 21px 18px !important;

        border-radius: 19px !important;

        overflow-wrap: anywhere !important;
    }

    .guidance-card h2 {
        font-size: 19px !important;

        line-height: 1.3 !important;
    }

    .guidance-card p,
    .guidance-card li {
        font-size: 14px !important;

        line-height: 1.6 !important;
    }


    /* ------------------------------------------------------
       Important
       ------------------------------------------------------ */

    .important-card {
        width: 100% !important;

        padding: 19px 17px !important;

        border-radius: 17px !important;
    }

    .important-card p {
        font-size: 13px !important;

        line-height: 1.6 !important;
    }


    /* ------------------------------------------------------
       Alternatives
       ------------------------------------------------------ */

    .alternatives-card {
        width: 100% !important;

        padding: 19px 16px !important;

        border-radius: 18px !important;
    }

    .alternative-item {
        font-size: 13px !important;

        padding: 11px !important;
    }


    /* ------------------------------------------------------
       About section
       ------------------------------------------------------ */

    .info-box {
        width: 100% !important;

        padding: 21px 18px !important;

        border-radius: 19px !important;
    }

    .info-box h3 {
        font-size: 19px !important;
    }

    .info-box p {
        font-size: 13px !important;

        line-height: 1.6 !important;
    }


    /* ------------------------------------------------------
       AI explanation
       ------------------------------------------------------ */

    .ai-details {
        width: 100% !important;

        border-radius: 18px !important;
    }

    .ai-details summary {
        padding: 17px !important;

        font-size: 14px !important;
    }

    .ai-details-content {
        padding: 0 17px 20px 17px !important;

        font-size: 13px !important;
    }

    .ai-step {
        padding: 15px !important;

        font-size: 13px !important;
    }


    /* ------------------------------------------------------
       Footer
       ------------------------------------------------------ */

    .footer {
        width: 100% !important;

        margin-top: 28px !important;

        padding: 22px 8px !important;

        font-size: 11px !important;

        line-height: 1.6 !important;
    }

    .creator-name {
        font-size: 14px !important;
    }


    /* ------------------------------------------------------
       Extra protection against horizontal overflow
       ------------------------------------------------------ */

    .gradio-container *,
    .gradio-container .block,
    .gradio-container .form,
    .gradio-container .wrap,
    .gradio-container .panel {
        max-width: 100% !important;
    }

}


/* ==========================================================
   VERY SMALL PHONES
   ========================================================== */

@media (max-width: 380px) {

    .gradio-container {
        padding-left: 9px !important;
        padding-right: 9px !important;
    }

    .hero {
        padding-left: 14px !important;
        padding-right: 14px !important;
    }

    .hero h1 {
        font-size: 27px !important;
    }

    .diagnosis-name {
        font-size: 25px !important;
    }

    .guidance-card {
        padding: 18px 15px !important;
    }

}


/* ==========================================================
   END CSS
   ==========================================================

"""


# ============================================================
# BUILD INTERFACE
# ============================================================

with gr.Blocks(
    title="PlantCare AI — Plant Disease Detection"
) as demo:

    # ========================================================
    # HERO
    # ========================================================

    gr.HTML("""
    <div class="hero">

        <div style="
            display:inline-block;
            padding:9px 18px;
            border-radius:999px;
            background:rgba(255,255,255,0.12);
            border:1px solid rgba(255,255,255,0.22);
            font-size:13px;
            font-weight:800;
            letter-spacing:2px;
            margin-bottom:18px;
        ">
            🌱 AI-POWERED AGRICULTURE
        </div>

        <h1>PlantCare AI</h1>

        <p>
            Intelligent Plant Disease Detection
        </p>

        <p class="subtitle">
            Upload a leaf image and describe the visible symptoms
            to receive an AI-assisted plant health assessment,
            practical care recommendations, and prevention guidance.
        </p>

    </div>
    """)


    # ========================================================
    # USER GUIDE
    # ========================================================

    gr.HTML("""
    <div class="section-title">
        🌿 How to get the best result
    </div>

    <div class="guidance-card">

        <h2>📷 1. Provide a clear leaf image</h2>

        <p>
        Take a close, well-lit photograph of the affected leaf.
        Try to keep the leaf clearly visible and avoid blurry or
        extremely dark images.
        </p>

        <h2>📝 2. Tell us what you see</h2>

        <p>
        Describe visible changes such as yellowing, spots,
        curling, powdery areas, drying, discoloration, or
        unusual marks.
        </p>

        <h2>🎯 3. Review the assessment</h2>

        <p>
        The system analyzes the available information and
        provides a possible condition together with practical
        next steps and prevention guidance.
        </p>

        <div style="
            margin-top:18px;
            padding:16px;
            border-radius:15px;
            background:#ecfdf5;
            border:1px solid #cce9dc;
        ">
            <b style="color:#047857;">
                ✓ Best image:
            </b>
            clear, well-lit, close-up leaf

            <br><br>

            <b style="color:#047857;">
                ✓ Best symptoms:
            </b>
            specific visible changes

            <br><br>

            <b style="color:#047857;">
                ✓ Best use:
            </b>
            educational and plant-care decision support
        </div>

    </div>
    """)


    # ========================================================
    # INPUT SECTION
    # ========================================================

    gr.HTML("""
    <div class="section-title">
        🔬 Analyze Your Plant
    </div>
    """)

    with gr.Row():

        with gr.Column(
            scale=1,
            elem_classes=["input-card"]
        ):

            gr.Markdown(
                "### 📷 Leaf Image"
            )

            image_input = gr.Image(
                type="pil",
                label="Upload a clear leaf image",
                height=360
            )

            gr.Markdown(
                "A clear close-up image helps the AI identify visual patterns."
            )


        with gr.Column(
            scale=1,
            elem_classes=["input-card"]
        ):

            gr.Markdown(
                "### 📝 Describe What You See"
            )

            symptoms_input = gr.Textbox(
                label="Visible symptoms",
                placeholder=(
                    "Example:\n"
                    "The leaves are turning yellow and have "
                    "small dark spots. Some leaves are curling."
                ),
                lines=9
            )

            gr.Markdown("""
            **Helpful details:**

            • Leaf color changes  
            • Spots or lesions  
            • Curling  
            • Powdery coating  
            • Drying or discoloration  
            • Whether symptoms are spreading
            """)


    # ========================================================
    # BUTTONS
    # ========================================================

    with gr.Row():

        predict_button = gr.Button(
            "🔍  Analyze My Plant",
            variant="primary",
            elem_classes=["analyze-button"]
        )

        clear_button = gr.Button(
            "↻  Start Over",
            variant="secondary",
            elem_classes=["clear-button"]
        )


    # ========================================================
    # RESULT
    # ========================================================

    gr.HTML("""
    <div class="section-title">
        🎯 Your Plant Health Assessment
    </div>
    """)

    diagnosis_output = gr.Markdown(
        visible=True,
        elem_classes=["result-card"]
    )


    # ========================================================
    # ALTERNATIVES
    # ========================================================

    other_results_output = gr.Markdown(
        visible=True,
        elem_classes=["top-card"]
    )


    # ========================================================
    # HIDDEN TECHNICAL OUTPUT
    # ========================================================

    technical_output = gr.Markdown(
        visible=False
    )


    # ========================================================
    # AI ASSESSMENT EXPLANATION
    # ========================================================

    gr.HTML("""
    <details class="ai-details">

        <summary>
            🧠 How the AI assessment works
        </summary>

        <div class="ai-details-content">

            <p>
            The technical process stays in the background.
            This explanation is provided only for users who want
            to understand how the assessment is produced.
            </p>

            <div class="ai-step">

                <strong>01 — Image analysis</strong>

                <br><br>

                The uploaded leaf image is analyzed for visual
                patterns that may be associated with the conditions
                the system has been trained to recognize.

            </div>

            <div class="ai-step">

                <strong>02 — Symptom analysis</strong>

                <br><br>

                When symptoms are provided, the description is
                processed as additional information about the
                visible condition of the plant.

            </div>

            <div class="ai-step">

                <strong>03 — Combined assessment</strong>

                <br><br>

                When both image and symptom information are
                available, the available evidence is combined
                to produce the displayed assessment.

            </div>

            <div class="ai-step">

                <strong>04 — Guidance</strong>

                <br><br>

                The detected condition is matched with
                user-friendly information covering what it is,
                possible causes, recommended next steps,
                and prevention guidance.

            </div>

        </div>

    </details>
    """)


    # ========================================================
    # ABOUT
    # ========================================================

    gr.HTML("""
    <div class="info-box">

        <h3>🌱 About PlantCare AI</h3>

        <p>
        PlantCare AI is designed to make plant-health assessment
        easier to understand. Instead of showing technical model
        outputs, the application presents the result in a
        user-friendly format with practical guidance.
        </p>

        <p>
        For the strongest assessment, provide both a clear leaf
        photograph and a short description of the visible symptoms.
        </p>

        <p>
        The system is intended for educational and decision-support
        purposes. AI predictions can be uncertain, so serious,
        rapidly spreading, or unusual plant problems should be
        checked by a qualified agricultural or plant-health specialist.
        </p>

    </div>
    """)


    # ========================================================
    # FOOTER / CREATOR
    # ========================================================

    gr.HTML("""
    <div class="footer">

        <div>
            🌿 <b>PlantCare AI</b>
        </div>

        <div>
            AI-assisted plant disease detection and care guidance
        </div>

        <div class="creator-name">
            Created by Absana Mehrin Barsha
        </div>

        <div style="margin-top:7px;">
            Machine Learning Lab • CSE 0619 321L(1)
        </div>

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
    print("STARTING PLANTCARE AI")
    print("Host:", RENDER_HOST)
    print("Port:", RENDER_PORT)
    print("==============================================")

    demo.launch(
        server_name=RENDER_HOST,
        server_port=RENDER_PORT,
        css=custom_css,
        theme=gr.themes.Soft(),
        show_error=True
    )
