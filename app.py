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
# RENDER CONFIGURATION
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
# CONFIDENCE LABEL
# ============================================================

def confidence_label(confidence):

    if confidence >= 0.85:
        return "High confidence"

    if confidence >= 0.70:
        return "Good confidence"

    if confidence >= 0.50:
        return "Moderate confidence"

    return "Low confidence"


# ============================================================
# CONFIDENCE COLOR CLASS
# ============================================================

def confidence_class(confidence):

    if confidence >= 0.85:
        return "high"

    if confidence >= 0.70:
        return "good"

    if confidence >= 0.50:
        return "moderate"

    return "low"


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

    if image is None:

        return (
            """
            <div class="empty-result">
                <div class="empty-icon">🌱</div>
                <h2>Ready to analyze your plant</h2>
                <p>
                    Upload a clear leaf image to begin the
                    disease detection process.
                </p>
            </div>
            """,
            "",
            ""
        )

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
    # FUSION
    # ========================================================

    if symptoms:

        print("")
        print("Analyzing described symptoms...")

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

    else:

        print("")
        print("No symptom description provided.")
        print("Using image result.")

        final_disease = vision_result[
            "disease"
        ]

        final_confidence = vision_result[
            "confidence"
        ]

        final_probabilities = vision_result[
            "probabilities"
        ]

    # ========================================================
    # INFORMATION
    # ========================================================

    disease_info = get_disease_info(
        final_disease
    )

    confidence_percent = (
        final_confidence * 100
    )

    confidence_text = confidence_label(
        final_confidence
    )

    confidence_type = confidence_class(
        final_confidence
    )

    # ========================================================
    # ACTION HTML
    # ========================================================

    action_html = ""

    for number, action in enumerate(
        disease_info["what_to_do"],
        start=1
    ):

        action_html += f"""
        <div class="action-item">
            <div class="action-number">{number}</div>
            <div class="action-text">{action}</div>
        </div>
        """

    # ========================================================
    # PREVENTION HTML
    # ========================================================

    prevention_html = ""

    for prevention in disease_info["prevention"]:

        prevention_html += f"""
        <div class="prevention-item">
            <span class="check-icon">✓</span>
            <span>{prevention}</span>
        </div>
        """

    # ========================================================
    # MAIN DIAGNOSIS
    # ========================================================

    diagnosis_html = f"""
    <div class="diagnosis-wrapper">

        <div class="diagnosis-top">

            <div class="diagnosis-label">
                <span class="pulse-dot"></span>
                AI ANALYSIS COMPLETE
            </div>

            <div class="diagnosis-icon">
                🌿
            </div>

            <div class="diagnosis-small">
                POSSIBLE PLANT CONDITION
            </div>

            <h1>{final_disease}</h1>

            <p class="diagnosis-description">
                {disease_info["what_is_it"]}
            </p>

        </div>


        <div class="confidence-section">

            <div class="confidence-heading">

                <span>Prediction confidence</span>

                <strong>
                    {confidence_percent:.1f}%
                </strong>

            </div>

            <div class="confidence-track">

                <div
                    class="confidence-fill {confidence_type}"
                    style="width:{min(confidence_percent, 100):.1f}%"
                ></div>

            </div>

            <div class="confidence-bottom">

                <span>{confidence_text}</span>

                <span>
                    Based on submitted plant information
                </span>

            </div>

        </div>


        <div class="information-grid">

            <div class="info-card why-card">

                <div class="info-card-icon">
                    🔎
                </div>

                <div>

                    <div class="info-card-title">
                        Why might this happen?
                    </div>

                    <div class="info-card-text">
                        {disease_info["why_happens"]}
                    </div>

                </div>

            </div>


            <div class="info-card care-card">

                <div class="info-card-icon">
                    💧
                </div>

                <div>

                    <div class="info-card-title">
                        What should you do?
                    </div>

                    <div class="info-card-text">
                        Follow the practical care
                        recommendations below.
                    </div>

                </div>

            </div>

        </div>


        <div class="result-section">

            <div class="section-heading">

                <span class="section-heading-icon">
                    🩺
                </span>

                <div>

                    <h2>Recommended Actions</h2>

                    <p>
                        Practical steps you can take to
                        manage the identified condition.
                    </p>

                </div>

            </div>

            <div class="actions-list">

                {action_html}

            </div>

        </div>


        <div class="result-section prevention-section">

            <div class="section-heading">

                <span class="section-heading-icon">
                    🛡️
                </span>

                <div>

                    <h2>Prevention & Plant Care</h2>

                    <p>
                        Simple practices that can help
                        maintain healthier plants.
                    </p>

                </div>

            </div>

            <div class="prevention-list">

                {prevention_html}

            </div>

        </div>


        <div class="result-disclaimer">

            <div class="disclaimer-icon">
                ⚠️
            </div>

            <div>

                <strong>
                    Important information
                </strong>

                <p>
                    This result is an AI-assisted prediction,
                    not a guaranteed laboratory diagnosis.
                    If symptoms are severe, spreading rapidly,
                    or unclear, consider confirmation from a
                    qualified agricultural specialist.
                </p>

            </div>

        </div>

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

        alternative_cards = ""

        for rank, item in enumerate(
            alternatives,
            start=2
        ):

            alternative_cards += f"""
            <div class="alternative-card">

                <div class="alternative-rank">
                    #{rank}
                </div>

                <div class="alternative-name">
                    {item["disease"]}
                </div>

                <div class="alternative-confidence">
                    {item["confidence"] * 100:.1f}%
                </div>

            </div>
            """

        other_results = f"""
        <div class="alternatives-wrapper">

            <div class="alternative-heading">

                <span>🔍</span>

                <div>

                    <h3>
                        Other possible conditions
                    </h3>

                    <p>
                        These were among the other
                        possibilities considered by
                        the prediction system.
                    </p>

                </div>

            </div>

            <div class="alternatives-list">

                {alternative_cards}

            </div>

            <div class="alternative-note">
                These alternatives are provided for
                reference and should not be treated
                as confirmed diagnoses.
            </div>

        </div>
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
        diagnosis_html,
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
# PREMIUM GREEN UI CSS
# ============================================================

custom_css = r"""

/* =========================================================
   GLOBAL
   ========================================================= */

:root {

    --forest: #064e3b;
    --forest-dark: #022c22;

    --emerald: #047857;
    --emerald-bright: #059669;
    --green: #16a34a;

    --mint: #d1fae5;
    --mint-light: #ecfdf5;

    --leaf: #22c55e;

    --cream: #f5fbf7;

    --text-dark: #12372a;
    --text: #36564a;
    --text-light: #6b8178;

    --border: #d8e9df;

    --white: #ffffff;

    --shadow:
        0 15px 45px rgba(6, 78, 59, 0.10);

    --shadow-small:
        0 7px 25px rgba(6, 78, 59, 0.08);
}


/* =========================================================
   MAIN APP BACKGROUND
   ========================================================= */

body {

    background:
        linear-gradient(
            180deg,
            #eefaf3 0%,
            #f7fcf9 35%,
            #edf8f1 100%
        ) !important;

}


.gradio-container {

    max-width: 1380px !important;

    margin: auto !important;

    padding:
        18px 28px 50px 28px !important;

    background:
        transparent !important;

}


/* =========================================================
   REMOVE EXCESS GRADIO VISUALS
   ========================================================= */

footer {

    display: none !important;

}

.contain {

    border: none !important;

}


/* =========================================================
   HERO
   ========================================================= */

.hero {

    position: relative;

    overflow: hidden;

    margin:
        0 0 35px 0;

    padding:
        60px 45px 55px 45px;

    border-radius:
        32px;

    background:

        radial-gradient(
            circle at 90% 10%,
            rgba(52, 211, 153, 0.35),
            transparent 28%
        ),

        radial-gradient(
            circle at 5% 95%,
            rgba(16, 185, 129, 0.22),
            transparent 30%
        ),

        linear-gradient(
            135deg,
            #022c22 0%,
            #064e3b 42%,
            #047857 75%,
            #059669 100%
        );

    color: white;

    box-shadow:
        0 22px 55px
        rgba(2, 44, 34, 0.23);

}


.hero::before {

    content: "";

    position: absolute;

    width: 260px;
    height: 260px;

    right: -90px;
    top: -100px;

    border:
        1px solid
        rgba(255,255,255,0.15);

    border-radius:
        50%;

}


.hero::after {

    content: "";

    position: absolute;

    width: 190px;
    height: 190px;

    left: -70px;
    bottom: -100px;

    border:
        1px solid
        rgba(255,255,255,0.12);

    border-radius:
        50%;

}


.hero-content {

    position: relative;

    z-index: 2;

    text-align: center;

}


.hero-badge {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding:
        8px 16px;

    margin-bottom:
        18px;

    border-radius:
        50px;

    background:
        rgba(255,255,255,0.12);

    border:
        1px solid
        rgba(255,255,255,0.20);

    backdrop-filter:
        blur(12px);

    font-size:
        12px;

    font-weight:
        700;

    letter-spacing:
        1.5px;

}


.hero h1 {

    font-size:
        50px !important;

    line-height:
        1.1 !important;

    font-weight:
        850 !important;

    letter-spacing:
        -1.5px;

    margin:
        0 0 18px 0 !important;

}


.hero-subtitle {

    max-width:
        760px;

    margin:
        0 auto 15px auto;

    font-size:
        19px;

    line-height:
        1.6;

    color:
        rgba(255,255,255,0.94);

}


.hero-description {

    max-width:
        690px;

    margin:
        auto;

    font-size:
        14px;

    line-height:
        1.6;

    color:
        rgba(255,255,255,0.72);

}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {

    display:
        flex;

    align-items:
        center;

    gap:
        12px;

    margin:
        30px 4px 16px 4px;

    color:
        var(--forest-dark);

    font-size:
        25px;

    font-weight:
        800;

}


.section-title-icon {

    width:
        42px;

    height:
        42px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        13px;

    background:
        linear-gradient(
            135deg,
            #d1fae5,
            #a7f3d0
        );

    font-size:
        21px;

    box-shadow:
        0 5px 15px
        rgba(16,185,129,0.12);

}


/* =========================================================
   INPUT CARDS
   ========================================================= */

.input-card {

    position:
        relative;

    background:
        rgba(255,255,255,0.92) !important;

    border:
        1px solid
        var(--border) !important;

    border-radius:
        24px !important;

    padding:
        24px !important;

    box-shadow:
        var(--shadow-small);

    transition:
        transform .25s ease,
        box-shadow .25s ease;

}


.input-card:hover {

    transform:
        translateY(-3px);

    box-shadow:
        0 15px 35px
        rgba(6,78,59,0.12);

}


.input-card h3 {

    color:
        var(--forest) !important;

    font-size:
        20px !important;

    font-weight:
        800 !important;

}


.input-card p {

    color:
        var(--text) !important;

}


/* =========================================================
   IMAGE UPLOAD
   ========================================================= */

.image-upload {

    min-height:
        330px;

    border:
        2px dashed
        #9acbb4 !important;

    border-radius:
        20px !important;

    background:
        linear-gradient(
            145deg,
            #f0fdf4,
            #ecfdf5
        ) !important;

    transition:
        all .25s ease;

}


.image-upload:hover {

    border-color:
        var(--emerald) !important;

    background:
        #e7f9ef !important;

}


/* =========================================================
   TEXTBOX
   ========================================================= */

.input-card textarea {

    min-height:
        190px !important;

    border:
        1px solid
        #cfe4d8 !important;

    border-radius:
        17px !important;

    background:
        #f9fdfb !important;

    color:
        var(--text-dark) !important;

    font-size:
        15px !important;

    line-height:
        1.6 !important;

}


.input-card textarea:focus {

    border-color:
        var(--emerald) !important;

    box-shadow:
        0 0 0 4px
        rgba(5,150,105,0.10) !important;

}


/* =========================================================
   BUTTONS
   ========================================================= */

.analyze-button {

    min-height:
        60px !important;

    border:
        none !important;

    border-radius:
        16px !important;

    background:
        linear-gradient(
            135deg,
            #047857,
            #059669
        ) !important;

    color:
        white !important;

    font-size:
        17px !important;

    font-weight:
        800 !important;

    box-shadow:
        0 10px 25px
        rgba(4,120,87,0.25) !important;

    transition:
        all .2s ease !important;

}


.analyze-button:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 14px 30px
        rgba(4,120,87,0.32) !important;

}


.clear-button {

    min-height:
        60px !important;

    border-radius:
        16px !important;

    border:
        1px solid
        #bddbc9 !important;

    background:
        #ffffff !important;

    color:
        var(--forest) !important;

    font-weight:
        750 !important;

}


.clear-button:hover {

    background:
        #ecfdf5 !important;

}


/* =========================================================
   DIAGNOSIS WRAPPER
   ========================================================= */

.diagnosis-wrapper {

    overflow:
        hidden;

    margin-top:
        10px;

    border-radius:
        30px;

    background:
        white;

    border:
        1px solid
        var(--border);

    box-shadow:
        var(--shadow);

}


/* =========================================================
   DIAGNOSIS HEADER
   ========================================================= */

.diagnosis-top {

    position:
        relative;

    text-align:
        center;

    padding:
        42px 30px 38px 30px;

    color:
        white;

    background:

        radial-gradient(
            circle at 20% 0%,
            rgba(52,211,153,0.25),
            transparent 28%
        ),

        linear-gradient(
            135deg,
            #064e3b,
            #047857
        );

}


.diagnosis-label {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        8px;

    padding:
        7px 13px;

    border-radius:
        30px;

    background:
        rgba(255,255,255,0.11);

    border:
        1px solid
        rgba(255,255,255,0.18);

    font-size:
        11px;

    font-weight:
        800;

    letter-spacing:
        1.3px;

}


.pulse-dot {

    width:
        8px;

    height:
        8px;

    border-radius:
        50%;

    background:
        #6ee7b7;

    box-shadow:
        0 0 0 5px
        rgba(110,231,183,0.12);

}


.diagnosis-icon {

    width:
        70px;

    height:
        70px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    margin:
        20px auto 12px auto;

    border-radius:
        22px;

    background:
        rgba(255,255,255,0.13);

    border:
        1px solid
        rgba(255,255,255,0.18);

    font-size:
        32px;

}


.diagnosis-small {

    font-size:
        11px;

    font-weight:
        750;

    letter-spacing:
        2px;

    opacity:
        0.75;

}


.diagnosis-top h1 {

    margin:
        8px 0 12px 0;

    color:
        white;

    font-size:
        38px;

    line-height:
        1.2;

    font-weight:
        850;

}


.diagnosis-description {

    max-width:
        720px;

    margin:
        auto;

    color:
        rgba(255,255,255,0.88);

    font-size:
        15px;

    line-height:
        1.6;

}


/* =========================================================
   CONFIDENCE
   ========================================================= */

.confidence-section {

    padding:
        25px 30px;

    background:
        #f8fdf9;

    border-bottom:
        1px solid
        var(--border);

}


.confidence-heading {

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

    margin-bottom:
        10px;

    color:
        var(--text);

    font-size:
        14px;

    font-weight:
        700;

}


.confidence-heading strong {

    color:
        var(--emerald);

    font-size:
        22px;

}


.confidence-track {

    width:
        100%;

    height:
        12px;

    overflow:
        hidden;

    border-radius:
        20px;

    background:
        #dcefe4;

}


.confidence-fill {

    height:
        100%;

    border-radius:
        20px;

    background:
        linear-gradient(
            90deg,
            #16a34a,
            #34d399
        );

    box-shadow:
        0 0 12px
        rgba(34,197,94,0.25);

}


.confidence-fill.moderate {

    background:
        linear-gradient(
            90deg,
            #65a30d,
            #a3e635
        );

}


.confidence-fill.low {

    background:
        linear-gradient(
            90deg,
            #ca8a04,
            #facc15
        );

}


.confidence-bottom {

    display:
        flex;

    justify-content:
        space-between;

    gap:
        15px;

    margin-top:
        9px;

    font-size:
        12px;

    color:
        var(--text-light);

}


/* =========================================================
   INFORMATION CARDS
   ========================================================= */

.information-grid {

    display:
        grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap:
        18px;

    padding:
        25px 30px 10px 30px;

}


.info-card {

    display:
        flex;

    gap:
        15px;

    padding:
        20px;

    border-radius:
        19px;

    border:
        1px solid
        var(--border);

}


.why-card {

    background:
        linear-gradient(
            135deg,
            #f0fdf4,
            #ecfdf5
        );

}


.care-card {

    background:
        linear-gradient(
            135deg,
            #ecfdf5,
            #d1fae5
        );

}


.info-card-icon {

    flex:
        0 0 auto;

    width:
        48px;

    height:
        48px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        15px;

    background:
        white;

    font-size:
        22px;

    box-shadow:
        0 5px 15px
        rgba(6,78,59,0.07);

}


.info-card-title {

    margin-bottom:
        5px;

    color:
        var(--forest);

    font-weight:
        800;

}


.info-card-text {

    color:
        var(--text);

    font-size:
        13px;

    line-height:
        1.55;

}


/* =========================================================
   RESULT SECTION
   ========================================================= */

.result-section {

    padding:
        30px;

}


.section-heading {

    display:
        flex;

    gap:
        14px;

    align-items:
        flex-start;

    margin-bottom:
        20px;

}


.section-heading-icon {

    flex:
        0 0 auto;

    width:
        50px;

    height:
        50px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        16px;

    background:
        linear-gradient(
            135deg,
            #d1fae5,
            #a7f3d0
        );

    font-size:
        24px;

}


.section-heading h2 {

    margin:
        0 0 5px 0;

    color:
        var(--forest-dark);

    font-size:
        22px;

    font-weight:
        850;

}


.section-heading p {

    margin:
        0;

    color:
        var(--text-light);

    font-size:
        13px;

}


/* =========================================================
   ACTION ITEMS
   ========================================================= */

.actions-list {

    display:
        flex;

    flex-direction:
        column;

    gap:
        11px;

}


.action-item {

    display:
        flex;

    align-items:
        center;

    gap:
        14px;

    padding:
        15px 17px;

    border-radius:
        15px;

    background:
        linear-gradient(
            90deg,
            #f0fdf4,
            #f8fdf9
        );

    border:
        1px solid
        #dcefe4;

}


.action-number {

    flex:
        0 0 auto;

    width:
        34px;

    height:
        34px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        11px;

    background:
        var(--emerald);

    color:
        white;

    font-size:
        13px;

    font-weight:
        800;

}


.action-text {

    color:
        var(--text);

    font-size:
        14px;

    line-height:
        1.5;

}


/* =========================================================
   PREVENTION
   ========================================================= */

.prevention-section {

    margin:
        0 30px 30px 30px;

    padding:
        25px !important;

    border-radius:
        20px;

    background:
        linear-gradient(
            135deg,
            #064e3b,
            #047857
        );

}


.prevention-section .section-heading h2 {

    color:
        white;

}


.prevention-section .section-heading p {

    color:
        rgba(255,255,255,0.72);

}


.prevention-section .section-heading-icon {

    background:
        rgba(255,255,255,0.13);

}


.prevention-list {

    display:
        grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap:
        10px;

}


.prevention-item {

    display:
        flex;

    align-items:
        flex-start;

    gap:
        10px;

    padding:
        13px;

    border-radius:
        13px;

    background:
        rgba(255,255,255,0.09);

    border:
        1px solid
        rgba(255,255,255,0.11);

    color:
        rgba(255,255,255,0.90);

    font-size:
        13px;

    line-height:
        1.45;

}


.check-icon {

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    flex:
        0 0 auto;

    width:
        22px;

    height:
        22px;

    border-radius:
        50%;

    background:
        #34d399;

    color:
        #064e3b;

    font-weight:
        900;

}


/* =========================================================
   DISCLAIMER
   ========================================================= */

.result-disclaimer {

    display:
        flex;

    gap:
        14px;

    margin:
        0 30px 30px 30px;

    padding:
        18px;

    border-radius:
        16px;

    background:
        #fffbeb;

    border:
        1px solid
        #f3e5b0;

}


.disclaimer-icon {

    flex:
        0 0 auto;

    font-size:
        22px;

}


.result-disclaimer strong {

    color:
        #795b00;

    font-size:
        14px;

}


.result-disclaimer p {

    margin:
        5px 0 0 0;

    color:
        #776b43;

    font-size:
        12px;

    line-height:
        1.55;

}


/* =========================================================
   ALTERNATIVE RESULTS
   ========================================================= */

.alternatives-wrapper {

    margin-top:
        20px;

    padding:
        25px;

    border-radius:
        23px;

    background:
        white;

    border:
        1px solid
        var(--border);

    box-shadow:
        var(--shadow-small);

}


.alternative-heading {

    display:
        flex;

    align-items:
        center;

    gap:
        13px;

    margin-bottom:
        18px;

}


.alternative-heading > span {

    width:
        44px;

    height:
        44px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        13px;

    background:
        #ecfdf5;

    font-size:
        21px;

}


.alternative-heading h3 {

    margin:
        0 0 3px 0;

    color:
        var(--forest);

    font-size:
        18px;

    font-weight:
        800;

}


.alternative-heading p {

    margin:
        0;

    color:
        var(--text-light);

    font-size:
        12px;

}


.alternatives-list {

    display:
        grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap:
        10px;

}


.alternative-card {

    position:
        relative;

    display:
        grid;

    grid-template-columns:
        42px 1fr auto;

    align-items:
        center;

    gap:
        10px;

    padding:
        14px;

    border-radius:
        14px;

    background:
        #f5fbf7;

    border:
        1px solid
        var(--border);

}


.alternative-rank {

    width:
        34px;

    height:
        34px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        10px;

    background:
        #d1fae5;

    color:
        var(--forest);

    font-size:
        12px;

    font-weight:
        800;

}


.alternative-name {

    color:
        var(--text-dark);

    font-size:
        13px;

    font-weight:
        750;

}


.alternative-confidence {

    color:
        var(--emerald);

    font-weight:
        800;

    font-size:
        13px;

}


.alternative-note {

    margin-top:
        15px;

    padding:
        11px 13px;

    border-radius:
        12px;

    background:
        #f0fdf4;

    color:
        var(--text-light);

    font-size:
        11px;

}


/* =========================================================
   EMPTY RESULT
   ========================================================= */

.empty-result {

    text-align:
        center;

    padding:
        60px 25px;

    border-radius:
        26px;

    border:
        1px dashed
        #b8d8c5;

    background:
        linear-gradient(
            135deg,
            #f0fdf4,
            #ecfdf5
        );

}


.empty-icon {

    font-size:
        48px;

    margin-bottom:
        10px;

}


.empty-result h2 {

    color:
        var(--forest);

    margin:
        0 0 8px 0;

    font-size:
        23px;

}


.empty-result p {

    color:
        var(--text-light);

    margin:
        0;

}


/* =========================================================
   GUIDE CARDS
   ========================================================= */

.guide-wrapper {

    margin-top:
        35px;

    padding:
        30px;

    border-radius:
        26px;

    background:
        linear-gradient(
            135deg,
            #ffffff,
            #f0fdf4
        );

    border:
        1px solid
        var(--border);

    box-shadow:
        var(--shadow-small);

}


.guide-header {

    text-align:
        center;

    margin-bottom:
        25px;

}


.guide-header h2 {

    margin:
        0 0 7px 0;

    color:
        var(--forest-dark);

    font-size:
        25px;

    font-weight:
        850;

}


.guide-header p {

    margin:
        0;

    color:
        var(--text-light);

    font-size:
        14px;

}


.guide-grid {

    display:
        grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap:
        15px;

}


.guide-card {

    padding:
        20px;

    border-radius:
        18px;

    background:
        white;

    border:
        1px solid
        var(--border);

    transition:
        transform .2s ease;

}


.guide-card:hover {

    transform:
        translateY(-3px);

}


.guide-number {

    width:
        40px;

    height:
        40px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        12px;

    background:
        linear-gradient(
            135deg,
            #d1fae5,
            #a7f3d0
        );

    color:
        var(--forest);

    font-size:
        17px;

    font-weight:
        850;

    margin-bottom:
        13px;

}


.guide-card h3 {

    margin:
        0 0 7px 0;

    color:
        var(--forest);

    font-size:
        16px;

    font-weight:
        800;

}


.guide-card p {

    margin:
        0;

    color:
        var(--text);

    font-size:
        12px;

    line-height:
        1.6;

}


/* =========================================================
   SYMPTOM TIPS
   ========================================================= */

.tip-box {

    margin-top:
        15px;

    padding:
        15px;

    border-radius:
        15px;

    background:
        #ecfdf5;

    border:
        1px solid
        #c9ead8;

}


.tip-box strong {

    color:
        var(--forest);

    font-size:
        13px;

}


.tip-box p {

    margin:
        6px 0 0 0;

    color:
        var(--text);

    font-size:
        12px;

    line-height:
        1.6;

}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    position:
        relative;

    margin-top:
        40px;

    padding:
        35px 20px;

    text-align:
        center;

    border-radius:
        24px;

    background:
        linear-gradient(
            135deg,
            #022c22,
            #064e3b
        );

    color:
        rgba(255,255,255,0.75);

}


.footer-logo {

    font-size:
        24px;

    margin-bottom:
        8px;

}


.footer-title {

    color:
        white;

    font-size:
        16px;

    font-weight:
        800;

}


.footer-subtitle {

    margin-top:
        5px;

    font-size:
        12px;

    color:
        rgba(255,255,255,0.55);

}


.footer-divider {

    width:
        80px;

    height:
        1px;

    margin:
        20px auto;

    background:
        rgba(255,255,255,0.16);

}


/* =========================================================
   MOBILE
   ========================================================= */


/* =========================================================
   FINAL COLLAPSED TECHNICAL PANEL
   ========================================================= */

.technical-details-final {
    margin: 34px 0 24px 0;
    border-radius: 24px;
    overflow: hidden;
    border: 2px solid #9ad8bb;
    background: linear-gradient(135deg, #effcf4 0%, #e2f7ea 100%);
    box-shadow: 0 14px 38px rgba(6, 78, 59, 0.10);
}

.technical-details-final summary {
    list-style: none;
    cursor: pointer;
    display: grid;
    grid-template-columns: 58px 1fr 42px;
    align-items: center;
    gap: 16px;
    padding: 20px 22px;
}

.technical-details-final summary::-webkit-details-marker { display: none; }

.technical-summary-icon {
    width: 52px;
    height: 52px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #064e3b, #059669);
    color: white;
    font-size: 25px;
    box-shadow: 0 8px 20px rgba(4,120,87,.22);
}

.technical-summary-kicker {
    font-size: 10px;
    font-weight: 850;
    letter-spacing: 1.5px;
    color: #2f7b5c;
    margin-bottom: 3px;
}

.technical-summary-title {
    font-size: 21px;
    font-weight: 850;
    color: #064e3b;
}

.technical-summary-subtitle {
    margin-top: 3px;
    font-size: 13px;
    color: #5e786c;
}

.technical-summary-arrow {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: white;
    color: #047857;
    font-size: 24px;
    font-weight: 900;
    transition: transform .2s ease;
}

.technical-details-final[open] .technical-summary-arrow {
    transform: rotate(180deg);
}

.technical-content-final {
    padding: 0 22px 22px 22px;
    border-top: 1px solid #bfe4cf;
}

.technical-intro-final {
    margin: 18px 0;
    color: #49665a;
    font-size: 13px;
    line-height: 1.6;
}

.technical-grid-final {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.technical-card-final {
    padding: 18px;
    border-radius: 17px;
    background: rgba(255,255,255,.78);
    border: 1px solid #cfe9d9;
}

.technical-number-final {
    color: #059669;
    font-weight: 900;
    font-size: 12px;
    margin-bottom: 7px;
}

.technical-card-final h3 {
    margin: 0 0 7px 0;
    color: #064e3b;
    font-size: 15px;
}

.technical-card-final p {
    margin: 0;
    color: #557065;
    font-size: 12px;
    line-height: 1.55;
}

.technical-facts-final {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 14px;
}

.technical-facts-final > div {
    padding: 13px;
    border-radius: 14px;
    background: #ffffff;
    border: 1px solid #d5ece0;
    text-align: center;
}

.technical-facts-final strong,
.technical-facts-final span {
    display: block;
}

.technical-facts-final strong {
    color: #047857;
    font-size: 14px;
}

.technical-facts-final span {
    margin-top: 3px;
    color: #678075;
    font-size: 11px;
}

.details-grid-single {
    grid-template-columns: 1fr !important;
}


.footer-creator {
    margin-top: 16px;
    color: #b9dfcb;
    font-size: 13px;
    letter-spacing: .2px;
}
.footer-creator strong {
    color: #ffffff;
    font-weight: 850;
}
@media (max-width: 800px) {

    .gradio-container {

        padding:
            10px 12px 30px 12px !important;

    }


    .hero {

        padding:
            42px 20px;

        border-radius:
            24px;

    }


    .hero h1 {

        font-size:
            32px !important;

        letter-spacing:
            -0.5px;

    }


    .hero-subtitle {

        font-size:
            15px;

    }


    .information-grid {

        grid-template-columns:
            1fr;

        padding:
            18px;

    }


    .result-section {

        padding:
            20px;

    }


    .prevention-section {

        margin:
            0 20px 20px 20px;

    }


    .prevention-list {

        grid-template-columns:
            1fr;

    }


    .result-disclaimer {

        margin:
            0 20px 20px 20px;

    }


    .guide-grid {

        grid-template-columns:
            1fr;

    }


    .alternatives-list {

        grid-template-columns:
            1fr;

    }


    .diagnosis-top h1 {

        font-size:
            29px;

    }


    .confidence-bottom {

        flex-direction:
            column;

        gap:
            3px;

    }

}


/* =========================================================
   SMALL MOBILE
   ========================================================= */

@media (max-width: 500px) {

    .hero h1 {

        font-size:
            28px !important;

    }


    .diagnosis-top {

        padding:
            32px 18px;

    }


    .diagnosis-top h1 {

        font-size:
            25px;

    }


    .confidence-section {

        padding:
            20px;

    }


    .section-heading h2 {

        font-size:
            19px;

    }


    .action-item {

        align-items:
            flex-start;

    }


    .guide-wrapper {

        padding:
            20px;

    }

}


/* =========================================================
   PREMIUM INFORMATION LAYER
   ========================================================= */

body {
    background:
        radial-gradient(circle at 8% 8%, rgba(167,243,208,.55), transparent 22%),
        radial-gradient(circle at 92% 20%, rgba(134,239,172,.32), transparent 24%),
        radial-gradient(circle at 18% 78%, rgba(187,247,208,.38), transparent 26%),
        linear-gradient(145deg, #e9f8ef 0%, #f7fcf8 42%, #e5f5eb 100%) !important;
    background-attachment: fixed !important;
}

.gradio-container {
    position: relative;
}

.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        radial-gradient(circle at 20% 30%, rgba(22,163,74,.035) 0 2px, transparent 2px),
        radial-gradient(circle at 70% 70%, rgba(6,78,59,.025) 0 2px, transparent 2px);
    background-size: 42px 42px, 58px 58px;
    z-index: -1;
}

.overview-panel,
.model-panel,
.details-card {
    border: 1px solid rgba(125, 174, 148, .34);
    box-shadow: 0 18px 50px rgba(6,78,59,.085);
    border-radius: 28px;
}

.overview-panel {
    padding: 30px;
    margin: -6px 0 34px;
    background: linear-gradient(135deg, rgba(255,255,255,.96), rgba(236,253,245,.96));
}

.overview-intro { max-width: 900px; margin: 0 auto 24px; text-align: center; }
.overview-kicker, .model-kicker {
    color: #047857;
    font-size: 11px;
    font-weight: 850;
    letter-spacing: 1.8px;
    margin-bottom: 8px;
}
.overview-intro h2, .model-heading h2 {
    color: #064e3b;
    font-size: 28px;
    margin: 0 0 9px;
    font-weight: 850;
}
.overview-intro p, .model-heading p { color: #47665a; line-height: 1.7; margin: 0; }

.quick-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
}
.quick-card {
    border-radius: 21px;
    padding: 20px;
    display: flex;
    gap: 14px;
    min-height: 125px;
    border: 1px solid rgba(6,78,59,.08);
    transition: transform .2s ease, box-shadow .2s ease;
}
.quick-card:hover { transform: translateY(-3px); box-shadow: 0 13px 28px rgba(6,78,59,.10); }
.quick-card-dark { background: linear-gradient(145deg,#064e3b,#047857); color:#fff; }
.quick-card-mint { background: linear-gradient(145deg,#d1fae5,#ecfdf5); color:#12372a; }
.quick-card-soft { background: linear-gradient(145deg,#f0fdf4,#dcfce7); color:#12372a; }
.quick-icon { font-size: 28px; line-height: 1; }
.quick-card h3 { margin: 0 0 6px; font-size: 17px; font-weight: 800; }
.quick-card p { margin: 0; font-size: 13px; line-height: 1.55; opacity: .9; }
.quick-card-dark p { color: rgba(255,255,255,.84); }

.quality-strip {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 10px;
    margin-top: 18px;
    padding: 14px 16px;
    border-radius: 16px;
    background: rgba(6,78,59,.055);
    color: #36564a;
    font-size: 12px;
}
.quality-strip span { color:#16a34a; font-weight:900; margin-right:6px; }

.model-panel {
    margin: 0 0 34px;
    padding: 30px;
    background: linear-gradient(145deg, #f4fdf7, #dff6e8);
}
.model-heading { display:flex; gap:16px; align-items:flex-start; margin-bottom:24px; }
.model-icon {
    width:54px; height:54px; flex:0 0 54px; border-radius:17px;
    display:flex; align-items:center; justify-content:center; font-size:27px;
    background:linear-gradient(135deg,#064e3b,#16a34a); box-shadow:0 10px 24px rgba(6,78,59,.18);
}
.model-flow { display:grid; grid-template-columns: 1fr auto 1fr auto 1fr; align-items:center; gap:12px; }
.model-step { background:rgba(255,255,255,.82); border:1px solid rgba(6,78,59,.10); border-radius:20px; padding:20px; min-height:170px; }
.model-step-final { background:linear-gradient(145deg,#ecfdf5,#d1fae5); border-color:#a7f3d0; }
.model-step-top { display:flex; justify-content:space-between; color:#047857; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
.model-step-top span { font-weight:900; opacity:.55; }
.model-step h3 { margin:12px 0 7px; color:#064e3b; font-size:18px; }
.model-step p { margin:0; color:#4a685c; font-size:13px; line-height:1.55; }
.flow-arrow { color:#059669; font-size:28px; font-weight:900; }
.model-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:16px; }
.stat { text-align:center; padding:15px 10px; background:rgba(255,255,255,.72); border-radius:16px; border:1px solid rgba(6,78,59,.08); }
.stat strong { display:block; color:#047857; font-size:19px; font-weight:850; }
.stat span { color:#60786d; font-size:11px; }

.details-grid { display:grid; grid-template-columns:1.25fr .9fr; gap:18px; margin-top:28px; }
.details-card { background:rgba(255,255,255,.94); padding:25px; }
.details-card-title { color:#064e3b; font-size:19px; font-weight:850; margin-bottom:7px; }
.details-card-subtitle { color:#60786d; font-size:13px; line-height:1.55; margin-top:0; }
.checklist-card { background:linear-gradient(145deg,#064e3b,#075e48); color:#fff; border:none; }
.checklist-card .details-card-title { color:#fff; }
.check-row { display:flex; gap:11px; align-items:flex-start; padding:11px 0; border-bottom:1px solid rgba(255,255,255,.10); }
.check-row:last-child { border-bottom:0; }
.check-row > span { width:28px; height:28px; border-radius:9px; background:rgba(255,255,255,.13); display:flex; align-items:center; justify-content:center; color:#a7f3d0; font-size:10px; font-weight:900; flex:0 0 28px; }
.check-row p { margin:0; color:rgba(255,255,255,.82); font-size:12px; line-height:1.55; }
.check-row b { color:#fff; }

/* Result cards become more visually prominent */
.result-card, .top-card { border-radius:26px !important; border-color:#cfe7d8 !important; box-shadow:0 18px 45px rgba(6,78,59,.09) !important; }
.diagnosis-top { background:linear-gradient(135deg,#022c22 0%,#064e3b 48%,#059669 100%) !important; }
.confidence-section { background:linear-gradient(180deg,#f0fdf4,#ecfdf5) !important; }
.info-card.why-card { background:linear-gradient(145deg,#f0fdf4,#dcfce7) !important; }
.info-card.care-card { background:linear-gradient(145deg,#ecfdf5,#d1fae5) !important; }

@media (max-width: 900px) {
    .quick-grid, .quality-strip, .model-stats { grid-template-columns:1fr; }
    .model-flow { grid-template-columns:1fr; }
    .flow-arrow { transform:rotate(90deg); justify-self:center; }
    .details-grid { grid-template-columns:1fr; }
}

@media (max-width: 600px) {
    .overview-panel, .model-panel, .details-card { padding:20px; border-radius:22px; }
    .overview-intro h2, .model-heading h2 { font-size:23px; }
    .model-heading { gap:11px; }
    .model-icon { width:46px; height:46px; flex-basis:46px; font-size:23px; }

    /* Prevent horizontal overflow on phones */
    *, *::before, *::after { box-sizing: border-box !important; }
    html, body { width:100% !important; max-width:100% !important; overflow-x:hidden !important; }
    .gradio-container {
        width:100% !important;
        max-width:100% !important;
        min-width:0 !important;
        padding-left:10px !important;
        padding-right:10px !important;
        overflow-x:hidden !important;
    }
    .gradio-row, .gradio-column, .input-card, .diagnosis-wrapper,
    .result-card, .top-card, .overview-panel, .details-card,
    .technical-details, .guide-wrapper, .alternatives-wrapper {
        width:100% !important;
        max-width:100% !important;
        min-width:0 !important;
    }
    .gradio-row {
        margin-left:0 !important;
        margin-right:0 !important;
        gap:12px !important;
    }
    .gradio-column {
        padding-left:0 !important;
        padding-right:0 !important;
    }
    .hero, .section-title, .confidence-heading, .confidence-bottom,
    .section-heading, .technical-summary, .technical-grid, .technical-stats,
    .details-grid, .quick-grid, .quality-strip {
        max-width:100% !important;
    }
    .hero { margin-left:0 !important; margin-right:0 !important; }
    .hero-subtitle, .hero-description, .diagnosis-description,
    .info-card-text, .action-text, .prevention-item, .check-row p {
        overflow-wrap:anywhere !important;
        word-break:normal !important;
    }
    .diagnosis-top h1 {
        overflow-wrap:anywhere !important;
    }
    .technical-details summary {
        display:flex !important;
        align-items:flex-start !important;
        gap:10px !important;
        flex-wrap:wrap !important;
    }
    .technical-details summary span {
        width:100% !important;
        margin-left:34px !important;
    }

/* =========================================================
   FINAL MOBILE OVERRIDE — PHONE SAFE LAYOUT
   ========================================================= */

html,
body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

gradio-app {
    display: block !important;
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

#root,
.main,
.gradio-container {
    width: 100vw !important;
    min-width: 0 !important;
    max-width: 100vw !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
}

.gradio-container {
    padding-left: 28px !important;
    padding-right: 28px !important;
}

/* Make every custom section calculate its width correctly. */
.hero,
.overview-panel,
.model-panel,
.guide-wrapper,
.details-grid,
.details-card,
.result-card,
.top-card,
.technical-details-final,
.footer,
.section-title {
    box-sizing: border-box !important;
    max-width: 100% !important;
}

/* Restore readable text on pale cards. */
.quick-card-mint,
.quick-card-soft,
.quick-card-mint h3,
.quick-card-soft h3,
.quick-card-mint p,
.quick-card-soft p,
.guide-card,
.guide-card h3,
.guide-card p,
.quality-strip,
.quality-strip b,
.quality-strip div,
.overview-intro,
.overview-intro h2,
.overview-intro p {
    color: #12372a !important;
}

.guide-card p,
.quick-card-mint p,
.quick-card-soft p {
    opacity: 1 !important;
}

@media (max-width: 700px) {

    html,
    body,
    gradio-app,
    #root,
    .main,
    .gradio-container {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    .gradio-container {
        padding: 8px 10px 28px 10px !important;
    }

    /* Never allow a Gradio row to preserve desktop width. */
    .gradio-row,
    .gradio-row.wrap,
    .gradio-column,
    .block,
    .form,
    .input-card {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
    }

    .gradio-row {
        flex-wrap: wrap !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    .gradio-column {
        flex: 1 1 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    /* Phone-friendly spacing. */
    .hero {
        width: 100% !important;
        margin: 0 0 20px 0 !important;
        padding: 34px 18px !important;
        border-radius: 22px !important;
    }

    .hero h1 {
        font-size: 30px !important;
        line-height: 1.08 !important;
        overflow-wrap: anywhere !important;
    }

    .hero-subtitle {
        font-size: 15px !important;
        line-height: 1.45 !important;
    }

    .hero-description {
        font-size: 13px !important;
        line-height: 1.55 !important;
    }

    .overview-panel,
    .guide-wrapper,
    .details-card,
    .model-panel {
        width: 100% !important;
        padding: 18px !important;
        border-radius: 20px !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    .overview-intro h2,
    .guide-header h2,
    .model-heading h2 {
        font-size: 21px !important;
        line-height: 1.25 !important;
    }

    .quick-grid,
    .guide-grid,
    .quality-strip,
    .details-grid,
    .model-stats,
    .technical-grid-final {
        grid-template-columns: 1fr !important;
        width: 100% !important;
        max-width: 100% !important;
    }

    .quick-card {
        width: 100% !important;
        min-width: 0 !important;
        min-height: 0 !important;
        padding: 17px !important;
        box-sizing: border-box !important;
    }

    .quick-card h3,
    .guide-card h3 {
        font-size: 16px !important;
    }

    .quick-card p,
    .guide-card p,
    .quality-strip,
    .check-row p {
        font-size: 12px !important;
        line-height: 1.55 !important;
        overflow-wrap: anywhere !important;
    }

    .quality-strip {
        padding: 13px !important;
    }

    .section-title {
        margin: 22px 2px 12px 2px !important;
        font-size: 21px !important;
        line-height: 1.25 !important;
        overflow-wrap: anywhere !important;
    }

    .input-card {
        padding: 15px !important;
        border-radius: 18px !important;
        margin: 0 !important;
    }

    .image-upload {
        width: 100% !important;
        max-width: 100% !important;
    }

    .tip-box {
        width: 100% !important;
        box-sizing: border-box !important;
    }

    .analyze-button,
    .clear-button {
        width: 100% !important;
        min-height: 52px !important;
    }

    /* Stack action buttons instead of squeezing them side-by-side. */
    .gradio-row:has(.analyze-button) {
        flex-direction: column !important;
        gap: 9px !important;
    }

    .result-card,
    .top-card {
        width: 100% !important;
        max-width: 100% !important;
        padding: 18px !important;
        border-radius: 20px !important;
        box-sizing: border-box !important;
    }

    .diagnosis-top {
        padding: 24px 18px !important;
    }

    .diagnosis-top h1 {
        font-size: 25px !important;
        overflow-wrap: anywhere !important;
    }

    .confidence-section,
    .info-card,
    .prevention-section,
    .result-disclaimer {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    .technical-details-final {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }

    .technical-details-final summary {
        width: 100% !important;
        box-sizing: border-box !important;
        padding: 16px !important;
    }

    .technical-summary-title {
        font-size: 15px !important;
    }

    .technical-summary-subtitle {
        font-size: 11px !important;
        line-height: 1.4 !important;
    }

    .footer {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        padding: 24px 12px !important;
        overflow-wrap: anywhere !important;
    }
}

@media (max-width: 380px) {
    .gradio-container {
        padding-left: 7px !important;
        padding-right: 7px !important;
    }

    .hero h1 {
        font-size: 27px !important;
    }

    .overview-panel,
    .guide-wrapper,
    .details-card,
    .model-panel {
        padding: 15px !important;
    }
}

}

/* =========================================================
   CLEAN PHONE UI — LIGHT BACKGROUND + ANALYZE SECTION FIX
   This is a mobile-only override. Desktop styles are untouched.
   ========================================================= */

@media (max-width: 700px) {

    /* ---- Force Gradio's theme itself to stay light ---- */
    :root,
    gradio-app,
    body,
    #root {
        --body-background-fill: #f5fbf7 !important;
        --background-fill-primary: #ffffff !important;
        --background-fill-secondary: #eef8f1 !important;
        --block-background-fill: #ffffff !important;
        --block-background-fill-secondary: #f7fcf9 !important;
        --panel-background-fill: #ffffff !important;
        --input-background-fill: #ffffff !important;
        --input-background-fill-focus: #ffffff !important;
        --button-secondary-background-fill: #ffffff !important;
        --button-secondary-background-fill-hover: #f0f8f3 !important;
        --border-color-primary: #d7e8dd !important;
        --border-color-accent: #a8d5ba !important;
        --body-text-color: #12372a !important;
        --body-text-color-subdued: #5b7467 !important;
        color-scheme: light !important;
    }

    html,
    body,
    gradio-app,
    #root,
    .main {
        background: #f5fbf7 !important;
        color: #12372a !important;
    }

    body {
        background-image:
            radial-gradient(circle at 10% 0%, rgba(167,243,208,.28), transparent 24%),
            radial-gradient(circle at 95% 12%, rgba(187,247,208,.20), transparent 25%),
            linear-gradient(180deg, #f3fbf6 0%, #f8fcf9 48%, #eef8f1 100%) !important;
        background-attachment: scroll !important;
    }

    gradio-app,
    #root,
    .main,
    .gradio-container {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        margin: 0 !important;
        box-sizing: border-box !important;
        overflow-x: hidden !important;
    }

    .gradio-container {
        padding: 10px 12px 30px !important;
        background: transparent !important;
    }

    .gradio-container::before {
        display: none !important;
    }

    /* ---- Clean every Gradio row/column on the phone ---- */
    .gradio-container .row,
    .gradio-container .gradio-row {
        display: flex !important;
        flex-direction: column !important;
        flex-wrap: nowrap !important;
        align-items: stretch !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        gap: 12px !important;
        margin: 0 !important;
        box-sizing: border-box !important;
    }

    .gradio-container .column,
    .gradio-container .gradio-column {
        display: flex !important;
        flex: 0 0 auto !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
        margin: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    /* ---- Make the Analyze Your Plant area a clean stack ---- */
    .section-title {
        width: 100% !important;
        max-width: 100% !important;
        margin: 24px 2px 12px !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }

    .section-title-icon {
        flex: 0 0 auto !important;
    }

    .input-card {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        margin: 0 !important;
        padding: 16px !important;
        border-radius: 20px !important;
        background: #ffffff !important;
        border: 1px solid #d6e8dc !important;
        box-shadow: 0 8px 24px rgba(6,78,59,.07) !important;
        overflow: hidden !important;
    }

    .input-card h3,
    .input-card p,
    .input-card label,
    .input-card span {
        max-width: 100% !important;
        overflow-wrap: anywhere !important;
    }

    /* ---- Image uploader: no black/dark frame ---- */
    .input-card .image-upload,
    .input-card [data-testid="image"],
    .input-card .image-container,
    .input-card .image-frame,
    .input-card .wrap,
    .input-card .block {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
        background: #f3fbf6 !important;
        color: #12372a !important;
    }

    .input-card .image-upload {
        min-height: 230px !important;
        height: auto !important;
        border-radius: 16px !important;
        border: 2px dashed #9acbb4 !important;
        overflow: hidden !important;
    }

    .input-card .image-upload img,
    .input-card .image-container img,
    .input-card img {
        max-width: 100% !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
    }

    /* ---- Symptoms box: comfortable phone size ---- */
    .input-card textarea,
    .input-card input,
    .input-card .textarea-container,
    .input-card .input-container {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
        background: #fbfefc !important;
        color: #12372a !important;
    }

    .input-card textarea {
        min-height: 175px !important;
        font-size: 15px !important;
        line-height: 1.55 !important;
    }

    .tip-box {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        overflow-wrap: anywhere !important;
    }

    /* ---- Analyze / Start Over: clean full-width buttons ---- */
    .gradio-container .row:has(.analyze-button),
    .gradio-container .gradio-row:has(.analyze-button) {
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        gap: 9px !important;
        margin: 4px 0 0 !important;
    }

    .analyze-button,
    .clear-button {
        display: flex !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        min-height: 54px !important;
        margin: 0 !important;
        box-sizing: border-box !important;
        border-radius: 15px !important;
    }

    /* ---- Result area: keep every card inside the viewport ---- */
    .diagnosis-wrapper,
    .result-card,
    .top-card,
    .confidence-section,
    .info-card,
    .prevention-section,
    .result-disclaimer,
    .technical-details-final {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
        overflow-wrap: anywhere !important;
    }

    .diagnosis-wrapper {
        border-radius: 20px !important;
        overflow: hidden !important;
    }

    .diagnosis-top {
        width: 100% !important;
        max-width: 100% !important;
        padding: 26px 16px !important;
        box-sizing: border-box !important;
    }

    .diagnosis-top h1 {
        font-size: 25px !important;
        line-height: 1.18 !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }

    .diagnosis-description,
    .info-card-text,
    .action-text,
    .prevention-item,
    .alternative-name,
    .alternative-note,
    .technical-card-final p,
    .check-row p {
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }

    .information-grid,
    .prevention-list,
    .technical-grid-final,
    .technical-facts-final,
    .details-grid,
    .quick-grid,
    .guide-grid,
    .quality-strip,
    .model-stats {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        grid-template-columns: 1fr !important;
        box-sizing: border-box !important;
    }

    .overview-panel,
    .model-panel,
    .guide-wrapper,
    .details-card,
    .footer {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
    }

    .overview-panel,
    .model-panel,
    .guide-wrapper,
    .details-card {
        padding: 18px !important;
        border-radius: 20px !important;
    }

    .quick-card,
    .guide-card,
    .details-card,
    .model-step,
    .stat,
    .quality-strip {
        min-width: 0 !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    .technical-details-final summary {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        padding: 14px !important;
        grid-template-columns: 40px minmax(0, 1fr) 28px !important;
        gap: 9px !important;
    }

    .technical-summary-title,
    .technical-summary-subtitle {
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }

    .footer {
        padding: 22px 10px !important;
        overflow-wrap: anywhere !important;
    }
}

"""


# ============================================================
# BUILD APPLICATION
# ============================================================

with gr.Blocks(
    title="PlantCare AI — Intelligent Plant Disease Detection"
) as demo:


    # ========================================================
    # HERO
    # ========================================================

    gr.HTML("""
    <div class="hero">

        <div class="hero-content">

            <div class="hero-badge">
                🌱 AI-POWERED AGRICULTURE
            </div>

            <h1>
                PlantCare AI
            </h1>

            <div class="hero-subtitle">
                Intelligent Plant Disease Detection
            </div>

            <div class="hero-description">
                Upload a leaf image and describe the visible
                symptoms to receive an AI-assisted plant health
                assessment, practical care recommendations,
                and prevention guidance.
            </div>

        </div>

    </div>
    """)


    # ========================================================
    # QUICK START + MODEL OVERVIEW
    # ========================================================

    gr.HTML(f"""
    <div class="overview-panel">

        <div class="overview-intro">
            <div class="overview-kicker">SMART • SIMPLE • EXPLAINABLE</div>
            <h2>🌱 Your quick guide to a better result</h2>
            <p>
                PlantCare AI is designed to give users a possible plant-health
                condition <b>plus the information needed to understand and act on it.</b>
                For the strongest assessment, provide both a clear leaf image and a
                short description of what you observe.
            </p>
        </div>

        <div class="quick-grid">
            <div class="quick-card quick-card-dark">
                <div class="quick-icon">📷</div>
                <div>
                    <h3>1. Show the leaf</h3>
                    <p>Use a bright, focused image. Keep the affected leaf large enough to see clearly.</p>
                </div>
            </div>

            <div class="quick-card quick-card-mint">
                <div class="quick-icon">📝</div>
                <div>
                    <h3>2. Tell us what you see</h3>
                    <p>Mention color changes, spots, curling, powder, drying, wilting, or spreading symptoms.</p>
                </div>
            </div>

            <div class="quick-card quick-card-soft">
                <div class="quick-icon">🎯</div>
                <div>
                    <h3>3. Read before acting</h3>
                    <p>Review the possible condition, confidence, reasons, recommended actions, and prevention tips.</p>
                </div>
            </div>
        </div>

        <div class="quality-strip">
            <div><span>✓</span><b>Best image:</b> clear, well-lit, close-up leaf</div>
            <div><span>✓</span><b>Best symptoms:</b> specific visible changes</div>
            <div><span>✓</span><b>Best use:</b> educational & decision support</div>
        </div>

    </div>
    """)

    # ========================================================
    # ANALYSIS TITLE
    # ========================================================

    gr.HTML("""
    <div class="section-title">

        <div class="section-title-icon">
            🔬
        </div>

        Analyze Your Plant

    </div>
    """)


    # ========================================================
    # INPUT AREA
    # ========================================================

    with gr.Row():

        with gr.Column(
            scale=1,
            elem_classes=["input-card"]
        ):

            gr.Markdown(
                "### 📷 Leaf Image"
            )

            gr.Markdown(
                "Upload a clear photo of the affected leaf."
            )

            image_input = gr.Image(
                type="pil",
                label="",
                height=340,
                elem_classes=["image-upload"]
            )

            gr.HTML("""
            <div class="tip-box">

                <strong>📸 For a better result</strong>

                <p>
                    Use a well-lit image where the leaf is
                    clearly visible. Avoid extremely blurry,
                    dark, or distant photographs.
                </p>

            </div>
            """)


        with gr.Column(
            scale=1,
            elem_classes=["input-card"]
        ):

            gr.Markdown(
                "### 📝 Describe What You See"
            )

            gr.Markdown(
                "Adding symptoms can provide additional context."
            )

            symptoms_input = gr.Textbox(
                label="",
                placeholder=(
                    "Example:\n\n"
                    "The leaves are turning yellow and have "
                    "small dark spots. Some leaves are curling "
                    "and the affected areas are becoming dry."
                ),
                lines=8
            )

            gr.HTML("""
            <div class="tip-box">

                <strong>💡 Helpful symptoms to mention</strong>

                <p>
                    Leaf color changes • spots • lesions •
                    curling • white powder • yellowing •
                    drying • wilting • unusual patterns
                </p>

            </div>
            """)


    # ========================================================
    # BUTTONS
    # ========================================================

    with gr.Row():

        predict_button = gr.Button(
            "🌿  Analyze My Plant",
            variant="primary",
            elem_classes=["analyze-button"]
        )

        clear_button = gr.Button(
            "↺  Start Over",
            variant="secondary",
            elem_classes=["clear-button"]
        )


    # ========================================================
    # RESULT TITLE
    # ========================================================

    gr.HTML("""
    <div class="section-title">

        <div class="section-title-icon">
            🎯
        </div>

        Plant Health Assessment

    </div>
    """)


    # ========================================================
    # DIAGNOSIS RESULT
    # ========================================================

    diagnosis_output = gr.HTML(
        value="""
        <div class="empty-result">

            <div class="empty-icon">
                🌱
            </div>

            <h2>
                Your plant assessment will appear here
            </h2>

            <p>
                Upload a leaf image and click
                <b>Analyze My Plant</b> to begin.
            </p>

        </div>
        """,
        elem_classes=["result-card"]
    )


    # ========================================================
    # ALTERNATIVE RESULTS
    # ========================================================

    other_results_output = gr.HTML(
        value="",
        elem_classes=["top-card"]
    )


    # ========================================================
    # USER GUIDE
    # ========================================================

    gr.HTML("""
    <div class="guide-wrapper">

        <div class="guide-header">

            <h2>
                🌱 How to Use PlantCare AI
            </h2>

            <p>
                Follow three simple steps to get the most
                useful assessment from the system.
            </p>

        </div>


        <div class="guide-grid">

            <div class="guide-card">

                <div class="guide-number">
                    1
                </div>

                <h3>
                    Upload a Leaf
                </h3>

                <p>
                    Choose a clear photograph of the leaf
                    showing the visible symptoms or damaged
                    area.
                </p>

            </div>


            <div class="guide-card">

                <div class="guide-number">
                    2
                </div>

                <h3>
                    Describe Symptoms
                </h3>

                <p>
                    Explain changes such as spots, yellowing,
                    curling, discoloration, powdery growth,
                    or drying.
                </p>

            </div>


            <div class="guide-card">

                <div class="guide-number">
                    3
                </div>

                <h3>
                    Review the Result
                </h3>

                <p>
                    Read the possible condition, confidence,
                    recommended actions, and prevention
                    guidance before deciding what to do next.
                </p>

            </div>

        </div>

    </div>
    """)


    # ========================================================
    # BEFORE YOU ANALYZE
    # ========================================================

    gr.HTML(f"""
    <div class="details-grid details-grid-single">

        <div class="details-card checklist-card">
            <div class="details-card-title">📸 Before you analyze</div>
            <div class="check-row"><span>01</span><p><b>Clean view:</b> make sure the leaf is visible and not heavily covered by fingers or objects.</p></div>
            <div class="check-row"><span>02</span><p><b>Good lighting:</b> avoid very dark photos, harsh glare, or strong shadows.</p></div>
            <div class="check-row"><span>03</span><p><b>Useful context:</b> describe what changed, where it appears, and whether it is spreading.</p></div>
            <div class="check-row"><span>04</span><p><b>Use the result carefully:</b> low-confidence or unusual cases deserve extra confirmation.</p></div>
        </div>

    </div>
    """)

    # ========================================================
    # DISCLAIMER
    # ========================================================

    gr.HTML("""
    <div class="result-disclaimer"
         style="margin-top:30px;">

        <div class="disclaimer-icon">
            ⚠️
        </div>

        <div>

            <strong>
                Responsible Use of AI Results
            </strong>

            <p>
                PlantCare AI provides an AI-assisted prediction
                based on the information submitted by the user.
                It is intended for educational and decision-support
                purposes and does not replace professional
                agricultural diagnosis. For serious, rapidly
                spreading, or uncertain plant problems, consult
                a qualified agricultural specialist.
            </p>

        </div>

    </div>
    """)


    # ========================================================
    # OPTIONAL TECHNICAL EXPLANATION — HIDDEN BY DEFAULT
    # ========================================================

    gr.HTML(f"""
    <details class="technical-details-final">
        <summary>
            <div class="technical-summary-icon">🧠</div>
            <div class="technical-summary-text">
                <div class="technical-summary-kicker">FOR CURIOUS USERS & STUDENTS</div>
                <div class="technical-summary-title">How the AI assessment works</div>
                <div class="technical-summary-subtitle">Click here to reveal the technology behind the result</div>
            </div>
            <div class="technical-summary-arrow">⌄</div>
        </summary>

        <div class="technical-content-final">
            <p class="technical-intro-final">
                This section is optional. It explains, in simple language, what happens behind the result without changing the practical guidance shown above.
            </p>

            <div class="technical-grid-final">
                <div class="technical-card-final">
                    <div class="technical-number-final">01</div>
                    <h3>📷 Image analysis</h3>
                    <p>A trained vision model examines visual patterns in the uploaded plant image.</p>
                </div>

                <div class="technical-card-final">
                    <div class="technical-number-final">02</div>
                    <h3>📝 Symptom analysis</h3>
                    <p>When you describe symptoms, a text model analyzes the words and patterns in that description.</p>
                </div>

                <div class="technical-card-final">
                    <div class="technical-number-final">03</div>
                    <h3>🔗 Combined assessment</h3>
                    <p>When both image and symptom information are available, the system combines the available evidence to produce the displayed result.</p>
                </div>
            </div>

            <div class="technical-facts-final">
                <div><strong>224 × 224</strong><span>Image input size</span></div>
                <div><strong>Image + Symptoms</strong><span>Uses both when provided</span></div>
                <div><strong>AI-assisted</strong><span>Not a laboratory diagnosis</span></div>
            </div>
        </div>
    </details>
    """)

    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML("""
    <div class="footer">

        <div class="footer-logo">
            🌿
        </div>

        <div class="footer-title">
            PlantCare AI
        </div>

        <div class="footer-subtitle">
            Intelligent Plant Disease Detection System
        </div>

        <div class="footer-divider"></div>

        <div class="footer-subtitle">
            AI-assisted plant health assessment
            <br>
            Clear, practical plant-health guidance
        </div>

        <div class="footer-divider"></div>

        <div class="footer-subtitle">
            Machine Learning Lab • CSE 0619 321L(1)
        </div>

        <div class="footer-divider"></div>

        <div class="footer-creator">
            Created by <strong>Absana Mehrin Barsha</strong>
        </div>

    </div>
    """)


    technical_state = gr.State("")


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
            technical_state
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
            technical_state
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    print("==============================================")
    print("STARTING PLANTCARE AI")
    print("==============================================")
    print("Host:", RENDER_HOST)
    print("Port:", RENDER_PORT)
    print("==============================================")

    demo.launch(
        server_name=RENDER_HOST,
        server_port=RENDER_PORT,
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="green",
            neutral_hue="slate"
        )
    )
