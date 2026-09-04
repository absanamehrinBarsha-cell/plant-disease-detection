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
    print("Image analysis started...")

    processed_image = preprocess_image(
        image
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
        f"Inference time: {inference_time:.3f}s"
    )

    print(
        f"Total image analysis: {total_time:.3f}s"
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

    text_features = tfidf_vectorizer.transform(
        [cleaned_text]
    )

    probabilities = calibrated_svm.predict_proba(
        text_features
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
        "Symptom prediction:",
        predicted_disease,
        f"({confidence * 100:.2f}%)"
    )

    print(
        f"Total symptom analysis: {total_time:.3f}s"
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

    return {

        "final_disease":
            final_disease,

        "final_confidence":
            final_confidence,

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
        ],

        "expert_help":
            "If the disease is spreading quickly or affecting many plants, consider consulting a local agricultural specialist."
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
        ],

        "expert_help":
            "Seek professional advice if symptoms continue to spread despite basic plant-care measures."
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
        ],

        "expert_help":
            "If symptoms are rapidly increasing, professional confirmation can help determine the most suitable treatment."
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
        ],

        "expert_help":
            "Because this can be a serious vine disease, professional confirmation is recommended for severe or persistent symptoms."
    },


    "Healthy": {

        "what_is_it":
            "The submitted image does not show clear signs of the diseases included in this system.",

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
        ],

        "expert_help":
            "If the plant continues to decline even though it appears healthy to the system, consider getting it checked by an agricultural specialist."
    },


    "Huanglongbing": {

        "what_is_it":
            "Huanglongbing, also called citrus greening, is a serious disease affecting citrus plants.",

        "why_happens":
            "It is caused by bacteria associated with citrus greening and is spread by insect vectors.",

        "what_to_do": [
            "Inspect the plant for additional symptoms.",
            "Manage affected plants according to local agricultural guidance.",
            "Monitor and manage insect vectors using appropriate local recommendations.",
            "Consult an agricultural specialist for confirmation."
        ],

        "prevention": [
            "Use healthy planting material.",
            "Monitor for insect vectors.",
            "Follow local citrus disease-management recommendations."
        ],

        "expert_help":
            "Professional confirmation is strongly recommended because citrus greening can seriously affect citrus production."
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
        ],

        "expert_help":
            "Rapidly spreading symptoms should be assessed promptly, especially when many plants are affected."
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
        ],

        "expert_help":
            "If the damage continues spreading, consider professional confirmation."
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
        ],

        "expert_help":
            "If symptoms continue despite improved ventilation and moisture management, seek agricultural advice."
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
        ],

        "expert_help":
            "If leaf damage continues or affects new growth, professional advice may help identify the underlying cause."
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
        ],

        "expert_help":
            "If the infection is severe or recurring, consult an agricultural specialist about suitable treatment options."
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
        ],

        "expert_help":
            "Professional confirmation is useful when leaf spotting is severe or difficult to distinguish from other diseases."
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
        ],

        "expert_help":
            "If the infestation becomes severe or spreads to nearby plants, seek advice about appropriate pest management."
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
        ],

        "expert_help":
            "Because viral diseases require careful management, professional confirmation is recommended for serious outbreaks."
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
        ],

        "expert_help":
            "If spotting continues to spread, professional confirmation can help distinguish it from similar leaf diseases."
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
        ],

        "expert_help":
            "If several plants show similar symptoms, seek agricultural advice to help confirm the cause and manage spread."
    }
}


# ============================================================
# DISEASE INFORMATION
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
            ],

            "expert_help":
                "Consider consulting a local agricultural specialist if the condition continues or worsens."
        }
    )


# ============================================================
# CONFIDENCE DESCRIPTION
# ============================================================

def get_confidence_level(confidence):

    percentage = confidence * 100

    if percentage >= 85:
        return (
            "High confidence",
            "The submitted information strongly matches the detected class."
        )

    elif percentage >= 70:
        return (
            "Moderate confidence",
            "The result is reasonably supported, but visual confirmation is recommended."
        )

    elif percentage >= 50:
        return (
            "Low confidence",
            "The result is uncertain. Try a clearer image and provide more symptoms."
        )

    else:
        return (
            "Very low confidence",
            "The system is uncertain about this result. Consider submitting better information."
        )


# ============================================================
# HTML ESCAPING
# ============================================================

def safe_text(text):

    text = str(text)

    replacements = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    }

    for key, value in replacements.items():
        text = text.replace(
            key,
            value
        )

    return text


# ============================================================
# MAIN USER-FACING PREDICTION
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

    <div class="empty-icon">📷</div>

    <h2>Upload a leaf image to begin</h2>

    <p>
        Please upload a clear photo of the affected leaf.
        You can also describe the symptoms for a more
        informative result.
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
    # IMAGE ANALYSIS
    # ========================================================

    print("Analyzing leaf image...")

    vision_result = vision_predict(
        image
    )

    # ========================================================
    # SYMPTOM ANALYSIS + FUSION
    # ========================================================

    if symptoms:

        print("Analyzing symptoms...")

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

    confidence_label, confidence_message = (
        get_confidence_level(
            final_confidence
        )
    )

    confidence_percentage = (
        final_confidence * 100
    )

    # ========================================================
    # DIAGNOSIS CARD
    # ========================================================

    diagnosis_card = f"""
<div class="diagnosis-main">

    <div class="diagnosis-header">

        <div class="diagnosis-icon">
            🌿
        </div>

        <div>

            <div class="result-label">
                POSSIBLE PLANT CONDITION
            </div>

            <h1>
                {safe_text(final_disease)}
            </h1>

            <p class="diagnosis-subtitle">
                Based on the submitted plant information
            </p>

        </div>

    </div>


    <div class="confidence-area">

        <div class="confidence-top">

            <span>
                AI confidence
            </span>

            <strong>
                {confidence_percentage:.1f}%
            </strong>

        </div>

        <div class="confidence-bar">

            <div
                class="confidence-fill"
                style="width:{min(confidence_percentage, 100):.1f}%"
            ></div>

        </div>

        <div class="confidence-label">
            {confidence_label}
        </div>

        <p class="confidence-message">
            {confidence_message}
        </p>

    </div>


    <div class="info-section">

        <div class="info-heading">
            <span class="info-number">01</span>

            <div>
                <h2>❓ What is it?</h2>
                <p class="heading-caption">
                    Understanding the possible condition
                </p>
            </div>
        </div>

        <p>
            {safe_text(disease_info["what_is_it"])}
        </p>

    </div>


    <div class="info-section">

        <div class="info-heading">
            <span class="info-number">02</span>

            <div>
                <h2>🔎 Why might this happen?</h2>
                <p class="heading-caption">
                    Common factors associated with the condition
                </p>
            </div>
        </div>

        <p>
            {safe_text(disease_info["why_happens"])}
        </p>

    </div>


    <div class="info-section action-section">

        <div class="info-heading">

            <span class="info-number">03</span>

            <div>
                <h2>🩺 What should I do?</h2>

                <p class="heading-caption">
                    Practical next steps
                </p>
            </div>

        </div>

        <div class="action-list">
"""

    for index, action in enumerate(
        disease_info["what_to_do"],
        start=1
    ):

        diagnosis_card += f"""
            <div class="action-item">

                <div class="action-check">
                    {index}
                </div>

                <div>
                    {safe_text(action)}
                </div>

            </div>
"""

    diagnosis_card += """
        </div>

    </div>


    <div class="info-section prevention-section">

        <div class="info-heading">

            <span class="info-number">04</span>

            <div>
                <h2>🛡️ How can I prevent it?</h2>

                <p class="heading-caption">
                    Good practices for future plant health
                </p>
            </div>

        </div>

        <div class="prevention-list">
"""

    for prevention in disease_info["prevention"]:

        diagnosis_card += f"""
            <div class="prevention-item">

                <span>✓</span>

                <p>
                    {safe_text(prevention)}
                </p>

            </div>
"""

    diagnosis_card += f"""
        </div>

    </div>


    <div class="expert-section">

        <div class="expert-icon">
            👨‍🌾
        </div>

        <div>

            <h3>
                When should you seek expert help?
            </h3>

            <p>
                {safe_text(disease_info["expert_help"])}
            </p>

        </div>

    </div>


    <div class="result-note">

        <strong>⚠️ Important:</strong>

        This result is an AI-assisted prediction, not a
        guaranteed laboratory diagnosis. If the plant is
        severely damaged, rapidly declining, or the result
        does not match what you observe, seek confirmation
        from a qualified agricultural specialist.

    </div>

</div>
"""

    # ========================================================
    # OTHER POSSIBILITIES
    # ========================================================

    top_predictions = get_top_predictions(
        final_probabilities,
        top_k=3
    )

    alternatives = []

    for item in top_predictions:

        if item["disease"] != final_disease:

            alternatives.append(
                item
            )

    other_results = ""

    if alternatives:

        other_results = """
<div class="alternatives-card">

    <div class="alternative-title">

        <span>🔍</span>

        <div>
            <h2>Other possible results</h2>

            <p>
                Other conditions the AI considered from the
                submitted information.
            </p>
        </div>

    </div>

    <div class="alternative-list">
"""

        for item in alternatives:

            disease_name = safe_text(
                item["disease"]
            )

            percentage = (
                item["confidence"] * 100
            )

            other_results += f"""
        <div class="alternative-item">

            <div class="alternative-name">

                <span class="small-leaf">
                    🌱
                </span>

                <strong>
                    {disease_name}
                </strong>

            </div>

            <div class="alternative-confidence">
                {percentage:.1f}%
            </div>

        </div>
"""

        other_results += """
    </div>

    <p class="alternative-note">
        These are alternative AI possibilities, not confirmed diagnoses.
    </p>

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
# PROFESSIONAL CSS
# ============================================================

custom_css = """

/* =========================================================
   GLOBAL
   ========================================================= */

:root {

    --primary: #0f766e;
    --primary-dark: #115e59;
    --primary-light: #ccfbf1;

    --green: #15803d;
    --green-light: #dcfce7;

    --text: #17251f;
    --muted: #64748b;

    --border: #e2e8f0;

    --background: #f7faf8;

    --card: #ffffff;

    --shadow:
        0 10px 35px rgba(15, 23, 42, 0.08);

    --shadow-large:
        0 20px 55px rgba(15, 23, 42, 0.12);
}


/* =========================================================
   MAIN CONTAINER
   ========================================================= */

.gradio-container {

    max-width: 1320px !important;

    margin: auto !important;

    padding:
        0 22px 50px 22px !important;

    background:
        linear-gradient(
            180deg,
            #f8fffb 0%,
            #f7faf8 45%,
            #ffffff 100%
        );
}


/* =========================================================
   HERO
   ========================================================= */

.hero {

    position: relative;

    overflow: hidden;

    text-align: center;

    padding:
        60px 35px 55px 35px;

    margin:
        10px 0 34px 0;

    border-radius: 30px;

    color: white;

    background:

        radial-gradient(
            circle at 15% 20%,
            rgba(255,255,255,0.14),
            transparent 30%
        ),

        radial-gradient(
            circle at 85% 80%,
            rgba(255,255,255,0.12),
            transparent 32%
        ),

        linear-gradient(
            135deg,
            #064e3b 0%,
            #047857 50%,
            #0d9488 100%
        );

    box-shadow:
        0 20px 55px
        rgba(6, 78, 59, 0.25);
}


.hero::before {

    content: "🌿";

    position: absolute;

    font-size: 130px;

    opacity: 0.08;

    left: 4%;

    top: -25px;

    transform:
        rotate(-15deg);
}


.hero::after {

    content: "🍃";

    position: absolute;

    font-size: 110px;

    opacity: 0.08;

    right: 5%;

    bottom: -30px;

    transform:
        rotate(18deg);
}


.hero h1 {

    position: relative;

    font-size:
        48px !important;

    line-height:
        1.15 !important;

    font-weight:
        850 !important;

    letter-spacing:
        -1.5px;

    margin:
        0 0 15px 0 !important;
}


.hero p {

    position: relative;

    font-size:
        19px !important;

    line-height:
        1.6 !important;

    margin:
        8px auto !important;

    max-width:
        800px;
}


.hero .subtitle {

    font-size:
        15px !important;

    opacity:
        0.86;

    max-width:
        720px;

    margin-top:
        14px !important;
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
        10px;

    font-size:
        26px;

    font-weight:
        800;

    color:
        var(--text);

    margin:
        30px 0 16px 0;
}


.section-subtitle {

    color:
        var(--muted);

    font-size:
        14px;

    margin:
        -8px 0 18px 0;
}


/* =========================================================
   INPUT CARDS
   ========================================================= */

.input-card {

    background:
        rgba(255,255,255,0.96) !important;

    border:
        1px solid
        var(--border) !important;

    border-radius:
        24px !important;

    padding:
        23px !important;

    box-shadow:
        var(--shadow);

    transition:
        all 0.25s ease;
}


.input-card:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 16px 40px
        rgba(15, 23, 42, 0.10);
}


.input-card h3 {

    color:
        var(--text) !important;

    font-size:
        19px !important;

    font-weight:
        750 !important;
}


/* =========================================================
   IMAGE UPLOAD
   ========================================================= */

.image-upload {

    border-radius:
        18px !important;

    overflow:
        hidden !important;

    border:
        2px dashed
        #99f6e4 !important;

    background:
        #f0fdfa !important;

    min-height:
        340px;
}


/* =========================================================
   TEXTBOX
   ========================================================= */

.input-card textarea {

    border-radius:
        15px !important;

    border:
        1px solid
        #cbd5e1 !important;

    line-height:
        1.6 !important;
}


.input-card textarea:focus {

    border-color:
        var(--primary) !important;

    box-shadow:
        0 0 0 3px
        rgba(13,148,136,0.12) !important;
}


/* =========================================================
   TIP BOX
   ========================================================= */

.tip-box {

    background:
        #f0fdfa;

    border:
        1px solid
        #99f6e4;

    border-radius:
        15px;

    padding:
        15px 18px;

    margin-top:
        14px;

    color:
        #134e4a;

    font-size:
        14px;

    line-height:
        1.6;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.analyze-button {

    min-height:
        60px !important;

    border-radius:
        16px !important;

    font-size:
        18px !important;

    font-weight:
        800 !important;

    margin-top:
        20px;

    box-shadow:
        0 10px 25px
        rgba(15,118,110,0.25);

    transition:
        all 0.25s ease !important;
}


.analyze-button:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 14px 30px
        rgba(15,118,110,0.30);
}


.clear-button {

    min-height:
        60px !important;

    border-radius:
        16px !important;

    font-size:
        17px !important;

    font-weight:
        700 !important;

    margin-top:
        20px;
}


/* =========================================================
   HOW IT WORKS
   ========================================================= */

.workflow {

    display:
        grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap:
        18px;

    margin:
        20px 0 35px 0;
}


.workflow-card {

    background:
        #ffffff;

    border:
        1px solid
        var(--border);

    border-radius:
        20px;

    padding:
        23px;

    box-shadow:
        var(--shadow);
}


.workflow-icon {

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
        15px;

    background:
        var(--primary-light);

    font-size:
        25px;

    margin-bottom:
        15px;
}


.workflow-card h3 {

    margin:
        0 0 8px 0;

    font-size:
        18px;

    color:
        var(--text);
}


.workflow-card p {

    margin:
        0;

    color:
        var(--muted);

    font-size:
        14px;

    line-height:
        1.6;
}


/* =========================================================
   DIAGNOSIS
   ========================================================= */

.diagnosis-main {

    background:
        #ffffff;

    border:
        1px solid
        var(--border);

    border-radius:
        28px;

    padding:
        34px;

    box-shadow:
        var(--shadow-large);

    margin-top:
        10px;

    overflow:
        hidden;
}


/* =========================================================
   DIAGNOSIS HEADER
   ========================================================= */

.diagnosis-header {

    display:
        flex;

    align-items:
        center;

    gap:
        20px;

    padding-bottom:
        25px;

    border-bottom:
        1px solid
        var(--border);
}


.diagnosis-icon {

    flex:
        0 0 auto;

    width:
        78px;

    height:
        78px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        22px;

    background:
        linear-gradient(
            135deg,
            #ccfbf1,
            #dcfce7
        );

    font-size:
        39px;

    box-shadow:
        inset 0 0 0 1px
        rgba(15,118,110,0.08);
}


.result-label {

    font-size:
        12px;

    letter-spacing:
        1.5px;

    font-weight:
        800;

    color:
        var(--primary);

    margin-bottom:
        5px;
}


.diagnosis-header h1 {

    font-size:
        38px !important;

    font-weight:
        850 !important;

    color:
        var(--text) !important;

    margin:
        0 !important;
}


.diagnosis-subtitle {

    margin:
        6px 0 0 0;

    color:
        var(--muted);

    font-size:
        14px;
}


/* =========================================================
   CONFIDENCE
   ========================================================= */

.confidence-area {

    margin:
        25px 0 10px 0;

    padding:
        20px;

    border-radius:
        18px;

    background:
        #f8fafc;

    border:
        1px solid
        #e2e8f0;
}


.confidence-top {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    margin-bottom:
        10px;

    color:
        #475569;

    font-size:
        14px;

    font-weight:
        650;
}


.confidence-top strong {

    color:
        var(--primary);

    font-size:
        21px;

    font-weight:
        850;
}


.confidence-bar {

    height:
        11px;

    width:
        100%;

    background:
        #e2e8f0;

    border-radius:
        999px;

    overflow:
        hidden;
}


.confidence-fill {

    height:
        100%;

    border-radius:
        999px;

    background:
        linear-gradient(
            90deg,
            #14b8a6,
            #15803d
        );

    transition:
        width 0.8s ease;
}


.confidence-label {

    margin-top:
        10px;

    color:
        var(--primary-dark);

    font-weight:
        800;

    font-size:
        14px;
}


.confidence-message {

    margin:
        4px 0 0 0;

    color:
        var(--muted);

    font-size:
        13px;

    line-height:
        1.5;
}


/* =========================================================
   INFORMATION SECTIONS
   ========================================================= */

.info-section {

    padding:
        28px 0;

    border-bottom:
        1px solid
        var(--border);
}


.info-section > p {

    color:
        #475569;

    line-height:
        1.75;

    font-size:
        15px;

    margin:
        14px 0 0 57px;
}


.info-heading {

    display:
        flex;

    align-items:
        center;

    gap:
        14px;
}


.info-number {

    flex:
        0 0 auto;

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
        #f0fdfa;

    color:
        var(--primary);

    font-weight:
        850;

    font-size:
        13px;
}


.info-heading h2 {

    margin:
        0;

    font-size:
        21px;

    color:
        var(--text);

    font-weight:
        800;
}


.heading-caption {

    margin:
        3px 0 0 0;

    font-size:
        12px;

    color:
        #94a3b8;
}


/* =========================================================
   ACTION LIST
   ========================================================= */

.action-list {

    margin:
        18px 0 0 57px;

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
        flex-start;

    gap:
        13px;

    padding:
        14px 16px;

    border-radius:
        14px;

    background:
        #f8fafc;

    border:
        1px solid
        #e2e8f0;

    color:
        #334155;

    font-size:
        14px;

    line-height:
        1.55;
}


.action-check {

    flex:
        0 0 auto;

    width:
        28px;

    height:
        28px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        9px;

    background:
        #dcfce7;

    color:
        #15803d;

    font-size:
        12px;

    font-weight:
        850;
}


/* =========================================================
   PREVENTION
   ========================================================= */

.prevention-list {

    margin:
        18px 0 0 57px;

    display:
        grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap:
        11px;
}


.prevention-item {

    display:
        flex;

    align-items:
        flex-start;

    gap:
        10px;

    padding:
        13px 15px;

    border-radius:
        13px;

    background:
        #f0fdf4;

    border:
        1px solid
        #bbf7d0;
}


.prevention-item span {

    color:
        #15803d;

    font-weight:
        900;

    font-size:
        17px;
}


.prevention-item p {

    margin:
        0;

    color:
        #365314;

    font-size:
        14px;

    line-height:
        1.5;
}


/* =========================================================
   EXPERT SECTION
   ========================================================= */

.expert-section {

    display:
        flex;

    align-items:
        flex-start;

    gap:
        16px;

    margin:
        27px 0;

    padding:
        20px;

    border-radius:
        18px;

    background:
        linear-gradient(
            135deg,
            #fff7ed,
            #fffbeb
        );

    border:
        1px solid
        #fed7aa;
}


.expert-icon {

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
        14px;

    background:
        #ffedd5;

    font-size:
        24px;
}


.expert-section h3 {

    margin:
        0 0 5px 0;

    color:
        #9a3412;

    font-size:
        16px;
}


.expert-section p {

    margin:
        0;

    color:
        #7c2d12;

    font-size:
        14px;

    line-height:
        1.6;
}


/* =========================================================
   IMPORTANT NOTE
   ========================================================= */

.result-note {

    padding:
        16px 18px;

    border-radius:
        15px;

    background:
        #f8fafc;

    color:
        #64748b;

    font-size:
        12px;

    line-height:
        1.6;

    border:
        1px solid
        #e2e8f0;
}


/* =========================================================
   ALTERNATIVE RESULTS
   ========================================================= */

.alternatives-card {

    background:
        #ffffff;

    border:
        1px solid
        var(--border);

    border-radius:
        22px;

    padding:
        25px;

    margin-top:
        18px;

    box-shadow:
        var(--shadow);
}


.alternative-title {

    display:
        flex;

    align-items:
        center;

    gap:
        13px;
}


.alternative-title > span {

    width:
        45px;

    height:
        45px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        13px;

    background:
        #f0fdfa;

    font-size:
        22px;
}


.alternative-title h2 {

    margin:
        0;

    color:
        var(--text);

    font-size:
        19px;

    font-weight:
        800;
}


.alternative-title p {

    margin:
        3px 0 0 0;

    color:
        var(--muted);

    font-size:
        12px;
}


.alternative-list {

    display:
        flex;

    flex-direction:
        column;

    gap:
        9px;

    margin-top:
        18px;
}


.alternative-item {

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    padding:
        13px 15px;

    border-radius:
        13px;

    background:
        #f8fafc;

    border:
        1px solid
        #e2e8f0;
}


.alternative-name {

    display:
        flex;

    align-items:
        center;

    gap:
        9px;

    color:
        #334155;

    font-size:
        14px;
}


.small-leaf {

    font-size:
        17px;
}


.alternative-confidence {

    color:
        var(--primary);

    font-weight:
        800;

    font-size:
        13px;
}


.alternative-note {

    margin:
        15px 0 0 0;

    color:
        #94a3b8;

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
        55px 25px;

    background:
        #ffffff;

    border:
        1px dashed
        #cbd5e1;

    border-radius:
        24px;

    box-shadow:
        var(--shadow);
}


.empty-icon {

    font-size:
        52px;

    margin-bottom:
        12px;
}


.empty-result h2 {

    color:
        var(--text);

    margin:
        0 0 8px 0;

    font-size:
        23px;
}


.empty-result p {

    max-width:
        560px;

    margin:
        auto;

    color:
        var(--muted);

    line-height:
        1.6;
}


/* =========================================================
   ABOUT SECTION
   ========================================================= */

.about-card {

    background:
        linear-gradient(
            135deg,
            #ecfdf5,
            #f0fdfa
        );

    border:
        1px solid
        #a7f3d0;

    border-radius:
        24px;

    padding:
        28px;

    margin-top:
        35px;
}


.about-card h2 {

    margin:
        0 0 10px 0;

    color:
        #064e3b;

    font-size:
        22px;
}


.about-card p {

    color:
        #365314;

    line-height:
        1.7;

    font-size:
        14px;
}


.about-grid {

    display:
        grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap:
        13px;

    margin-top:
        20px;
}


.about-item {

    background:
        rgba(255,255,255,0.7);

    border:
        1px solid
        rgba(16,185,129,0.15);

    border-radius:
        15px;

    padding:
        16px;
}


.about-item strong {

    display:
        block;

    color:
        #065f46;

    margin-bottom:
        5px;

    font-size:
        14px;
}


.about-item span {

    color:
        #64748b;

    font-size:
        12px;

    line-height:
        1.5;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {

    text-align:
        center;

    margin-top:
        42px;

    padding:
        28px 10px;

    border-top:
        1px solid
        #e2e8f0;

    color:
        #94a3b8;

    font-size:
        12px;

    line-height:
        1.7;
}


.footer strong {

    color:
        #475569;

    font-size:
        14px;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 850px) {

    .workflow {

        grid-template-columns:
            1fr;

    }

    .about-grid {

        grid-template-columns:
            1fr;

    }

    .prevention-list {

        grid-template-columns:
            1fr;

    }

}


@media (max-width: 700px) {

    .gradio-container {

        padding:
            0 11px 30px 11px !important;

    }


    .hero {

        padding:
            38px 20px;

        border-radius:
            22px;

    }


    .hero h1 {

        font-size:
            32px !important;

        letter-spacing:
            -0.8px;

    }


    .hero p {

        font-size:
            15px !important;

    }


    .diagnosis-main {

        padding:
            22px 18px;

        border-radius:
            21px;

    }


    .diagnosis-header {

        align-items:
            flex-start;

    }


    .diagnosis-icon {

        width:
            58px;

        height:
            58px;

        border-radius:
            17px;

        font-size:
            29px;

    }


    .diagnosis-header h1 {

        font-size:
            27px !important;

    }


    .info-section > p {

        margin-left:
            0;

    }


    .action-list {

        margin-left:
            0;

    }


    .prevention-list {

        margin-left:
            0;

    }


    .expert-section {

        padding:
            16px;

    }


    .section-title {

        font-size:
            22px;

    }

}


"""


# ============================================================
# BUILD INTERFACE
# ============================================================

with gr.Blocks(
    title="Plant Disease Detection AI"
) as demo:

    # ========================================================
    # HERO
    # ========================================================

    gr.HTML("""
    <div class="hero">

        <h1>
            🌿 Plant Disease Detection AI
        </h1>

        <p>
            Understand your plant's health with
            AI-powered image and symptom analysis.
        </p>

        <p class="subtitle">
            Upload a clear leaf photo, describe what you see,
            and receive a possible diagnosis with practical
            guidance for the next steps.
        </p>

    </div>
    """)


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    gr.HTML("""
    <div class="section-title">
        🧠 How the system works
    </div>

    <div class="section-subtitle">
        A simple three-step process designed to make
        plant health analysis easy to understand.
    </div>

    <div class="workflow">

        <div class="workflow-card">

            <div class="workflow-icon">
                📷
            </div>

            <h3>
                1. Upload a leaf photo
            </h3>

            <p>
                Provide a clear image showing the affected
                part of the plant. A well-lit image gives
                the system better visual information.
            </p>

        </div>


        <div class="workflow-card">

            <div class="workflow-icon">
                📝
            </div>

            <h3>
                2. Describe the symptoms
            </h3>

            <p>
                Mention visible changes such as spots,
                yellowing, curling, powdery surfaces,
                drying, or unusual discoloration.
            </p>

        </div>


        <div class="workflow-card">

            <div class="workflow-icon">
                🌱
            </div>

            <h3>
                3. Get practical guidance
            </h3>

            <p>
                The system provides a possible condition,
                confidence information, explanations,
                suggested actions, and prevention tips.
            </p>

        </div>

    </div>
    """)


    # ========================================================
    # INPUT SECTION
    # ========================================================

    gr.HTML("""
    <div class="section-title">
        🔎 Analyze your plant
    </div>

    <div class="section-subtitle">
        For the best result, provide both a clear image
        and a short description of what you observe.
    </div>
    """)


    with gr.Row():

        # ====================================================
        # IMAGE
        # ====================================================

        with gr.Column(
            scale=1,
            elem_classes=["input-card"]
        ):

            gr.Markdown(
                "### 📷 Upload Leaf Image"
            )

            image_input = gr.Image(
                type="pil",
                label="Plant leaf image",
                height=350,
                elem_classes=["image-upload"]
            )

            gr.HTML("""
            <div class="tip-box">

                <strong>📸 Better image tips</strong>

                <br>

                • Use good lighting<br>
                • Keep the leaf clearly visible<br>
                • Avoid blurry photos<br>
                • Show the affected area closely

            </div>
            """)


        # ====================================================
        # SYMPTOMS
        # ====================================================

        with gr.Column(
            scale=1,
            elem_classes=["input-card"]
        ):

            gr.Markdown(
                "### 📝 Describe What You See"
            )

            symptoms_input = gr.Textbox(
                label="Plant symptoms",
                placeholder=(
                    "Example:\n\n"
                    "The leaves are turning yellow and curling. "
                    "There are small dark spots on the leaves "
                    "and the affected areas appear to be spreading."
                ),
                lines=10
            )

            gr.HTML("""
            <div class="tip-box">

                <strong>💡 What should you describe?</strong>

                <br>

                • Leaf color changes<br>
                • Spots or lesions<br>
                • Curling or wilting<br>
                • White or powdery coating<br>
                • Dry or brown areas<br>
                • Whether the problem is spreading

            </div>
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
    # RESULTS
    # ========================================================

    gr.HTML("""
    <div class="section-title">
        🎯 Your Plant Health Result
    </div>

    <div class="section-subtitle">
        Review the possible condition and the recommended
        next steps below.
    </div>
    """)


    diagnosis_output = gr.HTML(
        value="""
        <div class="empty-result">

            <div class="empty-icon">
                🌱
            </div>

            <h2>
                Your result will appear here
            </h2>

            <p>
                Upload a leaf image and click
                <strong>Analyze My Plant</strong>
                to begin.
            </p>

        </div>
        """,
        elem_classes=["result-card"]
    )


    # ========================================================
    # OTHER RESULTS
    # ========================================================

    other_results_output = gr.HTML(
        value="",
        elem_classes=["top-card"]
    )


    # ========================================================
    # ABOUT SYSTEM
    # ========================================================

    gr.HTML("""
    <div class="about-card">

        <h2>
            🌱 About this Plant Health Assistant
        </h2>

        <p>
            This application uses machine learning to assist
            with plant disease identification. It analyzes
            visual information from a leaf image and can also
            use the symptoms provided by the user.
        </p>

        <p>
            The goal is not only to provide a possible disease
            name, but also to help users understand the condition
            and take sensible next steps.
        </p>


        <div class="about-grid">

            <div class="about-item">

                <strong>
                    👁️ Image Analysis
                </strong>

                <span>
                    A MobileNetV2-based vision model analyzes
                    the submitted plant image.
                </span>

            </div>


            <div class="about-item">

                <strong>
                    📝 Symptom Analysis
                </strong>

                <span>
                    A TF-IDF and calibrated SVM pipeline
                    analyzes the described symptoms.
                </span>

            </div>


            <div class="about-item">

                <strong>
                    🧠 Combined Assessment
                </strong>

                <span>
                    When both inputs are provided, their
                    prediction information is combined to
                    produce the final result.
                </span>

            </div>

        </div>

    </div>
    """)


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML("""
    <div class="footer">

        <strong>
            🌿 Plant Disease Detection AI
        </strong>

        <br>

        AI-assisted plant health identification and guidance

        <br><br>

        16-class plant disease classification

        <br>

        Machine Learning Lab • CSE 0619 321L(1)

        <br><br>

        ⚠️ AI results are informational and should not
        replace professional agricultural diagnosis.

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
            gr.Markdown(visible=False)
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
            gr.Markdown(visible=False)
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    print("==============================================")
    print("STARTING PLANT DISEASE DETECTION APPLICATION")
    print("==============================================")
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
