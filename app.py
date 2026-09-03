import os

# ============================================================
# RENDER CPU / TENSORFLOW PERFORMANCE CONFIGURATION
# ============================================================

# Disable XLA/JIT compilation.
# Render's CPU was spending a very long time compiling
# the MobileNetV2 inference graph.
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

# Limit TensorFlow CPU threads on small Render instances.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import re
import numpy as np
import tensorflow as tf

# Explicitly disable TensorFlow JIT.
tf.config.optimizer.set_jit(False)

import joblib
import gradio as gr

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

vision_model = tf.keras.models.load_model(
    vision_path
)

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

print("Models loaded successfully.")
print("Number of classes:", len(classes))
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

    image = tf.convert_to_tensor(image)

    # Remove alpha channel if present
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

    print("Vision Agent: preprocessing image...")

    processed_image = preprocess_image(
        image
    )

    print("Vision Agent: running MobileNetV2...")

    prediction = vision_model.predict(
        processed_image,
        verbose=0
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

    print(
        "Vision Agent:",
        predicted_disease,
        f"({confidence * 100:.2f}%)"
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

    print("Text Agent: processing symptoms...")

    cleaned_text = clean_text(
        symptoms
    )

    # Handle empty text safely
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

    print(
        "Text Agent:",
        predicted_disease,
        f"({confidence * 100:.2f}%)"
    )

    return {
        "disease": predicted_disease,
        "confidence": confidence,
        "class_index": predicted_index,
        "probabilities": probabilities
    }


# ============================================================
# TOP-3 PREDICTIONS
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
#
# IMPORTANT:
# This function receives already-computed predictions.
# It DOES NOT run the models again.
# ============================================================

def fusion_predict(
    vision_result,
    text_result,
    vision_weight=0.5,
    text_weight=0.5
):

    print("Fusion Agent: combining predictions...")

    vision_probabilities = np.asarray(
        vision_result["probabilities"]
    )

    text_probabilities = np.asarray(
        text_result["probabilities"]
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

            decision = "Both Agents Agree"

        else:

            decision = (
                "Both Agents Agree - Low Confidence"
            )

    else:

        if (
            vision_result["confidence"]
            >=
            text_result["confidence"]
        ):

            decision = "Vision Agent Preferred"

        else:

            decision = "Text Agent Preferred"

    print(
        "Fusion Agent:",
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
# WEB PREDICTION FUNCTION
# ============================================================

def web_predict(
    image,
    symptoms
):

    print("")
    print("==============================================")
    print("STARTING PLANT ANALYSIS")
    print("==============================================")

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if image is None:

        print("No image supplied.")

        return (
            "⚠️ Please upload a leaf image.",
            "",
            "",
            "",
            "",
            "",
            ""
        )

    # --------------------------------------------------------
    # CLEAN SYMPTOMS
    # --------------------------------------------------------

    if symptoms is None:
        symptoms = ""

    symptoms = symptoms.strip()

    # --------------------------------------------------------
    # VISION PREDICTION
    #
    # RUN ONLY ONCE
    # --------------------------------------------------------

    print("")
    print("STEP 1: Vision Agent")

    vision_result = vision_predict(
        image
    )

    vision_disease = vision_result[
        "disease"
    ]

    vision_confidence = vision_result[
        "confidence"
    ]

    # --------------------------------------------------------
    # TEXT PREDICTION
    # --------------------------------------------------------

    if symptoms:

        print("")
        print("STEP 2: Text Agent")

        text_result = text_predict(
            symptoms
        )

        text_disease = text_result[
            "disease"
        ]

        text_confidence = text_result[
            "confidence"
        ]

        # ----------------------------------------------------
        # FUSION
        #
        # IMPORTANT:
        # Use existing results.
        # Do NOT call vision_predict() or text_predict()
        # again.
        # ----------------------------------------------------

        print("")
        print("STEP 3: Fusion Agent")

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

        decision = fusion_result[
            "decision"
        ]

        agents_agree = fusion_result[
            "agents_agree"
        ]

        verification_needed = fusion_result[
            "verification_needed"
        ]

        final_probabilities = fusion_result[
            "final_probabilities"
        ]

    # --------------------------------------------------------
    # IMAGE ONLY
    # --------------------------------------------------------

    else:

        print("")
        print("No symptoms provided.")
        print("Using Vision Agent only.")

        text_disease = "Not provided"
        text_confidence = 0.0

        final_disease = vision_disease
        final_confidence = vision_confidence

        decision = "Vision Agent Only"

        agents_agree = False

        verification_needed = False

        final_probabilities = vision_result[
            "probabilities"
        ]

    # --------------------------------------------------------
    # TOP 3 PREDICTIONS
    # --------------------------------------------------------

    print("")
    print("STEP 4: Generating Top-3 predictions")

    top_predictions = get_top_predictions(
        final_probabilities,
        top_k=3
    )

    top3_text = ""

    for rank, item in enumerate(
        top_predictions,
        1
    ):

        top3_text += (
            f"**{rank}. {item['disease']}**"
            f" — {item['confidence'] * 100:.2f}%\n\n"
        )

    # --------------------------------------------------------
    # AGREEMENT STATUS
    # --------------------------------------------------------

    if symptoms:

        if agents_agree:

            agreement_text = (
                "🟢 **Both Agents Agree**"
            )

        else:

            agreement_text = (
                "🟠 **Agents Disagree**"
            )

    else:

        agreement_text = (
            "🔵 **Vision Agent Only**"
        )

    # --------------------------------------------------------
    # VERIFICATION STATUS
    # --------------------------------------------------------

    if verification_needed:

        verification_text = (
            "⚠️ **Further Verification Recommended**"
        )

    else:

        verification_text = (
            "✅ **No Further Verification Required**"
        )

    # --------------------------------------------------------
    # RESULT CARD
    # --------------------------------------------------------

    result_card = f"""
# 🌿 {final_disease}

### Final Confidence
## {final_confidence * 100:.2f}%

{agreement_text}

{verification_text}
"""

    # --------------------------------------------------------
    # VISION CARD
    # --------------------------------------------------------

    vision_card = f"""
### 👁️ Vision Agent

**Prediction:** {vision_disease}

**Confidence:** {vision_confidence * 100:.2f}%
"""

    # --------------------------------------------------------
    # TEXT + FUSION CARDS
    # --------------------------------------------------------

    if symptoms:

        text_card = f"""
### 📝 Text Agent

**Prediction:** {text_disease}

**Confidence:** {text_confidence * 100:.2f}%
"""

        fusion_card = f"""
### 🔗 Fusion Decision

**Decision:** {decision}

**Final Prediction:** {final_disease}

**Final Confidence:** {final_confidence * 100:.2f}%

**Agents Agree:** {agents_agree}

**Verification Needed:** {verification_needed}
"""

    else:

        text_card = """
### 📝 Text Agent

No symptoms were provided.
"""

        fusion_card = """
### 🔗 Fusion Decision

Image-only prediction was used.
"""

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("")
    print("==============================================")
    print("ANALYSIS COMPLETE")
    print("==============================================")

    return (
        result_card,
        vision_card,
        text_card,
        fusion_card,
        top3_text,
        agreement_text,
        verification_text
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
        "",
        "",
        "",
        "",
        ""
    )


# ============================================================
# PROFESSIONAL WEBSITE CSS
# ============================================================

custom_css = """

/* ----------------------------------------------------------
   MAIN PAGE
---------------------------------------------------------- */

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    padding: 0 25px 35px 25px !important;
}


/* ----------------------------------------------------------
   HERO SECTION
---------------------------------------------------------- */

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


/* ----------------------------------------------------------
   SECTION HEADERS
---------------------------------------------------------- */

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin: 15px 0 12px 0;
}


/* ----------------------------------------------------------
   INPUT CARDS
---------------------------------------------------------- */

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


/* ----------------------------------------------------------
   IMAGE UPLOAD
---------------------------------------------------------- */

.image-upload {
    border-radius: 18px !important;
    overflow: hidden !important;
}


/* ----------------------------------------------------------
   BUTTONS
---------------------------------------------------------- */

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


/* ----------------------------------------------------------
   RESULT AREA
---------------------------------------------------------- */

.result-card {
    border-radius: 22px !important;

    padding: 28px !important;

    border: 1px solid #d1d5db !important;

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.07);

    margin-top: 12px;
}

.result-card h1 {
    font-size: 34px !important;
}


/* ----------------------------------------------------------
   AGENT CARDS
---------------------------------------------------------- */

.agent-card {
    border-radius: 18px !important;

    padding: 20px !important;

    border: 1px solid #d1d5db !important;

    box-shadow:
        0 5px 18px rgba(0, 0, 0, 0.05);
}


/* ----------------------------------------------------------
   TOP PREDICTIONS
---------------------------------------------------------- */

.top-card {
    border-radius: 18px !important;

    padding: 22px !important;

    border: 1px solid #d1d5db !important;
}


/* ----------------------------------------------------------
   INFO BOX
---------------------------------------------------------- */

.info-box {
    border-radius: 18px;

    padding: 22px;

    margin: 25px 0;

    border: 1px solid #d1d5db;
}


/* ----------------------------------------------------------
   FOOTER
---------------------------------------------------------- */

.footer {
    text-align: center;

    margin-top: 40px;

    padding: 25px 10px;

    font-size: 13px;

    opacity: 0.75;

    border-top: 1px solid #d1d5db;
}


/* ----------------------------------------------------------
   MOBILE
---------------------------------------------------------- */

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
# BUILD PROFESSIONAL INTERFACE
# ============================================================

with gr.Blocks(
    title="Plant Disease Detection System",
    css=custom_css,
    theme=gr.themes.Soft()
) as demo:

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    gr.HTML("""
    <div class="hero">

        <h1>🌿 Plant Disease Detection</h1>

        <p>
            AI-powered plant health analysis using
            Computer Vision and Symptom Analysis
        </p>

        <p class="subtitle">
            MobileNetV2&nbsp; • &nbsp;TF-IDF&nbsp; • &nbsp;SVM
            &nbsp; • &nbsp;Fusion Agent
        </p>

    </div>
    """)


    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    gr.HTML("""
    <div class="section-title">
        🔎 Analyze Your Plant
    </div>
    """)

    with gr.Row():

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SYMPTOMS
        # ----------------------------------------------------

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


    # --------------------------------------------------------
    # ACTION BUTTONS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    gr.HTML("""
    <div class="section-title">
        🎯 Diagnosis Result
    </div>
    """)

    result_output = gr.Markdown(
        visible=True,
        elem_classes=["result-card"]
    )


    # --------------------------------------------------------
    # AGENT RESULTS
    # --------------------------------------------------------

    with gr.Row():

        vision_output = gr.Markdown(
            visible=True,
            elem_classes=["agent-card"]
        )

        text_output = gr.Markdown(
            visible=True,
            elem_classes=["agent-card"]
        )


    fusion_output = gr.Markdown(
        visible=True,
        elem_classes=["agent-card"]
    )


    # --------------------------------------------------------
    # TOP 3
    # --------------------------------------------------------

    top3_output = gr.Markdown(
        visible=True,
        elem_classes=["top-card"]
    )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    with gr.Row():

        agreement_output = gr.Markdown(
            visible=True,
            elem_classes=["agent-card"]
        )

        verification_output = gr.Markdown(
            visible=True,
            elem_classes=["agent-card"]
        )


    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    gr.HTML("""
    <div class="info-box">

        <h3>🌱 About This System</h3>

        <p>
        This system combines image-based and symptom-based
        machine learning predictions to assist with
        plant disease identification.
        </p>

        <p>
        When the two agents disagree or confidence is low,
        further verification is recommended.
        </p>

    </div>
    """)


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.HTML("""
    <div class="footer">

        <b>🌿 Plant Disease Detection System</b>

        <br><br>

        MobileNetV2 • TF-IDF • SVM • Fusion Agent

        <br>

        16-class plant disease classification

        <br><br>

        Machine Learning Lab • CSE 0619 321L(1)

    </div>
    """)


    # --------------------------------------------------------
    # BUTTON EVENTS
    # --------------------------------------------------------

    predict_button.click(
        fn=web_predict,
        inputs=[
            image_input,
            symptoms_input
        ],
        outputs=[
            result_output,
            vision_output,
            text_output,
            fusion_output,
            top3_output,
            agreement_output,
            verification_output
        ]
    )


    clear_button.click(
        fn=clear_interface,
        inputs=[],
        outputs=[
            image_input,
            symptoms_input,
            result_output,
            vision_output,
            text_output,
            fusion_output,
            top3_output,
            agreement_output,
            verification_output
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
        server_port=RENDER_PORT
    )
