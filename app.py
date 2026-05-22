"""
Smart Image Reality Checker - Streamlit Web App
Upload an image and get instant predictions!
"""

import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras

st.set_page_config(
    page_title="Smart Image Reality Checker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS Styling
st.markdown("""
    <style>
        /* Main background and text colors */
        :root {
            --primary-color: #6366f1;
            --secondary-color: #ec4899;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --light-bg: #f8fafc;
            --dark-text: #1e293b;
            --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Overall styling */
        .main {
            background: linear-gradient(135deg, #f0f9ff 0%, #f0fdf4 50%, #fef3c7 100%);
            padding: 2rem;
        }
        
        /* Title styling */
        h1 {
            background: linear-gradient(135deg, #6366f1, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.5rem;
            font-size: 3rem !important;
            font-weight: 800 !important;
            letter-spacing: -1px;
        }
        
        /* Subtitle styling */
        .subtitle {
            text-align: center;
            color: #64748b;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 500;
        }
        
        /* Card styling */
        .info-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            border-left: 4px solid #6366f1;
            box-shadow: var(--card-shadow);
            margin: 1rem 0;
        }
        
        .success-card {
            background: #f0fdf4;
            border-left: 4px solid #10b981;
        }
        
        .warning-card {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
        }
        
        .error-card {
            background: #fef2f2;
            border-left: 4px solid #ef4444;
        }
        
        /* Section headers */
        .section-header {
            color: #1e293b;
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #6366f1, #ec4899);
            border-radius: 10px;
        }
        
        /* Metric styling */
        .metric-card {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: var(--card-shadow);
        }
        
        /* Divider styling */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
            margin: 2rem 0;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 12px rgba(99, 102, 241, 0.4) !important;
        }
        
        /* File uploader styling */
        .stFileUploader {
            border: 2px dashed #6366f1 !important;
            border-radius: 12px !important;
            padding: 2rem !important;
            background: linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%);
        }
        
        /* Text styling */
        .description-text {
            color: #64748b;
            line-height: 1.6;
            font-size: 1rem;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.9rem;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 2px solid #e2e8f0;
        }
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = 'outputs/best_model.keras'
CLASS_INDICES_PATH = 'outputs/class_indices.json'
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        
        class_names = {v: k for k, v in class_indices.items()}
        
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you've trained the model first using 'python train.py'")
        return None, None

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_image(model, image, class_names):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index] * 100)
    
    all_probabilities = {
        class_names[i]: float(predictions[0][i] * 100)
        for i in range(len(class_names))
    }
    
    return predicted_class, confidence, all_probabilities

def main():
    st.markdown("<h1>🤖 Smart Image Reality Checker</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Instantly detect AI-generated images vs real photographs using deep learning</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    with st.spinner("⚙️ Loading model..."):
        model, class_names = load_model()
    
    if model is None:
        st.stop()
    
    st.markdown(
        "<div class='info-card success-card'><b>✅ Model loaded successfully!</b> Ready to analyze images.</div>",
        unsafe_allow_html=True
    )
    
    col_space1, col_main, col_space2 = st.columns([0.5, 2, 0.5])
    
    with col_main:
        st.markdown(
            "<div class='section-header'>📤 Upload Your Image</div>",
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png'],
            help="📸 Upload a clear image (JPG, JPEG, or PNG) for best results"
        )
    
    if uploaded_file is not None:
        st.markdown("---")
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            st.markdown(
                "<div class='section-header'>🖼️ Original Image</div>",
                unsafe_allow_html=True
            )
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, clamp=True)
        
        with st.spinner("🔍 Analyzing image... This may take a moment..."):
            predicted_class, confidence, all_probabilities = predict_image(
                model, image, class_names
            )
        
        with col2:
            st.markdown(
                "<div class='section-header'>🎯 Prediction Results</div>",
                unsafe_allow_html=True
            )
            
            if predicted_class.lower() == 'ai':
                st.markdown(
                    f"""
                    <div class='info-card error-card' style='text-align: center; padding: 2rem;'>
                        <h3 style='color: #ef4444; margin: 0;'>🤖 AI-Generated</h3>
                        <p style='color: #991b1b; margin-top: 0.5rem;'>This image appears to be AI-generated</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='info-card success-card' style='text-align: center; padding: 2rem;'>
                        <h3 style='color: #10b981; margin: 0;'>📷 Real Photograph</h3>
                        <p style='color: #065f46; margin-top: 0.5rem;'>This image appears to be a real photograph</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("")
            
            col_metric1, col_metric2 = st.columns([1, 1.5])
            with col_metric1:
                st.metric(
                    label="📊 Confidence",
                    value=f"{confidence:.1f}%",
                    label_visibility="visible"
                )
            
            st.markdown("")
            st.progress(confidence / 100)
            
            st.markdown("")
            st.markdown(
                "<div class='section-header'>💡 Confidence Level</div>",
                unsafe_allow_html=True
            )
            if confidence >= 90:
                st.markdown(
                    "<div class='info-card success-card'><b>🎯 Very High Confidence</b> - The model is very certain about this prediction.</div>",
                    unsafe_allow_html=True
                )
            elif confidence >= 70:
                st.markdown(
                    "<div class='info-card success-card'><b>✅ High Confidence</b> - The model is fairly certain about this prediction.</div>",
                    unsafe_allow_html=True
                )
            elif confidence >= 50:
                st.markdown(
                    "<div class='info-card warning-card'><b>⚠️ Moderate Confidence</b> - The model has some uncertainty.</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='info-card warning-card'><b>⚠️ Low Confidence</b> - The model is uncertain. Results may be unreliable.</div>",
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
        st.markdown(
            "<div class='section-header'>📊 Detailed Probability Analysis</div>",
            unsafe_allow_html=True
        )
        
        prob_col1, prob_col2 = st.columns([1.5, 3])
        
        with prob_col1:
            st.markdown("**Classification Breakdown**")
            for class_name, probability in sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{class_name.upper()}**")
                st.progress(probability / 100)
                st.markdown(f"<p style='color: #64748b; margin: -10px 0 10px 0; font-size: 0.9rem;'>{probability:.2f}%</p>", unsafe_allow_html=True)
        
        with prob_col2:
            st.markdown("**Score Details**")
            st.markdown("<div style='padding: 1rem; background: #f1f5f9; border-radius: 8px;'>", unsafe_allow_html=True)
            for class_name, probability in sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True):
                bar_color = "#10b981" if class_name.lower() == "real" else "#ef4444"
                st.markdown(
                    f"""
                    <div style='margin-bottom: 1rem;'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                            <span style='font-weight: 600; color: #1e293b;'>{class_name.capitalize()}</span>
                            <span style='color: {bar_color}; font-weight: 700;'>{probability:.2f}%</span>
                        </div>
                        <div style='height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden;'>
                            <div style='height: 100%; width: {probability}%; background: {bar_color}; border-radius: 4px;'></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        st.markdown("")
        st.markdown(
            "<div class='info-card' style='text-align: center; padding: 2rem;'>"
            "<p style='font-size: 1.1rem; color: #64748b;'>👆 <b>Upload an image to get started!</b></p>"
            "<p style='color: #94a3b8;'>Drag and drop or click to browse your files</p>"
            "</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("")
        st.markdown("---")
        
        col_left, col_right = st.columns(2, gap="large")
        
        with col_left:
            st.markdown(
                "<div class='section-header'>📝 How to Use</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <div class='info-card' style='border-left-color: #6366f1;'>
                <ol style='line-height: 2; color: #475569; margin: 0;'>
                <li><b>Upload</b> an image using the file uploader above</li>
                <li><b>Wait</b> for the model to analyze (usually < 1 second)</li>
                <li><b>View</b> the prediction and confidence score</li>
                <li><b>Review</b> detailed probability breakdown</li>
                </ol>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_right:
            st.markdown(
                "<div class='section-header'>ℹ️ About This Project</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <div class='info-card' style='border-left-color: #ec4899;'>
                <p style='color: #475569; margin: 0; line-height: 1.6;'>
                This <b>Smart Image Reality Checker</b> uses a deep learning model trained with 
                transfer learning (MobileNetV2) to classify images as either AI-generated 
                or real photographs with high accuracy.
                </p>
                <p style='color: #94a3b8; margin-top: 1rem; margin-bottom: 0;'><b>Features:</b></p>
                <ul style='color: #475569; margin-top: 0.5rem; padding-left: 1.25rem;'>
                <li>⚡ Fast inference (< 1 second)</li>
                <li>🎯 High accuracy with confidence scores</li>
                <li>🎨 Easy-to-use web interface</li>
                <li>📊 Detailed probability breakdown</li>
                <li>🔄 Real-time predictions</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        st.markdown(
            "<div class='section-header'>🚀 Getting Started Tips</div>",
            unsafe_allow_html=True
        )
        
        tips_col1, tips_col2, tips_col3 = st.columns(3, gap="small")
        
        with tips_col1:
            st.markdown(
                """
                <div class='info-card' style='text-align: center; border-left-color: #10b981;'>
                <h4 style='color: #10b981; margin-top: 0;'>📸 Best Format</h4>
                <p style='color: #64748b; font-size: 0.95rem;'>JPG, JPEG, or PNG format works best for optimal results.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with tips_col2:
            st.markdown(
                """
                <div class='info-card' style='text-align: center; border-left-color: #6366f1;'>
                <h4 style='color: #6366f1; margin-top: 0;'>🎯 Clear Images</h4>
                <p style='color: #64748b; font-size: 0.95rem;'>Use clear, well-lit images for more accurate predictions.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with tips_col3:
            st.markdown(
                """
                <div class='info-card' style='text-align: center; border-left-color: #ec4899;'>
                <h4 style='color: #ec4899; margin-top: 0;'>📊 Confidence Level</h4>
                <p style='color: #64748b; font-size: 0.95rem;'>Higher confidence (>70%) indicates more reliable predictions.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    st.markdown(
        """
        <div class='footer'>
        <p style='margin-bottom: 0.5rem;'>💜 Built with <b>TensorFlow</b> & <b>Streamlit</b></p>
        <p style='font-size: 0.85rem; color: #cbd5e1; margin: 0;'>Smart Image Reality Checker • Detecting reality, one image at a time</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()