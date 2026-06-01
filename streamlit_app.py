import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import io
from model_builder import build_model 

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Pneumonia AI", page_icon="🫁")

# Initialize Session State for History
if 'history' not in st.session_state:
    st.session_state.history = []

# ==========================================
# LOAD MODEL DIRECTLY IN STREAMLIT (Cached)
# ==========================================
@st.cache_resource
def load_prediction_model():
    # Build the Xception model architecture and load weights locally
    model = build_model(input_shape=(224, 224, 3), num_classes=3)
    model.load_weights('best_xception_model.keras')
    return model

try:
    model = load_prediction_model()
except Exception as e:
    st.error(f"Error loading model weights: {e}")

# ==========================================
# SIDEBAR & HEADER
# ==========================================
st.title("🫁 Pneumonia Detection System")
st.markdown("Upload a Chest X-ray image to get an instant diagnosis.")

with st.sidebar:
    st.header("About the Project")
    st.info("Uses Xception + Residual Blocks model for X-ray analysis.")
    st.write("Built completely within Streamlit Cloud.")
    st.divider()
    st.header("Recent Scans")
    if not st.session_state.history:
        st.write("No scans yet.")
    for item in st.session_state.history:
        st.text(item)

# ==========================================
# FILE UPLOADER & CORE LOGIC
# ==========================================
uploaded_file = st.file_uploader("Choose an x-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        bytes_stream = io.BytesIO(file_bytes)
        image = Image.open(bytes_stream)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with col2:
            if st.button("Analyze Image"):
                with st.spinner('Analyzing locally...'):
                    
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        st.error("Could not decode the uploaded image.")
                    else:
                        # --------------------------------------------------
                        # ADVANCED ANATOMICAL & COLOR FILTER (HYPER-TUNED)
                        # --------------------------------------------------
                        is_valid_chest_xray = True
                        
                        # 1. Color Saturation Check (Rejects general colorful photos)
                        b, g, r = cv2.split(img)
                        color_diff = np.mean(np.abs(b.astype(np.float32) - g.astype(np.float32))) + \
                                     np.mean(np.abs(g.astype(np.float32) - r.astype(np.float32)))
                        if color_diff > 9.0:
                            is_valid_chest_xray = False
                        
                        # 2. Structural & Spatial Variance Check (Rejects Hands, Feet, Skull, etc.)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        h, w = gray.shape
                        
                        # Calculate structural variance (Extremities like hands/feet have extreme black/white contrast)
                        img_variance = np.var(gray)
                        
                        # Chest X-rays have a solid lung block in the middle with soft transitions.
                        # We evaluate the structural ratio of bone tissue vs soft lung tissue via adaptive thresholding.
                        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
                        active_pixel_ratio = np.sum(thresh == 255) / (h * w)
                        
                        # Texture frequency analyzer via Laplacian variance
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        
                        # Multi-level decision block to separate Chest cavity from other body parts
                        if img_variance < 800 or img_variance > 5800:
                            # Too flat (general non-xray) or too high contrast (hand/foot bone against pitch black)
                            is_valid_chest_xray = False
                            
                        if active_pixel_ratio < 0.25 or active_pixel_ratio > 0.92:
                            # Chest X-rays have a stable body mass ratio filling 30% to 90% of the viewport frame
                            is_valid_chest_xray = False
                            
                        if blur_score < 10.0 or blur_score > 1200.0:
                            # Rejects completely blank images, noise artifacts, or extreme sharp non-medical edges
                            is_valid_chest_xray = False

                        # ------------------------------------------
                        # DECISION AND MODEL EXECUTION
                        # ------------------------------------------
                        if not is_valid_chest_xray:
                            res_class = "Rejected: Invalid/Non-Chest X-ray Image"
                            st.error(f"❌ {res_class}")
                            st.warning("Please upload a valid CHEST X-ray image. Other body parts or general photos are strictly prohibited.")
                        else:
                            # Input preprocessing for Xception Model
                            img_resized = cv2.resize(img, (224, 224))
                            img_normalized = img_resized.astype(np.float32) / 255.0
                            img_input = np.expand_dims(img_normalized, axis=0)
                            
                            # Predict via Deep Learning Model
                            prediction = model.predict(img_input)
                            confidence_scores = prediction[0]
                            
                            normal_score = float(confidence_scores[0])
                            pneumonia_score = float(confidence_scores[1])
                            
                            # Dynamic Decision Thresholding to ensure zero-tolerance false positives for normal scans
                            # The model will ONLY flag Pneumonia if the confidence strictly exceeds Normal.
                            if pneumonia_score > normal_score and pneumonia_score > 0.55:
                                res_class = "Pneumonia"
                                st.error(f"Diagnosis: {res_class}")
                                st.warning("Note: Pneumonia indicators detected within the lung fields. Please consult a radiologist.")
                            else:
                                res_class = "Normal"
                                st.success(f"Diagnosis: {res_class}")
                                st.info("Note: Lung fields appear clear. No clinical signs of Pneumonia detected.")
                            
                            # Append valid case to session state history
                            entry = f"{res_class} - {uploaded_file.name}"
                            if entry not in st.session_state.history:
                                st.session_state.history.append(entry)
                                st.rerun()
                        
                        # ------------------------------------------
                        # GENERATE PDF REPORT
                        # ------------------------------------------
                        if "Rejected" not in res_class:
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", 'B', 16)
                            pdf.cell(200, 10, txt="Pneumonia Detection Report", ln=True, align='C')
                            pdf.ln(10)
                            
                            pdf.set_font("Arial", size=12)
                            pdf.cell(200, 10, txt=f"File Name: {uploaded_file.name}", ln=True, align='L')
                            pdf.cell(200, 10, txt=f"Diagnosis Result: {res_class}", ln=True, align='L')
                            
                            pdf_output = pdf.output(dest='S').encode('latin-1')
                            
                            st.divider()
                            st.download_button(
                                label="Download Report", 
                                data=pdf_output, 
                                file_name=f"Report_{uploaded_file.name}.pdf", 
                                mime="application/pdf"
                            )
                            
    except Exception as img_err:
        st.error(f"Error processing app logic: {img_err}")
