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
                        # BALANCED VALIDATION LOGIC
                        # --------------------------------------------------
                        is_valid_image = True
                        
                        # Color Saturation Check (Detects and rejects colorful general photos instantly)
                        b, g, r = cv2.split(img)
                        color_diff = np.mean(np.abs(b.astype(np.float32) - g.astype(np.float32))) + \
                                     np.mean(np.abs(g.astype(np.float32) - r.astype(np.float32)))
                        
                        # If the image has high color variation, it's definitely not a medical X-ray
                        if color_diff > 15.0:
                            is_valid_image = False

                        # ------------------------------------------
                        # DECISION AND MODEL RUN
                        # ------------------------------------------
                        if not is_valid_image:
                            res_class = "Rejected: Invalid/Non-X-ray Image"
                            st.error(f"❌ {res_class}")
                            st.warning("Please upload a valid chest X-ray image. General colorful photos are not allowed.")
                        else:
                            # Preprocessing for the deep learning model
                            img_resized = cv2.resize(img, (224, 224))
                            img_normalized = img_resized.astype(np.float32) / 255.0
                            img_input = np.expand_dims(img_normalized, axis=0)
                            
                            # Model prediction execution
                            prediction = model.predict(img_input)
                            confidence_scores = prediction[0]
                            
                            normal_score = float(confidence_scores[0])
                            pneumonia_score = float(confidence_scores[1])
                            
                            if pneumonia_score > normal_score:
                                res_class = "Pneumonia"
                                st.error(f"Diagnosis: {res_class}")
                                st.warning("Note: Pneumonia detected. Please consult a doctor immediately.")
                            else:
                                res_class = "Normal"
                                st.success(f"Diagnosis: {res_class}")
                                st.info("Note: No signs of Pneumonia detected. Stay healthy!")
                            
                            # Update recent scans session history
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
