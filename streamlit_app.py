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
                        # STRICT FILTER: CHEST X-RAY VALIDATION LOGIC
                        # --------------------------------------------------
                        is_valid_chest_xray = True
                        
                        # 1. Color Saturation Check (Filters out regular colorful images)
                        b, g, r = cv2.split(img)
                        color_diff = np.mean(np.abs(b.astype(np.float32) - g.astype(np.float32))) + \
                                     np.mean(np.abs(g.astype(np.float32) - r.astype(np.float32)))
                        if color_diff > 12.0:  # Relaxed slightly to prevent monochrome compression noise rejection
                            is_valid_chest_xray = False
                        
                        # 2. Image Edge and Texture Filter
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray, 30, 150)
                        edge_density = np.sum(edges) / (img.shape[0] * img.shape[1])
                        if edge_density > 6.0:  # Increased threshold to allow fine rib structures in real X-rays
                            is_valid_chest_xray = False
                            
                        # 3. Optimized Anatomical Region Analysis
                        h, w = gray.shape
                        
                        # Measure mean brightness in the central lung/heart region
                        center_roi = gray[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
                        center_mean = np.mean(center_roi)
                        
                        # Calculate bone-to-background pixel occupancy via binary thresholding
                        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
                        bone_pixels = np.sum(thresh == 255) / (h * w)
                        
                        # Balanced heuristic checks tailored for chest dynamic ranges
                        if center_mean < 25 or center_mean > 230: 
                            is_valid_chest_xray = False
                        if bone_pixels < 0.15 or bone_pixels > 0.95: 
                            is_valid_chest_xray = False

                        # ------------------------------------------
                        # DECISION AND MODEL RUN
                        # ------------------------------------------
                        if not is_valid_chest_xray:
                            res_class = "Rejected: Invalid/Non-Chest X-ray Image"
                            st.error(f"❌ {res_class}")
                            st.warning("Please upload a valid CHEST X-ray image to get a diagnosis. Other body parts or general photos are not allowed.")
                        else:
                            # Model execution triggers only if the image passes balanced validation checks
                            img_resized = cv2.resize(img, (224, 224))
                            img_normalized = img_resized.astype(np.float32) / 255.0
                            img_input = np.expand_dims(img_normalized, axis=0)
                            
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
