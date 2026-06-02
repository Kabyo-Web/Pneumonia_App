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
st.set_page_config(page_title="Pneumonia Diagnosis Pro", page_icon="🩺")

# Load Model
@st.cache_resource
def load_prediction_model():
    model = build_model(input_shape=(224, 224, 3), num_classes=3)
    model.load_weights('best_xception_model.keras')
    return model

model = load_prediction_model()

st.title("🩺 Clinical Pneumonia Screening System")
st.markdown("Professional-grade diagnostic tool for Chest X-ray analysis.")

# ==========================================
# CORE LOGIC
# ==========================================
uploaded_file = st.file_uploader("Upload Chest X-ray...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(file_bytes))
    st.image(image, caption='Uploaded Scan', use_column_width=True)
    
    if st.button("Run Diagnostic Analysis"):
        with st.spinner('Validating and Analyzing...'):
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # --- 1. SMART GATEKEEPER (Validation Logic) ---
            is_valid = True
            
            # Check for non-X-ray colorful images
            b, g, r = cv2.split(img)
            color_std = np.std(b) + np.std(g) + np.std(r)
            if color_std > 25: # High color variation = General image
                is_valid = False
            
            # Check for anatomical structure (Hands/Feet/Others have extreme contrast)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray)
            if variance < 600 or variance > 7000: # X-rays fall within specific variance range
                is_valid = False

            # --- 2. DIAGNOSIS (If Valid) ---
            if not is_valid:
                st.error("❌ REJECTED: Invalid/Non-Chest X-ray Image detected.")
                st.info("System only accepts standard Chest X-ray scans. Please upload a valid thoracic scan.")
            else:
                # Prediction
                img_resized = cv2.resize(img, (224, 224))
                img_normalized = img_resized.astype(np.float32) / 255.0
                img_input = np.expand_dims(img_normalized, axis=0)
                
                prediction = model.predict(img_input)
                # Assuming index 0=Normal, 1=Pneumonia
                if prediction[0][1] > prediction[0][0]:
                    res = "Pneumonia Detected"
                    st.error(f"## Result: {res}")
                    st.warning("Recommendation: Immediate clinical correlation required.")
                else:
                    res = "Normal"
                    st.success(f"## Result: {res}")
                    st.info("Status: No pulmonary signs of Pneumonia found.")

                # Generate Report
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, txt="Clinical Diagnosis Report", ln=True, align='C')
                pdf.ln(10)
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Analysis Result: {res}", ln=True)
                
                st.download_button("Download Official Report", pdf.output(dest='S').encode('latin-1'), 
                                   f"Report_{uploaded_file.name}.pdf", "application/pdf")
