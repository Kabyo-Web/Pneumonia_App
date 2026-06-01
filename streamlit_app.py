import streamlit as st
import requests
from PIL import Image
from fpdf import FPDF
import io  # Required to handle bytes buffer safely in cloud environment

# ==========================================
# CONFIGURATION & BACKEND URL
# ==========================================
# Use this URL for local testing.
# When deploying the backend to a cloud hosting platform (e.g., Render, Railway),
# change this URL to your live cloud deployment API link.
BACKEND_URL = "https://pneumonia-app-cjae.onrender.com/predict" 

# Page configuration
st.set_page_config(page_title="Pneumonia AI", page_icon="🫁")

# Initialize Session State for History
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.title("🫁 Pneumonia Detection System")
st.markdown("Upload a Chest X-ray image to get an instant diagnosis.")

# Sidebar for information and History
with st.sidebar:
    st.header("About the Project")
    st.info("Uses Xception + Residual Blocks model for X-ray analysis.")
    st.write("Built with FastAPI and Streamlit.")
    st.divider()
    st.header("Recent Scans")
    if not st.session_state.history:
        st.write("No scans yet.")
    for item in st.session_state.history:
        st.text(item)

# File Uploader
uploaded_file = st.file_uploader("Choose an x-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # 1. Safely extract bytes and create an isolated stream clone to prevent pointer loss
        file_bytes = uploaded_file.getvalue()
        bytes_stream = io.BytesIO(file_bytes)
        
        # 2. Open the image from the isolated bytes stream clone
        image = Image.open(bytes_stream)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            # Fixed: Changed 'use_container_width' to 'use_column_width' for older Streamlit versions
            st.image(image, caption='Uploaded X-ray', use_column_width=True)
        
        with col2:
            if st.button("Analyze Image"):
                with st.spinner('Analyzing...'):
                    # 3. Structure file data correctly using the locked byte array
                    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                    
                    try:
                        # Send HTTP POST request to the backend API
                        response = requests.post(BACKEND_URL, files=files)
                        
                        if response.status_code == 200:
                            data = response.json()
                            res_class = data.get("result")
                            
                            if res_class:
                                # ------------------------------------------
                                # DISPLAY RESULT
                                # ------------------------------------------
                                # 1. If the image is rejected by validation filters
                                if res_class.startswith("Rejected") or "Invalid" in res_class:
                                    st.error(f"❌ {res_class}")
                                    st.warning("Please upload a valid chest X-ray image to get a diagnosis.")
                                    
                                # 2. If Pneumonia is detected by the AI model
                                elif res_class == "Pneumonia":
                                    st.error(f"Diagnosis: {res_class}")
                                    st.warning("Note: Pneumonia detected. Please consult a doctor immediately.")
                                    
                                # 3. If the X-ray is diagnosed as Normal
                                elif res_class == "Normal":
                                    st.success(f"Diagnosis: {res_class}")
                                    st.info("Note: No signs of Pneumonia detected. Stay healthy!")
                                    
                                else:
                                    st.warning(f"Result: {res_class}")
                                
                                # ------------------------------------------
                                # ADD TO HISTORY (Skip entry if image was rejected)
                                # ------------------------------------------
                                if not (res_class.startswith("Rejected") or "Invalid" in res_class):
                                    entry = f"{res_class} - {uploaded_file.name}"
                                    if entry not in st.session_state.history:
                                        st.session_state.history.append(entry)
                                        # Instantly refresh Streamlit interface to update history sidebar
                                        st.rerun()
                                
                                # ------------------------------------------
                                # GENERATE PDF REPORT
                                # ------------------------------------------
                                pdf = FPDF()
                                pdf.add_page()
                                pdf.set_font("Arial", 'B', 16)
                                pdf.cell(200, 10, txt="Pneumonia Detection Report", ln=True, align='C')
                                pdf.ln(10) # Line spacing
                                
                                pdf.set_font("Arial", size=12)
                                pdf.cell(200, 10, txt=f"File Name: {uploaded_file.name}", ln=True, align='L')
                                pdf.cell(200, 10, txt=f"Diagnosis Result: {res_class}", ln=True, align='L')
                                
                                # Stream PDF output directly into byte format
                                pdf_output = pdf.output(dest='S').encode('latin-1')
                                
                                st.divider()
                                st.download_button(
                                    label="Download Report", 
                                    data=pdf_output, 
                                    file_name=f"Report_{uploaded_file.name}.pdf", 
                                    mime="application/pdf"
                                )
                            else:
                                st.error("Unexpected response payload structure from server.")
                        else:
                            st.error(f"Backend Error: Status code {response.status_code}. Make sure your backend server is up.")
                            
                    except Exception as e:
                        st.error(f"Connection error: Could not reach the backend API. {e}")
                        
    except Exception as img_err:
        st.error(f"Error processing the uploaded image file: {img_err}")