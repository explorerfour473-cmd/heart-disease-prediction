import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. โหลดไฟล์สมองกลที่เราเทรนไว้
@st.cache_resource
def load_assets():
    model_columns = joblib.load('model_columns.pkl')
    scaler = joblib.load('scaler.pkl')
    ensemble_model = joblib.load('ensemble_model.pkl')
    nn_model = load_model('nn_model.keras')
    return model_columns, scaler, ensemble_model, nn_model

model_columns, scaler, ensemble_model, nn_model = load_assets()

# 2. ตั้งค่าเมนูด้านข้าง (Sidebar)
st.sidebar.title("เมนูหลัก")
page = st.sidebar.radio("เลือกหน้าต่างทำงาน:", 
    ["1. ทฤษฎี Machine Learning", 
     "2. ทฤษฎี Neural Network", 
     "3. ทดสอบ Machine Learning", 
     "4. ทดสอบ Neural Network"]
)

# --- หน้าที่ 1: อธิบายทฤษฎี ML ---
if page == "1. ทฤษฎี Machine Learning":
    st.title("📖 ทฤษฎี Machine Learning แบบ Ensemble")
    st.write("โปรเจคนี้ใช้ข้อมูลโรคหัวใจจาก UCI (ถูกแบ่งเป็น 2 ชุดคือ ประวัติผู้ป่วย และ ผลตรวจสุขภาพ นำมา Join กัน)")
    st.write("เราใช้เทคนิค **Ensemble Learning (Voting Classifier)** โดยนำโมเดล 3 ตัวมาช่วยกันโหวตตัดสินใจ ได้แก่:")
    st.markdown("""
    1. **Logistic Regression:** วิเคราะห์ความสัมพันธ์เชิงเส้น
    2. **Decision Tree:** สร้างต้นไม้ตัดสินใจแบบมีเงื่อนไข
    3. **K-Nearest Neighbors (KNN):** วิเคราะห์จากเพื่อนบ้านที่ใกล้ที่สุด
    """)
    st.write("**แหล่งอ้างอิง:** ข้อมูลจาก Kaggle (Heart Disease UCI), ทฤษฎีจาก Scikit-Learn Documentation")

# --- หน้าที่ 2: อธิบายทฤษฎี NN ---
elif page == "2. ทฤษฎี Neural Network":
    st.title("🧠 ทฤษฎี Neural Network")
    st.write("โครงข่ายประสาทเทียม (Multi-Layer Perceptron) ที่ออกแบบไว้ มีโครงสร้างดังนี้:")
    st.markdown("""
    - **Input Layer:** รับข้อมูลที่ผ่านการทำความสะอาดและ Scaling แล้ว
    - **Hidden Layer 1:** มี 16 โหนด (Activation: ReLU) เพื่อสกัดคุณลักษณะแฝง
    - **Hidden Layer 2:** มี 8 โหนด (Activation: ReLU)
    - **Output Layer:** มี 1 โหนด (Activation: Sigmoid) ให้ผลลัพธ์เป็นความน่าจะเป็น (0 ถึง 1)
    """)
    st.write("**แหล่งอ้างอิง:** ทฤษฎี Deep Learning พื้นฐาน, ไลบรารี TensorFlow/Keras")

# --- ฟังก์ชันฟอร์มกรอกข้อมูล (ใช้ร่วมกันหน้าที่ 3 และ 4) ---
def get_user_input():
    st.write("### กรอกข้อมูลคนไข้เพื่อประเมินความเสี่ยง")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("อายุ (Age)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("เพศ (Sex)", ["Male", "Female"])
        cp = st.selectbox("อาการเจ็บหน้าอก (Chest Pain)", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        trestbps = st.number_input("ความดันโลหิต (Resting Blood Pressure)", min_value=50, max_value=250, value=120)
    with col2:
        chol = st.number_input("คอเลสเตอรอล (Cholesterol)", min_value=100, max_value=600, value=200)
        thalch = st.number_input("อัตราการเต้นหัวใจสูงสุด (Max Heart Rate)", min_value=60, max_value=220, value=150)
        fbs = st.selectbox("น้ำตาลในเลือด > 120 (Fasting Blood Sugar)", ["TRUE", "FALSE"])
        exang = st.selectbox("เจ็บหน้าอกเวลาออกกำลังกาย (Exercise Induced Angina)", ["TRUE", "FALSE"])
    
    # สร้างเป็น DataFrame
    user_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 
        'chol': chol, 'thalch': thalch, 'fbs': fbs, 'exang': exang,
        # ค่าอื่นๆ ที่ไม่ได้กรอก ให้ใส่ค่าเริ่มต้นกลางๆ เพื่อไม่ให้ Error
        'restecg': 'normal', 'oldpeak': 0.0, 'slope': 'flat', 'ca': 0, 'thal': 'normal'
    }
    input_df = pd.DataFrame([user_data])
    
    # แปลงตัวหนังสือเป็นตัวเลข (ให้เหมือนตอนเทรน)
    input_df = pd.get_dummies(input_df)
    # จัดคอลัมน์ให้ตรงกับพิมพ์เขียว
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    # ปรับสเกลตัวเลข
    scaled_input = scaler.transform(input_df)
    return scaled_input

# --- หน้าที่ 3: ทดสอบ ML ---
elif page == "3. ทดสอบ Machine Learning":
    st.title("⚙️ ทดสอบโมเดล Machine Learning")
    scaled_data = get_user_input()
    
    if st.button("🔍 ทำนายผลด้วย Machine Learning"):
        prediction = ensemble_model.predict(scaled_data)
        if prediction[0] == 1:
            st.error("🚨 ผลการทำนาย: มีความเสี่ยงเป็นโรคหัวใจ!")
        else:
            st.success("✅ ผลการทำนาย: ปกติ (ไม่มีความเสี่ยง)")

# --- หน้าที่ 4: ทดสอบ NN ---
elif page == "4. ทดสอบ Neural Network":
    st.title("⚙️ ทดสอบโมเดล Neural Network")
    scaled_data = get_user_input()
    
    if st.button("🔍 ทำนายผลด้วย Neural Network"):
        prediction_prob = nn_model.predict(scaled_data)[0][0]
        if prediction_prob > 0.5:
            st.error(f"🚨 ผลการทำนาย: มีความเสี่ยงเป็นโรคหัวใจ! (ความมั่นใจ {prediction_prob*100:.2f}%)")
        else:
            st.success(f"✅ ผลการทำนาย: ปกติ (ความเสี่ยงเพียง {prediction_prob*100:.2f}%)")