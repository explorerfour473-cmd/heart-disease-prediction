import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go  # 🎨 เพิ่มไลบรารีสำหรับทำกราฟ

# 1. โหลดไฟล์สมองกลที่เราเทรนไว้
@st.cache_resource
def load_assets():
    model_columns = joblib.load('model_columns.pkl')
    scaler = joblib.load('scaler.pkl')
    ensemble_model = joblib.load('ensemble_model.pkl')
    nn_model = load_model('nn_model.keras')
    return model_columns, scaler, ensemble_model, nn_model

model_columns, scaler, ensemble_model, nn_model = load_assets()

# --- ฟังก์ชันฟอร์มกรอกข้อมูล ---
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
    
    user_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 
        'chol': chol, 'thalch': thalch, 'fbs': fbs, 'exang': exang,
        'restecg': 'normal', 'oldpeak': 0.0, 'slope': 'flat', 'ca': 0, 'thal': 'normal'
    }
    input_df = pd.DataFrame([user_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    scaled_input = scaler.transform(input_df)
    return scaled_input

# 2. ตั้งค่าเมนูด้านข้าง (Sidebar)
st.sidebar.title("เมนูหลัก")
page = st.sidebar.radio("เลือกหน้าต่างทำงาน:", 
    ["1. ทฤษฎี Machine Learning", 
     "2. ทฤษฎี Neural Network", 
     "3. ทดสอบ Machine Learning", 
     "4. ทดสอบ Neural Network"]
)

if page == "1. ทฤษฎี Machine Learning":
    st.title("📖 ทฤษฎี Machine Learning แบบ Ensemble")
    st.write("โปรเจคนี้ใช้ข้อมูลโรคหัวใจจาก UCI นำมาประมวลผลด้วยเทคนิค **Ensemble Learning (Voting Classifier)** โดยนำโมเดล 3 ตัวมาช่วยกันโหวตตัดสินใจ ได้แก่ Logistic Regression, Decision Tree และ KNN")

elif page == "2. ทฤษฎี Neural Network":
    st.title("🧠 ทฤษฎี Neural Network")
    st.write("โครงข่ายประสาทเทียม (Multi-Layer Perceptron) ที่ออกแบบไว้ มีโครงสร้าง 3 ชั้น (16 -> 8 -> 1 โหนด) เพื่อสกัดคุณลักษณะแฝงและให้ผลลัพธ์เป็นความน่าจะเป็น (0-100%)")

elif page == "3. ทดสอบ Machine Learning":
    st.title("⚙️ ทดสอบโมเดล Machine Learning")
    scaled_data = get_user_input()
    
    if st.button("🔍 ทำนายผลด้วย Machine Learning"):
        prediction = ensemble_model.predict(scaled_data)
        if prediction[0] == 1:
            st.error("🚨 ผลการทำนาย: มีความเสี่ยงเป็นโรคหัวใจ!")
        else:
            st.success("✅ ผลการทำนาย: ปกติ (ไม่มีความเสี่ยง)")
            
        st.info("💡 ข้อสังเกต: โมเดล Machine Learning แบบ Hard Voting จะฟันธงแค่ 'เป็น' หรือ 'ไม่เป็น' เท่านั้น (ไม่มีเปอร์เซ็นต์ความน่าจะเป็น)")

# --- หน้าที่ 4: ทดสอบ NN (เพิ่มกราฟหน้าปัดเข็มไมล์) ---
elif page == "4. ทดสอบ Neural Network":
    st.title("⚙️ ทดสอบโมเดล Neural Network")
    scaled_data = get_user_input()
    
    if st.button("🔍 ทำนายผลด้วย Neural Network"):
        prediction_prob = nn_model.predict(scaled_data)[0][0]
        percent_risk = prediction_prob * 100
        
        # 🎨 สร้างกราฟเข็มไมล์ (Gauge Chart)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = percent_risk,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "ระดับความเสี่ยงการเป็นโรคหัวใจ", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkgray"}, # สีของเข็ม
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': "#a5d6a7"},    # โซนปลอดภัย (เขียว)
                    {'range': [40, 70], 'color': "#fff59d"},    # โซนเฝ้าระวัง (เหลือง)
                    {'range': [70, 100], 'color': "#ef9a9a"}    # โซนอันตราย (แดง)
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': percent_risk
                }
            }
        ))
        
        # แสดงกราฟบนหน้าเว็บ
        st.plotly_chart(fig, use_container_width=True)

        # แสดงข้อความสรุปด้านล่างกราฟ
        if prediction_prob > 0.5:
            st.error(f"🚨 สรุปผล: มีความเสี่ยงเป็นโรคหัวใจ! (ความมั่นใจ {percent_risk:.2f}%)")
        else:
            st.success(f"✅ สรุปผล: ปกติ (ความเสี่ยงเพียง {percent_risk:.2f}%)")
