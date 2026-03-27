import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import shap  # 🔍 เพิ่มไลบรารี AI ให้เหตุผล
import matplotlib.pyplot as plt # เอาไว้ช่วยวาดกราฟเบื้องหลัง

# 1. ตั้งค่าหน้าเพจ
st.set_page_config(
    page_title="ระบบประเมินความเสี่ยงโรคหัวใจ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ตกแต่ง CSS ให้ดูเป็นแอปพลิเคชันทางการแพทย์
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        color: #0d47a1;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 20px;
        color: #1565c0;
        border-bottom: 2px solid #e3f2fd;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    .shap-explanation {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# 3. โหลดไฟล์สมองกล
@st.cache_resource
def load_assets():
    model_columns = joblib.load('model_columns.pkl')
    scaler = joblib.load('scaler.pkl')
    ensemble_model = joblib.load('ensemble_model.pkl')
    nn_model = load_model('nn_model.keras')
    return model_columns, scaler, ensemble_model, nn_model

model_columns, scaler, ensemble_model, nn_model = load_assets()

# --- ฟังก์ชันฟอร์มกรอกข้อมูล (จัด Layout ใหม่เป็นหมวดหมู่) ---
def get_user_input():
    st.markdown('<div class="section-header">แบบฟอร์มบันทึกข้อมูลผู้ป่วย (Patient Data Form)</div>', unsafe_allow_html=True)
    
    # หมวดที่ 1: ข้อมูลทั่วไป
    st.markdown("**1. ข้อมูลทั่วไปและอาการเบื้องต้น**")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("อายุ (Age)", min_value=1, max_value=120, value=50)
    with col2:
        sex = st.selectbox("เพศ (Sex)", ["Male", "Female"])
    with col3:
        cp = st.selectbox("ลักษณะการเจ็บหน้าอก (Chest Pain Type)", 
                          ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    
    # หมวดที่ 2: ผลตรวจทางห้องปฏิบัติการ
    st.markdown("**2. ผลการตรวจทางห้องปฏิบัติการ (Lab Results)**")
    col4, col5, col6 = st.columns(3)
    with col4:
        trestbps = st.number_input("ความดันโลหิตขณะพัก (Resting BP - mmHg)", min_value=50, max_value=250, value=120)
    with col5:
        chol = st.number_input("คอเลสเตอรอล (Cholesterol - mg/dl)", min_value=100, max_value=600, value=200)
    with col6:
        fbs = st.selectbox("น้ำตาลในเลือดขณะอดอาหาร > 120 (Fasting Blood Sugar)", ["FALSE", "TRUE"])
        
    # หมวดที่ 3: ผลการทดสอบสมรรถภาพ
    st.markdown("**3. ผลการทดสอบสมรรถภาพหัวใจ (Stress Test)**")
    col7, col8 = st.columns(2)
    with col7:
        thalch = st.number_input("อัตราการเต้นหัวใจสูงสุด (Max Heart Rate)", min_value=60, max_value=220, value=150)
    with col8:
        exang = st.selectbox("เจ็บหน้าอกขณะออกกำลังกาย (Exercise Angina)", ["FALSE", "TRUE"])
    
    st.markdown("---")
    
    user_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 
        'chol': chol, 'thalch': thalch, 'fbs': fbs, 'exang': exang,
        'restecg': 'normal', 'oldpeak': 0.0, 'slope': 'flat', 'ca': 0, 'thal': 'normal'
    }
    input_df = pd.DataFrame([user_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    scaled_input = scaler.transform(input_df)
    
    # ส่งคืนทั้ง scaled_input และ DataFrame ต้นฉบับ (เอาไว้ทำ SHAP)
    return scaled_input, input_df

# 4. ตั้งค่าเมนูด้านข้าง (Sidebar) แบบทางการ
st.sidebar.markdown("### ระบบประเมินสุขภาพ")
st.sidebar.markdown("โปรดเลือกเมนูที่ต้องการใช้งาน:")
page = st.sidebar.radio("", 
    ["หน้าหลัก - ทฤษฎี Machine Learning", 
     "หน้าหลัก - ทฤษฎี Neural Network", 
     "ระบบประเมิน - Machine Learning", 
     "ระบบประเมิน - Neural Network"]
)
st.sidebar.markdown("---")
st.sidebar.caption("พัฒนาโดย: นายธิติภูมิ บุญภูมิ  6704062612235")

# ==========================================
# --- หน้าทฤษฎี (คงเดิม) ---
# ==========================================
if page == "หน้าหลัก - ทฤษฎี Machine Learning":
    st.markdown('<div class="main-header">ทฤษฎีการสร้างโมเดล Machine Learning</div>', unsafe_allow_html=True)
    st.write("โปรเจคนี้ใช้ชุดข้อมูล Heart Disease จาก UCI โดยมีขั้นตอนการพัฒนาโมเดลตั้งแต่การเตรียมข้อมูลไปจนถึงการเทรน ดังนี้:")
    st.header("Step 1: การเตรียมข้อมูล (Data Preprocessing)")
    st.write("คอมพิวเตอร์ไม่สามารถเข้าใจข้อมูลที่เป็นตัวหนังสือหรือช่องว่างได้ เราจึงต้องทำความสะอาดและแปลงข้อมูลก่อน:")
    st.code("""
# ตัวอย่างโค้ดการเตรียมข้อมูลด้วย Pandas และ Scikit-Learn
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. โหลดข้อมูลและแปลงข้อความเป็น 0,1
df = pd.read_csv('heart_disease_uci.csv')
df_encoded = pd.get_dummies(df)

# 2. ปรับสเกลข้อมูล (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
    """, language='python')
    st.header("Step 2: สร้างโมเดล Ensemble Learning")
    st.code("""
# ตัวอย่างโค้ดการสร้างโมเดล Ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

# สร้างโมเดลลูก 3 ตัว
model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = KNeighborsClassifier()

# มัดรวมเป็น Ensemble Model แล้วสั่งเทรน (fit)
ensemble_model = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2), ('knn', model3)], 
    voting='hard'
)
ensemble_model.fit(X_train_scaled, y_train)
    """, language='python')

elif page == "หน้าหลัก - ทฤษฎี Neural Network":
    st.markdown('<div class="main-header">ทฤษฎีการสร้างโครงข่ายประสาทเทียม (Neural Network)</div>', unsafe_allow_html=True)
    st.write("Neural Network (โครงข่ายประสาทเทียม) เลียนแบบการทำงานของสมองมนุษย์ โดยรับข้อมูลเข้ามา ประมวลผลผ่านชั้นต่างๆ แล้วส่งผลลัพธ์ออกมาเป็นความน่าจะเป็น")
    st.header("Step 1: ออกแบบโครงสร้าง (Architecture)")
    st.code("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nn_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
    """, language='python')

# ==========================================
# --- ระบบประเมิน - Machine Learning (เพิ่มระบบให้เหตุผล) ---
# ==========================================
elif page == "ระบบประเมิน - Machine Learning":
    st.markdown('<div class="main-header">ระบบประเมินความเสี่ยง (Machine Learning)</div>', unsafe_allow_html=True)
    scaled_data, raw_data = get_user_input()  # รับข้อมูล 2 แบบ
    
    if st.button("ประมวลผลข้อมูล (Run Analysis)", type="primary"):
        st.markdown('<div class="section-header">ผลการประเมินและการวิเคราะห์ปัจจัย (Analysis Result)</div>', unsafe_allow_html=True)
        
        # 1. ทำนายผลปกติ
        prediction = ensemble_model.predict(scaled_data)
        if prediction[0] == 1:
            st.error("คำเตือน: ระบบตรวจพบความเสี่ยงของการเกิดโรคหัวใจ แนะนำให้ปรึกษาแพทย์")
        else:
            st.success("ผลการประเมิน: ปกติ (ไม่พบความเสี่ยงในระดับที่น่ากังวล)")
        
        st.write("---")
        
        # 2. 🔍 เริ่มระบบ AI ให้เหตุผล (SHAP)
        st.subheader("ปัจจัยที่มีผลต่อการทำนาย (Explanation Factors)")
        
        # คำนวณ SHAP Values (ใช้ KernelExplainer สำหรับ Ensemble model)
        # เนื่องจากคำนวณนาน เราจะใช้ข้อมูลปัจจุบันตัวเดียวเป็นพื้นฐานเพื่อความเร็ว
        with st.spinner("โมเดลกำลังวิเคราะห์เหตุผลเชิงลึก... (อาจใช้เวลา 10-20 วินาที)"):
            explainer = shap.Explainer(ensemble_model.predict, scaled_data)
            shap_values = explainer(scaled_data)
        
        # แปลงค่า SHAP เป็นเปอร์เซ็นต์ความเข้าใจง่าย (เชิงบวก=เสี่ยงเพิ่ม, เชิงลบ=เสี่ยงลด)
        current_shap_values = shap_values.values[0]
        feature_names = model_columns
        
        # กรองเอาเฉพาะปัจจัยที่มีผลเยอะๆ มาโชว์ (top 10)
        shap_data = pd.DataFrame({
            'Feature': feature_names,
            'Influence': current_shap_values
        })
        shap_data = shap_data.sort_values(by='Influence', ascending=True).tail(10) # เอาตัวที่มีผลมากสุด (ทั้งบวกและลบ)
        
        # กำหนดสี (แดง=เสี่ยงเพิ่ม, เขียว=เสี่ยงลด)
        shap_data['Color'] = shap_data['Influence'].apply(lambda x: '#d62728' if x > 0 else '#2ca02c')
        shap_data['Label'] = shap_data['Influence'].apply(lambda x: 'เพิ่มความเสี่ยง' if x > 0 else 'ลดความเสี่ยง')
        
        # สร้างกราฟ Plotly Horizontal Bar Chart
        fig_shap = go.Figure(data=[
            go.Bar(
                x=shap_data['Influence'],
                y=shap_data['Feature'],
                orientation='h',
                marker_color=shap_data['Color'],
                text=shap_data['Label'],
                textposition='auto',
            )
        ])
        
        fig_shap.update_layout(
            title_text='กราฟวิเคราะห์ปัจจัยที่มีผลต่อคำทำนาย (SHAP Importance)',
            xaxis=dict(title='ระดับอิทธิพลต่อคำทำนาย (Influence Scale)'),
            yaxis=dict(title='ปัจจัยตรวจสุขภาพ'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400,
            margin=dict(l=150) # เว้นที่ด้านซ้ายให้ชื่อตัวแปร
        )
        
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # เพิ่มคำอธิบายกราฟแบบเข้าใจง่าย
        st.markdown("""
        <div class="shap-explanation">
        <strong>💡 วิธีอ่านกราฟ:</strong>
        <ul>
            <li><span style="color:#d62728; font-weight:bold;">แท่งสีแดงพุ่งไปทางขวา:</span> แปลว่าปัจจัยนี้ <strong>"ทำให้คุณเสี่ยงเป็นโรคหัวใจเพิ่มขึ้น"</strong> (ยิ่งยาว ยิ่งอันตราย)</li>
            <li><span style="color:#2ca02c; font-weight:bold;">แท่งสีเขียวพุ่งไปทางซ้าย:</span> แปลว่าปัจจัยนี้ <strong>"ช่วยลดความเสี่ยงหรือทำให้คุณดูปกติ"</strong></li>
        </ul>
        โมเดล AI ตัดสินใจจากผลรวมของแท่งสีแดงและสีเขียวทั้งหมดนี้ครับ
        </div>
        """, unsafe_allow_html=True)
        
        st.write("---")
        st.info("ข้อสังเกต: โมเดล Machine Learning แบบ Hard Voting จะจำแนกผลลัพธ์เป็นกลุ่มเด็ดขาด (เป็น/ไม่เป็น) โดยไม่มีการแสดงค่าความน่าจะเป็น")

# ==========================================
# --- ระบบประเมิน - Neural Network (คงเดิม) ---
# ==========================================
elif page == "ระบบประเมิน - Neural Network":
    st.markdown('<div class="main-header">ระบบประเมินความเสี่ยง (Neural Network)</div>', unsafe_allow_html=True)
    scaled_data, raw_data = get_user_input()
    
    if st.button("ประมวลผลข้อมูลเชิงลึก (Run Deep Analysis)", type="primary"):
        prediction_prob = nn_model.predict(scaled_data)[0][0]
        prob_risk = prediction_prob * 100
        prob_normal = 100 - prob_risk
        
        st.markdown('<div class="section-header">รายงานผลการวิเคราะห์เชิงลึก (Deep Analysis Report)</div>', unsafe_allow_html=True)
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label="โอกาสที่จะอยู่ในเกณฑ์ปกติ", value=f"{prob_normal:.1f}%")
        with col_res2:
            st.metric(label="โอกาสที่จะมีความเสี่ยงโรคหัวใจ", value=f"{prob_risk:.1f}%", 
                      delta="พบความเสี่ยง" if prob_risk > 50 else "ปลอดภัย", 
                      delta_color="inverse")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['ปกติ (Normal)', 'มีความเสี่ยง (Risk)'],
                y=[prob_normal, prob_risk],
                marker_color=['#2ca02c', '#d62728'],
                text=[f"{prob_normal:.1f}%", f"{prob_risk:.1f}%"],
                textposition='auto',
                width=0.4
            )
        ])
        fig.update_layout(
            yaxis=dict(title='ความน่าจะเป็น (%)', range=[0, 100]),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=30, b=0)
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
        
        st.plotly_chart(fig, use_container_width=True)
