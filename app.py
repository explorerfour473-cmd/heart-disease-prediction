import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

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

# ==========================================
# --- หน้าที่ 1: อธิบายทฤษฎี ML (อัปเกรดเนื้อหา) ---
# ==========================================
if page == "1. ทฤษฎี Machine Learning":
    st.title("📖 ทฤษฎีการสร้าง Machine Learning")
    st.write("โปรเจคนี้ใช้ชุดข้อมูล Heart Disease จาก UCI โดยมีขั้นตอนการพัฒนาโมเดลตั้งแต่การเตรียมข้อมูลไปจนถึงการเทรน ดังนี้:")
    
    st.header("Step 1: การเตรียมข้อมูล (Data Preprocessing)")
    st.write("คอมพิวเตอร์ไม่สามารถเข้าใจข้อมูลที่เป็นตัวหนังสือหรือช่องว่างได้ เราจึงต้องทำความสะอาดและแปลงข้อมูลก่อน:")
    st.markdown("""
    - **จัดการค่าว่าง (Missing Values):** ลบหรือเติมข้อมูลในช่องที่ว่างเปล่า
    - **แปลงข้อความเป็นตัวเลข (One-Hot Encoding):** แปลงคอลัมน์เช่น เพศ, อาการเจ็บหน้าอก ให้เป็นตัวเลข 0 และ 1
    - **ปรับสเกลข้อมูล (Feature Scaling):** ปรับช่วงของตัวเลข (เช่น อายุ 50, คอเลสเตอรอล 200) ให้มาอยู่ในสเกลเดียวกัน โมเดลจะได้ไม่งง
    """)
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
    st.write("เราใช้เทคนิค  **Voting Classifier** โดยนำโมเดล 3 ตัวมาช่วยกันโหวตตัดสินใจ (Hard Voting) เพื่อให้ผลลัพธ์แม่นยำกว่าการใช้โมเดลเดียว")
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

# ==========================================
# --- หน้าที่ 2: อธิบายทฤษฎี NN (อัปเกรดเนื้อหา) ---
# ==========================================
elif page == "2. ทฤษฎี Neural Network":
    st.title("🧠 ทฤษฎีการสร้าง Neural Network")
    st.write("Neural Network (หรือโครงข่ายประสาทเทียม) เลียนแบบการทำงานของสมองมนุษย์  โดยรับข้อมูลเข้ามา ประมวลผลผ่านชั้นต่างๆ (Layers) แล้วส่งผลลัพธ์ออกมาเป็นความน่าจะเป็น")

    st.header("Step 1: ออกแบบโครงสร้าง (Architecture)")
    st.write("เราใช้ไลบรารี TensorFlow/Keras ในการสร้างโมเดลแบบ Multi-Layer Perceptron (MLP) โดยมีโครงสร้าง 3 ชั้น:")
    st.markdown("""
    1. **Hidden Layer 1:** มี 16 โหนด (ใช้ฟังก์ชันกระตุ้น `ReLU` เพื่อหาความสัมพันธ์ที่ซับซ้อน)
    2. **Hidden Layer 2:** มี 8 โหนด (ใช้ `ReLU` กรองข้อมูลให้แคบลง)
    3. **Output Layer:** มี 1 โหนด (ใช้ `Sigmoid` บีบผลลัพธ์ให้อยู่ระหว่าง 0 ถึง 1 หรือก็คือ 0-100%)
    """)
    st.code("""
# ตัวอย่างโค้ดการสร้างโครงสร้าง Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nn_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid') # ออก 1 โหนดเพื่อบอกว่า เป็น/ไม่เป็น
])
    """, language='python')

    st.header("Step 2: คอมไพล์และเทรนโมเดล (Compile & Train)")
    st.write("หลังจากสร้างโครงสร้างเสร็จ ต้องกำหนดวิธีการเรียนรู้ (Optimizer) และวิธีการวัดความผิดพลาด (Loss Function) จากนั้นจึงโยนข้อมูลเข้าไปสอน (Train)")
    st.code("""
# กำหนดการตั้งค่าการเรียนรู้
nn_model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', # เหมาะกับงานทายผล 2 ทาง (0 หรือ 1)
    metrics=['accuracy']
)

# สั่งเทรนโมเดล (ให้โมเดลดูข้อมูลวนไปมา 50 รอบ)
nn_model.fit(
    X_train_scaled, y_train, 
    epochs=50, 
    batch_size=16,
    verbose=0 # ซ่อนข้อความระหว่างเทรนไม่ให้รกรุงรัง
)
    """, language='python')

# ==========================================
# --- หน้าที่ 3: ทดสอบ ML (คงเดิม) ---
# ==========================================
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

# ==========================================
# --- หน้าที่ 4: ทดสอบ NN (มีกราฟ คงเดิม) ---
# ==========================================
elif page == "4. ทดสอบ Neural Network":
    st.title("⚙️ ทดสอบโมเดล Neural Network")
    scaled_data = get_user_input()
    
    if st.button("🔍 ทำนายผลด้วย Neural Network"):
        prediction_prob = nn_model.predict(scaled_data)[0][0]
        percent_risk = prediction_prob * 100
        
        # กราฟเข็มไมล์ (Gauge Chart)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = percent_risk,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "ระดับความเสี่ยงการเป็นโรคหัวใจ", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkgray"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': "#a5d6a7"},    
                    {'range': [40, 70], 'color': "#fff59d"},    
                    {'range': [70, 100], 'color': "#ef9a9a"}    
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': percent_risk
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)

        if prediction_prob > 0.5:
            st.error(f"🚨 สรุปผล: มีความเสี่ยงเป็นโรคหัวใจ! (ความมั่นใจ {percent_risk:.2f}%)")
        else:
            st.success(f"✅ สรุปผล: ปกติ (ความเสี่ยงเพียง {percent_risk:.2f}%)")
