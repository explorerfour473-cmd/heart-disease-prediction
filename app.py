import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# 1. ตั้งค่าหน้าเพจ (ต้องอยู่บรรทัดแรกสุดของ Streamlit)
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
    .card-text {
        font-size: 16px;
        color: #424242;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- ระบบจัดการหน้า (Session State Navigation) ---
if 'menu_option' not in st.session_state:
    st.session_state['menu_option'] = "หน้าแรก (Home)"

def navigate_to(page_name):
    st.session_state['menu_option'] = page_name

# 3. โหลดไฟล์สมองกล
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
    
    return scaled_input, user_data

# --- ฟังก์ชันเสริม: วาดกราฟ Radar Chart ---
def show_radar_chart(user_data):
    categories = ['ความดันโลหิต', 'คอเลสเตอรอล', 'อัตราหัวใจสูงสุด', 'ความดันโลหิต']
    user_values = [user_data['trestbps'], user_data['chol'], user_data['thalch'], user_data['trestbps']]
    normal_values = [120, 200, 150, 120]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=normal_values,
        theta=categories,
        fill=None,
        name='ค่ามาตรฐานคนปกติ (Normal)',
        line=dict(color='#2ca02c', width=3, dash='dash')
    ))

    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='ข้อมูลของคุณ (Your Data)',
        line_color='#d62728',
        opacity=0.7 
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 300]) 
        ),
        showlegend=True,
        margin=dict(t=30, b=30, l=30, r=30)
    )
    
    st.markdown('**กราเปรียบเทียบสุขภาพ (Health Radar)**')
    st.write("เปรียบเทียบข้อมูลของคุณกับเกณฑ์มาตรฐาน (เส้นปะสีเขียว) หากพื้นที่สีแดงทะลุกรอบเส้นปะออกไปมาก ควรเฝ้าระวังเป็นพิเศษครับ")
    st.plotly_chart(fig, use_container_width=True)

# 4. ตั้งค่าเมนูด้านข้าง (Sidebar)
st.sidebar.markdown("### ระบบประเมินสุขภาพ")
# ผูก Radio button เข้ากับ Session State เพื่อให้กดปุ่มในหน้าหลักแล้วเมนูเปลี่ยนตาม
page = st.sidebar.radio("โปรดเลือกเมนูที่ต้องการใช้งาน:", 
    ["หน้าแรก (Home)", 
     "หน้าหลัก - ทฤษฎี Machine Learning", 
     "หน้าหลัก - ทฤษฎี Neural Network", 
     "ระบบประเมิน - Machine Learning", 
     "ระบบประเมิน - Neural Network"],
    key='menu_option'
)
st.sidebar.markdown("---")
st.sidebar.caption("พัฒนาโดย: นายธิติภูมิ บุญภูมิ 6704062612235")

# ==========================================
# --- หน้าแรก (Home) ---
# ==========================================
if page == "หน้าแรก (Home)":
    st.markdown('<div class="main-header">ยินดีต้อนรับสู่ระบบประเมินความเสี่ยงโรคหัวใจ</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>แอปพลิเคชันนี้ประยุกต์ใช้เทคโนโลยีปัญญาประดิษฐ์ (AI) เพื่อวิเคราะห์ความเสี่ยงจากข้อมูลสุขภาพของคุณ <br>โปรดเลือกเมนูที่คุณต้องการใช้งานด้านล่างนี้</p>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">โหมดเรียนรู้ (Theory & Concept)</div>', unsafe_allow_html=True)
        st.info("ทำความเข้าใจเบื้องหลังการทำงานของ AI และขั้นตอนการเตรียมข้อมูลเพื่อสอนโมเดลสมองกล")
        st.button("ทฤษฎี Machine Learning (Ensemble)", 
                  on_click=navigate_to, args=("หน้าหลัก - ทฤษฎี Machine Learning",), 
                  use_container_width=True)
        st.button("ทฤษฎี Neural Network (Deep Learning)", 
                  on_click=navigate_to, args=("หน้าหลัก - ทฤษฎี Neural Network",), 
                  use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">โหมดประเมินความเสี่ยง (Assessment)</div>', unsafe_allow_html=True)
        st.success("ทดลองกรอกข้อมูลสุขภาพเพื่อประเมินความเสี่ยงโรคหัวใจด้วยตัวคุณเอง")
        st.button("ประเมินด้วย Machine Learning", 
                  type="primary", 
                  on_click=navigate_to, args=("ระบบประเมิน - Machine Learning",), 
                  use_container_width=True)
        st.button("ประเมินด้วย Neural Network", 
                  type="primary", 
                  on_click=navigate_to, args=("ระบบประเมิน - Neural Network",), 
                  use_container_width=True)

# ==========================================
# --- หน้าทฤษฎี ML ---
# ==========================================
elif page == "หน้าหลัก - ทฤษฎี Machine Learning":
    st.markdown('<div class="main-header">ทฤษฎีการสร้าง Machine Learning</div>', unsafe_allow_html=True)
    st.write("โปรเจคนี้ใช้ชุดข้อมูล Heart Disease จาก UCI โดยมีขั้นตอนการพัฒนาโมเดลตั้งแต่การเตรียมข้อมูลไปจนถึงการเทรน ดังนี้:")
    
    st.header("Step 1: การเตรียมข้อมูล (Data Preprocessing)")
    st.write("คอมพิวเตอร์ไม่สามารถเข้าใจข้อมูลที่เป็นตัวหนังสือหรือช่องว่างได้ เราจึงต้องทำความสะอาดและแปลงข้อมูลก่อน:")
    st.markdown("""
    - **จัดการค่าว่าง (Missing Values):** ลบหรือเติมข้อมูลในช่องที่ว่างเปล่า
    - **แปลงข้อความเป็นตัวเลข (One-Hot Encoding):** แปลงคอลัมน์เช่น เพศ, อาการเจ็บหน้าอก ให้เป็นตัวเลข 0 และ 1
    - **ปรับสเกลข้อมูล (Feature Scaling):** ปรับช่วงของตัวเลขให้มาอยู่ในสเกลเดียวกัน
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
    st.write("เราใช้เทคนิค Voting Classifier โดยนำโมเดล 3 ตัวมาช่วยกันโหวตตัดสินใจ (Hard Voting) เพื่อให้ผลลัพธ์แม่นยำกว่าการใช้โมเดลเดียว")
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
# --- หน้าทฤษฎี NN ---
# ==========================================
elif page == "หน้าหลัก - ทฤษฎี Neural Network":
    st.markdown('<div class="main-header">ทฤษฎีการสร้าง Neural Network</div>', unsafe_allow_html=True)
    st.write("Neural Network (โครงข่ายประสาทเทียม) เลียนแบบการทำงานของสมองมนุษย์ โดยรับข้อมูลเข้ามา ประมวลผลผ่านชั้นต่างๆ แล้วส่งผลลัพธ์ออกมาเป็นความน่าจะเป็น")

    st.header("Step 1: ออกแบบโครงสร้าง (Architecture)")
    st.write("เราใช้ไลบรารี TensorFlow/Keras ในการสร้างโมเดลแบบ Multi-Layer Perceptron (MLP) โดยมีโครงสร้าง 3 ชั้น:")
    st.markdown("""
    1. **Hidden Layer 1:** มี 16 โหนด (ใช้ฟังก์ชันกระตุ้น ReLU เพื่อหาความสัมพันธ์ที่ซับซ้อน)
    2. **Hidden Layer 2:** มี 8 โหนด (ใช้ ReLU กรองข้อมูลให้แคบลง)
    3. **Output Layer:** มี 1 โหนด (ใช้ Sigmoid บีบผลลัพธ์ให้อยู่ระหว่าง 0 ถึง 1 หรือก็คือ 0-100%)
    """)
    st.code("""
# ตัวอย่างโค้ดการสร้างโครงสร้าง Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nn_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
    """, language='python')

    st.header("Step 2: คอมไพล์และเทรนโมเดล (Compile & Train)")
    st.write("หลังจากสร้างโครงสร้างเสร็จ ต้องกำหนดวิธีการเรียนรู้ (Optimizer) และวิธีการวัดความผิดพลาด (Loss Function) จากนั้นจึงทำการสอน (Train)")
    st.code("""
# กำหนดการตั้งค่าการเรียนรู้
nn_model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', # เหมาะกับงานทายผล 2 ทาง
    metrics=['accuracy']
)

# สั่งเทรนโมเดล
nn_model.fit(
    X_train_scaled, y_train, 
    epochs=50, 
    batch_size=16,
    verbose=0
)
    """, language='python')

# ==========================================
# --- หน้าประเมิน ML ---
# ==========================================
elif page == "ระบบประเมิน - Machine Learning":
    st.markdown('<div class="main-header">ระบบประเมินความเสี่ยง (Machine Learning)</div>', unsafe_allow_html=True)
    scaled_data, user_data = get_user_input()
    
    if st.button("ประมวลผลข้อมูล (Run Analysis)", type="primary"):
        prediction = ensemble_model.predict(scaled_data)
        st.markdown('<div class="section-header">ผลการประเมิน (Assessment Result)</div>', unsafe_allow_html=True)
        if prediction[0] == 1:
            st.error("คำเตือน: ระบบตรวจพบความเสี่ยงของการเกิดโรคหัวใจ แนะนำให้ปรึกษาแพทย์")
        else:
            st.success("ผลการประเมิน: ปกติ (ไม่พบความเสี่ยงในระดับที่น่ากังวล)")
        
        show_radar_chart(user_data)
        
        st.info("ข้อสังเกต: โมเดล Machine Learning แบบ Hard Voting จะจำแนกผลลัพธ์เป็นกลุ่มเด็ดขาด (เป็น/ไม่เป็น) โดยไม่มีการแสดงค่าความน่าจะเป็น")

# ==========================================
# --- หน้าประเมิน NN ---
# ==========================================
elif page == "ระบบประเมิน - Neural Network":
    st.markdown('<div class="main-header">ระบบประเมินความเสี่ยง (Neural Network)</div>', unsafe_allow_html=True)
    scaled_data, user_data = get_user_input()
    
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
        st.markdown("---")
        
        show_radar_chart(user_data)
