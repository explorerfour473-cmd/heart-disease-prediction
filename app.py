import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

st.set_page_config(
    page_title="ระบบประเมินความเสี่ยงโรคหัวใจ",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .alert-box {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

if 'menu_option' not in st.session_state:
    st.session_state['menu_option'] = "หน้าแรก (Home)"

def navigate_to(page_name):
    st.session_state['menu_option'] = page_name

@st.cache_resource
def load_assets():
    model_columns = joblib.load('model_columns.pkl')
    scaler = joblib.load('scaler.pkl')
    ensemble_model = joblib.load('ensemble_model.pkl')
    nn_model = load_model('nn_model.keras')
    return model_columns, scaler, ensemble_model, nn_model

model_columns, scaler, ensemble_model, nn_model = load_assets()

def get_user_input():
    st.markdown('<div class="section-header">แบบฟอร์มบันทึกข้อมูลผู้ป่วย (Patient Data Form)</div>', unsafe_allow_html=True)
    
    st.markdown("**1. ข้อมูลทั่วไปและอาการเบื้องต้น**")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("อายุ (Age)", min_value=1, max_value=120, value=50)
    with col2:
        sex = st.selectbox("เพศ (Sex)", ["Male", "Female"])
    with col3:
        cp = st.selectbox("ลักษณะการเจ็บหน้าอก (Chest Pain Type)", 
                          ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    
    st.markdown("**2. ผลการตรวจทางห้องปฏิบัติการ (Lab Results)**")
    col4, col5, col6 = st.columns(3)
    with col4:
        trestbps = st.number_input("ความดันโลหิตขณะพัก (Resting BP - mmHg)", min_value=50, max_value=250, value=120)
    with col5:
        chol = st.number_input("คอเลสเตอรอล (Cholesterol - mg/dl)", min_value=100, max_value=600, value=200)
    with col6:
        fbs = st.selectbox("น้ำตาลในเลือดขณะอดอาหาร > 120 (Fasting Blood Sugar)", ["FALSE", "TRUE"])
        
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

def show_personal_health_dashboard(user_data):
    st.markdown('<div class="section-header">แดชบอร์ดสรุปสุขภาพส่วนบุคคล (Personal Health Dashboard)</div>', unsafe_allow_html=True)
    st.write("แสดงค่าสุขภาพของคุณเปรียบเทียบกับช่วงเกณฑ์: สีเขียว (ปกติ), สีเหลือง (เฝ้าระวัง) และสีแดง (ความเสี่ยงสูง)")
    st.write("") 
    
    col1, col2, col3 = st.columns(3)
    
    chart_layout = dict(height=80, margin=dict(t=10, b=10, l=15, r=15))
    
    with col1:
        st.markdown(f"**ความดันโลหิต (Resting BP)**<br><span style='font-size:24px; color:#0d47a1; font-weight:bold;'>{user_data['trestbps']}</span> mmHg", unsafe_allow_html=True)
        fig_bp = go.Figure(go.Indicator(
            mode = "gauge",
            value = user_data['trestbps'],
            gauge = {
                'shape': "bullet",
                'axis': {'range': [None, 250]},
                'bar': {'color': "#2c3e50", 'thickness': 0.4}, 
                'steps': [
                    {'range': [0, 120], 'color': "#d4edda"},   
                    {'range': [120, 130], 'color': "#fff3cd"}, 
                    {'range': [130, 250], 'color': "#f8d7da"}  
                ],
                'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': 120}
            }
        ))
        fig_bp.update_layout(**chart_layout)
        st.plotly_chart(fig_bp, use_container_width=True)

    with col2:
        st.markdown(f"**คอเลสเตอรอล (Cholesterol)**<br><span style='font-size:24px; color:#0d47a1; font-weight:bold;'>{user_data['chol']}</span> mg/dl", unsafe_allow_html=True)
        fig_chol = go.Figure(go.Indicator(
            mode = "gauge",
            value = user_data['chol'],
            gauge = {
                'shape': "bullet",
                'axis': {'range': [None, 400]},
                'bar': {'color': "#2c3e50", 'thickness': 0.4},
                'steps': [
                    {'range': [0, 200], 'color': "#d4edda"},   
                    {'range': [200, 240], 'color': "#fff3cd"}, 
                    {'range': [240, 400], 'color': "#f8d7da"}  
                ],
                'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': 200}
            }
        ))
        fig_chol.update_layout(**chart_layout)
        st.plotly_chart(fig_chol, use_container_width=True)

    max_hr = 220 - user_data['age'] 
    with col3:
        st.markdown(f"**อัตราหัวใจสูงสุด (Max: {max_hr})**<br><span style='font-size:24px; color:#0d47a1; font-weight:bold;'>{user_data['thalch']}</span> bpm", unsafe_allow_html=True)
        fig_hr = go.Figure(go.Indicator(
            mode = "gauge",
            value = user_data['thalch'],
            gauge = {
                'shape': "bullet",
                'axis': {'range': [None, 220]},
                'bar': {'color': "#2c3e50", 'thickness': 0.4},
                'steps': [
                    {'range': [0, max_hr * 0.5], 'color': "#e2e3e5"},         
                    {'range': [max_hr * 0.5, max_hr * 0.85], 'color': "#d4edda"}, 
                    {'range': [max_hr * 0.85, 220], 'color': "#f8d7da"}       
                ],
                'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': max_hr}
            }
        ))
        fig_hr.update_layout(**chart_layout)
        st.plotly_chart(fig_hr, use_container_width=True)

    st.markdown("---")
    st.markdown("**สรุปการวิเคราะห์ค่าสุขภาพเบื้องต้น:**")
    
    if user_data['trestbps'] > 130:
        st.warning(f"• **ความดันโลหิต:** {user_data['trestbps']} mmHg (อยู่ในเกณฑ์ **สูง**) ควรลดการบริโภคโซเดียมและปรึกษาแพทย์")
    elif user_data['trestbps'] > 120:
        st.info(f"• **ความดันโลหิต:** {user_data['trestbps']} mmHg (อยู่ในเกณฑ์ **ค่อนข้างสูง**) ควรเฝ้าระวังพฤติกรรมการทานอาหาร")
    else:
        st.success(f"• **ความดันโลหิต:** {user_data['trestbps']} mmHg (อยู่ในเกณฑ์ **ปกติ**)")

    if user_data['chol'] > 240:
        st.warning(f"• **คอเลสเตอรอล:** {user_data['chol']} mg/dl (อยู่ในเกณฑ์ **สูง**) เสี่ยงต่อการเกิดหลอดเลือดอุดตัน ควรหลีกเลี่ยงอาหารมัน/ทอด")
    elif user_data['chol'] > 200:
        st.info(f"• **คอเลสเตอรอล:** {user_data['chol']} mg/dl (อยู่ในเกณฑ์ **เริ่มสูง**) ควรเริ่มควบคุมอาหาร")
    else:
        st.success(f"• **คอเลสเตอรอล:** {user_data['chol']} mg/dl (อยู่ในเกณฑ์ **ปกติ**)")

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
    
    st.markdown('**กราฟเปรียบเทียบสุขภาพรวม (Health Radar)**')
    st.write("เปรียบเทียบข้อมูลของคุณกับเกณฑ์มาตรฐาน (เส้นปะสีเขียว) หากพื้นที่สีแดงทะลุกรอบเส้นปะออกไปมาก ควรเฝ้าระวังเป็นพิเศษครับ")
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("### ระบบประเมินสุขภาพ")
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

elif page == "หน้าหลัก - ทฤษฎี Machine Learning":
    st.markdown('<div class="main-header">แนวทางการพัฒนาและทฤษฎี Machine Learning</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. แหล่งอ้างอิงข้อมูล (Data Source)")
    st.write("โครงการนี้ใช้ชุดข้อมูล **Heart Disease Dataset** จาก **UCI Machine Learning Repository** (Cleveland database) ซึ่งประกอบด้วยข้อมูลผู้ป่วยจำนวน 303 ราย และคุณลักษณะ (Features) ที่เกี่ยวข้องกับสุขภาพหัวใจจำนวน 13 ตัวแปร เช่น อายุ, เพศ, ความดันโลหิต, คอเลสเตอรอล และผลตรวจทางห้องปฏิบัติการอื่นๆ")

    st.markdown("### 2. การเตรียมข้อมูลและขั้นตอนการพัฒนาโมเดล (Data Preparation & Development Steps)")
    st.write("เพื่อให้คอมพิวเตอร์สามารถเรียนรู้และสร้างโมเดลที่มีประสิทธิภาพ ได้มีการดำเนินการตามขั้นตอนดังต่อไปนี้:")
    st.markdown("""
    - **การทำความสะอาดข้อมูล (Data Cleaning):** ตรวจสอบและจัดการค่าสูญหาย (Missing Values) ในชุดข้อมูล
    - **การแปลงข้อมูลกลุ่ม (Categorical Encoding):** แปลงข้อมูลตัวอักษรให้เป็นตัวเลขด้วยเทคนิค One-Hot Encoding (เช่น การแปลงประเภทการเจ็บหน้าอกเป็น 0 และ 1) เพื่อให้โมเดลทางคณิตศาสตร์สามารถประมวลผลได้
    - **การปรับสเกลข้อมูล (Feature Scaling):** ปรับค่าตัวเลขที่มีหน่วยแตกต่างกันให้อยู่ในสเกลมาตรฐานเดียวกันด้วยวิธี `StandardScaler` เพื่อป้องกันไม่ให้ตัวแปรที่มีค่ามากมีอิทธิพลต่อโมเดลมากเกินไป
    - **การแบ่งชุดข้อมูล (Train-Test Split):** แบ่งข้อมูลสำหรับฝึกสอนโมเดล (Training Set) 80% และข้อมูลสำหรับทดสอบ (Testing Set) 20% เพื่อใช้วัดประสิทธิภาพการทำนายผล
    """)

    st.markdown("### 3. ทฤษฎีของอัลกอริทึมที่พัฒนา (Algorithm Theory)")
    st.write("ระบบนี้ประยุกต์ใช้เทคนิค **Ensemble Learning** แบบ **Hard Voting Classifier** ซึ่งเป็นการสร้างความแม่นยำด้วยการนำโมเดลพื้นฐาน 3 โมเดลมาทำงานร่วมกัน ได้แก่:")
    st.markdown("""
    - **Logistic Regression:** วิเคราะห์ความสัมพันธ์เชิงเส้นเพื่อแยกแยะกลุ่มเป้าหมาย
    - **Decision Tree Classifier:** สร้างเงื่อนไขการตัดสินใจในรูปแบบโครงสร้างต้นไม้
    - **K-Nearest Neighbors (KNN):** จัดกลุ่มข้อมูลโดยพิจารณาจากข้อมูลเพื่อนบ้านที่อยู่ใกล้เคียงที่สุด
    """)
    st.write("เมื่อได้รับข้อมูลผู้ป่วยใหม่ โมเดลทั้ง 3 ตัวจะทำการประเมินและโหวตคำตอบร่วมกัน ระบบจะเลือกคำตอบจาก 'เสียงข้างมาก' (Majority Vote) ทำให้ได้ผลลัพธ์การคัดกรองที่มีความเสถียรและแม่นยำสูงกว่าการใช้โมเดลเดียว")

    st.code("""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = KNeighborsClassifier()

ensemble_model = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2), ('knn', model3)], 
    voting='hard'
)
ensemble_model.fit(X_train_scaled, y_train)
    """, language='python')

elif page == "หน้าหลัก - ทฤษฎี Neural Network":
    st.markdown('<div class="main-header">แนวทางการพัฒนาและทฤษฎี Neural Network</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. แหล่งอ้างอิงข้อมูล (Data Source)")
    st.write("ชุดข้อมูลที่นำมาใช้ฝึกสอนโครงข่ายประสาทเทียม คือ **Heart Disease Dataset** จาก **UCI Machine Learning Repository** เช่นเดียวกัน โดยนำข้อมูลที่ผ่านกระบวนการทำความสะอาด แปลงค่า และปรับสเกลข้อมูลมาตรฐานแล้ว (Standardized Data) มาใช้เป็นข้อมูลขาเข้า (Input)")

    st.markdown("### 2. ทฤษฎีของอัลกอริทึมที่พัฒนา (Algorithm Theory)")
    st.write("**Artificial Neural Network (ANN)** หรือโครงข่ายประสาทเทียม เป็นสถาปัตยกรรมทางคอมพิวเตอร์ที่จำลองการทำงานของเซลล์ประสาทมนุษย์ สำหรับโครงการนี้ได้ออกแบบโครงสร้างในรูปแบบ **Multi-Layer Perceptron (MLP)** ซึ่งประกอบด้วยชั้นการประมวลผลดังนี้:")
    st.markdown("""
    - **Input Layer:** รับค่าตัวแปรอิสระที่ผ่านการปรับสเกลแล้วเข้าสู่ระบบ
    - **Hidden Layers:** ทำหน้าที่สกัดความสัมพันธ์ที่ซับซ้อนและไม่เป็นเส้นตรง (Non-linear) โดยใช้ฟังก์ชันกระตุ้น (Activation Function) แบบ **ReLU (Rectified Linear Unit)** จำนวน 2 ชั้นย่อย (ขนาด 16 และ 8 โหนด)
    - **Output Layer:** ชั้นแสดงผลลัพธ์สุดท้าย ใช้ 1 โหนดร่วมกับฟังก์ชันกระตุ้นแบบ **Sigmoid** เพื่อบีบอัดค่าผลลัพธ์ให้อยู่ในรูปของ 'ความน่าจะเป็น' ที่มีค่าระหว่าง 0 ถึง 1 (สามารถตีความเปอร์เซ็นต์ความเสี่ยง 0-100% ได้)
    """)

    st.markdown("### 3. ขั้นตอนการพัฒนาโมเดล (Model Development Steps)")
    st.markdown("""
    1. **ออกแบบโครงสร้าง (Model Architecture):** สร้างชั้น Dense Layers ตามโครงสร้าง MLP
    2. **กำหนดวิธีการเรียนรู้ (Compile):** - **Optimizer:** ใช้อัลกอริทึม `Adam` เพื่อปรับน้ำหนัก (Weights) ของเครือข่ายให้ลู่เข้าหาจุดที่เหมาะสมที่สุดได้อย่างรวดเร็ว
       - **Loss Function:** ใช้ `binary_crossentropy` ซึ่งออกแบบมาเฉพาะสำหรับการวัดค่าความผิดพลาดในงานจำแนกประเภทแบบ 2 คลาส (เป็น/ไม่เป็น โรคหัวใจ)
    3. **ฝึกสอนโมเดล (Training):** นำชุดข้อมูล Training Set ป้อนเข้าสู่โมเดล (กำหนดจำนวนรอบ Epochs และ Batch Size) เพื่อให้โมเดลค่อยๆ เรียนรู้และปรับน้ำหนัก
    4. **ประเมินผลและใช้งาน (Evaluation & Prediction):** ประเมินความแม่นยำด้วย Testing Set และนำโมเดลไปวิเคราะห์ความน่าจะเป็นเชิงลึกในหน้าการประเมิน
    """)

    st.code("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nn_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

nn_model.fit(
    X_train_scaled, y_train, 
    epochs=50, 
    batch_size=16,
    verbose=0
)
    """, language='python')

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
        
        show_personal_health_dashboard(user_data)
        st.markdown("---")
        show_radar_chart(user_data)
        
        st.info("ข้อสังเกต: โมเดล Machine Learning แบบ Hard Voting จะจำแนกผลลัพธ์เป็นกลุ่มเด็ดขาด (เป็น/ไม่เป็น) โดยไม่มีการแสดงค่าความน่าจะเป็น")

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
        
        show_personal_health_dashboard(user_data)
        st.markdown("---")
        show_radar_chart(user_data)
