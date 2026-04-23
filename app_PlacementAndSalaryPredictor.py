import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# ================== LOAD MODEL ==================
classifier = joblib.load('artifacts/classifier.pkl')
regressor = joblib.load('artifacts/regressor.pkl')

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AI Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# ================== STYLE ==================
st.markdown("""
    <style>
        .main-title {
            font-size:40px;
            font-weight:700;
            color:#4F8BF9;
            text-align:center;
        }
        .sub-title {
            text-align:center;
            color:gray;
            margin-bottom:30px;
        }
        .card {
            padding:20px;
            border-radius:15px;
            background-color:#f9f9f9;
            box-shadow:0px 2px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown('<div class="main-title">🎓 AI Placement & Salary Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict your placement chances and expected salary using Machine Learning</div>', unsafe_allow_html=True)

st.divider()

# ================== INPUT SECTION ==================
with st.container():
    st.markdown("### 👤 Personal Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.radio("Gender", ["Male", "Female"])
        branch = st.selectbox("Branch", ["CSE", "ECE", "IT", "ME", "CE"])

    with col2:
        cgpa = st.number_input("CGPA", 5.0, 10.0)
        backlogs = st.number_input("Backlogs", 0, 5)
        study_hours_per_day = st.number_input("Study Hours/Day", 0.0, 10.0)

    with col3:
        attendance_percentage = st.number_input("Attendance %", 44.7, 99.2)
        sleep_hours = st.number_input("Sleep Hours", 4.0, 9.0)
        stress_level = st.number_input("Stress Level", 1, 10)

st.divider()

with st.container():
    st.markdown("### 📊 Academic Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenth_percentage = st.number_input("10th %", 50.0, 100.0)
        twelfth_percentage = st.number_input("12th %", 50.0, 100.0)

    with col2:
        coding_skill_rating = st.number_input("Coding Skill", 1, 5)
        communication_skill_rating = st.number_input("Communication Skill", 1, 5)

    with col3:
        aptitude_skill_rating = st.number_input("Aptitude Skill", 1, 5)

st.divider()

with st.container():
    st.markdown("### 🚀 Experience & Activities")

    col1, col2, col3 = st.columns(3)

    with col1:
        projects_completed = st.number_input("Projects", 0, 8)
        internships_completed = st.number_input("Internships", 0, 4)

    with col2:
        hackathons_participated = st.number_input("Hackathons", 0, 6)
        certifications_count = st.number_input("Certifications", 0, 9)

    with col3:
        extracurricular_involvement = st.selectbox("Extracurricular", ["Low", "Medium", "High"])

st.divider()

with st.container():
    st.markdown("### 🌍 Lifestyle & Background")

    col1, col2, col3 = st.columns(3)

    with col1:
        part_time_job = st.radio("Part Time Job", ["Yes", "No"])
        internet_access = st.radio("Internet Access", ["Yes", "No"])

    with col2:
        family_income_level = st.radio("Family Income", ["Low", "Medium", "High"])

    with col3:
        city_tier = st.radio("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

# ================== PREDICTION ==================
st.divider()

if st.button("🚀 Predict Now", use_container_width=True):

    data = {
        "gender": gender,
        "branch": branch,
        "cgpa": cgpa,
        "tenth_percentage": tenth_percentage,
        "twelfth_percentage": twelfth_percentage,
        "backlogs": backlogs,
        "study_hours_per_day": study_hours_per_day,
        "attendance_percentage": attendance_percentage,
        "projects_completed": projects_completed,
        "internships_completed": internships_completed,
        "coding_skill_rating": coding_skill_rating,
        "communication_skill_rating": communication_skill_rating,
        "aptitude_skill_rating": aptitude_skill_rating,
        "hackathons_participated": hackathons_participated,
        "certifications_count": certifications_count,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "part_time_job": part_time_job,
        "family_income_level": family_income_level,
        "city_tier": city_tier,
        "internet_access": internet_access,
        "extracurricular_involvement": extracurricular_involvement
    }

    df = pd.DataFrame([data])

    # align columns with training data
    df = df.reindex(columns=classifier.feature_names_in_, fill_value=0)

    # ================== CLASS PREDICTION ==================
    class_pred = classifier.predict(df)[0]
    result = "🎉 Placed" if class_pred == 1 else "❌ Not Placed"
    st.success(f"Placement Result: {result}")

    # ================== SALARY PREDICTION ==================
    salary_pred = regressor.predict(df)[0] if class_pred == 1 else 0

    st.metric("💰 Predicted Salary (LPA)", f"{salary_pred:.2f}")

    # ================== VISUALIZATION ==================
    st.markdown("### 📊 Salary Visualization")

    if class_pred == 1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=salary_pred,
            title={'text': "Predicted Salary (LPA)"},
            gauge={
                'axis': {'range': [0, 20]},
                'bar': {'color': "#4F8BF9"},
                'steps': [
                    {'range': [0, 7], 'color': "#ffcccc"},
                    {'range': [7, 14], 'color': "#fff2cc"},
                    {'range': [14, 20], 'color': "#ccffcc"},
                ],
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No salary prediction because student is Not Placed.")