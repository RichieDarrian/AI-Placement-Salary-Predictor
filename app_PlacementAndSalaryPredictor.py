import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# ================== LOAD MODEL ==================
classifier = joblib.load('classifier.pkl')
regressor = joblib.load('regressor.pkl')

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
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
        backlogs = st.slider("Backlogs", 0, 10, 0)
        study_hours_per_day = st.slider("Study Hours/Day", 0.0, 12.0, 3.0, 0.5)

    with col3:
        attendance_percentage = st.slider("Attendance %", 0.0, 100.0, 85.0, 0.1)
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 6.0, 0.5)
        stress_level = st.slider("Stress Level", 1, 10, 5)

st.divider()

# ================== ACADEMIC ==================
with st.container():
    st.markdown("### 📊 Academic Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        tenth_percentage = st.slider("10th %", 0.0, 100.0, 75.0)
        twelfth_percentage = st.slider("12th %", 0.0, 100.0, 75.0)

    with col2:
        coding_skill_rating = st.slider("Coding Skill", 1, 10, 6)
        communication_skill_rating = st.slider("Communication Skill", 1, 10, 6)

    with col3:
        aptitude_skill_rating = st.slider("Aptitude Skill", 1, 10, 6)

st.divider()

# ================== EXPERIENCE ==================
with st.container():
    st.markdown("### 🚀 Experience & Activities")
    col1, col2, col3 = st.columns(3)

    with col1:
        projects_completed = st.slider("Projects", 0, 20, 2)
        internships_completed = st.slider("Internships", 0, 10, 1)

    with col2:
        hackathons_participated = st.slider("Hackathons", 0, 10, 0)
        certifications_count = st.slider("Certifications", 0, 20, 1)

    with col3:
        extracurricular_involvement = st.selectbox("Extracurricular", ["Low", "Medium", "High"])

st.divider()

# ================== LIFESTYLE ==================
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

    df = df.reindex(columns=classifier.feature_names_in_, fill_value=0)

    # ================== PREDICTION ==================
    class_pred = classifier.predict(df)[0]
    result = "🎉 Placed" if class_pred == 1 else "❌ Not Placed"
    st.success(f"Placement Result: {result}")

    salary_pred = regressor.predict(df)[0] if class_pred == 1 else 0
    st.metric("💰 Predicted Salary (LPA)", f"{salary_pred:.2f}")

    # ================== GAUGE ==================
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

        if salary_pred < 7:
            feedback = "🔴 Low Range: Gaji tergolong rendah, biasanya entry-level atau skill masih basic."
        elif salary_pred < 14:
            feedback = "🟡 Medium Range: Gaji menengah, biasanya sudah punya pengalaman atau skill cukup."
        else:
            feedback = "🟢 High Range: Gaji tinggi, menunjukkan skill sangat kuat atau kandidat unggul."

        st.markdown("### 📊 Salary Interpretation")
        st.info(feedback)
    else:
        st.warning("No salary prediction because student is Not Placed.")
