#!/usr/bin/env python3
"""
Streamlit Frontend for Academic Risk Predictor
Interactive web app for student risk assessment
"""
import streamlit as st
import requests
import json
from typing import Dict

# Page configuration
st.set_page_config(
    page_title="Academic Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

# Sidebar
st.sidebar.title("🎓 Academic Risk Predictor")
st.sidebar.write("**Professional Edition**")
st.sidebar.write("Predict student academic performance using psychosocial factors")

# Health check
def check_api_health():
    """Check if FastAPI backend is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=3)
        return response.status_code == 200
    except:
        return False

# Display health status
api_healthy = check_api_health()
if api_healthy:
    st.sidebar.success(" Backend Connected")
else:
    st.sidebar.error("Backend Not Available")
    st.sidebar.warning("Start FastAPI server: `uv run uvicorn api:app --host 0.0.0.0 --port 8000`")

# Main content
st.title("🎓 Student Academic Risk Predictor")
st.write("Assess academic performance based on psychosocial factors using Random Forest AI model")

# Category mapping not needed for current numeric input approach
reverse_mapping = {}

# Define questions with clear labels for UI
questions = {
    'parent_guardian_support_education_learning': 'Parent/Guardian Support in Education',
    'positive_supportive_relation_with_lecturers': 'Positive Relation with Lecturers',
    'avoid_engaging_risky_behavior': 'Avoid Risky Behavior',
    'family_status_stable_supportive_in_academic_pursuits': 'Family Support for Academic Pursuits',
    'parent_guardian_encourages_academic_pursuit': 'Parental Encouragement of Academic Pursuit',
    'no_significant_family_issues_interfere_academic_pursuit': 'No Family Issues Interfering with Studies',
    'strong_sense_belonging_university_community': 'Sense of Belonging at University',
    'university_provide_adequate_support_academic_pursuit': 'University Academic Support',
    'feel_safe_comfortable_in_school_environment': 'Safe and Comfortable School Environment'
}

# Likert scale labels
responses = {
    1: "Strongly Disagree",
    2: "Disagree",
    3: "Neutral",
    4: "Agree",
    5: "Strongly Agree"
}

# Input form (collapsible)
with st.expander("Student Assessment Form", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Information")
        gender = st.selectbox("Gender", options=["Male", "Female"], index=0, help="Select student gender")
        level = st.selectbox("Academic Level", options=["400L", "300L", "200L", "100L", "500L"], index=0,
                           help="Select current academic level")

    with col2:
        st.subheader("Family Environment")
        parent_support = st.slider(questions['parent_guardian_support_education_learning'], 1, 5, 3,
                                 help="How much parental support does the student receive?")
        family_stable = st.slider(questions['family_status_stable_supportive_in_academic_pursuits'], 1, 5, 3,
                                help="Is the family environment stable and supportive?")
        parent_encourages = st.slider(questions['parent_guardian_encourages_academic_pursuit'], 1, 5, 3,
                                    help="Do parents actively encourage academic achievement?")
        family_issues = st.slider(questions['no_significant_family_issues_interfere_academic_pursuit'], 1, 5, 2,
                                help="Are there family issues affecting the student's studies?")

    st.subheader("🎓 University Environment")
    col3, col4 = st.columns(2)
    with col3:
        lecturer_relation = st.slider(questions['positive_supportive_relation_with_lecturers'], 1, 5, 3,
                                    help="How positive are student-lecturer relationships?")
        belonging = st.slider(questions['strong_sense_belonging_university_community'], 1, 5, 3,
                            help="Does the student feel a sense of belonging?")
    with col4:
        university_support = st.slider(questions['university_provide_adequate_support_academic_pursuit'], 1, 5, 4,
                                     help="How adequate is university academic support?")
        safe_environment = st.slider(questions['feel_safe_comfortable_in_school_environment'], 1, 5, 3,
                                   help="Does the student feel safe at university?")

    st.subheader("Personal Behavior")
    risky_behavior = st.slider(questions['avoid_engaging_risky_behavior'], 1, 5, 3,
                              help="How well does the student avoid risky behaviors?")

    # Action buttons
    col5, col6 = st.columns([1, 1])
    with col5:
        predict_button = st.button("Predict Academic Risk", type="primary", use_container_width=True)

    with col6:
        clear_button = st.button("Clear Form", use_container_width=True)

# Clear form functionality
if clear_button:
    st.experimental_rerun()

# Prediction logic
if predict_button:
    if not api_healthy:
        st.error("Cannot make prediction: Backend API not available")
        st.info("Start the FastAPI server: `uv run uvicorn api:app --host 0.0.0.0 --port 8000`")
        st.stop()

    # Prepare data matching API Pydantic model field names
    data = {
        'gender': 1 if gender == "Male" else 2,
        'level': {"100L": 1, "200L": 2, "300L": 3, "400L": 4, "500L": 5}.get(level, 4),
        'parent_guardian_support_education_learning': parent_support,
        'positive_supportive_relation_with_lecturers': lecturer_relation,
        'avoid_engaging_risky_behavior': risky_behavior,
        'family_status_stable_supportive_in_academic_pursuits': family_stable,
        'parent_guardian_encourages_academic_pursuit': parent_encourages,
        'no_significant_family_issues_interfere_academic_pursuit': family_issues,
        'strong_sense_belonging_university_community': belonging,
        'university_provide_adequate_support_academic_pursuit': university_support,
        'feel_safe_comfortable_in_school_environment': safe_environment
    }

    # Show progress
    with st.spinner("Analyzing student data with AI model..."):
        try:
            response = requests.post("http://localhost:8000/predict", json=data, timeout=10)

            if response.status_code == 200:
                result = response.json()

                # Display results
                st.success("Prediction Complete!")

                col_result, _ = st.columns([2, 1])
                with col_result:
                    st.subheader("Prediction Result")

                    # Main prediction card
                    st.metric("Predicted CGPA Class", result['prediction'])

                    # Risk assessment based on CGPA class
                    if result['prediction'] in ['Fail', 'Pass', 'Third Class']:
                        st.error("High Risk: Student at risk of poor academic performance")
                    elif result['prediction'] == 'Second Class Lower':
                        st.warning("Medium Risk: Moderate academic performance expected")
                    elif result['prediction'] in ['Second Class Upper', 'First Class']:
                        st.success("Low Risk: Strong academic performance predicted")
                    else:
                        st.info("Prediction Status: Assessment completed")

                    # Show input summary
                    with st.expander("Input Summary", expanded=False):
                        st.write(f"**Gender:** {gender}")
                        st.write(f"**Level:** {level}")
                        st.write(f"**Parent Support:** {responses[parent_support]}")
                        st.write(f"**Family Stability:** {responses[family_stable]}")
                        st.write(f"**Lecturer Relations:** {responses[lecturer_relation]}")
                        st.write(f"**University Support:** {responses[university_support]}")
                        st.write(f"**Risk Avoidance:** {responses[risky_behavior]}")

            else:
                st.error(f"Prediction failed: HTTP {response.status_code}")
                st.write(response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")
            st.info("Make sure the FastAPI server is running on port 8000")

# Footer
st.markdown("---")
st.subheader("Recent Predictions")

if st.button(" Load Prediction History"):
    try:
        response = requests.get("http://localhost:8000/predictions", timeout=5)
        if response.status_code == 200:
            predictions = response.json()

            if not predictions:
                st.info("No previous predictions found.")
            else:
                # Display last 5 predictions
                st.write("**Recent Evaluations:**")
                for i, pred in enumerate(predictions[:5]):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.write(f"**ID {pred['id']}**")
                    with col_b:
                        st.write(f" {pred['prediction']}")
                    with col_c:
                        timestamp = pred['timestamp'].split('T')[0] if pred['timestamp'] else "N/A"
                        st.write(f"{timestamp}")

        else:
            st.error("Failed to load prediction history")

    except Exception as e:
        st.error(f"Error loading history: {e}")

# Footer info
st.markdown("---")
st.caption("Academic Risk Predictor v1.0 | Powered by Random Forest AI")
st.caption("Focus on early identification and intervention for at-risk students")
