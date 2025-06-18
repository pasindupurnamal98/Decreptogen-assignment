import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

st.set_page_config(page_title="Personality Predictor", layout="wide")

st.title("Introvert/Extrovert Personality Predictor")

st.write("""
### Enter your characteristics to predict if you're an introvert or extrovert
Please fill in all the fields below:
""")

# Create input fields
col1, col2 = st.columns(2)

with col1:
    time_alone = st.slider("Time spent alone (hours per day)", 1, 11, 5)
    stage_fear = st.selectbox("Do you have stage fear?", ["Yes", "No"])
    social_events = st.slider("Social event attendance (per month)", 1, 10, 5)
    going_outside = st.slider("Frequency of going outside (days per week)", 1, 7, 4)

with col2:
    drained_socializing = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])
    friends_circle = st.slider("Number of close friends", 1, 15, 7)
    post_frequency = st.slider("Social media post frequency (per week)", 1, 10, 5)

# Create prediction button
if st.button("Predict Personality"):
    try:
        # Load the complete pipeline
        pipeline = joblib.load('best_personality_model_pipeline.pkl')
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Time_spent_Alone': [time_alone],
            'Stage_fear': [stage_fear],
            'Social_event_attendance': [social_events],
            'Going_outside': [going_outside],
            'Drained_after_socializing': [drained_socializing],
            'Friends_circle_size': [friends_circle],
            'Post_frequency': [post_frequency]
        })

        # Make prediction using the pipeline
        prediction = pipeline.predict(input_data)
        
        # Load model info to get prediction probabilities
        model_info = pickle.load(open('comprehensive_model_info.pkl', 'rb'))
        
        # Display result with styling
        st.markdown("### Prediction Result")
        result = "Introvert" if prediction[0] == 0 else "Extrovert"
        
        # Get prediction probabilities
        prediction_proba = pipeline.predict_proba(input_data)[0]
        intro_prob = prediction_proba[0]
        extro_prob = prediction_proba[1]
        
        if result == "Introvert":
            st.markdown(f"""
            <div style='background-color: #E8F4F9; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #2C3E50;'>You are predicted to be an: {result}</h3>
                <p>Confidence: {intro_prob:.2%}</p>
                <p>Introverts tend to:</p>
                <ul>
                    <li>Recharge by spending time alone</li>
                    <li>Prefer deeper one-on-one conversations</li>
                    <li>Think before speaking</li>
                    <li>Feel drained after social interactions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: #F9E8E8; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #2C3E50;'>You are predicted to be an: {result}</h3>
                <p>Confidence: {extro_prob:.2%}</p>
                <p>Extroverts tend to:</p>
                <ul>
                    <li>Gain energy from social interactions</li>
                    <li>Enjoy group activities</li>
                    <li>Think while speaking</li>
                    <li>Feel energized after social events</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Display model performance metrics
        st.markdown("### Model Performance")
        st.markdown(f"""
        - Model Type: {model_info['selected_model_name']}
        - Test Accuracy: {model_info['performance_metrics']['test_accuracy']:.2%}
        - F1-Score: {model_info['performance_metrics']['test_f1']:.2%}
        """)

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Add explanation section
st.markdown("""
### About the Features
- **Time spent alone**: Average hours spent alone per day
- **Stage fear**: Whether you experience anxiety when speaking in public
- **Social event attendance**: Number of social events attended per month
- **Going outside**: Days per week spent outside home
- **Drained after socializing**: Whether you feel tired after social interactions
- **Friends circle size**: Number of close friends
- **Post frequency**: Number of social media posts per week
""")

# Add footer
st.markdown("""
---
Created with ❤️ using Streamlit
""")