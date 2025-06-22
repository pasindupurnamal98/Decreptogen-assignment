import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import time

# Page config
st.set_page_config(page_title="✨ Personality Predictor", page_icon="🧠", layout="wide")

# Title section
st.markdown("""
<div style='text-align: center; padding: 20px 0 10px 0;'>
    <h1 style='color:#4A4A4A;'>🧠 Introvert vs. Extrovert Personality Predictor</h1>
    <p style='font-size:18px; color: #666;'>Discover your dominant personality trait based on your lifestyle preferences</p>
</div>
""", unsafe_allow_html=True)

# Instructions
st.markdown("### 📝 Please provide the following details:")
st.markdown("---")
st.markdown("#### 👇 Personal Habits")
col1, col2 = st.columns(2)

with col1:
    time_alone = st.slider("⏱️ Time spent alone (hours/day)", 0, 11, 5)
    stage_fear = st.selectbox("🎤 Do you have stage fear?", ["Yes", "No"])
    social_events = st.slider("🎉 Social event attendance (per month)", 0, 10, 5)
    going_outside = st.slider("🚶 Frequency of going outside (days/week)", 0, 7, 4)

with col2:
    drained_socializing = st.selectbox("😩 Feel drained after socializing?", ["Yes", "No"])
    friends_circle = st.slider("👥 Number of close friends", 0, 15, 7)
    post_frequency = st.slider("📱 Social media post frequency (per week)", 0, 10, 5)

st.markdown("---")

# 🌟 Predict button
if st.button("🔍 Predict My Personality"):
    try:
        with st.spinner("🔍 Analyzing your personality... Please wait!"):
            time.sleep(2)

            # Load the saved pipeline and model metadata
            pipeline = joblib.load('best_personality_model_pipeline.pkl')
            model_info = pickle.load(open('comprehensive_model_info.pkl', 'rb'))
            label_encoder = model_info['label_encoder']

            # Format input data
            input_data = pd.DataFrame({
                'Time_spent_Alone': [time_alone],
                'Stage_fear': [stage_fear],
                'Social_event_attendance': [social_events],
                'Going_outside': [going_outside],
                'Drained_after_socializing': [drained_socializing],
                'Friends_circle_size': [friends_circle],
                'Post_frequency': [post_frequency]
            })

            # Perform prediction
            prediction = pipeline.predict(input_data)              # e.g. [0]
            prediction_proba = pipeline.predict_proba(input_data)[0]  # e.g. [0.84, 0.16]

            # Get index of predicted class
            pred_index = prediction[0]

            # Decode actual class label
            result = label_encoder.inverse_transform([pred_index])[0]

            # ✅ Robust confidence extraction based on ordered mapping
            class_order = label_encoder.transform(label_encoder.classes_)  # e.g. [0, 1] or [1, 0]
            proba_index = list(class_order).index(pred_index)
            confidence = prediction_proba[proba_index]

        # 🎨 Display results
        bg_color = "#E8F4F9" if result == "Introvert" else "#F9E8E8"
        emoji = "🌙" if result == "Introvert" else "🌞"

        personality_traits = """
            <li>Recharge by spending time alone</li>
            <li>Prefer deep one-on-one conversations</li>
            <li>Think before speaking</li>
            <li>May feel drained after social interactions</li>
        """ if result == "Introvert" else """
            <li>Gain energy from social interactions</li>
            <li>Enjoy group activities</li>
            <li>Think while speaking</li>
            <li>Feel energized after social events</li>
        """

        st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 25px; border-radius: 10px; margin-top: 10px;">
                <h2 style="color:#2C3E50;">{emoji} You are predicted to be an: <span style="color: #0073e6;">{result}</span></h2>
                <p style="font-size:16px;"><strong>Confidence:</strong> {confidence:.2%}</p>
                <p><strong>Personality Traits:</strong></p>
                <ul style="line-height: 1.6;">{personality_traits}</ul>
            </div>
        """, unsafe_allow_html=True)

        # 🔧 Optional debugging info
        with st.expander("🔍 See details (for developers)"):
            st.write("🎯 Label Mapping:", label_encoder.classes_)
            st.write("🔢 Prediction (Encoded):", prediction[0])
            st.write("📊 Prediction Probabilities:", prediction_proba)
            st.write("🎯 Correct Confidence Index:", proba_index)
            st.write("✅ Confidence Used:", f"{confidence:.4f}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")

# ℹ️ Feature explanation
st.markdown("---")
st.markdown("### ℹ️ About the Features")
st.markdown("""
- **⏱️ Time spent alone**: Average hours you enjoy solitude each day  
- **🎤 Stage fear**: Fear or nervousness while speaking in public  
- **🎉 Social event attendance**: Number of events you attend in a month  
- **🚶 Going outside**: Weekly frequency of spending time outside the house  
- **😩 Drained after socializing**: Feeling exhausted post social events  
- **👥 Friends circle size**: Number of close friends you have  
- **📱 Post frequency**: Weekly count of your social media posts  
""")

# Footer
st.markdown("""
---
<div style='text-align: center; padding-top: 10px; color: gray;'>
    Created by <strong>Pasindu Purnamal</strong> with ❤️ using <strong>Streamlit</strong><br>
    © 2025 Personality Predictor. All rights reserved.
</div>
""", unsafe_allow_html=True)