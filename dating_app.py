import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt




# Set up the app
st.set_page_config(page_title="Dating Match Predictor", layout="wide")
st.title("💘 Dating Match Prediction Tool")

# Load model and components (in a real app, these would be loaded from files)
# For demo purposes, we'll assume these are available
# model = pickle.load(open('xgb_model.pkl', 'rb'))
# pca = pickle.load(open('pca.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))

# For this example, we'll create mock functions
def predict_match(features):
    """Mock prediction function - replace with actual model"""
    # In real implementation:
    # features_scaled = scaler.transform(features)
    # features_pca = pca.transform(features_scaled)
    # return model.predict_proba(features_pca)[:,1]
    return np.random.random()

# Sidebar for user input
with st.sidebar:
    st.header("Input Parameters")
    
    gender = st.radio("Your Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    age = st.slider("Your Age", 18, 60, 30)
    age_o = st.slider("Partner's Age", 18, 60, 30)
    samerace = st.checkbox("Same Race", value=True)
    
    st.subheader("Your Ratings (1-10)")
    attr = st.slider("Attractiveness", 1, 10, 5)
    intel = st.slider("Intelligence", 1, 10, 5)
    fun = st.slider("Fun", 1, 10, 5)
    amb = st.slider("Ambition", 1, 10, 5)
    
    st.subheader("Partner's Ratings (1-10)")
    attr_o = st.slider("Partner Attractiveness", 1, 10, 5)
    intel_o = st.slider("Partner Intelligence", 1, 10, 5)
    fun_o = st.slider("Partner Fun", 1, 10, 5)
    amb_o = st.slider("Partner Ambition", 1, 10, 5)
    
    int_corr = st.slider("Interest Correlation", 0.0, 1.0, 0.5, 0.01)
    attr_gap = st.slider("Attractiveness Gap", 0, 10, 2)
    interaction = st.slider("Interaction Score", 0, 10, 5)
    like_o = st.slider("How Much You Like Partner", 0, 10, 5)
    prob_o = st.slider("Probability Partner Likes You", 0.0, 1.0, 0.5, 0.01)
    dec_o = st.slider("Partner Decision Score", 0, 10, 5)
    cluster_attr = st.selectbox("Cluster Attribute Match", [0, 1])

# Create feature dataframe
def create_features():
    features = {
        'gender': gender,
        'age': age,
        'age_o': age_o,
        'samerace': int(samerace),
        'int_corr': int_corr,
        'total_attr_compat': (attr * attr_o) / 10,
        'personality_alignment': (intel * intel_o + fun * fun_o) / 20,
        'attr_perception_gap': abs(attr - attr_o),
        'ambition_gap': abs(amb - amb_o),
        'gender_attr_interaction': gender * attr_gap,
        'age_compatibility': 1 / (1 + abs(age - age_o)),
        'cluster_attr_match': cluster_attr,
        'attr_gap': attr_gap,
        'interaction': interaction,
        'like_o': like_o,
        'prob_o': prob_o,
        'dec_o': dec_o
    }
    return pd.DataFrame([features])

# Main app
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Match Prediction")
    
    if st.button("Predict Match Probability"):
        input_df = create_features()
        
        # Display the engineered features
        with st.expander("View Engineered Features"):
            st.dataframe(input_df)
        
        # Make prediction
        prediction = predict_match(input_df)
        match_prob = prediction[0] * 100
        
        # Display result
        st.metric("Match Probability", f"{match_prob:.1f}%")
        
        # Visual gauge
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(['Match'], [match_prob], color='#FF4B4B')
        ax.set_xlim(0, 100)
        ax.set_title("Match Probability")
        st.pyplot(fig)
        
        # Interpretation
        if match_prob > 70:
            st.success("🔥 High compatibility! Good chances for a match.")
        elif match_prob > 40:
            st.warning("💡 Moderate compatibility. Worth exploring further.")
        else:
            st.info("🤔 Lower compatibility. Consider other factors.")

with col2:
    st.subheader("Compatibility Breakdown")
    
    # Calculate compatibility scores
    attr_compat = (attr * attr_o) / 10
    personality_align = (intel * intel_o + fun * fun_o) / 20
    age_comp = 1 / (1 + abs(age - age_o)) * 100
    
    st.metric("Attraction Compatibility", f"{attr_compat*10:.1f}/10")
    st.metric("Personality Alignment", f"{personality_align*10:.1f}/10")
    st.metric("Age Compatibility", f"{age_comp:.1f}%")
    
    # Radar chart for personality traits
    traits = ['Attraction', 'Intelligence', 'Fun', 'Ambition']
    you = [attr, intel, fun, amb]
    partner = [attr_o, intel_o, fun_o, amb_o]
    
    angles = np.linspace(0, 2*np.pi, len(traits), endpoint=False)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, you, 'o-', label="You")
    ax.plot(angles, partner, 'o-', label="Partner")
    ax.fill(angles, you, alpha=0.25)
    ax.fill(angles, partner, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, traits)
    ax.set_title("Personality Trait Comparison")
    ax.legend()
    st.pyplot(fig)

# Add some explanations
with st.expander("How this prediction works"):
    st.write("""
    This model uses advanced machine learning to predict dating compatibility based on:
    - Personal attributes and preferences
    - Compatibility scores across multiple dimensions
    - Perception gaps between partners
    - Demographic and behavioral factors
    
    The model was trained on real-world speed dating data using XGBoost with PCA dimensionality reduction.
    """)
    
    st.image("https://miro.medium.com/max/1400/1*Z54JgbS4DUwWSknhDCvNTQ.png", 
             caption="Model Architecture Overview", width=400)

# Add footer
st.markdown("---")
st.caption("Note: This is a demo application. For real-world use, the model should be properly trained and validated.")