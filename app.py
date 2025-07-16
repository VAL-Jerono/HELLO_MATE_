import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import json

# Config
st.set_page_config(page_title="💘 Dating Match Predictor", layout="wide")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/final_optimized_xgb_model.pkl')
        with open('model/selected_features.json', 'r') as f:
            feature_info = json.load(f)
        return model, feature_info
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

def engineer_features(input_dict):
    """Converts raw input features into model-ready features"""
    # Calculate all required features
    features = {
        'personality_alignment': (input_dict['intel']*input_dict['intel_o'] + 
                               input_dict['fun']*input_dict['fun_o'] + 
                               input_dict['amb']*input_dict['amb_o']) / 30,
        'total_attr_compat': (input_dict['attr'] + input_dict['attr_o']) / 2,
        'like_o': input_dict['like_o'],
        'samerace': input_dict['samerace'],
        'ambition_gap': abs(input_dict['amb'] - input_dict['amb_o']),
        'age_compatibility': 1 / (1 + abs(input_dict['age'] - input_dict['age_o'])),
        'attr_perception_gap': abs(input_dict['attr'] - input_dict['attr_o']),
        'gender': input_dict['gender'],
        'prob_o': input_dict['prob_o'],
        'age_o': input_dict['age_o'],
        'int_corr': input_dict['int_corr'],
        'age': input_dict['age'],
        'attr_gap': input_dict['attr_o'] - input_dict['attr'],
        'gender_attr_interaction': input_dict['gender'] * (input_dict['attr'] + input_dict['attr_o']) / 2
    }
    
    # Must match the EXACT order used during training
    feature_order = [
        'personality_alignment', 'total_attr_compat', 'like_o', 'samerace',
        'ambition_gap', 'age_compatibility', 'attr_perception_gap', 'gender',
        'prob_o', 'age_o', 'int_corr', 'age', 'attr_gap', 'gender_attr_interaction'
    ]
    
    # Return both the ordered features AND the raw input data
    return [features[feature] for feature in feature_order], features

def explain_prediction(prob, raw_features, feature_importance):
    percent = round(prob * 100, 1)
    
    # Get top 3 influential factors
    sorted_features = sorted(
        zip(feature_importance.keys(), feature_importance.values()),
        key=lambda x: x[1], 
        reverse=True
    )[:3]
    
    factors = []
    for feat, imp in sorted_features:
        if feat == 'personality_alignment':
            score = raw_features['personality_alignment']
            factors.append(f"personality match ({score:.1f}/10)")
        elif feat == 'total_attr_compat':
            score = raw_features['total_attr_compat']
            factors.append(f"attractiveness compatibility ({score:.1f}/10)")
        elif feat == 'int_corr':
            score = raw_features['int_corr']
            factors.append(f"interest correlation ({score:.2f})")
        elif feat == 'age_compatibility':
            gap = abs(1/raw_features['age_compatibility'] - 1)  # Convert back to age gap
            factors.append(f"age gap ({int(gap)} years)")
    
    factors_str = ", ".join(factors)
    
    if percent >= 75:
        emoji = "😍"
        adjective = "excellent"
        advice = "This looks like a great match! Consider reaching out."
    elif percent >= 50:
        emoji = "😊"
        adjective = "good"
        advice = "Potential for a good connection. Worth exploring further."
    elif percent >= 30:
        emoji = "🤔"
        adjective = "moderate"
        advice = "Some compatibility exists but may require more effort."
    else:
        emoji = "💔"
        adjective = "low"
        advice = "Limited compatibility detected. You might want to keep looking."
    
    explanation = (
        f"{emoji} **Match Probability: {percent}%** ({adjective})\n\n"
        f"**Key Factors:** {factors_str}\n\n"
        f"**Advice:** {advice}"
    )
    
    return explanation, percent

def display_feature_importance(feature_importance):
    df = pd.DataFrame({
        'Feature': feature_importance.keys(),
        'Importance': feature_importance.values()
    }).sort_values('Importance', ascending=False)
    
    st.subheader("🧠 What Matters Most")
    st.dataframe(df.style.format({'Importance': '{:.3f}'}), height=400)
    
    st.write("""
    *Understanding the factors:*
    - **Personality Alignment**: How well your personalities match
    - **Total Attr Compat**: Combined attractiveness score
    - **Int Corr**: Shared interests correlation
    - **Age Compatibility**: Inverse of age difference
    """)

def main():
    st.title("💘 Scientific Dating Match Predictor")
    st.write("""
    This predictor uses machine learning to analyze compatibility factors based on real psychological research.
    """)
    
    model, feature_info = load_model()
    if model is None:
        return
    
    with st.sidebar:
        st.header("👤 Your Profile")
        gender = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        age = st.slider("Age", 18, 60, 30)
        attr = st.slider("Attractiveness (1-10)", 1, 10, 7, 
                        help="How physically attractive would others rate you?")
        intel = st.slider("Intelligence (1-10)", 1, 10, 7)
        fun = st.slider("Fun (1-10)", 1, 10, 7)
        amb = st.slider("Ambition (1-10)", 1, 10, 7)
        
        st.header("💑 Partner Profile")
        age_o = st.slider("Partner Age", 18, 60, 30)
        attr_o = st.slider("Partner Attractiveness (1-10)", 1, 10, 7)
        intel_o = st.slider("Partner Intelligence (1-10)", 1, 10, 7)
        fun_o = st.slider("Partner Fun (1-10)", 1, 10, 7)
        amb_o = st.slider("Partner Ambition (1-10)", 1, 10, 7)
        
        st.header("🤝 Compatibility Factors")
        samerace = st.checkbox("Same Race", value=True)
        int_corr = st.slider("Interest Correlation", 0.0, 1.0, 0.5, 0.01,
                            help="How similar your interests are (0-1 scale)")
        like_o = st.slider("How Much You Like Partner", 1, 10, 5)
        prob_o = st.slider("Probability Partner Likes You", 0.0, 1.0, 0.5, 0.01)
        
        if st.checkbox("Show Advanced Options"):
            cluster = st.selectbox("Your Personality Cluster", options=range(5))
            attr1_s = st.selectbox("Partner Personality Cluster", options=range(5))
        else:
            cluster = 0
            attr1_s = 0
    
    if st.button("🔮 Calculate Match Probability", use_container_width=True):
        with st.spinner("Analyzing compatibility..."):
            input_data = {
                'gender': gender,
                'age': age,
                'age_o': age_o,
                'samerace': int(samerace),
                'int_corr': int_corr,
                'attr': attr,
                'attr_o': attr_o,
                'intel': intel,
                'intel_o': intel_o,
                'fun': fun,
                'fun_o': fun_o,
                'amb': amb,
                'amb_o': amb_o,
                'like_o': like_o,
                'prob_o': prob_o
            }

            try:
                features, raw_features = engineer_features(input_data)
                # Convert to 2D array (1 sample × 14 features)
                features_array = [features]
                probability = model.predict_proba(features_array)[0][1]
                
                st.success("Analysis Complete!")
                st.balloons()
                
                explanation, percent = explain_prediction(
                    probability, 
                    raw_features,
                    feature_info['feature_importance']
                )
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("## 💌 Match Report")
                    st.markdown(explanation)
                    
                    # Visual gauge - placeholder text if images aren't available
                    try:
                        gauge = Image.open('gauge.png') if percent >= 50 else Image.open('gauge_low.png')
                        st.image(gauge, caption=f"Match Quality: {percent}%", width=300)
                    except:
                        st.write(f"Visual gauge: {percent}% match")
                
                with col2:
                    display_feature_importance(feature_info['feature_importance'])
                
                with st.expander("📊 Technical Details"):
                    st.write("Model Input Features:", features)
                    st.write("Raw Values:", raw_features)
                    st.write(f"Model Used: Optimized XGBoost (AUC-PR: 0.6376)")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("""
                Common issues:
                - Missing required input fields
                - Invalid value ranges
                - Model version mismatch
                """)

if __name__ == "__main__":
    main()