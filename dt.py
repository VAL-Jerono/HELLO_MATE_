import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import json
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="💘 Dating Match Predictor", layout="wide")
st.title("💘 Scientific Dating Match Predictor")
st.write("This predictor uses machine learning to analyze compatibility factors based on real psychological research.")

# --- Load Model & Feature Info ---
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

# --- Feature Engineering ---
def engineer_features(input_dict):
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

    feature_order = [
        'personality_alignment', 'total_attr_compat', 'like_o', 'samerace',
        'ambition_gap', 'age_compatibility', 'attr_perception_gap', 'gender',
        'prob_o', 'age_o', 'int_corr', 'age', 'attr_gap', 'gender_attr_interaction'
    ]
    return [features[feature] for feature in feature_order], features

# --- Explain Prediction ---
def explain_prediction(prob, raw_features, feature_importance):
    percent = round(prob * 100, 1)
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    factors = []

    for feat, _ in sorted_features:
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
            gap = abs(1/raw_features['age_compatibility'] - 1)
            factors.append(f"age gap ({int(gap)} years)")

    factors_str = ", ".join(factors)

    if percent >= 75:
        emoji, adjective, advice = "😍", "excellent", "This looks like a great match! Consider reaching out."
    elif percent >= 50:
        emoji, adjective, advice = "😊", "good", "Potential for a good connection. Worth exploring further."
    elif percent >= 30:
        emoji, adjective, advice = "🤔", "moderate", "Some compatibility exists but may require more effort."
    else:
        emoji, adjective, advice = "💔", "low", "Limited compatibility detected. You might want to keep looking."

    explanation = (
        f"{emoji} **Match Probability: {percent}%** ({adjective})\n\n"
        f"**Key Factors:** {factors_str}\n\n"
        f"**Advice:** {advice}"
    )
    return explanation, percent

# --- Feature Importance Display ---
def display_feature_importance(feature_importance):
    df = pd.DataFrame({
        'Feature': feature_importance.keys(),
        'Importance': feature_importance.values()
    }).sort_values('Importance', ascending=False)

    st.subheader("🧠 Feature Importance")
    st.dataframe(df.style.format({'Importance': '{:.3f}'}), height=400)

# --- Radar Chart ---
def plot_radar_chart(you, partner):
    traits = ['Attraction', 'Intelligence', 'Fun', 'Ambition']
    angles = np.linspace(0, 2*np.pi, len(traits), endpoint=False).tolist()
    you += [you[0]]
    partner += [partner[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
    ax.plot(angles, you, 'o-', label='You')
    ax.plot(angles, partner, 'o-', label='Partner')
    ax.fill(angles, you, alpha=0.25)
    ax.fill(angles, partner, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), traits)
    ax.set_title("Personality Trait Comparison")
    ax.legend()
    return fig

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("👤 Your Profile")
    gender = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    age = st.slider("Age", 18, 60, 30)
    attr = st.slider("Attractiveness", 1, 10, 7)
    intel = st.slider("Intelligence", 1, 10, 7)
    fun = st.slider("Fun", 1, 10, 7)
    amb = st.slider("Ambition", 1, 10, 7)

    st.header("💑 Partner Profile")
    age_o = st.slider("Partner Age", 18, 60, 30)
    attr_o = st.slider("Partner Attractiveness", 1, 10, 7)
    intel_o = st.slider("Partner Intelligence", 1, 10, 7)
    fun_o = st.slider("Partner Fun", 1, 10, 7)
    amb_o = st.slider("Partner Ambition", 1, 10, 7)

    st.header("🤝 Compatibility Factors")
    samerace = st.checkbox("Same Race", value=True)
    int_corr = st.slider("Interest Correlation", 0.0, 1.0, 0.5, 0.01)
    like_o = st.slider("How Much You Like Partner", 1, 10, 5)
    prob_o = st.slider("Probability Partner Likes You", 0.0, 1.0, 0.5, 0.01)

# --- Main Logic ---
model, feature_info = load_model()
if model:
    if st.button("🔮 Calculate Match Probability", use_container_width=True):
        with st.spinner("Analyzing compatibility..."):
            input_data = {
                'gender': gender, 'age': age, 'age_o': age_o, 'samerace': int(samerace),
                'int_corr': int_corr, 'attr': attr, 'attr_o': attr_o,
                'intel': intel, 'intel_o': intel_o, 'fun': fun, 'fun_o': fun_o,
                'amb': amb, 'amb_o': amb_o, 'like_o': like_o, 'prob_o': prob_o
            }

            try:
                features, raw_features = engineer_features(input_data)
                features_array = [features]
                probability = model.predict_proba(features_array)[0][1]

                st.success("Analysis Complete!")
                st.balloons()

                explanation, percent = explain_prediction(probability, raw_features, feature_info['feature_importance'])

                col1, col2 = st.columns([1.2, 1])

                with col1:
                    st.markdown("## 💌 Match Report")
                    st.markdown(explanation)

                    fig_bar, ax = plt.subplots(figsize=(6, 1.5))
                    ax.barh(['Match Probability'], [percent], color='#FF4B4B')
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Probability (%)")
                    st.pyplot(fig_bar)

                with col2:
                    st.subheader("📊 Compatibility Breakdown")
                    attr_compat = raw_features['total_attr_compat']
                    personality = raw_features['personality_alignment']
                    age_compat = raw_features['age_compatibility'] * 100

                    st.metric("Attraction Compatibility", f"{attr_compat:.1f}/10")
                    st.metric("Personality Alignment", f"{personality:.1f}/10")
                    st.metric("Age Compatibility", f"{age_compat:.1f}%")

                    fig_radar = plot_radar_chart([attr, intel, fun, amb], [attr_o, intel_o, fun_o, amb_o])
                    st.pyplot(fig_radar)

                    display_feature_importance(feature_info['feature_importance'])

                with st.expander("🧠 Full Feature View"):
                    st.json(raw_features)

                with st.expander("📊 Technical Notes"):
                    st.write("Model: Optimized XGBoost")
                    st.write("Features:", features)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Note: This is a demo app using real speed dating data. Model predictions are for entertainment and exploratory purposes.")
