import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Config
st.set_page_config(page_title=" 💘 Dating Match Predictor", layout="wide")

@st.cache_resource
def load_model():
    with open('model/xgb_model.pkl', 'rb') as f:
        return pickle.load(f)

def engineer_features(input_dict):
    # Calculate all potential features
    features = {
        # Direct features
        'gender': input_dict['gender'],
        'age': input_dict['age'],
        'age_o': input_dict['age_o'],
        'samerace': input_dict['samerace'],
        'int_corr': input_dict['int_corr'],
        'attr_gap': input_dict['attr_gap'],
        'like_o': input_dict['like_o'],
        'prob_o': input_dict['prob_o'],
        'dec_o': input_dict['dec_o'],
        
        # Engineered features
        'total_attr_compat': (input_dict['attr'] * input_dict['attr_o']) / 10,
        'personality_alignment': (input_dict['intel']*input_dict['intel_o'] + 
                                 input_dict['fun']*input_dict['fun_o']) / 20,
        'attr_perception_gap': abs(input_dict['attr'] - input_dict['attr_o']),
        'ambition_gap': abs(input_dict['amb'] - input_dict['amb_o']),
        'gender_attr_interaction': input_dict['gender'] * input_dict['attr_gap'],
        'age_compatibility': 1 / (1 + abs(input_dict['age'] - input_dict['age_o'])),
        'cluster_attr_match': int(input_dict.get('cluster', 0) == input_dict.get('attr1_s', 0))
    }
    
    # Create DataFrame and select only the features the model expects
    df = pd.DataFrame([features])
    
    # Get the features your model actually expects
    expected_features = [
        'gender', 'age', 'age_o', 'samerace', 'int_corr',
        'total_attr_compat', 'personality_alignment',
        'attr_perception_gap', 'ambition_gap',
        'gender_attr_interaction', 'age_compatibility',
        'cluster_attr_match', 'attr_gap'
    ]
    
    # Fix any naming inconsistencies
    if 'gender_attr_interaction' not in df.columns and 'gender_attr_interaction' in expected_features:
        df['gender_attr_interaction'] = df['gender_attr_interaction']
    
    return df[expected_features]

def main():
    st.title("💘 Dating Match Prediction")
    model = load_model()
    
    with st.sidebar:
        st.header("Input Parameters")
        
        # Personal Attributes
        gender = st.radio("Your Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        age = st.slider("Your Age", 18, 60, 30)
        attr = st.slider("Your Attractiveness (1-10)", 1, 10, 7)
        intel = st.slider("Your Intelligence (1-10)", 1, 10, 7)
        fun = st.slider("Your Fun (1-10)", 1, 10, 7)
        amb = st.slider("Your Ambition (1-10)", 1, 10, 7)
        
        # Partner Attributes
        age_o = st.slider("Partner Age", 18, 60, 30)
        attr_o = st.slider("Partner Attractiveness (1-10)", 1, 10, 7)
        intel_o = st.slider("Partner Intelligence (1-10)", 1, 10, 7)
        fun_o = st.slider("Partner Fun (1-10)", 1, 10, 7)
        amb_o = st.slider("Partner Ambition (1-10)", 1, 10, 7)
        
        # Additional Features
        samerace = st.checkbox("Same Race", value=True)
        int_corr = st.slider("Interest Correlation", 0.0, 1.0, 0.5, 0.01)
        like_o = st.slider("How Much You Like Partner", 1, 10, 5)
        prob_o = st.slider("Probability Partner Likes You", 0.0, 1.0, 0.5, 0.01)
        dec_o = st.slider("Partner Decision Score", 1, 10, 5)
        
        # Cluster features (optional)
        if st.checkbox("Advanced Cluster Features"):
            cluster = st.selectbox("Your Cluster", options=range(5))
            attr1_s = st.selectbox("Partner Cluster", options=range(5))
        else:
            cluster = 0
            attr1_s = 0
    
    if st.button("Predict Match Probability"):
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
            'attr_gap': abs(attr - attr_o),
            'cluster': cluster,
            'attr1_s': attr1_s,
            'like_o': like_o,
            'prob_o': prob_o,
            'dec_o': dec_o
        }
        
        try:
            features = engineer_features(input_data)
            prediction = model.predict_proba(features)[0][1]
            
            st.success(f"Match Probability: {prediction*100:.1f}%")
            
            with st.expander("Feature Details"):
                st.write("Engineered Features:", features)
                st.write("Feature Importance:", model.feature_importances_)
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.error("Common issues:")
            st.error("- Feature name mismatches")
            st.error("- Missing required features")
            st.error("- Incorrect feature calculations")

if __name__ == "__main__":
    main()