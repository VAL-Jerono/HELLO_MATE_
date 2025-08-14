import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Enhanced Config with custom styling
st.set_page_config(
    page_title="💘 VonDetta LoveMatch AI - Scientific Dating Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning visuals
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.4);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .compatibility-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(168, 237, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .success-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 2rem;
        border-radius: 20px;
        color: #2c3e50;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(132, 250, 176, 0.4);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 20px;
        color: #8b4513;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(252, 182, 159, 0.4);
    }
    
    .danger-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 2rem;
        border-radius: 20px;
        color: #8b0000;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(255, 154, 158, 0.4);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .sidebar .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e1e8ed;
    }
    
    .sidebar .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .feature-importance-chart {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .animated-heart {
        animation: heartbeat 1.5s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes heartbeat {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(102, 126, 234, 0); }
        100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/final_optimized_xgb_model.pkl')
        with open('model/selected_features.json', 'r') as f:
            feature_info = json.load(f)
        return model, feature_info
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {str(e)}")
        return None, None

def create_animated_gauge(percentage):
    """Creates a beautiful animated gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "💕 Match Compatibility"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "rgba(102, 126, 234, 0.8)"},
            'steps': [
                {'range': [0, 25], 'color': "rgba(255, 154, 158, 0.3)"},
                {'range': [25, 50], 'color': "rgba(252, 182, 159, 0.3)"},
                {'range': [50, 75], 'color': "rgba(168, 237, 234, 0.3)"},
                {'range': [75, 100], 'color': "rgba(132, 250, 176, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#2c3e50", 'family': "Inter"},
        height=400
    )
    return fig

def create_compatibility_radar(raw_features):
    """Creates a radar chart for compatibility factors"""
    categories = ['Personality\nAlignment', 'Attractiveness\nCompatibility', 
                 'Age\nCompatibility', 'Interest\nCorrelation', 'Ambition\nAlignment']
    
    values = [
        raw_features['personality_alignment'] * 10,
        raw_features['total_attr_compat'],
        raw_features['age_compatibility'] * 10,
        raw_features['int_corr'] * 10,
        10 - raw_features['ambition_gap']  # Invert so higher is better
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Compatibility',
        line_color='rgba(102, 126, 234, 0.8)',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                gridcolor="rgba(255, 255, 255, 0.3)",
                linecolor="rgba(255, 255, 255, 0.3)"
            ),
            angularaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.3)",
                linecolor="rgba(255, 255, 255, 0.3)"
            )
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#2c3e50", 'family': "Inter"},
        height=500
    )
    return fig

def create_feature_importance_chart(feature_importance):
    """Creates an interactive feature importance chart"""
    df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=True)
    
    # Create beautiful feature names
    feature_names = {
        'personality_alignment': '🧠 Personality Match',
        'total_attr_compat': '✨ Attractiveness Sync',
        'int_corr': '🎯 Interest Alignment',
        'age_compatibility': '📅 Age Harmony',
        'like_o': '💖 Your Attraction Level',
        'samerace': '🌍 Cultural Similarity',
        'ambition_gap': '🚀 Ambition Balance',
        'prob_o': '💫 Mutual Interest Probability'
    }
    
    df['Pretty_Name'] = df['Feature'].map(lambda x: feature_names.get(x, x.replace('_', ' ').title()))
    
    fig = px.bar(df, x='Importance', y='Pretty_Name', orientation='h',
                 title="🔍 What Makes You Compatible?",
                 color='Importance',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#2c3e50", 'family': "Inter"},
        height=600,
        showlegend=False,
        title_font_size=20,
        title_x=0.5
    )
    
    fig.update_traces(
        marker_line_color='rgba(255,255,255,0.8)',
        marker_line_width=1.5,
        opacity=0.8
    )
    
    return fig

def engineer_features(input_dict):
    """Converts raw input features into model-ready features"""
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

def explain_prediction(prob, raw_features, feature_importance):
    percent = round(prob * 100, 1)
    
    sorted_features = sorted(
        zip(feature_importance.keys(), feature_importance.values()),
        key=lambda x: x[1], 
        reverse=True
    )[:3]
    
    factors = []
    for feat, imp in sorted_features:
        if feat == 'personality_alignment':
            score = raw_features['personality_alignment']
            factors.append(f"🧠 personality harmony ({score:.1f}/1.0)")
        elif feat == 'total_attr_compat':
            score = raw_features['total_attr_compat']
            factors.append(f"✨ attractiveness sync ({score:.1f}/10)")
        elif feat == 'int_corr':
            score = raw_features['int_corr']
            factors.append(f"🎯 shared interests ({score:.2f})")
        elif feat == 'age_compatibility':
            gap = abs(1/raw_features['age_compatibility'] - 1) if raw_features['age_compatibility'] > 0 else 0
            factors.append(f"📅 age harmony ({gap:.0f} year gap)")
    
    factors_str = " • ".join(factors)
    
    if percent >= 75:
        return f"""
        <div class="success-card">
            <h2>🎉 Excellent Match! ({percent}%)</h2>
            <p><strong>Key Strengths:</strong> {factors_str}</p>
            <p>✨ This looks like a fantastic connection! The stars are aligned for romance. Consider reaching out with confidence!</p>
        </div>
        """, "success"
    elif percent >= 50:
        return f"""
        <div class="compatibility-card">
            <h2>😊 Good Potential! ({percent}%)</h2>
            <p><strong>Compatibility Factors:</strong> {factors_str}</p>
            <p>💫 There's genuine potential here! While not perfect, this match shows promise worth exploring further.</p>
        </div>
        """, "good"
    elif percent >= 30:
        return f"""
        <div class="warning-card">
            <h2>🤔 Moderate Compatibility ({percent}%)</h2>
            <p><strong>Mixed Signals:</strong> {factors_str}</p>
            <p>⚖️ Some compatibility exists, but success may require extra effort and understanding from both sides.</p>
        </div>
        """, "moderate"
    else:
        return f"""
        <div class="danger-card">
            <h2>💔 Limited Compatibility ({percent}%)</h2>
            <p><strong>Challenges:</strong> {factors_str}</p>
            <p>🔍 This match faces significant compatibility challenges. You might want to explore other options.</p>
        </div>
        """, "low"

def create_profile_summary(input_data):
    """Creates a beautiful profile summary"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Your Profile")
        st.markdown(f"""
        <div class="metric-card">
            <h4>{"👩" if input_data['gender'] == 0 else "👨"} {input_data['age']} years old</h4>
            <p>Attractiveness: {'⭐' * input_data['attr']}</p>
            <p>Intelligence: {'🧠' * input_data['intel']}</p>
            <p>Fun Level: {'🎉' * input_data['fun']}</p>
            <p>Ambition: {'🚀' * input_data['amb']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💑 Partner Profile")
        st.markdown(f"""
        <div class="metric-card">
            <h4>👤 {input_data['age_o']} years old</h4>
            <p>Attractiveness: {'⭐' * input_data['attr_o']}</p>
            <p>Intelligence: {'🧠' * input_data['intel_o']}</p>
            <p>Fun Level: {'🎉' * input_data['fun_o']}</p>
            <p>Ambition: {'🚀' * input_data['amb_o']}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Beautiful header
    st.markdown("""
    <div class="main-header">
        <h1><span class="animated-heart">💘</span>  VonDetta LoveMatch AI</h1>
        <p>Discover your romantic compatibility with cutting-edge machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    model, feature_info = load_model()
    if model is None:
        st.error("🚨 Unable to load the AI model. Please check your model files.")
        return
    
    with st.sidebar:
        st.markdown("### 🎭 Create Your Dating Profile")
        
        st.markdown("#### 👤 About You")
        gender = st.radio("Gender", [0, 1], format_func=lambda x: "👩 Female" if x == 0 else "👨 Male")
        age = st.slider("Age", 18, 60, 30, help="Your current age")
        
        st.markdown("#### ✨ Your Qualities (1-10 scale)")
        attr = st.slider("💅 Attractiveness", 1, 10, 7, help="How physically attractive would others rate you?")
        intel = st.slider("🧠 Intelligence", 1, 10, 7, help="Your intellectual capabilities")
        fun = st.slider("🎉 Fun Factor", 1, 10, 7, help="How entertaining and enjoyable you are")
        amb = st.slider("🚀 Ambition", 1, 10, 7, help="Your drive and career aspirations")
        
        st.markdown("---")
        st.markdown("#### 💑 Your Ideal Partner")
        age_o = st.slider("Partner Age", 18, 60, 30)
        
        st.markdown("#### ✨ Partner Qualities (1-10 scale)")
        attr_o = st.slider("💅 Partner Attractiveness", 1, 10, 7)
        intel_o = st.slider("🧠 Partner Intelligence", 1, 10, 7)
        fun_o = st.slider("🎉 Partner Fun Factor", 1, 10, 7)
        amb_o = st.slider("🚀 Partner Ambition", 1, 10, 7)
        
        st.markdown("---")
        st.markdown("#### 🤝 Relationship Dynamics")
        samerace = st.checkbox("🌍 Same Cultural Background", value=True)
        int_corr = st.slider("🎯 Interest Similarity", 0.0, 1.0, 0.5, 0.01,
                           help="How similar are your interests and hobbies?")
        like_o = st.slider("💖 Your Attraction Level", 1, 10, 5,
                          help="How much do you like this person?")
        prob_o = st.slider("💫 Mutual Interest Probability", 0.0, 1.0, 0.5, 0.01,
                          help="What's the chance they like you back?")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 🎯 Quick Stats")
        age_gap = abs(age - age_o)
        attr_gap = abs(attr - attr_o)
        
        st.metric("Age Gap", f"{age_gap} years", 
                 delta=f"{'Low' if age_gap <= 3 else 'High'} impact")
        st.metric("Attractiveness Gap", f"{attr_gap} points",
                 delta=f"{'Well matched' if attr_gap <= 2 else 'Different levels'}")
        st.metric("Interest Alignment", f"{int_corr:.0%}",
                 delta=f"{'Strong' if int_corr > 0.7 else 'Moderate' if int_corr > 0.4 else 'Weak'}")
    
    with col1:
        if st.button("🔮 Analyze Compatibility", use_container_width=True, help="Click to predict your match!"):
            with st.spinner("🔍 Analyzing your romantic compatibility..."):
                # Add dramatic pause for effect
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                input_data = {
                    'gender': gender, 'age': age, 'age_o': age_o,
                    'samerace': int(samerace), 'int_corr': int_corr,
                    'attr': attr, 'attr_o': attr_o,
                    'intel': intel, 'intel_o': intel_o,
                    'fun': fun, 'fun_o': fun_o,
                    'amb': amb, 'amb_o': amb_o,
                    'like_o': like_o, 'prob_o': prob_o
                }

                try:
                    features, raw_features = engineer_features(input_data)
                    features_array = [features]
                    probability = model.predict_proba(features_array)[0][1]
                    
                    progress_bar.empty()
                    st.balloons()
                    
                    # Main result display
                    explanation, match_type = explain_prediction(
                        probability, raw_features, feature_info['feature_importance']
                    )
                    
                    st.markdown(explanation, unsafe_allow_html=True)
                    
                    # Create tabs for detailed analysis
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Compatibility Gauge", "🕸️ Compatibility Radar", "📈 Key Factors", "👥 Profile Summary"])
                    
                    with tab1:
                        gauge_fig = create_animated_gauge(probability * 100)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        if match_type == "success":
                            st.success("🎉 Congratulations! This is an exceptional match!")
                        elif match_type == "good":
                            st.info("😊 This shows genuine potential for a great relationship!")
                        elif match_type == "moderate":
                            st.warning("🤔 There are some compatibility challenges to consider.")
                        else:
                            st.error("💔 This match may face significant hurdles.")
                    
                    with tab2:
                        radar_fig = create_compatibility_radar(raw_features)
                        st.plotly_chart(radar_fig, use_container_width=True)
                        st.markdown("""
                        **Understanding Your Compatibility Radar:**
                        - **Larger area = Better overall compatibility**
                        - Each axis represents a key compatibility factor
                        - Balanced shapes indicate well-rounded matches
                        """)
                    
                    with tab3:
                        importance_fig = create_feature_importance_chart(feature_info['feature_importance'])
                        st.plotly_chart(importance_fig, use_container_width=True)
                        
                        st.markdown("### 🔬 Science Behind the Prediction")
                        st.info("""
                        This AI model was trained on real dating data using advanced machine learning. 
                        The algorithm considers personality psychology, social compatibility theory, and 
                        statistical relationship patterns to make predictions.
                        """)
                    
                    with tab4:
                        create_profile_summary(input_data)
                        
                        with st.expander("🔧 Technical Details"):
                            st.json({
                                "Model Type": "Optimized XGBoost",
                                "Performance": "AUC-PR: 0.6376",
                                "Features Used": len(features),
                                "Prediction Confidence": f"{probability:.3f}",
                                "Raw Feature Values": {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in raw_features.items()}
                            })
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    with st.expander("🔍 Troubleshooting"):
                        st.markdown("""
                        **Common issues:**
                        - Missing model files (check `model/` directory)
                        - Invalid input ranges
                        - Feature engineering errors
                        
                        **Quick fixes:**
                        - Ensure all sliders have valid values
                        - Check that model files are properly loaded
                        - Verify feature names match training data
                        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
        <p>💡 <strong>LoveMatch AI</strong> - Where science meets romance</p>
        <p>Built with machine learning, powered by data science, inspired by love 💕</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()