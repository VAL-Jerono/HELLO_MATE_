# 💘 Dating Match Predictor: Decoding the Science of Connection

This project uses data science and machine learning to explore the factors that drive romantic connections. By analyzing a comprehensive speed dating dataset, we build a predictive model that estimates the probability of a match between two people. The final result is an interactive web application built with Streamlit.

## 📝 Project Overview

The modern dating world is complex. This project aims to cut through the noise by answering key questions with data:
- What traits do people prioritize when meeting someone new?
- What are the key ingredients for a long-lasting relationship?
- Can we build a model to accurately predict if two people will match?

## 📊 The Data

This project utilizes two primary datasets:
1.  **Speed Dating Data:** Contains attributes and ratings from thousands of speed dating interactions, including self-perceptions, partner ratings, and final match decisions.
2.  **Marriage/Divorce Data:** Provides insights into the factors that contribute to relationship success and failure over the long term.

## 🛠️ Project Pipeline

The project follows a standard data science workflow:

1.  **Exploratory Data Analysis (EDA):**
    -   Analyzed what traits (Attractiveness, Intelligence, Fun, etc.) are most valued.
    -   Investigated gender differences in selectivity.
    -   Identified key factors for long-term relationship success (e.g., Common Interests, Loyalty).

2.  **Feature Engineering:**
    -   This was a critical step to move beyond raw data. We created powerful new features to capture the *interaction* between partners:
        -   `personality_alignment`: A composite score of mutual intelligence and fun ratings.
        -   `total_attr_compat`: The combined attractiveness score between two people.
        -   `age_compatibility`: A score that decreases as the age gap grows.
        -   `ambition_gap`: The absolute difference in ambition ratings.

3.  **Model Development & Training:**
    -   **Challenge:** The dataset is highly imbalanced (only ~17% of interactions result in a match).
    -   **Initial Models:** Started with a baseline Random Forest, which struggled with the class imbalance (poor recall for matches).
    -   **Advanced Techniques:**
        -   Used **SMOTE** (Synthetic Minority Over-sampling Technique) to create more "match" examples for the model to learn from.
        -   Trained an **XGBoost Classifier**, which is excellent for handling complex, tabular data.
        -   Implemented **class weighting** (`scale_pos_weight`) to force the model to pay more attention to the minority class (matches).
    -   **Hyperparameter Tuning:** Used `RandomizedSearchCV` to find the optimal settings for the XGBoost model, maximizing its predictive power (specifically the AUC-ROC score).

4.  **Model Evaluation:**
    -   The final model achieved an **AUC-ROC of 0.92**, indicating excellent predictive capability.
    -   It successfully balanced precision and recall, correctly identifying over **70% of actual matches**—a huge improvement from the baseline.

5.  **Deployment:**
    -   The trained XGBoost model, along with the list of selected features, was saved using `joblib` and `json`.
    -   An interactive web application was built using **Streamlit** to allow users to input their profiles and get a live match prediction.

## 🚀 How to Run the App

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file containing libraries like streamlit, pandas, scikit-learn, xgboost, imblearn, etc.)*

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

4.  **Open your browser** and navigate to the local URL provided (usually `http://localhost:8501` ).

## 🧠 Key Findings

-   **First Impressions are about more than looks:** While attractiveness is the initial hook, a fun and intelligent personality is what people truly value.
-   **Compatibility is a two-way street:** The most predictive features were not individual traits but *interaction effects*—how two people's attributes align.
-   **Communication is king:** The biggest reasons for relationship failure are ego and poor communication, not a lack of love or shared interests.
