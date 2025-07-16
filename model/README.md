
# 💘 VON Dating Match Predictor

Welcome to the **VON Dating Match Predictor** — a machine learning-powered web application that evaluates compatibility between individuals based on real psychological research and personal preferences.

This project is playful but rooted in real data science, built with **Streamlit**, and demonstrates the full cycle of deploying a trained model to production.

---

## 🎯 Overview

**VON Dating Match Predictor** allows users to input personality traits, preferences, or answers to survey-like questions and receive a **compatibility score** with another person. It leverages machine learning to simulate relationship compatibility analysis and presents results in an engaging, visual format.

---

## 🧠 Core Features

- 🚀 **Fast and responsive UI** using [Streamlit](https://streamlit.io/)
- 🧮 **Machine Learning** model trained with [XGBoost](https://xgboost.ai/)
- 🔍 **Feature interpretation** based on real psychological datasets
- 📊 Visual output using `matplotlib` (can be extended with radar charts or PCA)
- 🧠 Backend model serialized with `joblib`
- 🎨 Customisable frontend with colour themes via `.streamlit/config.toml`

---

## 🛠️ Tech Stack

| Area | Technology |
|------|------------|
| Frontend | Streamlit (v1.22.0) |
| Backend | Python 3.11 |
| Machine Learning | XGBoost, Scikit-learn |
| Model Persistence | Joblib |
| Visualisation | Matplotlib |
| Deployment | Streamlit Cloud or local server |
| Package Management | `uv` + `pip` |
| Data Handling | pandas, numpy |

---

## 📦 Installation

### 🔐 Requirements

Make sure you have Python ≥ 3.11 and pip or `uv`.

### 📁 Clone the Repository

```bash
git clone https://github.com/yourusername/hello_mate_.git
cd hello_mate_
````

### 🧰 Set Up the Environment

Using pip:

```bash
pip install -r requirements.txt
```

Or using `uv` (recommended if using Streamlit Cloud):

```bash
uv pip install -r requirements.txt
```

---

## 🏃‍♀️ Running the App

```bash
streamlit run dt.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 🧪 Example Model Input & Output

This app expects numeric or categorical inputs based on the questionnaire structure of the underlying dataset (e.g., hobbies, attractiveness, sincerity, shared interests, etc.).

The model returns:

* A **match prediction score**
* A **confidence level**
* (Optional) **Feature importance visualisation**

---

## 🧯 Challenges Faced

| Challenge                       | Solution                                                                                                   |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: joblib`   | Added `joblib==1.3.2` to `requirements.txt`                                                                |
| TOML config conflict            | Fixed duplicate `[server]` block in `.streamlit/config.toml`                                               |
| Pillow build failure            | Switched to `pillow==11.3.0` for compatibility                                                             |
| Inotify watch limit error       | Caused by excessive folder watching in Linux. Solved via config tuning or by avoiding unnecessary watchers |
| `XGBoost` model loading warning | The model was serialized using `joblib`; ideally, we would use `Booster.save_model()` in future            |

---

## 📁 Directory Structure

```plaintext
hello_mate_/
├── dt.py                       # Main Streamlit app
├── model.pkl                  # Pre-trained XGBoost model
├── requirements.txt           # All required dependencies
├── .streamlit/
│   └── config.toml            # UI theming and server settings
└── README.md                  # You're here!
```

---

## ❤️ Why This App?

This project was inspired by a desire to:

* Combine **psychology and machine learning**
* Make **data science relatable and fun**
* Practice **real-world ML deployment**
* Create a shareable tool that could engage friends or partners

Whether you're testing compatibility with your crush or exploring ML for relationships, this app makes it both fun and insightful.

---

## 🙋🏽‍♀️ About the Author

**Valerie Jerono (Jeron)**
Data Scientist & Underwriting Intern | MSc Data Science & Analytics
Researcher at @iLabAfrica | Lover of music, code, and behavior analysis
🌍 Nairobi, Kenya

---

## 📬 Feedback / Contributions

Feel free to fork this repo, raise an issue, or suggest new features like:

* Radar compatibility charts
* PCA-based clustering for personality types
* A chatbot-style questionnaire UI

---

## 📜 License

MIT License. Feel free to use, remix, and share!

---

## 🌟 Show Your Support

If you like this project:

* ⭐ Star it on GitHub
* 🐦 Share it on Twitter or LinkedIn
* 🍰 Fork it for your own love experiment!

---

> *“Science meets romance — one prediction at a time.”*

