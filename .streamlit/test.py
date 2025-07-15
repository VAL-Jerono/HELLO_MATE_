# test.py
import joblib
import pandas as pd
from sklearn.metrics import classification_report

def test_model():
    model = joblib.load('model.pkl')
    test_data = pd.read_csv("test_data.csv")
    
    # Run predictions
    X_test = test_data.drop('match', axis=1)
    y_pred = model.predict(X_test)
    
    print(classification_report(test_data['match'], y_pred))

if __name__ == "__main__":
    test_model()