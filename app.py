
from flask import Flask, render_template, request
import joblib

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    text = request.form['news']
    vect_text = vectorizer.transform([text])
    try:
        prob = model.predict_proba(vect_text)[0][1]
    except:
        prob = 1 if model.predict(vect_text)[0] == 1 else 0
    prediction = model.predict(vect_text)[0]
    label = "REAL" if prediction == 1 else "FAKE"
    confidence = round(prob * 100, 2)
    return render_template("index.html", prediction=prediction, prediction_label=label,
                           confidence=confidence, input_text=text)

if __name__ == "__main__":
    app.run(debug=True)
