from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle
from datetime import datetime

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Needed for session

# Load model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# ----------------------------
# 1️⃣ Welcome Page
# ----------------------------
@app.route('/')
def welcome():
    return render_template('welcome.html')


@app.route('/start', methods=['POST'])
def start():
    session['patient_name'] = request.form['patient_name']
    session['phone'] = request.form['phone']
    return redirect(url_for('home'))


# ----------------------------
# 2️⃣ Prediction Form Page
# ----------------------------
@app.route('/home')
def home():
    return render_template('index.html')


# ----------------------------
# 3️⃣ Prediction Logic
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    # Convert medical values to float
    features = [float(x) for x in form_data.values()]
    final_features = np.array([features])
    final_features = scaler.transform(final_features)

    prediction = model.predict(final_features)
    probability = model.predict_proba(final_features)[0][1] * 100

    if prediction[0] == 0:
        result = "No Heart Disease Detected."
    else:
        result = "Heart Disease Detected. Please consult a cardiologist."

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return render_template(
        'result.html',
        prediction_text=result,
        probability=f"{probability:.2f}",
        form_data=form_data,
        date=current_date,
        patient_name=session.get('patient_name'),
        phone=session.get('phone')
    )
import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
