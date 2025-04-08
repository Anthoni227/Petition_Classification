from flask import Flask, render_template, request
import pickle
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_mail import Mail, Message

app = Flask(__name__)

# Configure Flask-Mail
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "anthoniv198@gmail.com"  # Replace with sender's email
app.config["MAIL_PASSWORD"] = "gyiwhazmwkhhomep"  # Replace with the app password
app.config["MAIL_DEFAULT_SENDER"] = "no-reply@example.com"  # Keeps sender hidden

mail = Mail(app)

# Load trained models and required objects
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

best_model_dept = load_pickle("department_model.pkl")  # LSTM Model
best_model_urgency = load_pickle("urgency_model.pkl")  # LSTM Model
tokenizer = load_pickle("tokenizer.pkl")  # Tokenizer
label_encoder_dept = load_pickle("label_encoder_dept.pkl")  # Label Encoder
label_encoder_urgency = load_pickle("label_encoder_urgency.pkl")  # Label Encoder

# Constants
MAX_LEN = 50  # Ensure this matches what was used in training

# Text preprocessing function
def preprocess_text(text):
    return re.sub(r"[^\w\s]", "", text.lower().strip())  # Removes punctuation, converts to lowercase

@app.route("/")
def login():
    return render_template("login_page.html")

@app.route("/index", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    petition_text = request.form.get("petition_text", "").strip()
    
    if not petition_text:
        return render_template("index.html", error="Please enter text.")

    # Preprocess and tokenize input text
    cleaned_text = preprocess_text(petition_text)
    text_seq = tokenizer.texts_to_sequences([cleaned_text])
    text_padded = pad_sequences(text_seq, maxlen=MAX_LEN)

    # LSTM Model Predictions
    department = label_encoder_dept.inverse_transform(
        [np.argmax(best_model_dept.predict(text_padded), axis=1)[0]]
    )[0]

    urgency = label_encoder_urgency.inverse_transform(
        [np.argmax(best_model_urgency.predict(text_padded), axis=1)[0]]
    )[0]

    # Send email with classified department and urgency
    send_email(department, urgency, petition_text)

    if department == "Education Department":
        return render_template("result1.html", department=department, urgency=urgency, petition_text=petition_text)
    else:
        return render_template("result.html", department=department, urgency=urgency, petition_text=petition_text)

def send_email(department, urgency, petition_text):
    recipient_email = "anthoniv2004@gmail.com"  # Replace with receiver's email

    subject = f"New Complaint - {department} (Urgency: {urgency})"
    body = f"""
    A new complaint has been arrived:

    Complaint Text:   {petition_text}
    Classified Department:   {department}
    Urgency Level:   {urgency}

    Please take appropriate action.
    """

    msg = Message(subject, recipients=[recipient_email], body=body)
    mail.send(msg)

if __name__ == "__main__":
    app.run(debug=True)
