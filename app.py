import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

with open("Vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open("lr_model.pkl", "rb") as model_file:
    lr_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['Feedback']
    new_data = vectorizer.transform([data])
    output = lr_model.predict(new_data)
    sentiment = "Positive" if output==1 else "Negative"
    prediction_text = f"🎬 Your Review: '{data}'\n  ✨ Sentiment: {sentiment}! Let the world know what you think!"
    prediction_text = prediction_text.replace("\n","<br>")
    return render_template("home.html", prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True, port=3000)









