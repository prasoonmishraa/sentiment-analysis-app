from flask import Flask, render_template, request  
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]

    data = vectorizer.transform([text])
    result = model.predict(data)[0]

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
