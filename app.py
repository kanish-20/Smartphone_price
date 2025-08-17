from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("smartphone_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        brand = request.form["brand"]
        model_name = request.form["model"]
        ram = int(request.form["ram"])
        storage = int(request.form["storage"])
        fiveg = request.form["fiveg"]

        # Prepare input dataframe
        input_data = pd.DataFrame([{
            "Brand": brand,
            "Model": model_name,
            "RAM": ram,
            "Storage": storage,
            "5G": fiveg
        }])

        # Predict price
        prediction = round(model.predict(input_data)[0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
