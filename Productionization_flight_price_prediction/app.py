from import_lib import *
from predict import predict  # Assuming you have a predict function in predict.py
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None  # Initialize prediction as None
    if request.method == 'POST':
        # Get input data from the form
        date = request.form.get('date')
        from_location = request.form.get('from')
        to_location = request.form.get('to')
        flighttype = request.form.get('flighttype')
        agency = request.form.get('agency')

        # Create a dictionary to store the input data
        data = {
            'date': date,
            'from': from_location,
            'to': to_location,
            'flighttype': flighttype,
            'agency': agency
        }

        # Perform prediction using the predict function
        try:
            prediction = predict(pd.DataFrame([data]))[0]  # Get the prediction value
        except Exception as e:
            print(f"Error during prediction: {e}")

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
