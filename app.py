from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Load the trained model
app = Flask(__name__)
model = pickle.load(open('kidney.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetching 18 input features from the HTML form
        input_features = [
            float(request.form['age']),
            float(request.form['bp']),
            float(request.form['al']),
            float(request.form['su']),
            float(request.form['rbc']),
            float(request.form['pc']),
            float(request.form['pcc']),
            float(request.form['ba']),
            float(request.form['bgr']),
            float(request.form['bu']),
            float(request.form['sc']),
            float(request.form['pot']),
            float(request.form['wc']),
            float(request.form['htn']),
            float(request.form['dm']),
            float(request.form['cad']),
            float(request.form['pe']),
            float(request.form['ane']),
        ]

        # Convert inputs into numpy array and reshape
        input_array = np.array([input_features])
        prediction = model.predict(input_array)
        if prediction[0] == 1 :
             result = "Kidney Disease Detected"
        else :
            result = "No Kidney Disease Detected"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
