from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    prediction = model.predict(final_features)
    
    output = 'Approved' if prediction[0] == 1 else 'Rejected'

    return render_template('index.html', prediction_text=f'Loan Status: {output}')

if __name__ == "__main__":
    app.run(debug=True)