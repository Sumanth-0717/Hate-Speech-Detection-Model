from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')

dt = joblib.load(r'models/trained_model.sav')
cv = joblib.load(r'models/count_vectorizer.sav')

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form['inputText']
        if not text:
            return render_template("test.html", prediction="No text provided")
        
        print('Prediction starts')
        
        transformed_text = cv.transform([text]).toarray()

    
        prediction = dt.predict(transformed_text)


        prediction_str = prediction[0]
        
        return render_template("test.html", prediction=prediction_str)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return render_template("test.html", prediction="Prediction failed")

if __name__ == '__main__':
    app.run(debug=True)
