from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email_text = request.form['email_text']
        email_vec = vectorizer.transform([email_text])
        prediction = model.predict(email_vec)[0]
        result = 'Spam' if prediction == 1 else 'Not Spam'
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
