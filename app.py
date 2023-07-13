from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the trained model
app = Flask(__name__)
model = joblib.load('model/model.pkl')

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    loan_repay = float(request.form.get('loan_repay'))
    secured_loan = float(request.form.get('secured_loan'))
    loan_term = float(request.form.get('loan_term'))
    credit_limit_usage = float(request.form.get('credit_limit_usage'))
    ac_bal = float(request.form.get('ac_bal'))  
        
    features = np.array([loan_repay, secured_loan, loan_term, credit_limit_usage, ac_bal]).reshape(1, -1)
    
    if(loan_repay==0):
        cibil1=270
    elif(loan_repay==1):
        cibil1=250
    elif(loan_repay==2):
        cibil1=230
    else:
        cibil1 = 100

    if(secured_loan==1):
        cibil2=112
    else:
        cibil2=50

    if(loan_term<=5):
        cibil3=112
    else:
        cibil3=50

    if(credit_limit_usage<=10):
        cibil4=225
    elif(25>=credit_limit_usage>=11):
        cibil4=200
    elif(40>=credit_limit_usage>=26):
        cibil4=180
    elif(60>=credit_limit_usage>=41):
        cibil4=150
    elif(80>=credit_limit_usage>=61):
        cibil4=100
    else:
        cibil4 = 50

    if(ac_bal>=10):
        cibil5=180
    elif(ac_bal<10):
        cibil5=50

    score = (cibil1+cibil2+cibil3+cibil4+cibil5)

    prediction = model.predict(features)[0]
    probability = round(model.predict_proba(features)[0][1] * 100, 2)
    confidence = round(model.score(features, [prediction]) * 100, 2)
    
    return render_template('result.html', score=score, prediction=prediction, probability=probability, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)

