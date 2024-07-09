from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

rmodel = joblib.load('model.pkl')

scaler = joblib.load('scaler.pkl')

encoders = {}
encoder_files = [
    'gender_encoder.pkl',
    'Partner_encoder.pkl',
    'Dependents_encoder.pkl',
    'PhoneService_encoder.pkl',
    'MultipleLines_encoder.pkl',
    'InternetService_encoder.pkl',
    'OnlineSecurity_encoder.pkl',
    'OnlineBackup_encoder.pkl',
    'DeviceProtection_encoder.pkl',
    'TechSupport_encoder.pkl',
    'StreamingTV_encoder.pkl',
    'StreamingMovies_encoder.pkl',
    'Contract_encoder.pkl',
    'PaperlessBilling_encoder.pkl',
    'PaymentMethod_encoder.pkl'
]

for file in encoder_files:
    encoders[file.split('_')[0]] = joblib.load(file)


@app.route('/', methods=["GET", "POST"])
def func():
    if request.method == "POST":
       
        senior = int(request.form.get("senior"))
        gender = request.form.get("gender")
        partner = request.form.get("partner")
        dependents = request.form.get("dependents")
        tenure = int(request.form.get("tenure"))
        phone = request.form.get("phone")
        multipleLines = request.form.get("multipleLines")
        internet = request.form.get("internet")
        onlineSecurity = request.form.get("onlineSecurity")
        onlineBackup = request.form.get("onlineBackup")
        deviceProtection = request.form.get("deviceProtection")
        techSupport = request.form.get("techSupport")
        streamingTV = request.form.get("streamingTV")
        streamingMovies = request.form.get("streamingMovies")
        contract = request.form.get("Contract")
        paperlessBilling = request.form.get("paperlessBilling")
        paymentMethod = request.form.get("paymentMethod")
        monthlycharges = float(request.form.get("monthlycharges"))
        totalcharges = float(request.form.get("totalcharges"))

      
        encoded_data = {
            'gender': encoders['gender'].transform([gender])[0],
            'Partner': encoders['Partner'].transform([partner])[0],
            'Dependents': encoders['Dependents'].transform([dependents])[0],
            'PhoneService': encoders['PhoneService'].transform([phone])[0],
            'MultipleLines': encoders['MultipleLines'].transform([multipleLines])[0],
            'InternetService': encoders['InternetService'].transform([internet])[0],
            'OnlineSecurity': encoders['OnlineSecurity'].transform([onlineSecurity])[0],
            'OnlineBackup': encoders['OnlineBackup'].transform([onlineBackup])[0],
            'DeviceProtection': encoders['DeviceProtection'].transform([deviceProtection])[0],
            'TechSupport': encoders['TechSupport'].transform([techSupport])[0],
            'StreamingTV': encoders['StreamingTV'].transform([streamingTV])[0],
            'StreamingMovies': encoders['StreamingMovies'].transform([streamingMovies])[0],
            'Contract': encoders['Contract'].transform([contract])[0],
            'PaperlessBilling': encoders['PaperlessBilling'].transform([paperlessBilling])[0],
            'PaymentMethod': encoders['PaymentMethod'].transform([paymentMethod])[0]
        }


        arr = np.array([[encoded_data['gender'], senior, encoded_data['Partner'], encoded_data['Dependents'],
                         tenure, encoded_data['PhoneService'], encoded_data['MultipleLines'],
                         encoded_data['InternetService'], encoded_data['OnlineSecurity'], encoded_data['OnlineBackup'],
                         encoded_data['DeviceProtection'], encoded_data['TechSupport'], encoded_data['StreamingTV'],
                         encoded_data['StreamingMovies'], encoded_data['Contract'], encoded_data['PaperlessBilling'],
                         encoded_data['PaymentMethod'], monthlycharges, totalcharges]])

        try:
             scaler = joblib.load('scaler.pkl')
    
        except FileNotFoundError:
             print("Scaler file not found. Please ensure 'scaler.pkl' exists in the current directory.")
             scaler = None
        except Exception as e:
             print(f"Error loading scaler: {e}")
             scaler = None  
        
        if scaler is not None:
            arr_scaled = scaler.transform(arr)
            y_pred = rmodel.predict(arr_scaled)

       
        if y_pred==1:
             return render_template('result.html', prediction="Yes")
        else:
             return render_template('result.html', prediction="No")

    return render_template('form.html')

if __name__ == "__main__":
    app.run(debug=True)