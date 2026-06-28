import pickle
import pandas as pd

model = pickle.load(open("credit_model.pkl","rb"))
encoder = pickle.load(open("target_encoder.pkl","rb"))
encoders = pickle.load(open("encoders.pkl","rb"))

def predict_credit(data):

    df = pd.DataFrame([data])

    categorical = [
        'Month',
        'Occupation',
        'Type_of_Loan',
        'Credit_Mix',
        'Payment_of_Min_Amount',
        'Payment_Behaviour'
    ]

    for col in categorical:
        df[col] = encoders[col].transform(df[col])

    pred = model.predict(df)

    return encoder.inverse_transform(pred)[0]