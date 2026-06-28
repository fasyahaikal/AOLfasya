import pandas as pd
import numpy as np
import pickle

class Preprocessor:

    def __init__(self):
        self.encoders = pickle.load(open("encoders.pkl","rb"))

    def transform(self, df):

        df = df.copy()

        # Encode categorical
        categorical = [
            'Month',
            'Occupation',
            'Type_of_Loan',
            'Credit_Mix',
            'Payment_of_Min_Amount',
            'Payment_Behaviour'
        ]

        for col in categorical:
            if col in self.encoders:
                df[col] = self.encoders[col].transform(df[col])

        return df