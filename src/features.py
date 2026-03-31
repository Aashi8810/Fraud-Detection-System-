

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    
    def fit(self, X: pd.DataFrame, y=None):
        
        self.median_amount_ = X['Transaction_Amount'].median()
        self.median_time_   = X['Time_of_Transaction'].median()
        self.mode_device_   = X['Device_Used'].mode()[0]
        self.mode_location_ = X['Location'].mode()[0]
        self.mode_payment_  = X['Payment_Method'].mode()[0]

        
        self.log_amt_75p_  = np.log1p(X['Transaction_Amount'].quantile(0.75))
        self.acct_age_25p_ = X['Account_Age'].quantile(0.25)

        return self  

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        
        X = X.copy()  
        X['Transaction_Amount']  = X['Transaction_Amount'].fillna(self.median_amount_)
        X['Time_of_Transaction'] = X['Time_of_Transaction'].fillna(self.median_time_)
        X['Device_Used']         = X['Device_Used'].fillna(self.mode_device_)
        X['Location']            = X['Location'].fillna(self.mode_location_)
        X['Payment_Method']      = X['Payment_Method'].fillna(self.mode_payment_)

        
        X['log_amount'] = np.log1p(X['Transaction_Amount'])

        
        X['time_of_day'] = pd.cut(
            X['Time_of_Transaction'],
            bins=[-1, 5, 11, 17, 23],
            labels=['night', 'morning', 'afternoon', 'evening']
        ).astype(str)

        X['is_unknown_device'] = (
            X['Device_Used'] == 'Unknown Device'
        ).astype(int)

        
        X['is_invalid_payment'] = (
            X['Payment_Method'] == 'Invalid Method'
        ).astype(int)

        
        X['high_amount_new_account'] = (
            (X['log_amount'] > self.log_amt_75p_) &
            (X['Account_Age'] < self.acct_age_25p_)
        ).astype(int)

        
        X['velocity_risk'] = pd.cut(
            X['Number_of_Transactions_Last_24H'],
            bins=[0, 3, 7, 11, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        ).astype(str)

        return X



NUMERIC_FEATURES = [
    'log_amount',
    'Time_of_Transaction',
    'Previous_Fraudulent_Transactions',
    'Account_Age',
    'Number_of_Transactions_Last_24H',
]


CATEGORICAL_FEATURES = [
    'Transaction_Type',
    'Device_Used',
    'Location',
    'Payment_Method',
    'time_of_day',
    'velocity_risk',
]


PASSTHROUGH_FEATURES = [
    'is_unknown_device',
    'is_invalid_payment',
    'high_amount_new_account',
]




def build_preprocessing_pipeline() -> Pipeline:
    
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(drop='first', sparse_output=False,
                                  handle_unknown='ignore'), CATEGORICAL_FEATURES),
            ('bin', 'passthrough', PASSTHROUGH_FEATURES),
        ],
        remainder='drop'  # drop Transaction_ID, User_ID, raw Transaction_Amount, etc.
    )

    pipeline = Pipeline(steps=[
        ('engineer', FraudFeatureEngineer()),   # Step 1: create all new features
        ('preprocessor', column_transformer),   # Step 2: scale + encode
    ])

    return pipeline




def get_feature_names(pipeline: Pipeline) -> list:
    
    ct = pipeline.named_steps['preprocessor']

    feature_names = []

    # Numeric features — names stay the same after StandardScaler
    feature_names.extend(NUMERIC_FEATURES)

    # Categorical features — OneHotEncoder creates new names per category
    ohe = ct.named_transformers_['cat']
    cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names.extend(cat_names)

    # Passthrough binary features — names unchanged
    feature_names.extend(PASSTHROUGH_FEATURES)

    return feature_names
