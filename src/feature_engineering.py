import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def prepare_valuation_features(data_path='data/us_housing_transactions.csv', test_size=0.2, random_state=42):
    """
    Loads raw housing transactions, constructs interaction ratios,
    encodes categorical MSAs, and executes train/test split.
    """
    df = pd.read_csv(data_path)
    
    # Engineered feature interactions
    df['price_per_sqft'] = df['sale_price_usd'] / df['sqft']
    df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']
    df['space_efficiency'] = df['sqft'] / df['bedrooms']
    
    # One-Hot Encoding for Metro Statistical Areas (MSAs)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    msa_encoded = encoder.fit_transform(df[['msa']])
    msa_encoded_df = pd.DataFrame(
        msa_encoded, 
        columns=encoder.get_feature_names_out(['msa'])
    )
    
    features = [
        'sqft', 'bedrooms', 'bathrooms', 'property_age', 
        'lot_size_sqft', 'school_rating', 'crime_index', 
        'dist_to_city_center_miles', 'property_tax_rate',
        'bed_bath_ratio', 'space_efficiency'
    ]
    
    X = pd.concat([df[features], msa_encoded_df], axis=1)
    y = df['sale_price_usd']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, encoder