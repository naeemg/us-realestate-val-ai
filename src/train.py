import os
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from feature_engineering import prepare_valuation_features

def train_valuation_model():
    print("Executing U.S. Real Estate Automated Valuation Model (AVM) Pipeline...")
    
    data_file = 'data/us_housing_transactions.csv'
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing input dataset at {data_file}. Run data/generate_us_housing_data.py first.")
        
    X_train, X_test, y_train, y_test, encoder = prepare_valuation_features(data_file)
    
    print(f"Training XGBoost Valuation Engine on {X_train.shape[0]} samples...")
    
    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=25
    )
    
    # Model evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    print("\n--- Model Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared Score (R²): {r2:.4f}")
    
    # Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_avm_model.pkl')
    print("Model artifact successfully saved to models/xgboost_avm_model.pkl")

if __name__ == '__main__':
    train_valuation_model()