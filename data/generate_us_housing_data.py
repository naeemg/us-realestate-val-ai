import os
import numpy as np
import pandas as pd

def generate_us_housing_dataset(num_samples=3500, random_seed=42):
    """
    Generates synthetic U.S. residential real estate transaction data
    incorporating structural, spatial, and localized economic features.
    """
    np.random.seed(random_seed)
    
    metro_areas = {
        'Austin-Round Rock, TX': {'base_price': 420000, 'tax_rate': 0.0181},
        'Miami-Fort Lauderdale, FL': {'base_price': 480000, 'tax_rate': 0.0098},
        'Seattle-Tacoma-Bellevue, WA': {'base_price': 650000, 'tax_rate': 0.0092},
        'Phoenix-Mesa-Chandler, AZ': {'base_price': 390000, 'tax_rate': 0.0062},
        'Atlanta-Sandy Springs, GA': {'base_price': 360000, 'tax_rate': 0.0115}
    }
    
    msa_names = list(metro_areas.keys())
    msa_choices = np.random.choice(msa_names, size=num_samples)
    
    sqft = np.random.normal(2100, 650, num_samples).astype(int)
    sqft = np.clip(sqft, 700, 6000)
    
    bedrooms = np.clip((sqft / 600) + np.random.choice([-1, 0, 1], num_samples), 1, 6).astype(int)
    bathrooms = np.clip((bedrooms * 0.75) + np.random.choice([0, 0.5, 1], num_samples), 1.0, 5.0)
    
    year_built = np.random.randint(1950, 2024, size=num_samples)
    property_age = 2026 - year_built
    
    lot_size_sqft = sqft * np.random.uniform(1.5, 4.5, num_samples)
    school_rating = np.random.randint(1, 11, size=num_samples)
    crime_index = np.random.uniform(10.0, 80.0, num_samples)
    dist_to_city_center_miles = np.random.uniform(1.2, 28.0, num_samples)
    
    base_prices = np.array([metro_areas[msa]['base_price'] for msa in msa_choices])
    tax_rates = np.array([metro_areas[msa]['tax_rate'] for msa in msa_choices])
    
    # Target variable valuation logic
    price = (
        base_prices 
        + (sqft * 185) 
        + (bedrooms * 12000) 
        + (bathrooms * 18000) 
        - (property_age * 1200) 
        + (school_rating * 14000) 
        - (dist_to_city_center_miles * 2500) 
        - (crime_index * 800) 
        + np.random.normal(0, 25000, num_samples)
    )
    price = np.clip(price, 120000, 2500000)
    
    df = pd.DataFrame({
        'msa': msa_choices,
        'sqft': sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'property_age': property_age,
        'lot_size_sqft': np.round(lot_size_sqft, 2),
        'school_rating': school_rating,
        'crime_index': np.round(crime_index, 2),
        'dist_to_city_center_miles': np.round(dist_to_city_center_miles, 2),
        'property_tax_rate': tax_rates,
        'sale_price_usd': np.round(price, 2)
    })
    
    os.makedirs('data', exist_ok=True)
    output_path = 'data/us_housing_transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully: {output_path} ({num_samples} records)")

if __name__ == '__main__':
    generate_us_housing_dataset()