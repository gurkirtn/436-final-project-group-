import joblib
import pandas as pd

model = joblib.load("pricing_model.joblib")

listing = pd.DataFrame([{
    "bedrooms": 2, "bathrooms_clean": 1.5, "beds": 3, "accommodates": 4,
    "review_scores_rating": 4.8, "number_of_reviews": 20,
    "pet_friendly": 0, "self_check_in": 1, "parking": 0,
    "hot_tub_pool": 0, "luxury_amenities": 0,
    "host_is_superhost": 1, "instant_bookable": 0,
    "room_type": "Entire home/apt", "property_type": "Entire home",
    "neighbourhood_cleansed": "78704",
}])

price = model.predict(listing)[0]
print(f"Predicted nightly price: ${price:.2f}")
