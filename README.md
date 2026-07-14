# Austin Airbnb Pricing & Upgrade-ROI Model

Predicts nightly listing price from property features and amenities, then
ranks candidate host upgrades (parking, pet-friendly, self check-in, hot
tub/pool, luxury amenities) by estimated ROI / payback period.

## Setup

```bash
pip install -r requirements.txt
```

## Data

This repo does **not** include `listings.csv` (34.8MB — too large / not
ours to redistribute). Download the current Austin detailed listings file
yourself:

1. Go to https://insideairbnb.com/get-the-data/
2. Find the "Austin" section, click the `listings.csv.gz` link under
   **Detailed Listings data**
3. Unzip it and place `listings.csv` in this folder

## Run

Train from scratch (reads `listings.csv`, prints RMSE/R^2, saves the model):
```bash
python airbnb_idss_model.py
```

Use the already-trained model (`pricing_model.joblib`, included in this repo)
without retraining:
```python
import joblib
model = joblib.load("pricing_model.joblib")
```

Sanity-check that the saved model loads and predicts reasonably:
```bash
python test_model.py
```

## Model

Linear Regression. Test set (1,528 held-out listings): RMSE $203.28, R^2 0.527.

**Known limitation:** some amenity coefficients (parking, self check-in,
pet-friendly) come out negative, most likely due to confounding with
listing tier rather than a genuine causal effect — budget listings tend to
advertise these amenities more prominently than luxury ones. Worth treating
as correlational, not causal.
