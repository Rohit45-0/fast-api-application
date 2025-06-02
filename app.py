from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sys
from google.cloud import bigquery
import datetime

# [Your custom transformer classes remain unchanged]

# Register custom classes for unpickling
custom_classes = [
    GroupMeanDifference,
    LogDensityVolumeCalculator,
    PricePerGramCalculator,
    AspectRatioCalculator,
    HierarchyAggregator,
    ColumnDropper
]
for cls in custom_classes:
    setattr(sys.modules['__main__'], cls.__name__, cls)

# FastAPI app setup
app = FastAPI()

pipeline = None
model = None
project_id = "your-project-id"  # Replace with your actual project ID
bq_client = bigquery.Client(project=project_id)

@app.on_event("startup")
def load_models():
    global pipeline, model
    pipeline = joblib.load('sklearn_preprocessing_pipeline.pkl')
    model = xgb.XGBClassifier()
    model.load_model('xgb_model.json')

class ProductData(BaseModel):
    product_url: str
    hierarchy: str
    product_name: str
    detail_product_title: str
    price_in_dollar: float
    asin: str
    brand_name: str
    length: float
    width: float
    height: float
    final_weights_in_grams: float
    original_price_in_dollar: float
    original_length: float
    original_width: float
    original_height: float
    original_final_weights_in_grams: float
    diff_price: float
    diff_length: float
    diff_width: float
    diff_height: float
    diff_weight: float

@app.post("/predict")
def predict(products: List[ProductData]):
    data = [p.dict() for p in products]
    df = pd.DataFrame(data)
    column_mapping = {
        'product_url': 'Product URL',
        'hierarchy': 'Hierarchy',
        'product_name': 'Product Name',
        'detail_product_title': 'Detail Product Title',
        'price_in_dollar': 'Price In Dollar',
        'asin': 'ASIN',
        'brand_name': 'BRAND Name',
        'length': 'Length',
        'width': 'Width',
        'height': 'Height',
        'final_weights_in_grams': 'Final Weights in Grams',
        'original_price_in_dollar': 'Original Price In Dollar',
        'original_length': 'Original Length',
        'original_width': 'Original Width',
        'original_height': 'Original Height',
        'original_final_weights_in_grams': 'Original Final Weights in Grams',
        'diff_price': 'Diff Price',
        'diff_length': 'Diff Length',
        'diff_width': 'Diff Width',
        'diff_height': 'Diff Height',
        'diff_weight': 'Diff Weight'
    }
    df.rename(columns=column_mapping, inplace=True)
    df_processed = pipeline.transform(df)
    predictions = model.predict(df_processed)
    probabilities = model.predict_proba(df_processed)[:, 1]
    labels = ['No Anomaly' if p == 0 else 'anomaly' for p in predictions]
    results = [{"prediction": label, "probability": float(prob)} for label, prob in zip(labels, probabilities)]

    # Prepare and insert rows into BigQuery
    rows_to_insert = []
    for product, result in zip(products, results):
        row = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "product_url": product.product_url,
            "prediction": result["prediction"],
            "probability": result["probability"]
        }
        rows_to_insert.append(row)

    table_ref = f"{project_id}.fastapi_results.predictions"
    errors = bq_client.insert_rows_json(table_ref, rows_to_insert)
    if errors:
        print(f"Errors inserting rows: {errors}")
    else:
        print("Rows inserted successfully")

    return {"results": results}
