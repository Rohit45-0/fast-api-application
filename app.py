import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
import sys
import os
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom transformer definitions
class GroupMeanDifference(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, value_col, output_col=None):
        self.group_col = group_col
        self.value_col = value_col
        self.output_col = output_col or f'{value_col}_diff_from_group_mean'

    def fit(self, X, y=None):
        self.group_means_ = X.groupby(self.group_col)[self.value_col].mean().to_dict()
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.output_col] = X_[self.value_col] - X_[self.group_col].map(self.group_means_)
        return X_

class LogDensityVolumeCalculator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 weight_col='Final Weights in Grams',
                 dim_cols=('Length', 'Width', 'Height'),
                 log_volume_col='log_volume',
                 log_density_col='log_density_proxy',
                 density_col='density_proxy',
                 log_weight_col='log_final_weight'):
        self.weight_col = weight_col
        self.dim_cols = dim_cols
        self.log_volume_col = log_volume_col
        self.log_density_col = log_density_col
        self.density_col = density_col
        self.log_weight_col = log_weight_col
        self.epsilon = 1e-6

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        volume = X_[self.dim_cols[0]] * X_[self.dim_cols[1]] * X_[self.dim_cols[2]]
        density_proxy = X_[self.weight_col] / (volume + self.epsilon)
        X_[self.log_volume_col] = np.log1p(volume + self.epsilon)
        X_[self.log_density_col] = np.log1p(density_proxy + self.epsilon)
        X_[self.log_weight_col] = np.log1p(X_[self.weight_col] + self.epsilon)
        X_[self.density_col] = density_proxy
        return X_

class PricePerGramCalculator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 price_col='Price In Dollar',
                 weight_col='Final Weights in Grams',
                 output_col='price_per_gram'):
        self.price_col = price_col
        self.weight_col = weight_col
        self.output_col = output_col
        self.epsilon = 1e-6

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.output_col] = X_[self.price_col] / (X_[self.weight_col] + self.epsilon)
        return X_

class AspectRatioCalculator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 length_col='Length',
                 width_col='Width',
                 height_col='Height',
                 epsilon=1e-6):
        self.length_col = length_col
        self.width_col = width_col
        self.height_col = height_col
        self.epsilon = epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['L_by_W'] = X_[self.length_col] / (X_[self.width_col] + self.epsilon)
        X_['L_by_H'] = X_[self.length_col] / (X_[self.height_col] + self.epsilon)
        X_['W_by_H'] = X_[self.width_col] / (X_[self.height_col] + self.epsilon)
        return X_

class HierarchyAggregator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 group_col='Hierarchy',
                 agg_config=None):
        self.group_col = group_col
        self.agg_config = agg_config or {
            'Hierarchy_Weight_Mean': ('Final Weights in Grams', 'mean'),
            'Hierarchy_Price_Std': ('Price In Dollar', 'std'),
            'Hierarchy_Count': ('Hierarchy', 'count')
        }

    def fit(self, X, y=None):
        self.agg_df_ = X.groupby(self.group_col).agg(**self.agg_config).reset_index()
        return self

    def transform(self, X):
        X_ = X.copy()
        X_ = X_.merge(self.agg_df_, on=self.group_col, how='left')
        return X_

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

# BigQuery setup
script_dir = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(script_dir, "circular-hawk-459707-b8-0c4956e384ac.json")

if not os.path.exists(credentials_path):
    logger.error(f"Service account key file not found at: {credentials_path}")
    raise FileNotFoundError(f"Service account key file not found: {credentials_path}")

credentials = service_account.Credentials.from_service_account_file(credentials_path)
bigquery_client = bigquery.Client(credentials=credentials, project='circular-hawk-459707-b8')

# Define the BigQuery table schema
desired_schema = [
    bigquery.SchemaField("Product URL", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("Hierarchy", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("Product Name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("Detail Product Title", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("Price In Dollar", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("ASIN", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("BRAND Name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("Length", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("Width", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("Height", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("Final Weights in Grams", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("Original Price In Dollar", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("Original Length", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("Original Width", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("Original Height", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("Original Final Weights in Grams", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("prediction", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("probability", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
]

def ensure_table_schema():
    """Ensure the BigQuery table exists and matches the desired schema."""
    table_id = f"{bigquery_client.project}.anomaly_detection.predictions"
    try:
        table = bigquery_client.get_table(table_id)
        current_fields = {field.name for field in table.schema}
        desired_fields = {field.name for field in desired_schema}
        missing_fields = desired_fields - current_fields

        if missing_fields:
            new_schema = table.schema[:]
            for field_name in missing_fields:
                field = next(f for f in desired_schema if f.name == field_name)
                new_schema.append(field)
            table.schema = new_schema
            bigquery_client.update_table(table, ["schema"])
            logger.info(f"Added missing fields to table schema: {missing_fields}")
        else:
            logger.info("Table schema is up to date.")
    except NotFound:
        table = bigquery.Table(table_id, schema=desired_schema)
        bigquery_client.create_table(table)
        logger.info(f"Created table {table_id} with full schema.")

# FastAPI app
app = FastAPI()
pipeline = None
model = None

@app.on_event("startup")
def load_models():
    global pipeline, model

    # Custom unpickler to handle classes defined in app.py
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__":
                return globals()[name]  # Use app.py's global namespace
            return super().find_class(module, name)

    # Load the pipeline using the custom unpickler
    with open('sklearn_preprocessing_pipeline.pkl', 'rb') as f:
        unpickler = CustomUnpickler(f)
        pipeline = unpickler.load()

    # Load the XGBoost model
    model = xgb.XGBClassifier()
    model.load_model('xgb_model.json')
    logger.info("Models loaded successfully")

    # Ensure BigQuery table schema
    ensure_table_schema()

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
    try:
        # Convert product data to dictionary and rename fields for BigQuery
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
        }

        # Prepare data for prediction
        data = [p.dict() for p in products]
        df = pd.DataFrame(data)
        df_for_prediction = df.rename(columns=column_mapping)
        df_processed = pipeline.transform(df_for_prediction)
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]
        labels = ['No Anomaly' if p == 0 else 'anomaly' for p in predictions]
        results = [{"prediction": label, "probability": float(prob)} for label, prob in zip(labels, probabilities)]

        # Prepare data for BigQuery
        bq_rows = [
            {
                column_mapping.get(k, k): v for k, v in product.dict().items()
            } | {
                "prediction": result["prediction"],
                "probability": result["probability"],
                "timestamp": datetime.utcnow().isoformat()
            }
            for product, result in zip(products, results)
        ]

        # Insert into BigQuery
        table_id = f"{bigquery_client.project}.anomaly_detection.predictions"
        errors = bigquery_client.insert_rows_json(table_id, bq_rows)
        if errors:
            logger.error(f"Errors inserting rows into BigQuery: {errors}")
        else:
            logger.info(f"Successfully inserted {len(bq_rows)} rows into {table_id}")

        return {"results": results}
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise
