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
from datetime import datetime
import sys
import os
import logging

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
    cls.__module__ = '__main__'
    setattr(sys.modules['__main__'], cls.__name__, cls)

# BigQuery setup
script_dir = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(script_dir, "circular-hawk-459707-b8-0c4956e384ac.json")

if not os.path.exists(credentials_path):
    logger.error(f"Service account key file not found at: {credentials_path}")
    raise FileNotFoundError(f"Service account key file not found: {credentials_path}")

credentials = service_account.Credentials.from_service_account_file(credentials_path)
bigquery_client = bigquery.Client(credentials=credentials, project='circular-hawk-459707-b8')

# Define the BigQuery table schema with all input features
desired_schema = [
    bigquery.SchemaField("product_url", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("hierarchy", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("product_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("detail_product_title", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("price_in_dollar", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("asin", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("brand_name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("length", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("width", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("height", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("final_weights_in_grams", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("original_price_in_dollar", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("original_length", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("original_width", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("original_height", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("original_final_weights_in_grams", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("diff_price", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("diff_length", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("diff_width", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("diff_height", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("diff_weight", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("prediction", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("probability", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
]

def create_table_if_not_exists():
    dataset_id = 'anomaly_detection'
    table_id = 'predictions'
    full_table_id = f"{bigquery_client.project}.{dataset_id}.{table_id}"
    try:
        bigquery_client.get_table(full_table_id)
        logger.info(f"Table {full_table_id} already exists.")
    except NotFound:
        dataset_ref = bigquery_client.dataset(dataset_id)
        try:
            bigquery_client.get_dataset(dataset_ref)
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            bigquery_client.create_dataset(dataset)
            logger.info(f"Created dataset {dataset_id}")
        table = bigquery.Table(full_table_id, schema=desired_schema)
        table = bigquery_client.create_table(table)
        logger.info(f"Created table {full_table_id}")

create_table_if_not_exists()

# FastAPI app
app = FastAPI()
pipeline = None
model = None

@app.on_event("startup")
def load_models():
    global pipeline, model
    pipeline = joblib.load('sklearn_preprocessing_pipeline.pkl')
    model = xgb.XGBClassifier()
    model.load_model('xgb_model.json')
    logger.info("Models loaded successfully")

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
        
        # Get probabilities instead of binary predictions
        probabilities = model.predict_proba(df_processed)[:, 1]
        
        # Set a custom threshold for anomaly detection
        threshold = 0.7  # Adjust this value as needed
        labels = ['No Anomaly' if prob < threshold else 'anomaly' for prob in probabilities]
        results = [{"prediction": label, "probability": float(prob)} for label, prob in zip(labels, probabilities)]

        # Prepare data for BigQuery with all input features
        bq_rows = [
            {
                "product_url": product.product_url,
                "hierarchy": product.hierarchy,
                "product_name": product.product_name,
                "detail_product_title": product.detail_product_title,
                "price_in_dollar": product.price_in_dollar,
                "asin": product.asin,
                "brand_name": product.brand_name,
                "length": product.length,
                "width": product.width,
                "height": product.height,
                "final_weights_in_grams": product.final_weights_in_grams,
                "original_price_in_dollar": product.original_price_in_dollar,
                "original_length": product.original_length,
                "original_width": product.original_width,
                "original_height": product.original_height,
                "original_final_weights_in_grams": product.original_final_weights_in_grams,
                "diff_price": product.diff_price,
                "diff_length": product.diff_length,
                "diff_width": product.diff_width,
                "diff_height": product.diff_height,
                "diff_weight": product.diff_weight,
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
