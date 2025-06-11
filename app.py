import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import xgboost as xgb
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery setup with authentication
script_dir = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(script_dir, "circular-hawk-459707-b8-0c4956e384ac.json")

# Verify the service account key file exists
if not os.path.exists(credentials_path):
    logger.error(f"Service account key file not found at: {credentials_path}")
    raise FileNotFoundError(f"Service account key file not found: {credentials_path}")

# Load credentials from the service account key file
credentials = service_account.Credentials.from_service_account_file(credentials_path)
bigquery_client = bigquery.Client(credentials=credentials, project='circular-hawk-459707-b8')

# Define the BigQuery table schema (already updated as per your input)
desired_schema = [
    bigquery.SchemaField("Product URL", "STRING", mode="NULLABLE"),
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
    bigquery.SchemaField("prediction", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("probability", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
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

# Initialize FastAPI app
app = FastAPI()
pipeline = None
model = None

@app.on_event("startup")
def load_models():
    """Load the preprocessing pipeline and XGBoost model on startup."""
    global pipeline, model
    pipeline = joblib.load('sklearn_preprocessing_pipeline.pkl')
    model = xgb.XGBClassifier()
    model.load_model('xgb_model.json')
    logger.info("Models loaded successfully")
    ensure_table_schema()  # Ensure the table schema is correct on startup

class ProductData(BaseModel):
    """Pydantic model for incoming product data."""
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
    """Endpoint to make predictions and store results in BigQuery."""
    try:
        # Define column mapping for BigQuery compatibility
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

        # Prepare rows for BigQuery insertion
        bq_rows = [
            {
                column_mapping[k]: v for k, v in product.dict().items() if k in column_mapping
            } | {
                "prediction": result["prediction"],
                "probability": result["probability"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            for product, result in zip(products, results)
        ]

        # Insert data into BigQuery
        table_id = f"{bigquery_client.project}.anomaly_detection.predictions"
        errors = bigquery_client.insert_rows_json(table_id, bq_rows)
        if errors:
            logger.error(f"Errors inserting rows into BigQuery: {errors}")
            raise Exception(f"BigQuery insertion failed: {errors}")
        else:
            logger.info(f"Successfully inserted {len(bq_rows)} rows into {table_id}")

        return {"results": results}
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise
