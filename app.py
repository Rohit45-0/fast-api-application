from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sys

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

# Register custom classes in __main__ for unpickling
custom_classes = [
    GroupMeanDifference,
    LogDensityVolumeCalculator,
    PricePerGramCalculator,
    AspectRatioCalculator,
    HierarchyAggregator,
    ColumnDropper
]
for cls in custom_classes:
    cls.__module__ = '__main__'  # Explicitly set the module to __main__
    setattr(sys.modules['__main__'], cls.__name__, cls)

# FastAPI app setup
app = FastAPI()

pipeline = None
model = None

@app.on_event("startup")
async def load_models():
    global pipeline, model
    try:
        pipeline = joblib.load('sklearn_preprocessing_pipeline.pkl')
        model = xgb.XGBClassifier()
        model.load_model('xgb_model.json')
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

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
async def predict(products: List[ProductData]):
    data = [p.dict() for p in products]
    df = pd.DataFrame(data)
    # Rename columns to match pipeline expectations
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
    return {"results": results}
