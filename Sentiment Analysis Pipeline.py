import pandas as pd
import re
import emoji
import boto3
import psycopg2
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
from summa import summarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import fastapi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Data Preprocessing
def preprocess_text(text):
    """Clean and normalize text data."""
    # Convert to lowercase
    text = text.lower()
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Custom Dataset for BERT
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. Load and Prepare Data
def load_data(file_path):
    """Load review data from CSV."""
    df = pd.read_csv(file_path)
    df['text'] = df['text'].apply(preprocess_text)
    return df['text'].tolist(), df['label'].tolist()

# 3. Store Data on S3
def upload_to_s3(file_path, bucket_name, s3_key):
    """Upload file to Amazon S3."""
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        logger.info(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        raise

# 4. Train BERT Model
def train_bert_model(texts, labels, model_name='bert-base-uncased'):
    """Fine-tune BERT model for sentiment classification."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 classes: positive, negative, neutral

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train model
    trainer.train()
    return model, tokenizer

# 5. Summarization
def summarize_reviews(texts):
    """Generate summaries using extractive and abstractive techniques."""
    # Extractive summarization with TextRank
    extractive_summaries = [summarizer.summarize(text, ratio=0.3) for text in texts]

    # Abstractive summarization with BART
    summarizer_model = pipeline('summarization', model='facebook/bart-large-cnn')
    abstractive_summaries = [summarizer_model(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text'] for text in texts]

    return extractive_summaries, abstractive_summaries

# 6. FastAPI Deployment
app = FastAPI()

class ReviewInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(review: ReviewInput):
    """Predict sentiment for a given review."""
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('./results')
        inputs = tokenizer(review.text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return {"sentiment": labels[prediction]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 7. Store Results in PostgreSQL
def store_in_postgres(review, sentiment):
    """Store sentiment results in PostgreSQL."""
    conn = psycopg2.connect(
        dbname="your_db",
        user="your_user",
        password="your_password",
        host="your_host",
        port="5432"
    )
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sentiments (review, sentiment) VALUES (%s, %s)", (review, sentiment))
    conn.commit()
    cursor.close()
    conn.close()

# 8. Visualization with Dash
def create_dashboard(sentiments, texts):
    """Create a Dash dashboard for sentiment visualization."""
    app_dash = Dash(__name__)
    df = pd.DataFrame({'text': texts, 'sentiment': sentiments})

    app_dash.layout = html.Div([
        html.H1("Sentiment Analysis Dashboard"),
        dcc.Graph(id='sentiment-bar'),
        dcc.Dropdown(
            id='sentiment-filter',
            options=[{'label': s, 'value': s} for s in df['sentiment'].unique()],
            value='all',
            multi=True
        )
    ])

    @app_dash.callback(
        Output('sentiment-bar', 'figure'),
        Input('sentiment-filter', 'value')
    )
    def update_graph(selected_sentiments):
        if isinstance(selected_sentiments, str):
            selected_sentiments = [selected_sentiments]
        if 'all' in selected_sentiments or not selected_sentiments:
            filtered_df = df
        else:
            filtered_df = df[df['sentiment'].isin(selected_sentiments)]
        fig = px.histogram(filtered_df, x='sentiment', title='Sentiment Distribution')
        return fig

    app_dash.run_server(debug=True)

# Main Execution
if __name__ == "__main__":
    # Load data
    file_path = 'reviews.csv'  # Replace with your dataset
    texts, labels = load_data(file_path)

    # Upload to S3
    upload_to_s3(file_path, 'your-bucket', 'raw/reviews.csv')

    # Train model
    model, tokenizer = train_bert_model(texts, labels)

    # Summarize reviews
    extractive_summaries, abstractive_summaries = summarize_reviews(texts[:10])  # Example with first 10 reviews

    # Deploy API (run separately with `uvicorn main:app --host 0.0.0.0 --port 8000`)
    # Store results in PostgreSQL (example)
    for text in texts[:10]:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}[prediction]
        store_in_postgres(text, sentiment)

    # Create dashboard
    sentiments = ['positive', 'negative', 'neutral'] * (len(texts) // 3)  # Dummy data for example
    create_dashboard(sentiments, texts)