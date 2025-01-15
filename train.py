# File: train_distilbert_exam.py

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

# Step 1: Load Dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path, sep=';')
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)
    return Dataset.from_pandas(train_data), Dataset.from_pandas(val_data)

# Step 2: Tokenize Dataset
def tokenize_function(examples, tokenizer):
    return tokenizer(examples['question'], truncation=True, padding="max_length", max_length=128)

# Step 3: Prepare Model and Tokenizer
def prepare_model_and_tokenizer(num_labels):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    return tokenizer, model

# Step 4: Train the Model
def train_model(train_dataset, val_dataset, tokenizer, model):
    # Tokenize datasets
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    
    # Save model and tokenizer
    trainer.save_model("./results/checkpoint-final")
    tokenizer.save_pretrained("./results/checkpoint-final")



# Step 5: Main Execution
if __name__ == "__main__":
    # File path to your dataset
    dataset_path = "sentences.csv"

    # Load datasets
    train_dataset, val_dataset = load_dataset(dataset_path)

    # Unique labels
    unique_labels = train_dataset.unique('label')
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    train_dataset = train_dataset.map(lambda x: {"label": label_to_id[x['label']]})
    val_dataset = val_dataset.map(lambda x: {"label": label_to_id[x['label']]})

    # Prepare model and tokenizer
    tokenizer, model = prepare_model_and_tokenizer(num_labels=len(unique_labels))

    # Train model
    train_model(train_dataset, val_dataset, tokenizer, model)
    
    

