from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline

label_mapping = {
    0: "discuss",
    1: "describe",
    2: "compare",
    3: "explain",
    4: "argue",
    5: "reason",
    6: "other"
}

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("./results/checkpoint-final")
model = DistilBertForSequenceClassification.from_pretrained("./results/checkpoint-final")

# Use the model in a pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Get predictions
predictions = classifier("Choose any one of The Canterbury Tales and show in what ways it was calculated to appeal to the interests of its audience.")

for pred in predictions:
    pred['label'] = label_mapping[int(pred['label'].split('_')[-1])]  # Map label ID to name
    print(pred)
