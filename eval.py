from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline

label_mapping = {
    0: "Discuss",
    1: "Describe",
    2: "Compare",
    3: "Explain",
    4: "Argue",
    6: "Reason",
    7: "Other"
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
