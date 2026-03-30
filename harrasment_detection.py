# harassment_detection.py

# --------------------------
# 1. Install required libraries (run once)
# --------------------------
# !pip install transformers datasets torch scikit-learn pandas tqdm openpyxl

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# --------------------------
# 2. Load Excel & Convert to CSV
# --------------------------
df = pd.read_excel("the_harassment_data.xlsx")
df.to_csv("harassment_dataset.csv", index=False)
print("Excel file converted to CSV successfully!")

# --------------------------
# 3. Add Labels (Keyword-based Initial Labels)
# --------------------------
labels = []
for text in df['text']:
    text_lower = str(text).lower()
    harassment_keywords = ['bitch', 'stupid', 'moron', 'fuck', 'idiot', 'useless', 'loser', 'worthless']
    if any(word in text_lower for word in harassment_keywords):
        labels.append("harassment")
    else:
        labels.append("normal")
df['labels'] = labels

df.to_csv("harassment_dataset_labeled.csv", index=False)
print("Labels added and saved to CSV!")

# --------------------------
# 4. Preprocessing
# --------------------------
df = df[df['labels'].isin(['harassment', 'normal'])]
df['labels'] = df['labels'].map({'normal': 0, 'harassment': 1})
df['text'] = df['text'].str.lower().str.strip()
df.drop_duplicates(subset=['text'], inplace=True)

# --------------------------
# 5. Train/Test Split
# --------------------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['labels'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['labels']
)

# --------------------------
# 6. Tokenization
# --------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# --------------------------
# 7. Dataset Class
# --------------------------
class HarassmentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = HarassmentDataset(train_encodings, train_labels)
test_dataset = HarassmentDataset(test_encodings, test_labels)

# --------------------------
# 8. Load Model
# --------------------------
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# --------------------------
# 9. Training Setup
# --------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
optimizer = AdamW(model.parameters(), lr=5e-5)

# --------------------------
# 10. Training Loop
# --------------------------
epochs = 3
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch {epoch+1}')
        loop.set_postfix(loss=loss.item())

# --------------------------
# 11. Evaluation
# --------------------------
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print("\nFinal Results:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-score: {f1*100:.2f}%")

# --------------------------
# 12. Save Model
# --------------------------
model.save_pretrained("harassment_model")
tokenizer.save_pretrained("harassment_tokenizer")