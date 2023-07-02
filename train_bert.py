import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from os import path
from datetime import datetime


# Load CSV.
df = pd.read_csv('data_3genre.csv', header=None, names=['label', 'lyric'], encoding='utf8', delimiter='|')

# Split datas into frames.
labels = df['label'].tolist()
lyrics = df['lyric'].tolist()

label_dict = {label: i for i, label in enumerate(set(labels))}
labels = [label_dict[label] for label in labels]

prev_model_path = 'lyrics_model'
if path.exists(prev_model_path):
    tokenizer = BertTokenizer.from_pretrained(prev_model_path)
    model = BertForSequenceClassification.from_pretrained(prev_model_path)
    print("Model Loaded.", model)
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_dict))
    print("New model created.", model)

# encode lyrics
encoded_lyrics = tokenizer.batch_encode_plus(
    lyrics,
    add_special_tokens=True,
    padding='longest',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

# lyrics class
class LyricsDataset(Dataset):
    def __init__(self, lyrics, labels, attention_masks):
        self.lyrics = lyrics
        self.labels = labels
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        return {
            'lyric': self.lyrics[idx],
            'label': self.labels[idx],
            'attention_mask': self.attention_masks[idx]
        }

# Create dataset
dataset = LyricsDataset(
    lyrics=encoded_lyrics['input_ids'],
    labels=labels,
    attention_masks=encoded_lyrics['attention_mask']
)

# Dataloader config
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


#epochs = 5
lr = 2e-5

optimizer = AdamW(model.parameters(), lr=lr)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epoch = 0

while True:
    model.train()
    total_loss = 0

    for batch in dataloader:
        lyrics = batch['lyric'].to(device)
        labels = batch['label'].to(device)
        attention_masks = batch['attention_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(
            lyrics,
            token_type_ids=None,
            attention_mask=attention_masks,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1} - Loss: {avg_loss}', datetime.now().strftime("%H:%M:%S"))
    # Save Model
    model.save_pretrained('lyrics_model')
    tokenizer.save_pretrained('lyrics_model')

    epoch += 1