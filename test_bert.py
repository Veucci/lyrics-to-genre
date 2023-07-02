import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and Tokenizer
model_path = "lyrics_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Test data
test_lyrics = [input("Input Your Lyrics: ")]

# Encode Lyrics
encoded_lyrics = tokenizer.batch_encode_plus(
    test_lyrics,
    add_special_tokens=True,
    padding='longest',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

# Using Device
input_ids = encoded_lyrics["input_ids"].to(device)
attention_mask = encoded_lyrics["attention_mask"].to(device)

# Predict
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

# Predict Genre
predicted_labels = torch.argmax(logits, dim=1)
int_to_genre = {0:'pop',
                1:'rock',
                2:'hiphop'}
                #3:'jazz',
                #4:'electronic',
                #5: 'country', 
                #6: 'classical', 
                #7: 'metal'}


# Output
for lyric, predicted_label in zip(test_lyrics, predicted_labels):
    print(f"Predicted Genre: {int_to_genre[predicted_label.item()]}")
