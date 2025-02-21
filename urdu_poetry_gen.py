import torch
import json
import re

# Load vocabulary
vocab_path = r"C:\Users\umars\Desktop\Urdu Poetry Generation\vocab.json"
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Reverse vocab mapping for decoding
idx_to_word = {idx: word for word, idx in vocab.items()}

# Define model class (same as training notebook)
class LSTMGhazalGenerator(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super(LSTMGhazalGenerator, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

# Load model
model_path = r"C:\Users\umars\Desktop\Urdu Poetry Generation\ghazal_generator.pth"
vocab_size = len(vocab)
if "UNK" not in vocab:
    vocab["UNK"] = len(vocab) - 1  # Ensure UNK stays within bounds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMGhazalGenerator(vocab_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 🔹 Urdu text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove special characters and punctuation (keep Urdu letters and spaces)
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # Normalize Urdu letters (example: different forms of 'ی' → standard 'ی')
    text = text.replace("ئ", "ی").replace("ك", "ک").replace("ہ", "ہ")
    
    return text

# Function to generate second misra
def generate_misra1_to_misra2(misra1, model, vocab, idx_to_word, max_len=15):
    # Clean input
    clean_misra1 = clean_text(misra1)

    # Tokenize
    input_tokens = [vocab.get(word, vocab["UNK"]) for word in clean_misra1.split()]
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Predict output
    with torch.no_grad():
        output = model(input_tensor)

    # Convert output to words
    predicted_tokens = torch.argmax(output, dim=2).squeeze(0).tolist()
    generated_misra2 = " ".join([idx_to_word.get(idx, "") for idx in predicted_tokens])

    return generated_misra2

# Ask user for input misra
user_misra1 = input("برائے مہربانی پہلا مصرع درج کریں: ")

# Generate second misra
generated_misra2 = generate_misra1_to_misra2(user_misra1, model, vocab, idx_to_word)

# Display full sher
print("\n📜 مکمل شعر:")
print(f"{user_misra1}")
print(f"{generated_misra2}")
