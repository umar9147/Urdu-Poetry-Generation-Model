import streamlit as st
import torch
import json
import re

# Define the model class
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

# Add the missing functions
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove special characters and punctuation (keep Urdu letters and spaces)
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # Normalize Urdu letters
    text = text.replace("ئ", "ی").replace("ك", "ک").replace("ہ", "ہ")
    
    return text

def generate_misra1_to_misra2(misra1, model, vocab, idx_to_word, max_len=15):
    device = torch.device("cpu")
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

# Set page configuration
st.set_page_config(
    page_title="Urdu Poetry Generator",
    page_icon="📜",
    layout="centered"
)

# Add custom CSS for Urdu text alignment
st.markdown("""
    <style>
    .urdu-text {
        text-align: right;
        direction: rtl;
        font-size: 20px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_vocab():
    # Load vocabulary
    vocab_path = r"C:\Users\umars\Desktop\Urdu Poetry Generation\vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    # Reverse vocab mapping for decoding
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    # Initialize model
    vocab_size = len(vocab)
    if "UNK" not in vocab:
        vocab["UNK"] = len(vocab) - 1
    
    device = torch.device("cpu")
    model = LSTMGhazalGenerator(vocab_size).to(device)
    
    # Load the trained weights
    model_path = r"C:\Users\umars\Desktop\Urdu Poetry Generation\ghazal_generator.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, vocab, idx_to_word

def main():
    # Title
    st.title("🎭 اردو شاعری جنریٹر")
    st.markdown("<p class='urdu-text'>خوش آمدید! براہ کرم پہلا مصرع درج کریں</p>", unsafe_allow_html=True)
    
    # Load model and vocab
    try:
        model, vocab, idx_to_word = load_model_and_vocab()
        
        # Input text box for first misra
        misra1 = st.text_input("", key="misra1_input")
        
        if st.button("شعر مکمل کریں"):
            if misra1:
                with st.spinner("شعر تیار کیا جا رہا ہے..."):
                    # Generate second misra
                    misra2 = generate_misra1_to_misra2(misra1, model, vocab, idx_to_word)
                    
                    # Display the complete sher
                    st.markdown("### 📜 مکمل شعر")
                    st.markdown(f"<p class='urdu-text'>{misra1}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='urdu-text'>{misra2}</p>", unsafe_allow_html=True)
            else:
                st.warning("براہ کرم پہلا مصرع درج کریں")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.write("This application uses a deep learning model to generate Urdu poetry. "
             "Enter the first line of a sher (couplet) and the model will generate "
             "the second line to complete it.")

if __name__ == "__main__":
    main() 