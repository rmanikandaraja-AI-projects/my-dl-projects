import torch
import torch.nn as nn
import torch.optim as optim

# ----- 1. Data -----
pairs = [
    ("வணக்கம்", "Hello"),
    ("நீங்கள் எப்படி இருக்கிறீர்கள்?", "How are you?"),
    ("நான் நன்றாக இருக்கிறேன்", "I am fine"),
    ("உங்கள் பெயர் என்ன?", "What is your name?"),
    ("என் பெயர் மணிகண்டராஜா", "My name is Manikandaraja"),
    ("நன்றி", "Thank you"),
    ("நீங்கள் எங்கே செல்கிறீர்கள்?", "Where are you going?"),
    ("நான் பள்ளிக்குச் செல்கிறேன்", "I am going to school"),
    ("அவர் ஒரு ஆசிரியர்", "He is a teacher"),
    ("இது ஒரு புத்தகம்", "This is a book"),
]

src_texts = [s for s, _ in pairs]
tgt_texts = [t for _, t in pairs]

# ----- 2. Tokenization -----
def build_vocab(sentences):
    words = set()
    for s in sentences:
        words.update(s.lower().split())
    word2idx = {w: i+2 for i, w in enumerate(sorted(words))}
    word2idx["<PAD>"] = 0
    word2idx["<SOS>"] = 1
    word2idx["<EOS>"] = len(word2idx)
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

src_vocab, src_idx2word = build_vocab(src_texts)
tgt_vocab, tgt_idx2word = build_vocab(tgt_texts)

def encode_sentence(s, vocab):
    return [vocab["<SOS>"]] + [vocab[w] for w in s.lower().split() if w in vocab] + [vocab["<EOS>"]]

src_encoded = [encode_sentence(s, src_vocab) for s in src_texts]
tgt_encoded = [encode_sentence(s, tgt_vocab) for s in tgt_texts]

def pad(seq, max_len):
    return seq + [0] * (max_len - len(seq))

src_maxlen = max(len(s) for s in src_encoded)
tgt_maxlen = max(len(s) for s in tgt_encoded)

src_padded = [pad(s, src_maxlen) for s in src_encoded]
tgt_padded = [pad(t, tgt_maxlen) for t in tgt_encoded]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_tensor = torch.tensor(src_padded).to(device)
tgt_tensor = torch.tensor(tgt_padded).to(device)

# ----- 3. Model -----

### (A) Encoder — return all LSTM outputs + final states
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        # src: [batch, src_len]
        embedded = self.embedding(src)  # [batch, src_len, emb_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: [batch, src_len, hid_dim]
        return outputs, hidden, cell


### (B) Add Attention Layer
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch, hid_dim]
        # encoder_outputs: [batch, src_len, hid_dim]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.permute(1, 0, 2)  # [batch, 1, hid_dim]
        hidden = hidden.repeat(1, src_len, 1)  # [batch, src_len, hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, src_len, hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch, src_len]
        return torch.softmax(attention, dim=1)  # attention weights


### (C) Decoder with Attention
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(hid_dim + emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, output_dim)
        self.attention = attention

    def forward(self, input_token, hidden, cell, encoder_outputs):
        # input_token: [batch, 1]
        embedded = self.embedding(input_token)  # [batch, 1, emb_dim]

        attn_weights = self.attention(hidden, encoder_outputs)  # [batch, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch, 1, src_len]

        context = torch.bmm(attn_weights, encoder_outputs)  # [batch, 1, hid_dim]
        lstm_input = torch.cat((embedded, context), dim=2)  # [batch, 1, emb_dim + hid_dim]

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output.squeeze(1))  # [batch, output_dim]

        return prediction, hidden, cell, attn_weights


### (D) Seq2Seq wrapper
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = len(tgt_vocab)
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        input_token = tgt[:, 0].unsqueeze(1)  # <SOS>

        for t in range(1, tgt_len):
            output, hidden, cell, attn = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1).unsqueeze(1)
            input_token = tgt[:, t].unsqueeze(1) if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

# ----- 4. Train -----
INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(tgt_vocab)
EMB_DIM = 64
HID_DIM = 128

attn = Attention(HID_DIM)
encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, attn)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 300
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(src_tensor, tgt_tensor)
    output_dim = output.shape[-1]
    output = output[:, 1:].reshape(-1, output_dim)
    tgt = tgt_tensor[:, 1:].reshape(-1)
    loss = criterion(output, tgt)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ----- 5. Inference -----
def translate(sentence):
    model.eval()
    with torch.no_grad():
        src = torch.tensor([pad(encode_sentence(sentence, src_vocab), src_maxlen)]).to(device)
        encoder_outputs, hidden, cell = model.encoder(src)
        input_token = torch.tensor([[tgt_vocab["<SOS>"]]]).to(device)
        output_sentence = []
        for _ in range(tgt_maxlen):
            output, hidden, cell, attn = model.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = output.argmax(1)
            word = tgt_idx2word[top1.item()]
            if word == "<EOS>":
                break
            output_sentence.append(word)
            input_token = top1.unsqueeze(1)
    return " ".join(output_sentence)

print(translate("நன்றி"))
print(translate("நான் பள்ளிக்குச் செல்கிறேன்"))
print(translate("நான் நன்றாக இருக்கிறேன்"))
