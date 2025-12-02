# Tamil → English Translator (Seq2Seq LSTM)  
### With and Without Attention | PyTorch Implementation

This project contains **two complete neural machine translation models** built from scratch using **PyTorch**:

1. **Tamil → English Translator (Without Attention)**  
2. **Tamil → English Translator (With Attention Mechanism)**  

These models use:
- **Encoder–Decoder architecture**
- **LSTM-based sequence modelling**
- **Teacher Forcing during training**
- **Greedy decoding during inference**

This project is meant for learning **NLP fundamentals**, understanding **Seq2Seq architecture**, and seeing how **attention improves translation quality**.

---

## 🚀 Project Features

### ✔️ **Model 1 — Seq2Seq Without Attention**
- Simple Encoder–Decoder using LSTM  
- Decoder receives only the last hidden state  
- Works but struggles with longer sentences  
- Demonstrates limitations of classic seq2seq  

### ✔️ **Model 2 — Seq2Seq With Attention**
- Uses Bahdanau-style additive attention  
- Decoder attends to *every encoder timestep*  
- Significantly better translations  
- Visualizable attention weights  

---

## 📁 Project Structure
```
MiniTranslatorWithNN/
│
├── src/
│   ├── translator_without_attention.py
│   ├── translator_with_attention.py
│
├── requirements.txt
└── README.md
```

---

## 🧠 How the Models Work

### 🔹 1. **Encoder**
- Tokenizes Tamil sentences  
- Converts them into embeddings  
- Passes them through an LSTM  
- Produces hidden + cell states  

### 🔹 2. **Decoder**
- Takes English tokens step-by-step  
- Predicts the next English word  

### 🔹 3. **Attention (Second Model Only)**
- Computes relevance between decoder state and all encoder outputs  
- Creates a “context vector”  
- Helps decoder focus on correct parts of Tamil sentence  

---

## 📊 Training Details

- Loss: `CrossEntropyLoss`  
- Optimizer: `Adam`  
- Embedding size: 64  
- Hidden size: 128  
- Epochs: 300  
- Teacher Forcing: 50%  

Training logs print every 50 epochs.

---

## 📝 Example Translations

| Tamil Input | English Output |
|-------------|----------------|
| நன்றி | Thank you |
| நான் பள்ளிக்குச் செல்கிறேன் | I am going to school |
| நீங்கள் எப்படி இருக்கிறீர்கள்? | How are you? |

With attention, translations become more accurate and fluent.

---

## ▶️ Run the Translator

### **Without Attention**
```bash
python translator_without_attention.py
```
## 👨‍💻 Author

Manikandaraja
Passionate about NLP, Deep Learning & building ML from scratch.