🧠 Transformer From Scratch — English → Urdu Neural Machine Translation

This repository contains my complete implementation of a **Transformer architecture built entirely from scratch in PyTorch**, applied to **English-to-Urdu translation**.

Unlike most end-to-end tutorials, I developed **each Transformer component in a separate file**, tested them individually, and then combined them into a clean and efficient final model (`transformer.py`).

---

## 🧩 Project Overview

I designed and implemented each part of the Transformer manually, without using any high-level APIs (like `nn.Transformer` from PyTorch).  
This helped me deeply understand how **attention, embeddings, and encoder-decoder mechanisms** actually work.

Here’s how the project evolved:

| Stage | Component | Description |
|--------|------------|-------------|
| 1️⃣ | `self_attention.py` | Implemented **Scaled Dot-Product Self-Attention** from scratch. |
| 2️⃣ | `multi_head_attention.py` | Extended self-attention to **Multi-Head Attention**, splitting and concatenating heads manually. |
| 3️⃣ | `layer_norm.py` | Wrote **Layer Normalization** by hand for stabilizing training. |
| 4️⃣ | `embedding.py` | Built **Token + Positional Embedding** logic using word indices and learnable embeddings. |
| 5️⃣ | `positional_encoding.py` | Added **sinusoidal positional encoding** to preserve word order in sequences. |
| 6️⃣ | `encoder.py` | Combined all components (attention, normalization, feed-forward) to form a **Transformer Encoder block**. |
| 7️⃣ | `decoder.py` | Implemented **Masked Decoder** with self-attention, cross-attention, and feed-forward layers. |
| 8️⃣ | `transformer.py` | Integrated everything into a final **Transformer model** with clean, modular structure. |
| 9️⃣ | `Transformer_Trainer.ipynb` | Trained the model on a bilingual dataset (English → Urdu). |
| 🔟 | Model saved as `transformer_eng_ur_epochX.pt` after training. |

---

## 🌐 English → Urdu Translation

The final Transformer was trained on a custom dataset:
- File: `english_to_urdu_dataset.xlsx`
- Columns: `eng`, `urdu`
- Total Clean Pairs: ~9,000

After preprocessing, tokenization, and training, the model learned to translate English sentences into Urdu text.

**Example:**
```

EN: how are you
UR: آپ کیسے ہیں؟

```

---

## 🏗️ Folder Structure

```

├── self_attention.py
├── multi_head_attention.py
├── layer_norm.py
├── embedding.py
├── positional_encoding.py
├── encoder.py
├── decoder.py
├── transformer.py                # Final clean and complete Transformer
├── Transformer_Trainer.ipynb     # Model training notebook
├── english_to_urdu_dataset.xlsx  # Bilingual dataset
├── transformer_eng_ur_epoch6.pt  # Trained model
├── README.md

````

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/transformer-eng-urdu.git
cd transformer-eng-urdu
pip install -r requirements.txt
````

---

## 🧠 Model Details

| Component                     | Description                                            |
| ----------------------------- | ------------------------------------------------------ |
| **Embedding**                 | Converts words into dense vectors                      |
| **Positional Encoding**       | Adds sequence order using sine/cosine patterns         |
| **Self-Attention**            | Learns dependencies within a sentence                  |
| **Multi-Head Attention**      | Allows parallel attention focus                        |
| **Layer Normalization**       | Stabilizes gradients and convergence                   |
| **Encoder–Decoder Structure** | Follows the original *Attention Is All You Need* paper |
| **Loss Function**             | CrossEntropyLoss                                       |
| **Optimizer**                 | Adam                                                   |

---

## 💬 Translation Example

```python
from transformer import Transformer
import torch

# Load model
transformer = Transformer(...)
transformer.load_state_dict(torch.load("/content/transformer_eng_ur_epoch6.pt"))
transformer.eval()

# Translate
print(translate_sentence("how are you", transformer, eng_tokenizer, urdu_tokenizer, index_to_urdu, device="cuda"))
```

**Output:**

```
EN: how are you
UR: آپ کیسے ہیں؟
```

---

## 📈 Future Enhancements

* [ ] Add Byte Pair Encoding (BPE) for better tokenization
* [ ] Train on a larger parallel corpus
* [ ] Visualize attention weights
* [ ] Deploy model with Streamlit or Gradio web UI

---

## 👨‍💻 Author

**Abdul Moiz**
🎯 *Aspiring AI/ML Engineer*
---
