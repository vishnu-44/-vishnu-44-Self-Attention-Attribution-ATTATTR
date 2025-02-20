# **Self-Attention Attribution: Interpreting Information Interactions Inside Transformer**

## **1️⃣ Introduction**
Transformers use **self-attention mechanisms** to capture dependencies between words in a sentence. However, interpreting these attention mechanisms is **challenging** because attention scores do not directly indicate importance.

This paper introduces **Self-Attention Attribution (ATTATTR)**, an **Integrated Gradients-based** method to:
- **Attribute importance** to self-attention connections.
- **Interpret information flow** between token pairs.
- **Prune redundant attention heads** while preserving model performance.

## **2️⃣ Comparison: Normal Integrated Gradients (IG) vs. Self-Attention Attribution (ATTATTR)**

### **Example Sentence**
Consider the sentence:

> "This paper introduces a new interpretation method."

Let's analyze how **Normal IG** and **Self-Attention Attribution (ATTATTR)** attribute importance.

---

### **Normal Integrated Gradients (IG)**

- **Measures feature importance** at the **token level**.
- Computes **gradients w.r.t. input embeddings**.

#### **Example: Normal IG at Token Level**

| Token   | IG Attribution |
|---------|--------------|
| **This** | 0.21 |
| **paper** | 0.35 |
| **introduces** | 0.25 |
| **a** | 0.05 |
| **new** | 0.08 |
| **interpretation** | 0.29 |
| **method** | 0.12 |

**Key Limitation:** Normal IG **ignores token interactions** (e.g., how "paper" interacts with "introduces").

---

### **Self-Attention Attribution (ATTATTR)**
- **Extends IG to token pairs** by computing gradients w.r.t. **attention scores** instead of input tokens.
- Uses **self-attention matrices** as the feature of interest.
- **Interpolates between**:
  - A **zero attention baseline**.
  - The **actual attention scores**.

#### **Example: Self-Attention Attribution at Token-Pair Level**

| Token 1 | Token 2 | Attention Attribution |
|---------|---------|----------------------|
| **This** | **paper** | 0.42 |
| **paper** | **introduces** | 0.51 |
| **introduces** | **a** | 0.12 |
| **a** | **new** | 0.07 |
| **new** | **interpretation** | 0.38 |
| **interpretation** | **method** | 0.22 |

**Key Advantage:**  
- **Captures interactions** (e.g., "paper" → "introduces" has strong attribution) which is helpful for better interpreting attention mechanisms.

---

This makes ATTATTR more useful for interpreting attention mechanisms in Transformers!

## **3️⃣ How ATTATTR Works**

### **Step 1: Define Self-Attention Attribution**
Self-Attention Attribution extends **Integrated Gradients (IG)** to measure the **importance of attention scores** rather than individual token embeddings.

It is defined as:

$$
\text{Attr}_h(A) = A_h \circ \int_{0}^{1} \frac{\partial F(\alpha A)}{\partial A_h} d\alpha \in \mathbb{R}^{n \times n}
$$

This equation follows the **Integrated Gradients approach**, but instead of working with token embeddings, it operates on **attention matrices**.

### **Step 2: Compute Integrated Gradients for Attention**

This measures **how much each attention score contributes** to the final prediction.

### **Step 3: Aggregate Importance Across Layers**
The final **self-attention attribution** is obtained by **summing across layers** and normalizing.

---

## **4️⃣ Approach I Used**
- I used a pretrained **BERT model and fine-tuned it on SST-2** for 2 epochs.
- Sentences are tokenized and passed through the model to extract self-attention scores.
- Computed IG along the path for 20 steps where we calculate the gradient of interpolated attention w.r.t. actual layer attention, then integrate and scale it.
- This gives attribution of all 144 heads with respect to each pair, and we take the max from each head.
- This method allows us to identify the most influential attention heads that contribute to the model’s final decision.
- In my approach for pruning, I used the top 3 attributions, meaning the top 3 heads from each layer that contribute to binary classification in SST.

---


