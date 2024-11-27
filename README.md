# Transformer

## Self-Attention Mechanism

Self-attention is a key component of transformer architectures that allows a model to weigh the importance of different parts of the input sequence when processing each element.

### How it Works

1. **Input Transformation**
   - For each input element, create three vectors:
     - **Query (Q)**: What the current element is looking for
     - **Key (K)**: What this element offers to others
     - **Value (V)**: The actual content of the element
   - These are created by multiplying the input with learned weight matrices (WQ, WK, WV)

2. **Attention Score Calculation**
   ```python
   scores = (Q × K^T) / sqrt(d_k)
   ```
   - Multiply Query with Key transpose to get compatibility scores
   - Scale by sqrt(d_k) to prevent softmax from having extremely small gradients
   - d_k is the dimension of the key vectors

3. **Attention Weights**
   - Apply softmax to scores to get probabilities
   - These weights determine how much each position will focus on other positions

4. **Final Output**
   ```python
   attention = softmax(scores) × V
   ```
   - Multiply attention weights with Values
   - This produces the final attention-weighted representation

### Benefits

- **Global Context**: Each position can attend to all other positions
- **Parallel Processing**: All attention computations can be done simultaneously
- **No Sequential Bottleneck**: Unlike RNNs, information can flow directly between any positions
- **Interpretable**: Attention weights show which parts of input are important for each output

### Multi-Head Attention

Multi-head attention is an enhancement to the basic attention mechanism that allows the model to capture different types of relationships between elements simultaneously.

#### How Multi-Head Attention Works

1. **Split Into Heads**
   - Instead of one attention operation, perform multiple in parallel
   - Split the input embedding dimension (d_model) into h heads
   - Each head works with dimension d_k = d_model/h

2. **Per-Head Processing**
   ```python
   # For each head i:
   head_i = Attention(Q_i, K_i, V_i)
   where:
   Q_i = W_Q_i × X
   K_i = W_K_i × X
   V_i = W_V_i × X
   ```
   - Each head has its own learnable weight matrices
   - Each head can specialize in different relationship patterns

3. **Combine Heads**
   ```python
   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
   ```
   - Concatenate outputs from all heads
   - Project back to original dimension using W_O

#### Benefits of Multi-Head Attention

1. **Diverse Feature Learning**
   - Different heads can learn different aspects:
     - Head 1 might focus on syntactic relationships
     - Head 2 might capture semantic similarities
     - Head 3 might learn positional patterns

2. **Parallel Processing**
   - All heads operate independently
   - Computation can be done in parallel using matrix operations

3. **Enhanced Representation**
   - Combines multiple views of the relationships
   - More robust than single-head attention
   - Can capture both fine and coarse-grained patterns

#### Typical Configuration

- Common settings in transformer models:
  - 8 attention heads (h=8)
  - If d_model = 512, each head operates on d_k = 64 dimensions
  - Total computation remains similar to single-head attention
  - Final output dimension matches input (d_model)

#### Implementation Note

The actual implementation often combines the projections for all heads into single matrices:
```python
# Instead of h separate d_model × d_k matrices
# Use one d_model × d_model matrix for each of Q, K, V
# Then reshape to separate the heads
```

This makes implementation more efficient while maintaining the same mathematical properties.

