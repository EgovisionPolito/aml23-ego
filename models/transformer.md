# Classification via Transformer documentation

## Overview

This document provides a detailed explanation of a PyTorch implementation for a Transformer-based model designed for multi-class classification tasks. The model is implemented as a subclass of `nn.Module` in PyTorch.

The encoder is a critical component in a Transformer model, responsible for processing input sequences and generating meaningful representations. It achieves this through self-attention mechanisms, positional encoding, and stacking multiple layers to capture both local and global dependencies.

### Usage of Transformer Encoder for Classification

Encoders in Transformers are neural network layers that process the input sequence and produce a continuous representation, or embedding, of the input. The decoder then uses these embeddings to generate the output sequence. The encoder typically consists of multiple self-attention and feed-forward layers, allowing the model to process and understand the input sequence effectively.
The self-attention mechanism allows the model to weigh the importance of different input sequence parts by calculating the embeddings' dot product. This mechanism is also known as multi-head attention.

- **Classification as a Sequence-to-One Task:** Classification tasks, in essence, involve making a prediction for the entire input sequence. As such, they can be conceptualized as sequence-to-one problems. The encoder, with its ability to capture comprehensive contextual information, is well-suited for this type of task.

- **Leveraging Global Context:** The final hidden states produced by the encoder encapsulate information about the entire input sequence. This global context is invaluable in classification tasks, where understanding the entirety of the sequence aids in making informed predictions.

- **Simplified Architecture:** By using only the encoder, the architecture is simplified. The omission of the decoder reduces computational complexity, streamlining the model for classification-specific requirements.

- **Classification Head:** Following the encoder, a classification head is added, typically implemented as a fully connected layer. This head maps the final hidden states to class probabilities, serving as the final layer responsible for making class predictions.

The decision to employ only the encoder in a Transformer model for classification is driven by the nature of classification tasks as sequence-to-one problems. The encoder's capability to capture global context and produce context-rich representations aligns with the requirements of such tasks. This design choice not only simplifies the architecture but also allows for efficient utilization of the Transformer's strengths in handling complex dependencies within sequences.

When using this encoder-only approach, it is crucial to incorporate a suitable classification head for producing accurate and meaningful predictions.

## Model Architecture

The model consists of the following components:

1. **Input Embedding Layer:**
   ```python
   self.embedding = nn.Linear(seq_length, hidden_size)
   ```
   - A linear layer is used for embedding, which implies a simple linear transformation of the input features (assuming each element in the sequence is a feature).
   - The input sequence length is seq_length, and the embedded size is hidden_size. This transformation is necessary to project the input features into a suitable space for the transformer.

2. **Transformer Encoder:**
    ```python
    encoder_layers = TransformerEncoderLayer(hidden_size, num_heads)
    self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
    ```
    - The transformer model is used with `num_layers` layers, each containing a `TransformerEncoderLayer`.
    - `hidden_size` is the size of the model's hidden states.
    - `num_heads` is the number of attention heads in the multi-head self-attention models.
    - This design choice allows the model to capture complex relationships and dependencies within the input sequence.

3. **Fully Connected Layer (FC) for Classification:**
    ```python
    self.fc = nn.Linear(hidden_size, num_classes)
    ```
    - After the transformer encoder, a linear layer (`nn.Linear`) is used to map the final hidden state to the number of output classes (`num_classes`) in order to make the final predictions for each class.

## Model usage (Forward)
```python
    x = self.embedding(x)
```
- `x` is the input tensor representing a sequence of features. The `nn.Linear(seq_length, hidden_size)` layer is applied to linearly transform the input features. This operation projects the features into a higher-dimensional space, where each feature is associated with a learnable weight.
</br></br>

```python
    x = x.unsqueeze(1)
    x = x.permute(1, 0, 2)
```
- The reshaping is done to prepare the input tensor for the Transformer encoder, the tensor is unsqueezed along the second dimension and permuted. 
</br></br>

```python
    transformer_output = self.transformer_encoder(x)
```
- The input is then passed through the transformer encoder (the encoder processes the input sequence, capturing dependencies and relationships).
</br></br>

```python
    feat = transformer_output[0, :, :]
    logits = self.fc(feat)
```
- The output of the transformer is a tensor of shape `(seq_length, batch_size, hidden_size)`. We extract the hidden states corresponding to the first token `([0, :, :])`. `feat` represents the features obtained from the transformer for further analysis or interpretation.
- The extracted features are passed through a fully connected layer (`self.fc`), which maps the features to the number of output classes. The result is a tensor `logits` with shape `(batch_size, num_classes)`.

