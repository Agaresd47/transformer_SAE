# GEN AI Implementation with SAE Model

## Goal

Use SAE to reduce the activation levels of negative emotions such as hatred and deceit to create safer outputs.

Inspired by: [/2024/scaling-monosemanticity/](https://transformer-circuits.pub/2024/scaling-monosemanticity/)


## Ideal Output

When changing the activation level of different nodes/neural, the outcome will be changed correspondingly.
One example from anthropic :


![image](https://github.com/user-attachments/assets/c43c6415-3138-4fe2-b82e-70d0cc6b89fd)

## Idea Implement

I will introduce Sparse Auto Encoder (SAE) in the regular transform model and try to regulaize the node usage of the model. Then use testing method like input same sentence with different meaning to record the node activation level to capture the emotion represenation of each node. Thus, I could control different node to change the output of the model.

## Usage of Current Code

The majority of function can be access through the  [try.ipynb](https://github.com/Agaresd47/transformer_SAE/blob/master/try.ipynb), which include activation function analysis method, inference with current checkpoint, and inference through Hugging Face function.
You can  use [train.py](https://github.com/Agaresd47/transformer_SAE/blob/master/train.py) to train the model from scratch and change hyper parameter. The main architecture can also be modified in trian.py. The code [inference.py](https://github.com/Agaresd47/transformer_SAE/blob/master/inference.py） has the code of inference. 



## Main Architecture

Algorithm: Transformer with SAE Regularization
Input: 
  - input_ids: Sequence of token IDs.
  - attention_mask: Mask indicating valid tokens in the sequence.

Output:
  - act_output: Predicted dialogue acts.
  - emotion_output: Predicted emotions.
  - kl_div: KL divergence loss for SAE regularization.

Parameters:
  - transformer: Pretrained Transformer encoder.
  - batch_norm: Batch normalization layer.
  - dropout: Dropout layer.
  - sparse_autoencoder: SAE module for regularization.
  - act_classifier: Classifier for dialogue acts.
  - emotion_classifier: Classifier for emotions.

Steps:
1. **Convert Attention Mask**  
   - Convert `attention_mask` to transformer-compatible format:  
     `transformer_mask = attention_mask.unsqueeze(1).unsqueeze(2)`

2. **Extract Transformer Features**  
   - Compute transformer encoder output:  
     `transformer_output = transformer.encoder(input_ids, transformer_mask)`

3. **Batch Normalization**  
   - Reshape `transformer_output` for batch normalization:  
     `batch_size, seq_len, feature_dim = transformer_output.size()`  
     `transformer_output = transformer_output.view(batch_size * seq_len, feature_dim)`  
   - Apply batch normalization:  
     `transformer_output = batch_norm(transformer_output)`  
   - Reshape back to original dimensions:  
     `transformer_output = transformer_output.view(batch_size, seq_len, feature_dim)`

4. **Apply Dropout**  
   - Regularize with dropout:  
     `transformer_output = dropout(transformer_output)`

5. **SAE Regularization**  
   - Reconstruct features and compute SAE regularization:  
     `reconstructed, kl_div, encoded = sparse_autoencoder(transformer_output)`

6. **Classification with Encoded Representations**  
   - Extract classification token representation:  
     `cls_output = reconstructed[:, 0, :]`  
   - Predict dialogue acts:  
     `act_output = act_classifier(cls_output)`  
   - Predict emotions:  
     `emotion_output = emotion_classifier(cls_output)`

7. **Return Outputs**  
   - Return predictions and regularization loss:  
     `return act_output, emotion_output, kl_div`



![image](https://github.com/user-attachments/assets/04142012-a4e9-4a02-8734-fdeee2274d89)

Hugging Face Demo: [https://huggingface.co/agaresd/GEN-AI-Final-project](https://huggingface.co/agaresd/GEN-AI-Final-project)

## Impact of the Project

Demonstrate the application of SAE with transformers.




### Example:

Demonstration Video: [https://youtu.be/iP9NNwDiwGk](https://youtu.be/iP9NNwDiwGk)




![image](https://github.com/user-attachments/assets/62b636bf-d641-4a91-861c-86d3edb50129)

![image](https://github.com/user-attachments/assets/f99298b6-5271-4483-a073-acf40ae2b2ce)

![image](https://github.com/user-attachments/assets/720a7b79-e380-44c1-ad7a-0bf4ce35c9a4)

## Next Steps

1. Design a loss function that effectively integrates SAE with text classification tasks.
2. Test the model using datasets with different emotional labels to identify nodes strongly correlated with the results.







## Related Papers

- Original Method: [https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)  
- *Key-Sparse Transformer for Multimodal Speech Emotion Recognition*: [https://arxiv.org/abs/2106.11532](https://arxiv.org/abs/2106.11532)  
- *CLDTA: Contrastive Learning based on Diagonal Transformer Autoencoder for Cross-Dataset EEG Emotion Recognition*: [https://arxiv.org/abs/2406.08081](https://arxiv.org/abs/2406.08081)  
- *EEG-Based Emotion Classification Using a Deep Neural Network and Sparse Autoencoder*: [https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2020.00043/full?utm_source=chatgpt.com](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2020.00043/full?utm_source=chatgpt.com)  
- *Convolutional Sparse Autoencoder for Emotion Recognition*: [https://link.springer.com/chapter/10.1007/978-3-031-27762-7_1?utm_source=chatgpt.com](https://link.springer.com/chapter/10.1007/978-3-031-27762-7_1?utm_source=chatgpt.com)








# Transformer
My own implementation Transformer model (Attention is All You Need - Google Brain, 2017)
<br><br>
![model](image/model.png)
<br><br>


## 1. Implementations

### 1.1 Positional Encoding

![model](image/positional_encoding.jpg)
   
    
```python
class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]         
```
<br><br>

### 1.2 Multi-Head Attention


![model](image/multi_head_attention.jpg)

```python
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
```
<br><br>

### 1.3 Scale Dot Product Attention

![model](image/scale_dot_product_attention.jpg)

```python
class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
```
<br><br>

### 1.4 Layer Norm

![model](image/layer_norm.jpg)
    
```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

```
<br><br>

### 1.5 Positionwise Feed Forward

![model](image/positionwise_feed_forward.jpg)
    
```python

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```
<br><br>

### 1.6 Encoder & Decoder Structure

![model](image/enc_dec.jpg)
    
```python
class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
```
<br>

```python
class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
```
<br>

```python
class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):    
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
```
<br>

```python        
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output
```
<br><br>

## 2. Experiments

I use Multi30K Dataset to train and evaluate model <br>
You can check detail of dataset [here](https://arxiv.org/abs/1605.00459) <br>
I follow original paper's parameter settings. (below) <br>

![conf](image/transformer-model-size.jpg)
### 2.1 Model Specification

* total parameters = 55,207,087
* model size = 215.7MB
* lr scheduling : ReduceLROnPlateau

#### 2.1.1 configuration

* batch_size = 128
* max_len = 256
* d_model = 512
* n_layers = 6
* n_heads = 8
* ffn_hidden = 2048
* drop_prob = 0.1
* init_lr = 0.1
* factor = 0.9
* patience = 10
* warmup = 100
* adam_eps = 5e-9
* epoch = 1000
* clip = 1
* weight_decay = 5e-4
<br><br>

### 2.2 Training Result

![image](saved/transformer-base/train_result.jpg)
* Minimum Training Loss = 2.852672759656864
* Minimum Validation Loss = 3.2048025131225586 
<br><br>

| Model | Dataset | BLEU Score |
|:---:|:---:|:---:|
| Original Paper's | WMT14 EN-DE | 25.8 |
| My Implementation | Multi30K EN-DE | 26.4 |

<br><br>


## 3. Reference
- [Attention is All You Need, 2017 - Google](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer - Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- [Data & Optimization Code Reference - Bentrevett](https://github.com/bentrevett/pytorch-seq2seq/)

<br><br>

## 4. Licence
    Copyright 2019 Hyunwoong Ko.
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
    http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
