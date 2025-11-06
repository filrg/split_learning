import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration constants
NUM_SAMPLES = 5000
num_labels = 6
vocab_size = 30522
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
intermediate_size = 3072
max_position_embeddings = 512
type_vocab_size = 2
dropout_prob = 0.1

# BertEmbeddings class
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout_prob):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# BertSdpaSelfAttention class
class BertSdpaSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(BertSdpaSelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, hidden_size = hidden_states.size()

        # Create query, key, value projections
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Perform attention score calculation
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Scale attention scores
        import math
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape attention_mask from [batch_size, seq_length] to [batch_size, 1, 1, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert 1s (valid tokens) to 0s and 0s (padding) to large negative values
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask

        # Apply softmax to get probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Calculate context by attending to values
        context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape back to [batch_size, seq_length, hidden_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

# BertSelfOutput class
class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# BertAttention class
class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(BertAttention, self).__init__()
        self.self = BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.output = BertSelfOutput(hidden_size, dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output

# BertIntermediate class
class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# BertOutput class
class BertOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_prob):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# BertPooler class
class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# BertClassifier class
class BertClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob=0.1):
        super(BertClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Complete BERT Model
class Bert_EMOTION(nn.Module):
    def __init__(self,start_layer= 0, end_layer= 27, vocab_size=30522, hidden_size=768, intermediate_size=3072,
                 num_attention_heads=12, num_labels=4, max_position_embeddings=512,
                 type_vocab_size=2, dropout_prob=0.1, num_hidden_layers=12):

        super(Bert_EMOTION, self).__init__()

        self.start_layer = start_layer
        self.end_layer = end_layer

        if (self.start_layer < 1) and (self.end_layer >= 1):
            self.layer1 = BertEmbeddings(vocab_size, hidden_size    , max_position_embeddings, type_vocab_size, dropout_prob)

        if (self.start_layer < 2) and (self.end_layer >= 2):
            self.layer2 = nn.ModuleList([
                BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob),
                BertSelfOutput(hidden_size, dropout_prob)
            ])

        if (self.start_layer < 3) and (self.end_layer >= 3):
            self.layer3 = nn.ModuleList([
                BertIntermediate(hidden_size, intermediate_size),
                BertOutput(hidden_size, intermediate_size, dropout_prob)
            ])

        if (self.start_layer < 4) and (self.end_layer >= 4):
            self.layer4 = nn.ModuleList([
                BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob),
                BertSelfOutput(hidden_size, dropout_prob)
            ])

        if (self.start_layer < 5) and (self.end_layer >= 5):
            self.layer5 = nn.ModuleList([
                BertIntermediate(hidden_size, intermediate_size),
                BertOutput(hidden_size, intermediate_size, dropout_prob)
            ])

        if (self.start_layer < 6) and (self.end_layer >= 6):
            self.layer6 = nn.ModuleList([
                BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob),
                BertSelfOutput(hidden_size, dropout_prob)
            ])

        if (self.start_layer < 7) and (self.end_layer >= 7):
            self.layer7 = nn.ModuleList([
                BertIntermediate(hidden_size, intermediate_size),
                BertOutput(hidden_size, intermediate_size, dropout_prob)
            ])

        if (self.start_layer < 8) and (self.end_layer >= 8):
            self.layer8 = nn.ModuleList([
                BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob),
                BertSelfOutput(hidden_size, dropout_prob)
            ])

        if (self.start_layer < 9) and (self.end_layer >= 9):
            self.layer9 = nn.ModuleList([
                BertIntermediate(hidden_size, intermediate_size),
                BertOutput(hidden_size, intermediate_size, dropout_prob)
            ])

        if (self.start_layer < 10) and (self.end_layer >= 10):
            self.layer10 = nn.ModuleList([
                BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob),
                BertSelfOutput(hidden_size, dropout_prob)
            ])

        if (self.start_layer < 11) and (self.end_layer >= 11):
            self.layer11 = nn.ModuleList([
                BertIntermediate(hidden_size, intermediate_size),
                BertOutput(hidden_size, intermediate_size, dropout_prob)
            ])

        if (self.start_layer < 12) and (self.end_layer >= 12):
            self.layer12 = nn.ModuleList([
                BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob),
                BertSelfOutput(hidden_size, dropout_prob)
            ])

        if (self.start_layer < 13) and (self.end_layer >= 13):
            self.layer13 = nn.ModuleList([
                BertIntermediate(hidden_size, intermediate_size),
                BertOutput(hidden_size, intermediate_size, dropout_prob)
            ])

        if (self.start_layer < 14) and (self.end_layer >= 14):
            self.layer14 = nn.ModuleList([
                BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob),
                BertSelfOutput(hidden_size, dropout_prob)
            ])

        if (self.start_layer < 15) and (self.end_layer >= 15):
            self.layer15 = nn.ModuleList([
                BertIntermediate(hidden_size, intermediate_size),
                BertOutput(hidden_size, intermediate_size, dropout_prob)
            ])

        if (self.start_layer < 16) and (self.end_layer >= 16):
            self.layer16 = nn.ModuleList([
                BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob),
                BertSelfOutput(hidden_size, dropout_prob)
            ])

        if (self.start_layer < 17) and (self.end_layer >= 17):
            self.layer17 = nn.ModuleList([
                BertIntermediate(hidden_size, intermediate_size),
                BertOutput(hidden_size, intermediate_size, dropout_prob)
            ])

        if (self.start_layer < 18) and (self.end_layer >= 18):
            self.layer18 = nn.ModuleList([
                BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob),
                BertSelfOutput(hidden_size, dropout_prob)
            ])

        if (self.start_layer < 19) and (self.end_layer >= 19):
            self.layer19 = nn.ModuleList([
                BertIntermediate(hidden_size, intermediate_size),
                BertOutput(hidden_size, intermediate_size, dropout_prob)
            ])

        if (self.start_layer < 20) and (self.end_layer >= 20):
            self.layer20 = nn.ModuleList([
                BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob),
                BertSelfOutput(hidden_size, dropout_prob)
            ])

        if (self.start_layer < 21) and (self.end_layer >= 21):
            self.layer21 = nn.ModuleList([
                BertIntermediate(hidden_size, intermediate_size),
                BertOutput(hidden_size, intermediate_size, dropout_prob)
            ])

        if (self.start_layer < 22) and (self.end_layer >= 22):
            self.layer22 = nn.ModuleList([
                BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob),
                BertSelfOutput(hidden_size, dropout_prob)
            ])

        if (self.start_layer < 23) and (self.end_layer >= 23):
            self.layer23 = nn.ModuleList([
                BertIntermediate(hidden_size, intermediate_size),
                BertOutput(hidden_size, intermediate_size, dropout_prob)
            ])

        if (self.start_layer < 24) and (self.end_layer >= 24):
            self.layer24 = nn.ModuleList([
                BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob),
                BertSelfOutput(hidden_size, dropout_prob)
            ])

        if (self.start_layer < 25) and (self.end_layer >= 25):
            self.layer25 = nn.ModuleList([
                BertIntermediate(hidden_size, intermediate_size),
                BertOutput(hidden_size, intermediate_size, dropout_prob)
            ])

        if (self.start_layer < 26) and (self.end_layer >= 26):
            self.layer26 = BertPooler(hidden_size)

        if (self.start_layer < 27) and (self.end_layer >= 27):
            self.layer27 = BertClassifier(hidden_size, num_labels, dropout_prob)

    def forward(self, x, attention_mask=None, token_type_ids=None):
        if (self.start_layer < 1) and (self.end_layer >= 1):
            x = self.layer1(x, token_type_ids)

        if (self.start_layer < 2) and (self.end_layer >= 2):
            x = self.layer2[1](self.layer2[0](x, attention_mask), x)

        if (self.start_layer < 3) and (self.end_layer >= 3):
            x = self.layer3[1](self.layer3[0](x), x)

        if (self.start_layer < 4) and (self.end_layer >= 4):
            x = self.layer4[1](self.layer4[0](x, attention_mask), x)

        if (self.start_layer < 5) and (self.end_layer >= 5):
            x = self.layer5[1](self.layer5[0](x), x)

        if (self.start_layer < 6) and (self.end_layer >= 6):
            x = self.layer6[1](self.layer6[0](x, attention_mask), x)

        if (self.start_layer < 7) and (self.end_layer >= 7):
            x = self.layer7[1](self.layer7[0](x), x)

        if (self.start_layer < 8) and (self.end_layer >= 8):
            x = self.layer8[1](self.layer8[0](x, attention_mask), x)

        if (self.start_layer < 9) and (self.end_layer >= 9):
            x = self.layer9[1](self.layer9[0](x), x)

        if (self.start_layer < 10) and (self.end_layer >= 10):
            x = self.layer10[1](self.layer10[0](x, attention_mask), x)

        if (self.start_layer < 11) and (self.end_layer >= 11):
            x = self.layer11[1](self.layer11[0](x), x)

        if (self.start_layer < 12) and (self.end_layer >= 12):
            x = self.layer12[1](self.layer12[0](x, attention_mask), x)

        if (self.start_layer < 13) and (self.end_layer >= 13):
            x = self.layer13[1](self.layer13[0](x), x)

        if (self.start_layer < 14) and (self.end_layer >= 14):
            x = self.layer14[1](self.layer14[0](x, attention_mask), x)

        if (self.start_layer < 15) and (self.end_layer >= 15):
            x = self.layer15[1](self.layer15[0](x), x)

        if (self.start_layer < 16) and (self.end_layer >= 16):
            x = self.layer16[1](self.layer16[0](x, attention_mask), x)

        if (self.start_layer < 17) and (self.end_layer >= 17):
            x = self.layer17[1](self.layer17[0](x), x)

        if (self.start_layer < 18) and (self.end_layer >= 18):
            x = self.layer18[1](self.layer18[0](x, attention_mask), x)

        if (self.start_layer < 19) and (self.end_layer >= 19):
            x = self.layer19[1](self.layer19[0](x), x)

        if (self.start_layer < 20) and (self.end_layer >= 20):
            x = self.layer20[1](self.layer20[0](x, attention_mask), x)

        if (self.start_layer < 21) and (self.end_layer >= 21):
            x = self.layer21[1](self.layer21[0](x), x)

        if (self.start_layer < 22) and (self.end_layer >= 22):
            x = self.layer22[1](self.layer22[0](x, attention_mask), x)

        if (self.start_layer < 23) and (self.end_layer >= 23):
            x = self.layer23[1](self.layer23[0](x), x)

        if (self.start_layer < 24) and (self.end_layer >= 24):
            x = self.layer24[1](self.layer24[0](x, attention_mask), x)

        if (self.start_layer < 25) and (self.end_layer >= 25):
            x = self.layer25[1](self.layer25[0](x), x)

        if (self.start_layer < 26) and (self.end_layer >= 26):
            x = self.layer26(x)

        if (self.start_layer < 27) and (self.end_layer >= 27):
            x = self.layer27(x)

        return x
