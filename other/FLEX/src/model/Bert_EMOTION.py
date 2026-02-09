import torch
import torch.nn as nn
import torch.nn.functional as F

class DotDict(dict):
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v
    def __delattr__(self, k): del self[k]

# AG_NEWS have 4 layers: World, Sports, Business, Sci/Tech
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

    def forward(self, hidden_states):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        import math
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

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

class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(BertAttention, self).__init__()
        self.self = BertSdpaSelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.output = BertSelfOutput(hidden_size, dropout_prob)

    def forward(self, hidden_states):
        self_output = self.self(hidden_states)
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

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

class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_prob):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads, dropout_prob)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size, intermediate_size, dropout_prob)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

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

class BertClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob=0.1):
        super(BertClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class Bert(nn.Module):
    def __init__( self, vocab_size=28996, hidden_size=768, num_attention_heads=12, intermediate_size=3072,
        max_position_embeddings=512, type_vocab_size=2, dropout_prob=0.1, layer_id=0, n_block=12
    ):
        super(Bert, self).__init__()
        self.layer_id = layer_id
        self.config = DotDict(
            model_type="bert",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            bos_token_id=101, eos_token_id=102, pad_token_id=0,
            is_encoder_decoder=False, tie_word_embeddings=False,
            use_return_dict=True, output_attentions=False, output_hidden_states=False
        )

        if self.layer_id == 1:
            self.embeddings = BertEmbeddings(vocab_size=vocab_size, hidden_size=hidden_size, max_position_embeddings=max_position_embeddings,
                                             type_vocab_size=type_vocab_size,dropout_prob=dropout_prob)
            self.layers = nn.ModuleList(
                [BertLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob)
                 for _ in range(n_block)]
            )
        elif self.layer_id == 2:
            self.layers = nn.ModuleList(
                [BertLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob)
                 for _ in range(n_block)]
            )
            self.pooler = BertPooler(hidden_size)
            self.dropout = nn.Dropout(dropout_prob)
            self.classifier = nn.Linear(hidden_size, 4)
        else:
            self.embeddings = BertEmbeddings(vocab_size=vocab_size, hidden_size=hidden_size,
                                             max_position_embeddings=max_position_embeddings,
                                             type_vocab_size=type_vocab_size, dropout_prob=dropout_prob)
            self.layers = nn.ModuleList(
                [BertLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob)
                 for _ in range(n_block)]
            )
            self.pooler = BertPooler(hidden_size)
            self.dropout = nn.Dropout(dropout_prob)
            self.classifier = nn.Linear(hidden_size, 4)

    def forward(self, input_ids, token_type_ids=None,**kwargs):

        if self.layer_id == 1:
            x = self.embeddings(input_ids, token_type_ids)
            for encode in self.layers:
                x = encode(x)
        elif self.layer_id == 2:
            x = input_ids
            for encode in self.layers:
                x = encode(x)
            x = self.pooler(x)
            x = self.dropout(x)
            x = self.classifier(x)
        else:
            x = self.embeddings(input_ids, token_type_ids)
            for encode in self.layers:
                x = encode(x)
            x = self.pooler(x)
            x = self.dropout(x)
            x = self.classifier(x)

        return x
