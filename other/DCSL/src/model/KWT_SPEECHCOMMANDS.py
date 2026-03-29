import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=1, mlp_dim=256):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        _x = self.ln1(x)
        x_attn = self.mha(_x, _x, _x)[0]
        x = x + x_attn
        x_mlp = self.mlp(self.ln2(x))
        x = x + x_mlp
        return x


class KWT_SPEECHCOMMANDS(nn.Module):
    """
    KWT-1: dim=64, mlp_dim=256, heads=1, layers=12
    Layers:
        1: Linear embedding (n_mfcc -> embed_dim)
        2: CLS token concatenation
        3: Positional embedding + Dropout
        4-15: 12x Transformer encoder blocks
        16: LayerNorm
        17: Classification head
    """
    def __init__(self, start_layer=0, end_layer=17):
        super().__init__()
        self.start_layer = start_layer
        self.end_layer = 17 if end_layer == -1 else end_layer
        
        n_mfcc = 40
        time_steps = 98
        embed_dim = 64
        num_heads = 1
        mlp_dim = 256
        num_classes = 10
        dropout = 0.1
        
        # Layer 1: Linear embedding
        if self.start_layer < 1 <= self.end_layer:
            self.layer1 = nn.Linear(n_mfcc, embed_dim)
        
        # Layer 2: CLS token
        if self.start_layer < 2 <= self.end_layer:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Layer 3: Positional embedding + Dropout
        if self.start_layer < 3 <= self.end_layer:
            self.pos_embed = nn.Parameter(torch.randn(1, time_steps + 1, embed_dim))
            self.dropout = nn.Dropout(dropout)
        
        # Layers 4-15: 12x Transformer encoder blocks
        for i in range(12):
            layer_idx = 4 + i
            if self.start_layer < layer_idx <= self.end_layer:
                setattr(self, f'layer{layer_idx}',
                        TransformerEncoderBlock(embed_dim, num_heads, mlp_dim))
        
        # Layer 16: LayerNorm
        if self.start_layer < 16 <= self.end_layer:
            self.layer16 = nn.LayerNorm(embed_dim)
        
        # Layer 17: Classification head
        if self.start_layer < 17 <= self.end_layer:
            self.layer17 = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        if hasattr(self, 'cls_token'):
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if hasattr(self, 'pos_embed'):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        
        # Layer 1: Linear embedding
        if self.start_layer < 1 <= self.end_layer:
            x = x.transpose(1, 2) 
            x = self.layer1(x)     
        
        # Layer 2: CLS token
        if self.start_layer < 2 <= self.end_layer:
            cls_token = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        # Layer 3: Positional embedding + Dropout
        if self.start_layer < 3 <= self.end_layer:
            x = x + self.pos_embed
            x = self.dropout(x)
        
        # Layers 4-15: 12x Transformer blocks
        for i in range(12):
            layer_idx = 4 + i
            if self.start_layer < layer_idx <= self.end_layer:
                x = getattr(self, f'layer{layer_idx}')(x)
        
        # Layer 16: LayerNorm on CLS token
        if self.start_layer < 16 <= self.end_layer:
            x = self.layer16(x[:, 0])
        
        # Layer 17: Classification
        if self.start_layer < 17 <= self.end_layer:
            x = self.layer17(x)
        
        return x
