import torch
import torch.nn as nn
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_dim=256):
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
        # Attention + residual
        _x = self.ln1(x)
        x_attn = self.mha(_x, _x, _x)[0]
        x = x + x_attn

        # MLP + residual
        x_mlp = self.mlp(self.ln2(x))
        x = x + x_mlp
        return x


class ViT_CIFAR10(nn.Module):

    def __init__(self, start_layer=0, end_layer=12):
        super(ViT_CIFAR10, self).__init__()
        self.start_layer = start_layer
        self.end_layer = end_layer

        img_size = 32
        patch_size = 4
        in_channels = 3
        embed_dim = 128
        num_classes = 10
        num_patches = (img_size // patch_size) ** 2

        if (self.start_layer < 1) and (self.end_layer >= 1):
            self.layer1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        if (self.start_layer < 2) and (self.end_layer >= 2):
            self.layer2 = nn.Flatten(2)

        if (self.start_layer < 3) and (self.end_layer >= 3):
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        if (self.start_layer < 4) and (self.end_layer >= 4):
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
            self.layer4 = nn.Identity()

        if (self.start_layer < 5) and self.end_layer >= 5:
            self.layer5 = TransformerEncoderBlock(embed_dim=128)

        if (self.start_layer < 6) and (self.end_layer >= 6):
            self.layer6 = TransformerEncoderBlock(embed_dim=128)

        if (self.start_layer < 7) and (self.end_layer >= 7):
            self.layer7 = TransformerEncoderBlock(embed_dim=128)

        if (self.start_layer < 8) and (self.end_layer >= 8):
            self.layer8 = TransformerEncoderBlock(embed_dim=128)

        if (self.start_layer < 9) and (self.end_layer >= 9):
            self.layer9 = TransformerEncoderBlock(embed_dim=128)

        if (self.start_layer < 10) and (self.end_layer >= 10):
            self.layer10 = TransformerEncoderBlock(embed_dim=128)

        if (self.start_layer < 11) and (self.end_layer >= 11):
            self.layer11 = nn.LayerNorm(embed_dim)

        if (self.start_layer < 12) and (self.end_layer >= 12):
            self.layer12 = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        if self.start_layer < 1 <= self.end_layer:
            x = self.layer1(x)

        if self.start_layer < 2 <= self.end_layer:
            x = self.layer2(x)
            x = x.transpose(1, 2)

        if self.start_layer < 3 <= self.end_layer:
            cls_token = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        if self.start_layer < 4 <= self.end_layer:
            x = self.layer4(x + self.pos_embed)

        if self.start_layer < 5 <= self.end_layer:
            x = self.layer5(x)

        if self.start_layer < 6 <= self.end_layer:
            x = self.layer6(x)

        if self.start_layer < 7 <= self.end_layer:
            x = self.layer7(x)

        if self.start_layer < 8 <= self.end_layer:
            x = self.layer8(x)

        if self.start_layer < 9 <= self.end_layer:
            x = self.layer9(x)

        if self.start_layer < 10 <= self.end_layer:
            x = self.layer10(x)

        if self.start_layer < 11 <= self.end_layer:
            x = self.layer11(x[:, 0])

        if self.start_layer < 12 <= self.end_layer:
            x = self.layer12(x)
        return x
