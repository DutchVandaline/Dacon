import torch
import torch.nn as nn
import torch.nn.functional as F

from PatchEmbedding import PatchEmbedding


class ChaosViT_Anomaly_Detection(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 num_transformer_layers: int = 12,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 mlp_dropout: float = 0.1,
                 embedding_dropout: float = 0.1,
                 mode: str = "anomaly_score"):  # "classification" or "anomaly_score"
        super().__init__()

        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, got {img_size}, {patch_size}"

        self.mode = mode  # Anomaly Detection 방식 선택

        self.num_patches = (img_size * img_size) // patch_size ** 2
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        self.transformer_encoder = nn.Sequential(*[ChaosTransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                                num_heads=num_heads,
                                                                                mlp_size=mlp_size,
                                                                                mlp_dropout=mlp_dropout)
                                                   for _ in range(num_transformer_layers)])

        # ✅ Anomaly Detection 방식: Patch-Level Score 예측
        if mode == "anomaly_score":
            self.anomaly_head = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, 1)  # Patch-Level Anomaly Score
            )
        else:
            # ✅ Binary Classification 방식: Normal vs. Anomaly
            self.classifier = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, 2)  # Binary Classification (Normal vs. Anomaly)
            )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)

        if self.mode == "anomaly_score":
            return self.anomaly_head(x[:, 1:])  # Patch-Level Anomaly Score
        else:
            return self.classifier(x[:, 0])  # Classification (Normal vs. Anomaly)


class ChaosTransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1,
                 attn_dropout: float = 0):
        super().__init__()

        self.msa_block = MultiHeadSelfChaosAttentionBlock(embedding_dim=embedding_dim,
                                                          num_heads=num_heads,
                                                          attn_dropout=attn_dropout)

        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


class MultiHeadSelfChaosAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int = 768, num_heads: int = 12, attn_dropout: float = 0.1,
                 num_iterations=5, r_min=3.8, r_max=4.0):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = embedding_dim // num_heads
        self.num_iterations = num_iterations

        self.r = nn.Parameter(torch.rand(num_heads, 1, 1) * (r_max - r_min) + r_min)

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)

        self.W_o = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(attn_dropout)

        self.scaling_factor = (self.d_k) ** 0.5

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape

        x = self.layer_norm(x)

        Q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scaling_factor

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = F.softmax(attention_scores, dim=-1)

        for _ in range(self.num_iterations):
            attention_scores = self.r * attention_scores * (1 - attention_scores)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        output = torch.matmul(attention_probs, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.num_heads * self.d_k)
        return self.W_o(output)


class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
