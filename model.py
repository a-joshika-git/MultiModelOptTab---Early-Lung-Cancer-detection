"""MultiModalOptTab model architecture (matches notebook training code)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from pytorch_tabnet.tab_network import TabNet


class EfficientNetB0Encoder(nn.Module):
    def __init__(self, embed_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        base = tv_models.efficientnet_b0(weights=None)
        base.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.features = base.features
        self.avgpool = base.avgpool
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.avgpool(self.features(x)))


class TabNetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        n_d: int = 128,
        n_a: int = 128,
        n_steps: int = 6,
        gamma: float = 1.3,
        momentum: float = 0.02,
    ):
        super().__init__()
        group_matrix = torch.eye(input_dim)
        self.tabnet = TabNet(
            input_dim=input_dim,
            output_dim=embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=1,
            n_independent=2,
            n_shared=2,
            epsilon=1e-15,
            virtual_batch_size=256,
            momentum=momentum,
            mask_type="entmax",
            group_attention_matrix=group_matrix,
        )

    def _fix_device(self) -> None:
        device = next(self.parameters()).device
        enc = self.tabnet.tabnet.encoder
        if enc.group_attention_matrix.device != device:
            enc.group_attention_matrix = enc.group_attention_matrix.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._fix_device()
        out, _ = self.tabnet(x)
        if isinstance(out, (list, tuple)):
            out = torch.stack(out).mean(0)
        return out


class BayesianDropout(nn.Module):
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        img_dim: int = 512,
        tab_dim: int = 256,
        fusion_dim: int = 512,
        n_heads: int = 4,
    ):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, fusion_dim)
        self.tab_proj = nn.Linear(tab_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            fusion_dim, n_heads, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, img_emb: torch.Tensor, tab_emb: torch.Tensor) -> torch.Tensor:
        q = self.img_proj(img_emb).unsqueeze(1)
        k = self.tab_proj(tab_emb).unsqueeze(1)
        attn_out, _ = self.cross_attn(q, k, k)
        return self.norm(attn_out.squeeze(1) + q.squeeze(1))


class MultiModalOptTab(nn.Module):
    def __init__(
        self,
        tab_input_dim: int,
        img_embed_dim: int = 512,
        tab_embed_dim: int = 256,
        fusion_dim: int = 512,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        bnn_hidden: int = 256,
        dropout_rate: float = 0.3,
        n_classes: int = 2,
    ):
        super().__init__()
        self.img_encoder = EfficientNetB0Encoder(img_embed_dim, dropout_rate)
        self.tab_encoder = TabNetEncoder(tab_input_dim, tab_embed_dim)
        self.cross_fusion = CrossAttentionFusion(
            img_embed_dim, tab_embed_dim, fusion_dim, n_heads
        )
        enc_layer = nn.TransformerEncoderLayer(
            fusion_dim,
            n_heads,
            fusion_dim * 4,
            dropout_rate,
            "gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_transformer_layers)
        self.bnn_head = nn.Sequential(
            nn.Linear(fusion_dim, bnn_hidden),
            nn.GELU(),
            BayesianDropout(dropout_rate),
            nn.LayerNorm(bnn_hidden),
            nn.Linear(bnn_hidden, bnn_hidden // 2),
            nn.GELU(),
            BayesianDropout(dropout_rate),
            nn.Linear(bnn_hidden // 2, n_classes),
        )

    def forward_once(self, img: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        fused = self.cross_fusion(self.img_encoder(img), self.tab_encoder(tab))
        tf_out = self.transformer(fused.unsqueeze(1)).squeeze(1)
        return self.bnn_head(tf_out)

    def forward(
        self, img: torch.Tensor, tab: torch.Tensor, mc_passes: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mc_passes == 1:
            return self.forward_once(img, tab), None
        probs = torch.stack(
            [F.softmax(self.forward_once(img, tab), -1) for _ in range(mc_passes)]
        )
        return torch.log(probs.mean(0) + 1e-8), probs.var(0).sum(1)


class GradCamWrapper(nn.Module):
    def __init__(self, full_model: MultiModalOptTab, tab_input_dim: int):
        super().__init__()
        self.model = full_model
        self.tab_input_dim = tab_input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_emb = self.model.img_encoder(x)
        tab = torch.zeros(x.size(0), self.tab_input_dim, device=x.device)
        tab_emb = self.model.tab_encoder(tab)
        fused = self.model.cross_fusion(img_emb, tab_emb)
        return self.model.bnn_head(self.model.transformer(fused.unsqueeze(1)).squeeze(1))
