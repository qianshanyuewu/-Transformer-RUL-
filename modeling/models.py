"""Shared model definitions for Chapter 3."""
from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_terms = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_terms)
        pe[:, 1::2] = torch.cos(positions * div_terms)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class PaperTransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_projection = nn.Linear(input_dim, d_model)
        self.tgt_projection = nn.Linear(1, d_model)
        self.src_pos_encoder = SinusoidalPositionalEncoding(d_model=d_model)
        self.tgt_pos_encoder = SinusoidalPositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_target_mask(self, seq_len: int, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def _encode_source(self, src):
        src_embed = self.dropout(self.src_pos_encoder(self.src_projection(src)))
        return self.transformer.encoder(src_embed)

    def _decode_target(self, decoder_input, memory):
        if decoder_input.dim() == 2:
            decoder_input = decoder_input.unsqueeze(-1)
        tgt_embed = self.dropout(self.tgt_pos_encoder(self.tgt_projection(decoder_input)))
        tgt_mask = self._build_target_mask(tgt_embed.size(1), tgt_embed.device)
        return self.transformer.decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)

    def forward(self, src, decoder_input):
        memory = self._encode_source(src)
        decoded = self._decode_target(decoder_input, memory)
        last_step = decoded[:, -1, :]
        return self.head(last_step).squeeze(-1)

    def autoregressive_predict(self, src, start_tokens=None, prediction_steps: int | None = None):
        batch_size, seq_len, _ = src.shape
        if prediction_steps is None:
            prediction_steps = max(int(seq_len) - 1, 0)

        if start_tokens is None:
            start_tokens = torch.ones((batch_size, 1, 1), dtype=src.dtype, device=src.device)
        elif start_tokens.dim() == 2:
            start_tokens = start_tokens.unsqueeze(-1)
        start_tokens = start_tokens.to(device=src.device, dtype=src.dtype)

        memory = self._encode_source(src)
        generated = start_tokens
        for _ in range(prediction_steps):
            decoded = self._decode_target(generated, memory)
            next_pred = self.head(decoded[:, -1:, :])
            generated = torch.cat((generated, next_pred), dim=1)

        decoded = self._decode_target(generated, memory)
        last_step = decoded[:, -1, :]
        return self.head(last_step).squeeze(-1)


class EncoderOnlyTransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.dropout(self.pos_encoder(x))
        x = self.encoder(x)
        last_step = x[:, -1, :]
        return self.head(last_step).squeeze(-1)


class RecurrentRegressor(nn.Module):
    def __init__(
        self,
        rnn_type: str,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn_type = rnn_type
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        outputs, _ = self.rnn(x)
        last_step = outputs[:, -1, :]
        return self.head(last_step).squeeze(-1)


class LSTMRegressor(RecurrentRegressor):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__(
            rnn_type="lstm",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )


class GRURegressor(RecurrentRegressor):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__(
            rnn_type="gru",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )


def build_model(model_name: str, input_dim: int, model_config=None):
    model_config = model_config or {}
    model_name = model_name.lower()
    if model_name in ("transformer", "paper_transformer"):
        return PaperTransformerRegressor(
            input_dim=input_dim,
            d_model=int(model_config.get("d_model", 64)),
            num_heads=int(model_config.get("num_heads", 4)),
            num_layers=int(model_config.get("num_layers", 2)),
            ffn_dim=int(model_config.get("ffn_dim", 128)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    if model_name == "encoder_only_transformer":
        return EncoderOnlyTransformerRegressor(
            input_dim=input_dim,
            d_model=int(model_config.get("d_model", 64)),
            num_heads=int(model_config.get("num_heads", 4)),
            num_layers=int(model_config.get("num_layers", 2)),
            ffn_dim=int(model_config.get("ffn_dim", 128)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    if model_name == "lstm":
        return LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=int(model_config.get("hidden_dim", 64)),
            num_layers=int(model_config.get("num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    if model_name == "gru":
        return GRURegressor(
            input_dim=input_dim,
            hidden_dim=int(model_config.get("hidden_dim", 64)),
            num_layers=int(model_config.get("num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    raise ValueError(f"unsupported model_name: {model_name}")
