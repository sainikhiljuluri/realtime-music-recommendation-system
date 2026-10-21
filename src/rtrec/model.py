"""Two-tower retrieval model in PyTorch.

The user tower combines a learned user embedding with the user's genre profile from
training history. The item tower combines item, genre, and artist embeddings. Both
towers output L2-normalized vectors, so the dot product is cosine similarity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int,
        n_artists: int,
        emb_dim: int = 32,
        hidden: int = 64,
        out_dim: int = 32,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim + n_genres, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.genre_emb = nn.Embedding(n_genres, 16)
        self.artist_emb = nn.Embedding(n_artists, 16)
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim + 32, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
        )
        self.temperature = temperature

    def encode_user(self, user_idx: torch.Tensor, genre_profile: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_emb(user_idx), genre_profile], dim=-1)
        return F.normalize(self.user_mlp(x), dim=-1)

    def encode_item(self, item_idx: torch.Tensor, genre: torch.Tensor, artist: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.item_emb(item_idx), self.genre_emb(genre), self.artist_emb(artist)], dim=-1)
        return F.normalize(self.item_mlp(x), dim=-1)

    def in_batch_softmax_loss(self, user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        """Each row's positive is its own item; other rows' items are negatives."""
        logits = user_vec @ item_vec.T / self.temperature
        labels = torch.arange(user_vec.size(0), device=user_vec.device)
        return F.cross_entropy(logits, labels)
