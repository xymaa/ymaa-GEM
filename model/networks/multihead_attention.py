# 自注意力机制qkv是由同一个x产生
# 自注意力关注序列内部的依赖关系
# 互注意力是由不同的x产生
# 互注意力关注两个不同序列之间的依赖关系（例如原文和译文）
from typing import Dict, Optional
import torch
from torch import Tensor, nn

class SelfMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, bias=True, scaling_factor=0.1):
        super(SelfMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_rate)
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.softmax = nn.Softmax(dim=-1)
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tensor:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        q = (
                q.view(bsz, tgt_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
                * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        assert k is not None

        src_len = k.size(1)

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            attn =self.dropout(self.softmax(attn_weights))
        else:
            attn_weights += attn_bias
            attn =self.dropout(self.softmax(attn_weights))

        if not return_attn:
            attn =self.dropout(self.softmax(attn_weights))
        else:
            attn_weights += attn_bias
            attn =self.dropout(self.softmax(attn_weights))

        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )

        o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn

class CrossMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout_rate=0.1,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax()
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn =self.dropout(self.softmax(attn_weights))
        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        return o
