import torch
import torch.nn as nn
import math

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model = 768, nhead = 16, num_layers = 4, seq_len = 50, dim_feedforward = 2048, dropout=0.1):
        super(TransformerLM, self).__init__()
        self.tokens_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(seq_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(d_model, nhead, dropout = dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                                 	nn.ReLU(),
                                                 	nn.Linear(dim_feedforward, d_model)))
            self.layer_norms_1.append(nn.LayerNorm(d_model, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(d_model, eps=1e-12))
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean = 0.0, std = 0.02)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, hidden = None, **kwargs):
        x = x.transpose(0, 1)
        positions = torch.arange(len(x), device = x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = torch.full((len(x), len(x)), -float('Inf'), device = h.device, dtype = h.dtype)
        attn_mask = torch.triu(attn_mask, diagonal = 1)
        for i, (layer_norm_1, attention, layer_norm_2, feed_forward) in enumerate(zip(self.layer_norms_1, self.attentions,
                                                                                        self.layer_norms_2, self.feed_forwards)):

            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        logits = self.lm_head(h)
        logits = logits.permute(1, 0, 2)
        return logits, (torch.zeros(self.d_model), )