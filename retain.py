import torch.nn as nn
import torch
from torch.autograd import Function

class Retain(nn.Module):
    def __init__(self, dict_len, embedding_dim,  transformer_hidden, attn_heads, transformer_dropout, transformer_layers):
        super().__init__()

        retain_embed = 2*embedding_dim
        self.softmax = torch.nn.Softmax(dim=-1)
        self.retain_embed = nn.Linear(dict_len, retain_embed)
        self.gru = nn.GRU(input_size = retain_embed, hidden_size = retain_embed, batch_first = True, bidirectional = False)
        self.retain_attn_a = nn.Linear(retain_embed, 1)
        self.gru_b = nn.GRU(input_size = retain_embed, hidden_size = retain_embed, batch_first = True, bidirectional = False)
        self.retain_attn_b = nn.Sequential(nn.Linear(retain_embed, retain_embed), nn.Tanh())
        self.pred_feat = nn.Sequential(nn.Linear(4*retain_embed, 2*retain_embed), nn.Tanh())
        self.classify_layer = nn.Linear(retain_embed, 1)

                
    def forward(self, batch_input, time_info, visit_length):
        batch_input = batch_input[:,0:50,:]
        input_embedding = self.retain_embed(batch_input)
        gru_output = self.gru(input_embedding)

        gru_embedding = gru_output[0]

        attn_a = self.retain_attn_a(gru_embedding).squeeze(-1)
        attn_a = self.softmax(attn_a).unsqueeze(-1).permute(0, 2, 1)

        gru_b_output = self.gru_b(input_embedding)
        gru_b_embedding = gru_b_output[0]
        attn_b = self.retain_attn_b(gru_b_embedding)

        attn_b_input = input_embedding * attn_b
        context = torch.matmul(attn_a, attn_b_input).squeeze(1)

        classify_score = self.classify_layer(context)
        binary_output = torch.sigmoid(classify_score)
        
        
        return binary_output


