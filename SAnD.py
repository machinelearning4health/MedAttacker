from transformer import *
import torch.nn as nn
import torch
from torch.autograd import Function

class SAnD(nn.Module):
    def __init__(self, dict_len, embedding_dim,  transformer_hidden, attn_heads, transformer_dropout, transformer_layers):
        super().__init__()

        self.dense_embed = nn.Linear(dict_len+1, embedding_dim)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.classify_layer = nn.Linear(embedding_dim,1)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(256, 4, 256 * 4, transformer_dropout) for _ in range(transformer_layers)])
        self.position = PositionalEmbedding(256, max_len = 50)
            
                
    def forward(self, batch_input, time_info, visit_length):
        batch_input = batch_input[:,0:50,:]
        time_info = time_info[:,0:50]

        batch_size = batch_input.shape[0]
        visit_size = batch_input.shape[1]
        disease_size = batch_input.shape[2]

        s_list = list(range(1,51))
        s_list.reverse()
        visit_weight = torch.tensor(s_list).cuda()
        visit_weight = torch.square(visit_weight/50).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)

        input_embedding = self.dense_embed(batch_input)
        positional_embedding = self.position(input_embedding)
        transformer_input = input_embedding + positional_embedding


        mask = None
        for transformer in self.transformer_blocks:
            transformer_input = transformer.forward(transformer_input, mask)

        final_feature = torch.matmul(visit_weight, transformer_input)
        final_feature = final_feature.squeeze(1)
        classify_score = self.classify_layer(final_feature)
        binary_output = torch.sigmoid(classify_score)

        return binary_output


