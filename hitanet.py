from transformer import *
import torch.nn as nn
import torch
from torch.autograd import Function

class HiTANet(nn.Module):
    def __init__(self, dict_len, embedding_dim,  transformer_hidden, attn_heads, transformer_dropout, transformer_layers):
        super().__init__()

        self.embeds = nn.Embedding(dict_len+1, embedding_dim, padding_idx = dict_len)

        self.dense_embed = nn.Linear(dict_len, embedding_dim)
        self.time_embed1 = nn.Linear(int(1), int(embedding_dim/4))
        self.time_embed2 = nn.Linear(int(embedding_dim/4), embedding_dim)
        self.local_attn = nn.Linear(embedding_dim, 1)        
        self.softmax = torch.nn.Softmax(dim=-1)

        self.global_time_embed1 = nn.Linear(int(1), int(embedding_dim/4))
        self.global_time_embed2 = nn.Linear(int(embedding_dim/4), int(embedding_dim/4))

        self.query_embed = nn.Sequential(nn.Linear(embedding_dim, int(embedding_dim/4)), nn.ReLU())
        self.z_trans = nn.Linear(embedding_dim,2)

        self.classify_layer = nn.Linear(embedding_dim,1)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(256, 4, 256 * 4, transformer_dropout) for _ in range(transformer_layers)])

            
                
    def forward(self, batch_input, time_info, visit_length):
        batch_input = batch_input[:,0:50,:]
        time_info = time_info[:,0:50]

        batch_size = batch_input.shape[0]
        visit_size = batch_input.shape[1]
        disease_size = batch_input.shape[2]

        input_embedding = self.dense_embed(batch_input)
        time_info = time_info.unsqueeze(-1)/180
        time_transform = self.time_embed1(time_info)
        time_transform = 1-torch.tanh(torch.square(time_transform))
        time_embedding = self.time_embed2(time_transform)
        local_time_embedding = input_embedding+time_embedding

        mask = None
        for transformer in self.transformer_blocks:
            local_time_embedding = transformer.forward(local_time_embedding, mask)

        local_attn_weight = self.local_attn(local_time_embedding).squeeze(-1)
        local_attn_weight = self.softmax(local_attn_weight)

        global_time_transform = self.global_time_embed1(time_info)
        global_time_transform = 1-torch.tanh(torch.square(global_time_transform))
        global_time_keys = self.global_time_embed2(global_time_transform)

        query = local_time_embedding[:,-1,:]
        query_embed = self.query_embed(query).unsqueeze(-1)
        
        global_attn = torch.matmul(global_time_keys, query_embed)
        global_attn_weight = self.softmax(local_attn_weight.squeeze(-1))

        attn_weight = self.softmax(self.z_trans(query)).unsqueeze(-1)
        attn_cat = torch.cat((local_attn_weight.unsqueeze(-1),global_attn_weight.unsqueeze(-1)),dim=-1)
        dynamic_weight = torch.matmul(attn_cat, attn_weight).permute(0,-1,1) 
        
        final_feature = torch.matmul(dynamic_weight, local_time_embedding).squeeze(1)
        classify_score = self.classify_layer(final_feature)

        binary_output = torch.sigmoid(classify_score)

        return binary_output


