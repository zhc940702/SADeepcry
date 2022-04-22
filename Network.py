import torch
import torch.nn.functional as F
from torch import nn
import math
Max_length = 1000
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size
        self.all_head_size = hidden_size * num_attention_heads

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores_1(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_2(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, int(self.attention_head_size / self.num_attention_heads))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores_1(mixed_query_layer)
        key_layer = self.transpose_for_scores_1(mixed_key_layer)
        value_layer = self.transpose_for_scores_1(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_head_size * self.num_attention_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size * num_attention_heads, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class multimodal_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(multimodal_Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, num_attention_heads, hidden_dropout_prob)

    def forward(self, input_tensor):
        self_output = self.self(input_tensor)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class Protein_Crystallization(nn.Module):
    def __init__(self, protein_sequence_dim, protein_finger_dim, hid_embed_dim, attention_number, dropout, **base_config_TextCNN):
        super(Protein_Crystallization, self).__init__()
        self.dropout = dropout
        self.pemb = nn.Linear(21, hid_embed_dim)
        self.protein_sequence_dim = protein_sequence_dim
        self.embed_dim = hid_embed_dim
        self.protein_finger_dim = protein_finger_dim
        # self.CNN_model = TextCNN(base_config_TextCNN)
        self.number_attention = attention_number

        self.protein_sequence_layer = nn.Linear(self.embed_dim * Max_length, self.embed_dim)
        self.protein_sequence_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.protein_sequence_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.protein_finger_layer = nn.Linear(self.protein_finger_dim, self.embed_dim)
        self.protein_finger_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.protein_finger_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.total_layer = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.total_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.total_layer_1 = nn.Linear(self.embed_dim, 2)
        self.sigmoid = nn.Sigmoid()
        self.attention = multimodal_Attention(self.embed_dim, self.number_attention, self.dropout, self.dropout)


    def forward(self, protein_sequence, fingerprint, device):

        p_emb = self.pemb(protein_sequence.to(torch.float32))
        p_feature = self.attention(p_emb)
        p_feature = p_feature.view(p_feature.shape[0], -1)
        p_feature = F.relu(self.protein_sequence_bn(self.protein_sequence_layer(p_feature)), inplace=True)
        p_feature = F.dropout(p_feature, training=self.training, p=self.dropout)
        p_feature = self.protein_sequence_layer_1(p_feature)
        fingerprint = F.relu(self.protein_finger_bn(self.protein_finger_layer(fingerprint.to(device))), inplace=True)
        fingerprint = F.dropout(fingerprint, training=self.training, p=self.dropout)
        fingerprint = self.protein_finger_layer_1(fingerprint)

        p_feature = torch.cat([p_feature, fingerprint], dim=1)

        p_feature = F.relu(self.total_bn(self.total_layer(p_feature)), inplace=True)
        p_feature = F.dropout(p_feature, training=self.training, p=self.dropout)
        p_feature = self.total_layer_1(p_feature)

        # print(self.total)
        return p_feature