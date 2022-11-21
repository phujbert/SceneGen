import torch.nn as nn
import torch


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


class GraphConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pooling='avg', mlp_normalization='none'):
        super(GraphConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling

        self.net1 = nn.Sequential(
            nn.Linear(3 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim + output_dim),
            nn.ReLU())

        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU())

    def forward(self, obj_vecs, pred_vecs, edges):

        dtype, device = obj_vecs.dtype, obj_vecs.device
        O, T = obj_vecs.size(0), pred_vecs.size(0)
        Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]

        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        new_t_vecs = self.net1(cur_t_vecs)

        new_s_vecs = new_t_vecs[:, :H]
        new_p_vecs = new_t_vecs[:, H:(H + Dout)]
        new_o_vecs = new_t_vecs[:, (H + Dout):(2 * H + Dout)]

        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        if self.pooling == 'avg':
            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

        new_obj_vecs = self.net2(pooled_obj_vecs)

        return new_obj_vecs, new_p_vecs


class GraphConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5, pooling='avg'):
        super(GraphConvNet, self).__init__()

        self.obj_embedding = nn.Embedding(185, 128)
        self.pred_embedding = nn.Embedding(11, 128)

        self.num_layers = num_layers
        self.gcn = nn.ModuleList()
        for _ in range(num_layers):
            self.gcn.append(GraphConv(input_dim, hidden_dim, output_dim, pooling=pooling))

    def forward(self, objs, preds, edges):
        obj_vecs = self.obj_embedding(objs)
        pred_vecs = self.pred_embedding(preds)
        for i in range(self.num_layers):
            layer = self.gcn[i]
            obj_vecs, pred_vecs = layer(obj_vecs, pred_vecs, edges)
        return obj_vecs
