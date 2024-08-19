from graph.layer import GATv2Conv

import torch.nn.functional as F
import torch.nn as nn


def full_block(in_features, out_features, bias=True, p_drop=0.2):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


class GATAutoEncoder(nn.Module):
    def __init__(self, hidden_dims):
        super(GATAutoEncoder, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATv2Conv(in_channels=in_dim, out_channels=num_hidden, add_self_loops=False, bias=False,
                               share_weights=True)
        self.conv2 = GATv2Conv(in_channels=num_hidden, out_channels=out_dim, add_self_loops=False, bias=False,
                               share_weights=True)

        self.conv3 = GATv2Conv(in_channels=out_dim, out_channels=num_hidden, add_self_loops=False, bias=False,
                               share_weights=True)
        self.conv4 = GATv2Conv(in_channels=num_hidden, out_channels=in_dim, add_self_loops=False, bias=False,
                               share_weights=True)

    def forward(self, features, edge_index):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = (self.conv2(h1, edge_index, attention=False))

        self.conv3.lin_l.data = self.conv2.lin_l.data.transpose(0, 1)
        self.conv3.lin_r.data = self.conv2.lin_r.data.transpose(0, 1)

        h3 = F.elu(self.conv3(h2, edge_index, tie_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)
        return h2, h4