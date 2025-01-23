import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from torch import Tensor
from torch.nn import Parameter
from torch.nn import init
from torch.nn import Module, Linear, ReLU, ModuleList, BatchNorm1d, Dropout
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value


class GATv2Conv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
    static attention problem of the standard
    :class:`~torch_geometric.conv.GATConv` layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in :class:`GATv2`, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_j \, \Vert \, \mathbf{e}_{i,j}]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(\mathbf{\Theta}
        [\mathbf{x}_i \, \Vert \, \mathbf{x}_k \, \Vert \, \mathbf{e}_{i,k}]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights
        self.bias = bias

        self.lin_l = Parameter(torch.zeros(size=(in_channels, out_channels)))
        init.xavier_normal_(self.lin_l.data, gain=1.414)

        # share weights by default
        self.lin_r = self.lin_l

        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        init.xavier_normal_(self.att_src.data, gain=1.414)
        init.xavier_normal_(self.att_dst.data, gain=1.414)

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        self._alpha = None
        self.attentions = None

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                attention=True,
                tie_attention=None,
                return_attention_weights: bool = None):
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            # x_l = self.lin_l(x).view(-1, H, C)
            x_l = torch.mm(x, self.lin_l).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None
        x = (x_l, x_r)

        if not attention:
            return x[0].mean(dim=1)

        if tie_attention is None:
            alpha_l = (x_l * self.att_src).sum(dim=-1)
            alpha_r = None if x_r is None else (x_r * self.att_dst).sum(dim=-1)
            alpha = (alpha_l, alpha_r)
            self.attentions = alpha
        else:
            alpha = tie_attention

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=None)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        # x = F.sigmoid(x)
        alpha = torch.sigmoid(alpha)
        # alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GraphSAGE(Module):
    def __init__(self, in_channels, hidden_dims, agg_class, dropout=0., num_samples=25, BN=False):
        super(GraphSAGE, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.agg_class = agg_class
        self.aggregators = ModuleList([agg_class(in_channels, in_channels)])
        self.aggregators.extend([agg_class(dim, dim) for dim in hidden_dims])

        self.dropout = dropout
        self.num_samples = num_samples
        self.BN = BN

        self.fcs = ModuleList([Linear(2 * in_channels, hidden_dims[0])])
        self.fcs.extend([Linear(2 * hidden_dims[i - 1], hidden_dims[i]) for i in range(1, len(hidden_dims))])

        if self.BN:
            self.bns = ModuleList([BatchNorm1d(hidden_dim) for hidden_dim in hidden_dims])
        else:
            self.bns = None

        self.dropout = Dropout(dropout)
        self.relu = ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, features, node_layers, mapping, rows):
        out = features
        for k in range(len(self.hidden_dims)):
            nodes = node_layers[k + 1]
            mapping = mapping[k]
            init_mapped_nodes = np.array([mapping[0][v] for v in nodes], dtype=np.int64)
            cur_rows = rows[init_mapped_nodes]
            aggregate = self.aggregators[k](out, nodes, mapping, cur_rows, self.num_samples)
            cur_mapped_nodes = np.array([mapping[v] for v in nodes], dtype=np.int64)

            out = torch.cat((out[cur_mapped_nodes, :], aggregate), dim=1)
            out = self.fcs[k](out)
            if k + 1 < len(self.hidden_dims):
                if self.BN:
                    out = self.bns[k](out)
                else:
                    out = self.relu(out)
                    out = self.dropout(out)
        return out


class Aggregator(Module):
    def __init__(self, in_channels: int=None, out_channels: int=None):
        super().__init__()
        self.input_dim = in_channels
        self.output_dim = out_channels

    def forward(self, features, nodes, mapping, rows, num_samples):
        mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        if num_samples == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [np.random.choice(row, min(len(row), num_samples), len(row) < num_samples) for row in mapped_rows]

        n = len(nodes)
        out = torch.zeros(n, self.output_dim)
        for i in range(n):
            if len(sampled_rows[i]) != 0:
                out[i, :] = self._aggregate(features[sampled_rows[i], :])
        return out

    def _aggregate(self, features):
        raise NotImplementedError


class PoolAggregator(Aggregator):

    def __init__(self, input_dim, output_dim):
        super(PoolAggregator, self).__init__(input_dim, output_dim)

        self.fc1 = Linear(input_dim, output_dim)
        self.relu = ReLU()

    def _aggregate(self, features):
        out = self.relu(self.fc1(features))
        return self._pool_fn(out)

    def _pool_fn(self, features):
        raise NotImplementedError


class MaxPoolAggregator(PoolAggregator):
    def _pool_fn(self, features):
        return torch.max(features, dim=0)[0]


class MeanPoolAggregator(PoolAggregator):
    def _pool_fn(self, features):
        return torch.mean(features, dim=0)
