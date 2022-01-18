import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from itertools import permutations
import scipy.sparse as sp

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
    
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx    


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    if torch.cuda.is_available():
        return torch.sparse.FloatTensor(indices, values, shape).cuda()
    else:
        return torch.sparse.FloatTensor(indices, values, shape)

class GraphFunc(nn.Module):
    def __init__(self, z_dim):
        super(GraphFunc, self).__init__()
        """
        DeepSets Function
        """
        self.gc1 = GraphConvolution(z_dim, z_dim * 4)
        self.gc2 = GraphConvolution(z_dim * 4, z_dim)
        self.z_dim = z_dim

    def forward(self, graph_input_raw, graph_label):
        """
        set_input, seq_length, set_size, dim
        """
        set_length, set_size, dim = graph_input_raw.shape
        assert(dim == self.z_dim)
        set_output_list = []
        
        for g_index in range(set_length):
            graph_input = graph_input_raw[g_index, :]
            # construct the adj matrix
            unique_class = np.unique(graph_label)
            edge_set = []
            for c in unique_class:
                current_index = np.where(graph_label == c)[0].tolist()
                if len(current_index) > 1:
                    edge_set.append(np.array(list(permutations(current_index, 2))))
            
            if len(edge_set) == 0:
                adj = sp.coo_matrix((np.array([0]), (np.array([0]), np.array([0]))),
                                    shape=(graph_label.shape[0], graph_label.shape[0]),
                                    dtype=np.float32)
            else:
                edge_set = np.concatenate(edge_set, 0)
                adj = sp.coo_matrix((np.ones(edge_set.shape[0]), (edge_set[:, 0], edge_set[:, 1])),
                                    shape=(graph_label.shape[0], graph_label.shape[0]),
                                    dtype=np.float32)        
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = normalize(adj + sp.eye(adj.shape[0]))
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            
            # do GCN process
            residual = graph_input
            graph_input = F.relu(self.gc1(graph_input, adj))
            graph_input = F.dropout(graph_input, 0.5, training=self.training)
            graph_input = self.gc2(graph_input, adj)        
            set_output = residual + graph_input
            set_output_list.append(set_output)
        
        return torch.stack(set_output_list)
def _l2norm(x, dim=1, keepdim=True):
    return x / (1e-16 + torch.norm(x, 2, dim, keepdim))

def l2distance(x, y):
    """
    Input:
        x [.., c, M_x]
        y [.., c, M_y]
    Return:
        ret [.., M_x, M_y]
    """
    
    assert x.shape[:-2] == y.shape[:-2]
    prefix_shape = x.shape[:-2]

    c, M_x = x.shape[-2:]
    M_y = y.shape[-1]
    
    x = x.view(-1, c, M_x)
    y = y.view(-1, c, M_y)

    x_t = x.transpose(1, 2)
    x_t2 = x_t.pow(2.0).sum(-1, keepdim=True)
    y2 = y.pow(2.0).sum(1, keepdim=True)

    ret = x_t2 + y2 - 2.0 * x_t@y
    ret = ret.view(prefix_shape + (M_x, M_y))
    return ret

def batched_index_select(input_, dim, index):
    for ii in range(1, len(input_.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input_.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input_, dim, index)

def multihot_embedding(x, k, dim=-1):
    _, indice_ = torch.topk(x, k, dim=dim)
    shape_ = list(x.shape)
    shape_[dim] = k
    e = torch.zeros_like(x).scatter_(dim, indice_, torch.ones((shape_), device=x.device))
    return e

def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))

def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module

class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.
    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})
    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn

