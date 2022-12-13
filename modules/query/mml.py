import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry
from .innerproduct_similarity import InnerproductSimilarity
from modules.utils import GraphFunc, _l2norm, l2distance, batched_index_select



@registry.Query.register("MML")
class MML(nn.Module):

    def __init__(self, in_channels, cfg):
        super().__init__()

        self.n_way = cfg.n_way
        self.k_shot = cfg.k_shot
        self.neighbor_k = cfg.model.nbnn_topk
        self.part_num = 13
        self.inner_simi = InnerproductSimilarity(cfg, metric='cosine')
        self.temperature = cfg.model.temperature
        self.criterion = nn.CrossEntropyLoss()
        self.embed_dim = 640
        self.graph_func = GraphFunc(self.embed_dim)

        self.Norm_layer = nn.BatchNorm1d(self.n_way , affine=True)
    def cal_covariance_matrix(self, feature):
        n_local_descriptor = torch.tensor(feature.size(1)).cuda()
        feature_mean = torch.mean(feature, 1, True)
        feature = feature - feature_mean
        cov_matrix = torch.matmul(feature.permute(0, 2, 1), feature)
        cov_matrix = torch.div(cov_matrix, n_local_descriptor - 1)

        return feature_mean, cov_matrix

    def wasserstein_distance_raw_Batch(self, mean1, cov1, mean2, cov2):
        mean_diff = mean1 - mean2.squeeze(1)  # 75 * 5 * 64
        cov_diff = cov1.unsqueeze(1) - cov2  # 75 * 5 * 64 * 64
        l2_norm_mean = torch.div(torch.norm(mean_diff, p=2, dim=2), mean1.size(2))  # 75 * 5
        l2_norm_cova = torch.div(torch.norm(cov_diff, p=2, dim=(2, 3)), mean1.size(2) * mean1.size(2))  # 75 * 5

        return l2_norm_mean + l2_norm_cova

    def _scores(self, support_xf, support_y, query_xf, query_y):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        # support_xf = support_xf.view(b, self.n_way, self.k_shot, -1, h, w)
        support_xf = support_xf.view((-1,) + support_xf.shape[-3:])
        support_xf = F.adaptive_avg_pool2d(support_xf, 1).view(b, self.n_way, self.k_shot, c)
        support_proto = support_xf.mean(-2)  # [b, self.n_way, c]

        query_xf = F.adaptive_avg_pool2d(query_xf.view(-1, c, h, w), 1).view(b, q, c)
        scores = -l2distance(query_xf.transpose(-2, -1).contiguous(), support_proto.transpose(-2, -1).contiguous())
        scores = scores.view(b * q, -1)
        return scores


    def forward(self, support_xf, support_y, query_xf, query_y, support_xf_part, query_xf_part):
        b, q, c, h, w = query_xf.shape
        s = support_xf.shape[1]

        #pixel-level
        innerproduct_matrix = self.inner_simi(support_xf, support_y, query_xf, query_y)
        topk_value, _ = torch.topk(innerproduct_matrix, 1, -1)  # [b, q, N, M_q, neighbor_k]
        similarity_matrix = topk_value.mean(-1).view(b, q, self.n_way, -1).sum(-1)
        pixel_similarity = similarity_matrix.view(b * q, self.n_way)

        # global-level
        global_similarity = self._scores(support_xf, support_y, query_xf, query_y)

        #part-level
        b, q, c, h, w = query_xf_part.shape
        s = support_xf_part.shape[1]
        query_xf_part = query_xf_part.view(b, q, c, -1).contiguous()

        support_xf_part = support_xf_part.contiguous().view(b, s, c, -1).transpose(-1, -2)
        support_xf_part = support_xf_part.contiguous().view(b, -1, c)

        graph_label = torch.arange(self.n_way * self.k_shot * self.part_num).long()
        support_xf_part = self.graph_func(support_xf_part, graph_label)
        support_xf_part = support_xf_part.contiguous().view(b, s, -1, c).transpose(-1, -2)
        support_xf_part = support_xf_part.contiguous().view(b, self.n_way, self.k_shot, c, -1).permute(0, 1, 3, 2, 4)
        support_xf_part = support_xf_part.contiguous().view(b, self.n_way, c, -1)
        support_xf_part = support_xf_part.unsqueeze(1)
        query_xf_part = query_xf_part.unsqueeze(2)

        support_xf_part = _l2norm(support_xf_part, dim=-2)
        query_xf_part = _l2norm(query_xf_part, dim=-2)
        query_xf_part = query_xf_part.transpose(-1, -2)

        patch_simi_matr = query_xf_part @ support_xf_part
        part_similarity = torch.topk(patch_simi_matr, 1, -1)[0]
        part_similarity = part_similarity.mean(-1).view(b, q, self.n_way, -1).sum(-1)
        part_similarity = part_similarity.view(b * q, self.n_way)

        part_similarity_c = F.softmax(part_similarity, dim=-1)
        pixel_similarity_c = F.softmax(pixel_similarity, dim=-1)
        global_similarity_c = F.softmax(global_similarity, dim=-1)
        sim = 1 * part_similarity_c + 0.5 * pixel_similarity_c + 0.5 * global_similarity_c
        query_y = query_y.view(b * q)
        if self.training:
            loss1 = self.criterion(part_similarity, query_y)
            loss2 = self.criterion(pixel_similarity, query_y)
            loss3 = self.criterion(global_similarity  / 64, query_y)
            loss = 1 * loss1 + 0.5 * loss2 + 0.5 * loss3
            return {"mml_loss": loss}

        else:
            _, predict_labels = torch.max(sim, 1)
            rewards = [1 if predict_labels[j] == query_y[j].to(predict_labels.device) else 0 for j in
                       range(len(query_y))]
            return rewards
