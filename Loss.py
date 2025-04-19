import torch
from torch import nn
from torch.nn import BCELoss
import torch.nn.functional as F
import numpy as np
from hyptorch.pmath import dist_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def contrastive_loss(x0, x1, tau, hyper_c):
    # x0 and x1 - positive pair
    # tau - temperature
    # hyp_c - hyperbolic curvature, "0" enables sphere mode

    if hyper_c == 0:
        dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
    else:
        dist_f = lambda x, y: -dist_matrix(x, y)
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    # dist = dist_f(x0, x0)
    # dist2 = -euclidean_distance(x0, x0)
    logits00 = dist_f(x0, x0) / tau - eye_mask
    logits00_numpy = logits00.detach().cpu().numpy()
    logits01 = dist_f(x0, x1) / tau
    logits = torch.cat([logits01, logits00], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    loss = F.cross_entropy(logits, target)
    stats = {
        "logits/min": logits01.min().item(),
        "logits/mean": logits01.mean().item(),
        "logits/max": logits01.max().item(),
        "logits/acc": (logits01.argmax(-1) == target).float().mean().item(),
        "logits": logits,
    }
    return loss, stats




def false_negative_mask(center_index_level):
    bs = center_index_level.shape[0]
    mask = torch.zeros(bs, bs, dtype=torch.uint8)
    for i in range(bs):
        for j in range(bs):
            mask[i][j] = 1 if center_index_level[i] == center_index_level[j] and i != j else 0
    # return torch.cat([mask, mask], dim=1)
    return mask


def instance_contrastive_loss(x0, x1, tau, center_index0, center_index1, hyper_c):
    level = center_index0.shape[1]
    if hyper_c == 0:
        dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
    else:
        dist_f = lambda x, y: -dist_matrix(x, y)
    loss_all = 0
    bsize = x0.shape[0]
    for i in range(level):
        torch.set_printoptions(edgeitems=torch.inf)
        center_index_level0 = center_index0[:, i]
        center_index_level1 = center_index1[:, i]
        mask0 = false_negative_mask(center_index_level0).cuda()
        mask1 = false_negative_mask(center_index_level1).cuda()
        mask01 = torch.cat([mask0, mask1], dim=1)
        mask10 = torch.cat([mask1, mask0], dim=1)

        eye_mask = torch.eye(bsize).cuda() * 1e20
        logits00 = dist_f(x0, x0) - eye_mask
        logits01 = dist_f(x0, x1)
        logits11 = dist_f(x1, x1) - eye_mask
        logits10 = dist_f(x1, x0)
        logits_1st = torch.cat([logits01, logits00], dim=1)
        pos_distances_1st = torch.diag(logits01)
        neg_distances_1st = torch.sum(logits01, dim=1) - pos_distances_1st

        # tau_1st = calculate_adaptive_temperature(pos_distances_1st, neg_distances_1st, base_temperature=tau)
        tau_1st = tau
        logits_1st = logits_1st - mask10 * 1e20
        positive_1st = torch.diag(logits01)
        logits_2nd = torch.cat([logits10, logits11], dim=1)
        pos_distances_2nd = torch.diag(logits10)
        neg_distances_2nd = torch.sum(logits10, dim=1) - pos_distances_2nd
        # tau_2nd = calculate_adaptive_temperature(pos_distances_2nd, neg_distances_2nd, base_temperature=tau)
        tau_2nd = tau
        logits_2nd = logits_2nd - mask01 * 1e20
        positive_2nd = torch.diag(logits10)
        loss = (-torch.log(torch.exp(positive_1st / tau_1st) / torch.sum(torch.exp(logits_1st / tau_1st), dim=1)) -torch.log(torch.exp(positive_2nd / tau_2nd) / torch.sum(torch.exp(logits_2nd / tau_2nd), dim=1)))/2
        loss_all += (1 / (i + 1)) * loss
    return loss_all / level

def parse_proto(results, num_cluster):
    """
    center_corresponding: NxLx128
    center_index : NxL center assignment of each sample at different hierarchies.
    :param results:
    :param num_cluster:
    :return:
    """
    level = len(num_cluster)
    dim = results['centroids'][0].shape[1]
    num_samples = results['im2cluster'][0].shape[0]
    center_index = np.zeros(shape=(num_samples, level))
    center_corresponding = np.zeros(shape=(num_samples, level, dim))
    for i in range(level):
        if i == 0:
            center_index[:, 0] = results['im2cluster'][0]
        else:
            center_index[:, i] = results['im2cluster'][i][center_index[:, i - 1].astype('int64')]
    for j in range(num_samples):
        for k in range(level):
            center_corresponding[j, k, :] = results['centroids'][k][center_index[j, k].astype('int64')]
    return center_corresponding, center_index


def parse_proto_test(args, g, results, num_cluster, tau=0.2, hyper_c=0):
    """
    center_corresponding: NxLx128
    center_index : NxL center assignment of each sample at different hierarchies.
    :param results:
    :param num_cluster:
    :return:
    """
    if hyper_c == 0:
        dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
    else:
        dist_f = lambda x, y: -dist_matrix(x, y, hyper_c)
    centers = results['centroids']
    level = len(centers)
    device = args.device
    dim = results['centroids'][0].shape[1]
    num_samples = g.size(0)
    center_index = np.zeros(shape=(num_samples, level))
    center_corresponding = np.zeros(shape=(num_samples, level, dim))
    for i in range(level):
        centers_level = torch.tensor(centers[i], dtype=torch.float32).to(device)
        if i == 0:
            score_p, max_indices = torch.max(torch.exp(dist_f(g, centers_level) / tau), dim=1)
            center_index[:, 0] = max_indices.detach().cpu().numpy().astype('int64')
        else:
            center_index[:, i] = results['im2cluster'][i][center_index[:, i - 1].astype('int64')]
    for j in range(num_samples):
        for k in range(level):
            center_corresponding[j, k, :] = results['centroids'][k][center_index[j, k].astype('int64')]
    return center_corresponding, center_index, -score_p


def calculate_adaptive_temperature(pos_distances, neg_distances, base_temperature=0.2):
    """
    Parameters:
    - pos_distances: Tensor
    - neg_distances: Tensor
    - base_temperature: float

    Returns:
    - adaptive_temperature
    """
    mean_pos_distance = pos_distances.mean().item()
    mean_neg_distance = neg_distances.mean().item()
    temperature = base_temperature * (mean_neg_distance / (mean_pos_distance - 1e-20))
    return max(temperature, base_temperature)

def contrastive_proto(args, z_i, z_j, center_corresponding_f, center_corresponding_s, results_f, results_s, tau=0.2, hyper_c=0):
    if hyper_c == 0:
        dist_f = lambda x, y: F.normalize(x, dim=1) @ F.normalize(y, dim=1).t()
    else:
        dist_f = lambda x, y: -dist_matrix(x, y, hyper_c)
    loss_p_all = 0
    centers_f = results_f['centroids']
    centers_s = results_s['centroids']
    level = len(centers_f)
    device = args.device
    ccf = torch.tensor(center_corresponding_f, dtype=torch.float32).to(device)
    ccs = torch.tensor(center_corresponding_s, dtype=torch.float32).to(device)
    for l in range(level):
        centers_level_s = torch.tensor(centers_s[l], dtype=torch.float32).to(device)
        pos_distances_i = torch.diag(dist_f(z_i, ccs[:, l, :]))
        neg_distances_i = torch.sum(dist_f(z_i, centers_level_s), dim=1)-pos_distances_i
        tau_i = calculate_adaptive_temperature(pos_distances_i, neg_distances_i, base_temperature=tau)
        # tau_i = args.tau
        positive_i = torch.diag(torch.exp(dist_f(z_i, ccs[:, l, :]) / tau_i))
        negative_i = torch.sum(torch.exp(dist_f(z_i, centers_level_s) / tau_i), dim=1) + torch.tensor(1e-20).to(device)
        centers_level_f = torch.tensor(centers_f[l], dtype=torch.float32).to(device)

        pos_distances_j = torch.diag(dist_f(z_j, ccf[:, l, :]))
        neg_distances_j = torch.sum(dist_f(z_j, centers_level_f), dim=1) - pos_distances_j
        # tau_j = args.tau
        tau_j = calculate_adaptive_temperature(pos_distances_j, neg_distances_j, base_temperature=tau)
        positive_j = torch.diag(torch.exp(dist_f(z_j, ccf[:, l, :]) / tau_j))
        negative_j = torch.sum(torch.exp(dist_f(z_j, centers_level_f) / tau_j), dim=1) + torch.tensor(1e-20).to(device)

        loss_level = -torch.log(positive_i / negative_i) - torch.log(positive_j / negative_j)
        loss_p_all += (1 / (l + 1)) * loss_level
    return loss_p_all / level


if __name__ == '__main__':
    a = torch.tensor([[0, 1], [1, 1], [0, 1], [1, 0]])
    # b=torch.eye(1024)
    print()
