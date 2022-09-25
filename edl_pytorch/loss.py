import torch
from torch import nn
import torch.nn.functional as F

lg = torch.lgamma
dg = torch.digamma

# Normal Inverse Gamma Negative Log-Likelihood
# from https://arxiv.org/abs/1910.02600:
# > we denote the loss, L^NLL_i as the negative logarithm of model
# > evidence ...
def nig_nll(y, gamma, v, alpha, beta):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()


# Normal Inverse Gamma regularization
# from https://arxiv.org/abs/1910.02600:
# > we formulate a novel evidence regularizer, L^R_i
# > scaled on the error of the i-th prediction
def nig_reg(y, gamma, v, alpha, _beta):
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()


# KL divergence of predicted parameters from uniform Dirichlet distribution
# from https://arxiv.org/abs/1806.01768
# code based on:
# https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
def dirichlet_reg(y, alpha):
    # dirichlet parameters after removal of non-misleading evidence (from the label)
    alpha = y + (1 - y) * alpha

    # uniform dirichlet distribution
    beta = torch.ones_like(alpha)

    sum_alpha = alpha.sum(-1)
    sum_beta = beta.sum(-1)

    t1 = sum_alpha.lgamma() - sum_beta.lgamma()
    t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
    t3 = alpha - beta
    t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

    kl = t1 - t2 + (t3 * t4).sum(-1)
    return kl.mean()


# Eq. (5) from https://arxiv.org/abs/1806.01768:
# Sum of squares loss
def dirichlet_mse(y, alpha):
    sum_alpha = alpha.sum(-1, keepdims=True)
    p = alpha / sum_alpha
    t1 = (y - p).pow(2).sum(-1)
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)
    mse = t1 + t2
    return mse.mean()


def evidential_classification(y, alpha, lamb=1.0):
    num_classes = alpha.shape[-1]
    y = F.one_hot(y, num_classes)
    return dirichlet_mse(y, alpha) + lamb * dirichlet_reg(y, alpha)


def evidential_regression(y, dist_params, lamb=1.0):
    return nig_nll(y, *dist_params) + lamb * nig_reg(y, *dist_params)
