import torch
from torch import nn
import torch.nn.functional as F
from dataset import LGCNDataset
from model import GCLRec


def polar_contrastive_loss(anchor, positive, negative, temperature=0.07):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)

    pos_sim = torch.clamp(torch.sum(anchor * positive, dim=-1), -1 + 1e-7, 1 - 1e-7)
    neg_sim = torch.clamp(torch.sum(anchor * negative, dim=-1), -1 + 1e-7, 1 - 1e-7)

    pos_theta = torch.acos(pos_sim)
    neg_theta = torch.acos(neg_sim)

    pos_score = torch.exp(-pos_theta / temperature)
    neg_score = torch.exp(-neg_theta / temperature)

    return -torch.log(pos_score / (pos_score + neg_score + 1e-8)).mean()


class AdaptiveEdgeSampler(nn.Module):
    def __init__(self, dim, k=10):
        super().__init__()
        self.k = k
        self.Wb = nn.Linear(dim, dim, bias=False)
        self.Wi = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, 1, bias=False)
        self.tau = 1.0

    def forward(self, basket_emb, item_emb):
        b = self.Wb(basket_emb)[:, None, :]
        i = self.Wi(item_emb)[None, :, :]
        scores = self.v(torch.tanh(b + i)).squeeze(-1)

        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(scores) + 1e-9) + 1e-9)
            scores = (scores + g) / self.tau

        pos_idx = torch.topk(scores, self.k, dim=-1).indices
        neg_idx = torch.topk(-scores, self.k, dim=-1).indices
        return pos_idx, neg_idx


class CausalPropagationLayer(nn.Module):
    def __init__(self, blend=0.5):
        super().__init__()
        self.blend = blend

    def estimate_causal_matrix(self, item_emb, basket_emb):
        obs = F.softmax(item_emb @ basket_emb.T, dim=-1)
        item_c = item_emb - item_emb.mean(dim=0, keepdim=True)
        basket_c = basket_emb - basket_emb.mean(dim=0, keepdim=True)
        intv = F.softmax(item_c @ basket_c.T, dim=-1)
        return F.relu(intv - obs)

    def compute_loss(self, item_emb, basket_emb):
        C = self.estimate_causal_matrix(item_emb, basket_emb)
        prop = C @ basket_emb
        refined = self.blend * prop + (1.0 - self.blend) * item_emb
        sparsity = torch.mean(C)
        consistency = F.mse_loss(refined, item_emb.detach())
        return 0.5 * sparsity + 0.5 * consistency


class CASCADERec(GCLRec):
    def __init__(self, config: dict, dataset: LGCNDataset):
        super().__init__(config, dataset)
        self.use_adaptive = config.get("use_adaptive", True)
        self.use_causal = config.get("use_causal", True)

        self.lambda_adaptive = float(config.get("lambda_adaptive", 0.15))
        self.lambda_causal = float(config.get("lambda_causal", 0.3))
        self.lambda_polar = float(config.get("lambda_polar", 0.8))

        self.temp = float(config.get("infoNCE_temp", 0.07))
        self.tau = float(config.get("gumbel_tau", 1.0))

        if self.use_adaptive:
            self.adaptive_head = AdaptiveEdgeSampler(
                dim=config["lgc_latent_dim"],
                k=config.get("num_aug", 1)
            )

        if self.use_causal:
            self.causal_layer = CausalPropagationLayer(
                blend=float(config.get("causal_blend", 0.5))
            )

        self.g_droped = None
        self.pos_g_droped = None
        self.neg_g_droped = None

    def set_gumbel_tau(self, tau: float):
        self.tau = tau
        if hasattr(self, "adaptive_head"):
            self.adaptive_head.tau = tau

    def forward(self, bseq, bseq_len, run_type="train"):
        out = super().forward(bseq, bseq_len, run_type)

        if not isinstance(out, tuple):
            return out

        logits, pos_emb, neg_emb, bseq_emb, item_logits, basket_logits = out
        device = logits.device

        adaptive_loss = torch.tensor(0.0, device=device)
        causal_loss = torch.tensor(0.0, device=device)
        polar_loss = torch.tensor(0.0, device=device)

        if self.use_adaptive and hasattr(self, "adaptive_head") and run_type == "train":
            adaptive_loss = torch.tensor(0.0, device=device)

        if pos_emb is not None and neg_emb is not None:
            polar_loss = polar_contrastive_loss(
                bseq_emb, pos_emb, neg_emb, temperature=self.temp
            )

        if self.use_causal and hasattr(self, "causal_layer"):
            causal_loss = self.causal_layer.compute_loss(
                self.embedding_item.weight,
                self.embedding_basket.weight
            )

        return {
            "logits": logits,
            "pos_emb": pos_emb,
            "neg_emb": neg_emb,
            "bseq_emb": bseq_emb,
            "item_logits": item_logits,
            "basket_logits": basket_logits,
            "adaptive_loss": adaptive_loss * self.lambda_adaptive,
            "causal_loss": causal_loss * self.lambda_causal,
            "polar_loss": polar_loss * self.lambda_polar,
        }