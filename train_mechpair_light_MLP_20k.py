
"""基于原始 train_mechpair_light_10k.py 的 20k/4-GPU 控制实验。

保留：BioData2Seq 原编码器、shared_mask 全局池化、原四项损失、原评估协议。
新增：固定20k索引、可选独立A+deploy+Floor、可选反应QKV局部注意力。
注意：这不是新版缓存加速训练实现，不能将新旧流程差异视作单一变量。
"""
import argparse
import hashlib
from pathlib import Path
import csv
import importlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# 旧脚本经常放在 dec-v2/，实验脚本放在其子目录时也能正确导入旧编码器。
import sys
sys.path.insert(0, os.environ.get("ENCODER_DIR", "/datapool/home/tanlijun/RAMER/dec-v2"))
from Data2seq_v2 import BioData2Seq


import sys
# 将项目根目录添加到 sys.path
project_root = "/datapool/home/tanlijun/RAMER" #  /root/RAMMER
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在可以导入
from cluster import run_test_clustering_analysis
# 不在这里提前导入旧 prepare_embeddings：原文件在 import 时会将
# CUDA_VISIBLE_DEVICES 改为 "0"，导致 4-GPU torchrun 的非零 rank 失去 GPU。
# 聚类时仅在 rank0 完成 CUDA/DDP 初始化后延迟导入，并恢复环境变量。

# torchrun --nproc_per_node=4 train_mechpair_light_10k.py \
#   --enable_cluster_eval \
#   --cluster_eval_every 1 \
#   --cluster_eval_name Price-149 \
#   --cluster_eval_json /root/RAMMER/zero_shot/data/Price-149.json \
#   --cluster_eval_truth_csv /root/RAMMER/zero_shot/data/price.csv

def setup_logging(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="", help="默认继承 Slurm 的 CUDA_VISIBLE_DEVICES")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int) # 8
    parser.add_argument("--embed_dim", default=1024, type=int)
    parser.add_argument("--repr_dim", default=512, type=int)
    parser.add_argument("--topk_residues", default=24, type=int)
    parser.add_argument("--save_dir", default="/datapool/home/tanlijun/RAMER/dec-v2/mechpair_light_10k-full")
    parser.add_argument("--val_ratio", default=0.0, type=float) # 0.05
    parser.add_argument("--seed", default=3408, type=int)
    parser.add_argument("--base_lr", default=1e-4, type=float)
    parser.add_argument("--reaction_lr", default=5e-6, type=float)
    parser.add_argument("--temperature_lr", default=2e-5, type=float)
    parser.add_argument("--global_weight", default=1.0, type=float)
    parser.add_argument("--local_weight", default=0.5, type=float)
    parser.add_argument("--struct_weight", default=0.3, type=float) # 0.3 0.8
    parser.add_argument("--export_weight", default=0.5, type=float)
    parser.add_argument("--save_top_k_ari", default=2, type=int)

    parser.add_argument("--json_file", default="/datapool/home/tanlijun/RAMER/data/uniprot_20W_struct_seq_reaction_121_without_new392.json") # /root/RAMMER/data/uniprot_20W_struct_seq_reaction_121_without_new392.json
    parser.add_argument("--rhea_smiles_file", default="/datapool/home/tanlijun/RAMER/data/updated_rhea-reaction-smiles.json") # /root/RAMMER/data/updated_rhea-reaction-smiles.json
    parser.add_argument("--graph_embedding_path", default="/datapool/home/tanlijun/GearNet/pdbs/features/graph") # /root/RAMMER/data/gearnet_10k
    parser.add_argument("--residue_embedding_path", default="/datapool/home/tanlijun/GearNet/pdbs/features/node")  # /root/RAMMER/data/gearnet_features_10k/node

    parser.add_argument("--enable_cluster_eval", action="store_true")
    parser.add_argument("--cluster_eval_every", default=1, type=int)
    parser.add_argument("--cluster_eval_name", default="Price-149", type=str)
    parser.add_argument("--cluster_eval_json", default="/datapool/home/tanlijun/RAMER/data/test_data/Price-149.json", type=str) # /root/RAMMER/zero_shot/data/Price-149.json
    parser.add_argument("--cluster_eval_truth_csv", default="/datapool/home/tanlijun/RAMER/data/test_data/price.csv", type=str) # /root/RAMMER/zero_shot/data/price.csv
    parser.add_argument("--subset_file", required=True,
                        help="与 M0/M3D-FLOOR 共用 subsets/fixed_20k_seed3408.json")
    parser.add_argument("--subset_seed", default=3408, type=int)
    parser.add_argument("--experiment", choices=["baseline", "floor"], default="floor")
    parser.add_argument("--selector", choices=["mlp", "qkv"], default="mlp")
    parser.add_argument("--qkv_heads", default=8, type=int)
    parser.add_argument("--num_slots", default=4, type=int)
    parser.add_argument("--slot_dim", default=256, type=int)
    parser.add_argument("--slot_heads", default=4, type=int)
    parser.add_argument("--deploy_weight", default=0.2, type=float)
    parser.add_argument("--a_weight", default=0.5, type=float)
    parser.add_argument("--diversity_weight", default=0.01, type=float)
    parser.add_argument("--diversity_floor", default=0.10, type=float)
    parser.add_argument("--diversity_floor_penalty", default=5.0, type=float)
    parser.add_argument("--save_full_checkpoints", action="store_true",
                        help="额外保存原始全权重 checkpoint（占用空间很大，默认仅保存增量）")
    args = parser.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.val_ratio < 0 or args.val_ratio >= 1:
        parser.error("val_ratio must be in [0, 1)")
    if args.embed_dim % args.qkv_heads or args.slot_dim % args.slot_heads:
        parser.error("qkv_heads/slot_heads 必须整除对应维度")
    if any(x < 0 for x in [args.deploy_weight,args.a_weight,args.diversity_weight,args.diversity_floor,args.diversity_floor_penalty]):
        parser.error("所有损失系数不能为负")
    return args


def set_seed(seed: int = 3408):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_ddp(local_rank: int):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")


def get_ddp_generator(seed: int = 3408):
    local_rank = int(os.environ["LOCAL_RANK"])
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


class BioDataset(Dataset):
    def __init__(self, json_file, rhea_smiles_mapping, graph_embedding_path, residue_embedding_path=None):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        with open(rhea_smiles_mapping, "r") as f:
            self.rhea_smiles_mapping = json.load(f)

        residue_file_map = {}
        if residue_embedding_path and os.path.isdir(residue_embedding_path):
            for filename in os.listdir(residue_embedding_path):
                if filename.endswith("_node.pt"):
                    acc = filename.replace("_node.pt", "")
                    residue_file_map[acc] = os.path.join(residue_embedding_path, filename)

        self.filtered, self.global_files, self.residue_files = [], [], []
        for item in self.data:
            accession = item.get("primaryAccession")
            if not accession:
                continue
            global_path = os.path.join(graph_embedding_path, f"{accession}_graph.pt")
            # global_path = os.path.join(graph_embedding_path, f"{accession}_graph_embedding.pt")
            if not os.path.isfile(global_path):
                continue
            residue_path = residue_file_map.get(accession)
            if residue_embedding_path and residue_path is None:
                continue
            if item.get("RHEA") not in self.rhea_smiles_mapping:
                continue

            self.filtered.append(item)
            self.global_files.append(global_path)
            self.residue_files.append(residue_path)

        self.data = self.filtered # [:10000]  # 仅使用前 10k 样本
        # self.global_files = self.global_files[:10000]
        # self.residue_files = self.residue_files[:10000]
        print(f"Loaded {len(self.data)} samples from {json_file}.")
        print(f"Found graph embeddings for {len(self.global_files)} samples.")
        print(f"Found residue embeddings for {len(self.residue_files)} samples.")
        print(f"Retained {len(self.data)} samples after structure/reaction filtering.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        reaction_str = self.rhea_smiles_mapping[item["RHEA"]]
        reactants, products = reaction_str.split(">>")
        reactant_smiles = reactants.split(".") if reactants else []
        product_smiles = products.split(".") if products else []

        graph_embedding = torch.load(self.global_files[idx], weights_only=True).float().unsqueeze(0)
        residue_embeddings = torch.load(self.residue_files[idx], weights_only=True).float()
        if residue_embeddings.dim() == 1:
            residue_embeddings = residue_embeddings.unsqueeze(0)
        residue_mask = torch.ones(residue_embeddings.size(0), dtype=torch.bool)

        return {
            "protein_sequence": item["sequence"],
            "reaction_smiles": (reactant_smiles, product_smiles),
            "gearnet_embedding": graph_embedding,
            "gearnet_residue_embedding": residue_embeddings,
            "gearnet_residue_mask": residue_mask,
            "primary_accession": item["primaryAccession"],
        }


def custom_collate_fn(batch):
    protein_sequences = [x["protein_sequence"] for x in batch]
    reaction_smiles = [x["reaction_smiles"] for x in batch]
    accessions = [x["primary_accession"] for x in batch]
    gearnet_embeddings = torch.cat([x["gearnet_embedding"] for x in batch], dim=0)

    residue_emb_list = [x["gearnet_residue_embedding"] for x in batch]
    residue_mask_list = [x["gearnet_residue_mask"] for x in batch]
    max_len = max(emb.size(0) for emb in residue_emb_list)

    padded_embeddings, padded_masks = [], []
    for emb, mask in zip(residue_emb_list, residue_mask_list):
        seq_len = emb.size(0)
        padded_emb = torch.zeros(max_len, emb.size(1), dtype=emb.dtype)
        padded_emb[:seq_len] = emb
        padded_mask = torch.zeros(max_len, dtype=torch.bool)
        padded_mask[:seq_len] = mask
        padded_embeddings.append(padded_emb)
        padded_masks.append(padded_mask)

    return {
        "protein_sequence": protein_sequences,
        "reaction_smiles": reaction_smiles,
        "gearnet_embedding": gearnet_embeddings,
        "gearnet_residue_embedding": torch.stack(padded_embeddings, dim=0),
        "gearnet_residue_mask": torch.stack(padded_masks, dim=0),
        "primary_accession": accessions,
    }


def split_dataset(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_ratio))
    return Subset(dataset, indices[n_val:]), Subset(dataset, indices[:n_val])


def get_fixed_20k_subset(dataset, path, subset_seed=3408):
    """强制使用新单文件 M0/M3/D 的相同 20k 过滤后索引，失败就报错。"""
    sha = hashlib.sha256()
    for item in dataset.data:
        reaction = item["RHEA"]
        record = [item["primaryAccession"], item["sequence"], reaction, reaction,
                  dataset.rhea_smiles_mapping[reaction]]
        sha.update(json.dumps(record, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
        sha.update(b'\n')
    fp = sha.hexdigest()
    info = json.loads(Path(path).read_text(encoding='utf-8'))
    indices = info['selected_indices']
    if (info.get('dataset_fingerprint') != fp or
        info.get('filtered_total') != len(dataset) or
        info.get('subset_size') != 20000 or
        info.get('subset_seed') != subset_seed or
        len(indices) != 20000 or len(set(indices)) != 20000 or
        min(indices) < 0 or max(indices) >= len(dataset)):
        raise ValueError("旧脚本过滤后的索引或数据指纹与现有 20k 清单不一致。拒绝隐式重新抽样！")
    return indices, fp


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float().unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (x * mask_f).sum(dim=1) / denom


def align_residue_modalities(seq_res, seq_mask, struct_res, struct_mask):
    max_shared_len = min(seq_res.size(1), struct_res.size(1))
    seq_res = seq_res[:, :max_shared_len]
    struct_res = struct_res[:, :max_shared_len]
    seq_mask = seq_mask[:, :max_shared_len]
    struct_mask = struct_mask[:, :max_shared_len]
    shared_mask = seq_mask & struct_mask
    return seq_res, struct_res, shared_mask


def batched_index_select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    bsz, _, dim = x.shape
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, dim)
    return torch.gather(x, 1, idx_exp)


class InfoNCELoss(nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp, dtype=torch.float32))

    def forward(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        self.temperature.data = torch.clamp(self.temperature.data, min=0.3)
        scaled = similarity_matrix / self.temperature
        log_prob = F.log_softmax(scaled, dim=-1)
        target = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)
        return -log_prob[torch.arange(similarity_matrix.size(0), device=similarity_matrix.device), target].mean()


def symmetric_nce(loss_fn: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    sim = x @ y.T
    return 0.5 * (loss_fn(sim) + loss_fn(sim.T))


class ReactionGuidedSelector(nn.Module):
    def __init__(self, dim: int, topk: int = 24):
        super().__init__()
        self.topk = topk
        self.score = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )

    def forward(self, residue_feats: torch.Tensor, reaction_global: torch.Tensor, mask: torch.Tensor):
        """
        Use differentiable attention pooling for training, while still reporting top-k indices for inspection.
        This avoids DDP unused-parameter errors caused by hard top-k selection disconnecting the scorer.
        """
        bsz, length, dim = residue_feats.shape
        r = reaction_global.unsqueeze(1).expand(bsz, length, dim)
        scores = self.score(torch.cat([residue_feats, r], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = attn * mask.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        local = torch.sum(attn.unsqueeze(-1) * residue_feats, dim=1)

        k = min(self.topk, length)
        topk_idx = scores.topk(k, dim=-1).indices
        return local, topk_idx, attn


class ReactionQKVSelector(nn.Module):
    """标准 scaled dot-product multi-head cross attention。

    query = reaction_global [B,1,D]
    key/value = residue_feats [B,L,D]
    mask = residue valid positions [B,L]
    output = cross-attention projected local vector [B,D]
    训练时不使用硬 top-k；top-k 仅用于观察。
    """
    def __init__(self, dim: int, topk: int = 24, heads: int = 8):
        super().__init__()
        self.topk = topk
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)

    def forward(self, residue_feats, reaction_global, mask):
        if not bool(mask.any(dim=1).all()):
            raise ValueError("QKV 注意力发现全 padding 样本")
        q = reaction_global.unsqueeze(1)
        local, weights = self.attention(
            query=q, key=residue_feats, value=residue_feats,
            key_padding_mask=~mask, need_weights=True,
            average_attn_weights=True,
        )
        attn = weights[:, 0, :]
        k = min(self.topk, residue_feats.shape[1])
        indices = attn.masked_fill(~mask, -1).topk(k, dim=-1).indices
        return local.squeeze(1), indices, attn


class MultiSlotFusion(nn.Module):
    """来自当前 M3D-FLOOR 的 A 融合：learned Q 读取全序列和全结构。"""
    def __init__(self, dim, repr_dim, num_slots, slot_dim, heads):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_slots, slot_dim) * .02)
        self.seq_projection = nn.Sequential(nn.Linear(dim, slot_dim), nn.LayerNorm(slot_dim))
        self.struct_projection = nn.Sequential(nn.Linear(dim, slot_dim), nn.LayerNorm(slot_dim))
        self.seq_attn = nn.MultiheadAttention(slot_dim, heads, dropout=0., batch_first=True)
        self.struct_attn = nn.MultiheadAttention(slot_dim, heads, dropout=0., batch_first=True)
        self.gate = nn.Linear(2 * slot_dim, slot_dim)
        self.mix = nn.Sequential(nn.Linear(2 * slot_dim, slot_dim), nn.GELU(), nn.LayerNorm(slot_dim))
        self.readout = nn.Sequential(nn.Linear(num_slots * slot_dim, repr_dim), nn.GELU(),
                                     nn.LayerNorm(repr_dim), nn.Linear(repr_dim, repr_dim))

    def forward(self, seq, seq_mask, struct, struct_mask):
        if not bool(seq_mask.any(dim=-1).all()) or not bool(struct_mask.any(dim=-1).all()):
            raise ValueError('A 需要至少一个有效的序列和结构残基')
        q = self.queries.unsqueeze(0).expand(seq.size(0), -1, -1)
        s = self.seq_projection(seq)
        g = self.struct_projection(struct)
        ss, sw = self.seq_attn(q, s, s, key_padding_mask=~seq_mask, need_weights=True)
        gg, gw = self.struct_attn(q, g, g, key_padding_mask=~struct_mask, need_weights=True)
        pair = torch.cat((ss, gg), dim=-1)
        gate = torch.sigmoid(self.gate(pair))
        fused = gate * ss + (1 - gate) * gg + self.mix(pair)

        def overlap(attn):
            if attn.size(1) == 1:
                return attn.new_zeros(())
            norm = F.normalize(attn.float(), dim=-1)
            gram = norm @ norm.transpose(1, 2)
            eye = torch.eye(gram.size(1), device=gram.device, dtype=torch.bool)
            return gram[:, ~eye].square().mean()

        return self.readout(fused.flatten(1)), .5 * (overlap(sw) + overlap(gw))


def floor_diversity(raw, floor, penalty):
    """与现有 M3D-FLOOR 完全一致的 soft floor 公式。"""
    above = torch.relu(raw - float(floor))
    below = torch.relu(float(floor) - raw)
    return above + float(penalty) * below.square()


class SequenceOnlyMechanismHead(nn.Module):
    """
    Final exported protein representation is sequence-only.
    Reaction and structure supervise the sequence representation during training.
    """
    def __init__(self, dim: int, repr_dim: int, topk: int,
                 selector_mode: str = "mlp", qkv_heads: int = 8):
        super().__init__()
        factory = (lambda: ReactionGuidedSelector(dim, topk=topk)) if selector_mode == "mlp" else \
                  (lambda: ReactionQKVSelector(dim, topk=topk, heads=qkv_heads))
        self.seq_selector = factory()
        self.struct_selector = factory()

        self.seq_global_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, repr_dim)
        )
        self.react_global_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, repr_dim)
        )
        self.seq_local_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, repr_dim)
        )
        self.struct_local_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, repr_dim)
        )
        self.export_proj = nn.Sequential(
            nn.Linear(repr_dim * 2, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, repr_dim)
        )

    def forward(self, seq_res, struct_res, shared_mask, reaction_global):
        seq_global = masked_mean(seq_res, shared_mask)
        seq_local, seq_topk, seq_attn = self.seq_selector(seq_res, reaction_global, shared_mask)
        struct_local, struct_topk, struct_attn = self.struct_selector(struct_res, reaction_global, shared_mask)

        z_func_seq = self.seq_global_proj(seq_global)
        z_func_rxn = self.react_global_proj(reaction_global)
        z_local_seq = self.seq_local_proj(seq_local)
        z_local_struct = self.struct_local_proj(struct_local)

        z_protein = self.export_proj(torch.cat([z_func_seq, z_local_seq], dim=-1))
        return {
            "z_func_seq": z_func_seq,
            "z_func_rxn": z_func_rxn,
            "z_local_seq": z_local_seq,
            "z_local_struct": z_local_struct,
            "z_protein": z_protein,
            "seq_topk": seq_topk,
            "struct_topk": struct_topk,
            "seq_attn": seq_attn,
            "struct_attn": struct_attn,
        }


@dataclass
class StepOutputs:
    total_loss: torch.Tensor
    global_loss: torch.Tensor
    local_loss: torch.Tensor
    struct_loss: torch.Tensor
    export_loss: torch.Tensor
    deploy_loss: torch.Tensor
    fusion_loss: torch.Tensor
    diversity_raw: torch.Tensor
    diversity_effective: torch.Tensor


METRIC_KEYS = ["total", "global", "local", "struct", "export", "deploy", "fusion", "diversity", "diversity_effective"]


def build_step_outputs(batch: Dict, protein_encoder, struct_res_encoder,
                       reaction_encoder, mech_head, loss_fn, fusion=None):
    # 首先获取不截断的残基张量；原始 head 仍使用 shared_mask 的旧池化方式。
    seq_all, seq_mask = protein_encoder(batch["protein_sequence"])
    rxn = reaction_encoder(batch["reaction_smiles"])
    reaction_global = rxn["global"]
    struct_all, struct_mask = struct_res_encoder(
        batch["gearnet_residue_embedding"].cuda(non_blocking=True),
        batch["gearnet_residue_mask"].cuda(non_blocking=True),
    )
    seq_res, struct_res, shared_mask = align_residue_modalities(
        seq_all, seq_mask, struct_all, struct_mask)
    out = mech_head(seq_res, struct_res, shared_mask, reaction_global)

    global_loss = symmetric_nce(loss_fn, out["z_func_seq"], out["z_func_rxn"])
    local_loss = symmetric_nce(loss_fn, out["z_local_seq"], out["z_func_rxn"])
    struct_loss = 1.0 - F.cosine_similarity(
        F.normalize(out["z_local_seq"], dim=-1),
        F.normalize(out["z_local_struct"], dim=-1),
        dim=-1,
    ).mean()
    export_loss = symmetric_nce(loss_fn, out["z_protein"], out["z_func_rxn"])
    zero = global_loss.new_zeros(())
    deploy_loss = symmetric_nce(loss_fn, masked_mean(seq_all, seq_mask), reaction_global) if fusion is not None else zero
    if fusion is not None:
        fused, diversity = fusion(seq_all, seq_mask, struct_all, struct_mask)
        fusion_loss = symmetric_nce(loss_fn, fused, out["z_func_rxn"])
    else:
        fusion_loss, diversity = zero, zero
    return dict(global_loss=global_loss, local_loss=local_loss,
                struct_loss=struct_loss, export_loss=export_loss,
                deploy_loss=deploy_loss, fusion_loss=fusion_loss, diversity=diversity)


def aggregate_losses(step, args):
    eff_div = (floor_diversity(step["diversity"], args.diversity_floor,
                               args.diversity_floor_penalty)
               if args.experiment == "floor" else step["diversity"].new_zeros(()))
    total = (args.global_weight * step["global_loss"]
             + args.local_weight * step["local_loss"]
             + args.struct_weight * step["struct_loss"]
             + args.export_weight * step["export_loss"])
    if args.experiment == "floor":
        total = total + args.deploy_weight * step["deploy_loss"] \
                      + args.a_weight * step["fusion_loss"] \
                      + args.diversity_weight * eff_div
    return StepOutputs(total, step["global_loss"], step["local_loss"],
                       step["struct_loss"], step["export_loss"],
                       step["deploy_loss"], step["fusion_loss"],
                       step["diversity"], eff_div)


def _get_values(outs):
    return [outs.total_loss, outs.global_loss, outs.local_loss, outs.struct_loss,
            outs.export_loss, outs.deploy_loss, outs.fusion_loss,
            outs.diversity_raw, outs.diversity_effective]


def train_one_epoch(model_dict, loader, optimizer, loss_fn, args):
    for m in model_dict.values():
        m.train()
    sums = torch.zeros(len(METRIC_KEYS), device="cuda", dtype=torch.float64)
    n = 0
    for batch in tqdm(loader, disable=dist.get_rank() != 0):
        step = build_step_outputs(batch, model_dict["protein_encoder"],
                                  model_dict["struct_res_encoder"],
                                  model_dict["reaction_encoder"],
                                  model_dict["mech_head"], loss_fn,
                                  model_dict.get("fusion"))
        outs = aggregate_losses(step, args)
        if not torch.isfinite(outs.total_loss):
            raise FloatingPointError("nonfinite training loss")
        optimizer.zero_grad()
        outs.total_loss.backward()
        optimizer.step()
        for i, x in enumerate(_get_values(outs)):
            sums[i] += x.detach().double()
        n += 1
    totals = torch.cat([sums, sums.new_tensor([n])])
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {k: float(totals[i] / max(1., float(totals[-1]))) for i, k in enumerate(METRIC_KEYS)}


@torch.no_grad()
def validate_one_epoch(model_dict, loader, loss_fn, args):
    for m in model_dict.values():
        m.eval()
    sums = torch.zeros(len(METRIC_KEYS), device="cuda", dtype=torch.float64)
    n = 0
    for batch in tqdm(loader, disable=True):
        if len(batch["protein_sequence"]) < 2:
            continue
        step = build_step_outputs(batch, model_dict["protein_encoder"],
                                  model_dict["struct_res_encoder"],
                                  model_dict["reaction_encoder"],
                                  model_dict["mech_head"], loss_fn,
                                  model_dict.get("fusion"))
        outs = aggregate_losses(step, args)
        for i, x in enumerate(_get_values(outs)):
            sums[i] += x.detach().double()
        n += 1
    totals = torch.cat([sums, sums.new_tensor([n])])
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {k: float(totals[i] / max(1., float(totals[-1]))) for i, k in enumerate(METRIC_KEYS)}


def build_checkpoint_state(epoch, model_dict, loss_fn):
    return {
        "epoch": epoch,
        "protein_encoder": model_dict["protein_encoder"].module.state_dict(),
        "struct_res_encoder": model_dict["struct_res_encoder"].module.state_dict(),
        "reaction_encoder": model_dict["reaction_encoder"].module.state_dict(),
        "mech_head": model_dict["mech_head"].module.state_dict(),
        "temperature": loss_fn.temperature.detach().cpu(),
    }


def sequence_delta(state):
    """编码器原始 encoder.* 参数键名；LoRA 和线性层完整保留。"""
    return {k: v.detach().cpu() for k, v in state.items()
            if not (k.startswith('encoder.model.') and '.lora_' not in k)}


def save_checkpoint(save_path, state):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


def _import_attr(module_names, attr_name):
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, attr_name):
                return getattr(module, attr_name)
        except Exception:
            continue
    return None


def get_cluster_eval_helpers():
    prepare_embeddings = _import_attr(["zero_shot.step1_get_ram_enzyme_embedding", "step1_get_ram_enzyme_embedding"], "prepare_embeddings")
    run_test_clustering_analysis = _import_attr(["cluster", "zero_shot.cluster"], "run_test_clustering_analysis")
    return prepare_embeddings, run_test_clustering_analysis


def append_cluster_eval_csv(csv_path: str, row: Dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fields = sorted(row.keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _flatten_cluster_results(results: Dict, epoch: int, dataset_name: str):
    row = {"epoch": epoch, "dataset": dataset_name, "status": "ok"}
    alias_map = {"ari": ["ari", "ARI"], "nmi": ["nmi", "NMI"], "ami": ["ami", "AMI"], "fmi": ["fmi", "FMI"], "purity": ["purity", "Purity"]}
    if isinstance(results, dict):
        for out_key, candidates in alias_map.items():
            for cand in candidates:
                if cand in results:
                    row[out_key] = results[cand]
                    break
        for k, v in results.items():
            if k not in row:
                row[k] = v
    else:
        row["status"] = "failed"
        row["note"] = str(results)
    return row


@torch.no_grad()
def run_cluster_eval_epoch(model_dict: Dict[str, nn.Module], epoch: int, args):
    """同一套旧 cluster.py 聚类规则；复用已在 GPU 中的序列模型导出均值。

    原始 prepare_embeddings 每轮重新构造 ProtT5，4GPU 下可能令 rank0 显存峰值
    超出容量；复用当前模型只改变评估实现，不改变训练、优化器和评分算法。
    """
    import h5py
    result_dir = Path(args.save_dir)/"clustering_results"
    tmp_dir = Path(args.save_dir)/"cluster_eval_tmp"
    result_dir.mkdir(parents=True,exist_ok=True)
    tmp_dir.mkdir(parents=True,exist_ok=True)
    try:
        raw = json.loads(Path(args.cluster_eval_json).read_text(encoding='utf8'))
        ids = []
        sequences = []
        for i, r in enumerate(raw):
            rid = r.get('id') or r.get('primaryAccession')
            seq = r.get('protein_sequence') or r.get('sequence')
            if not isinstance(rid,str) or not isinstance(seq,str):
                raise ValueError(f'聚类测试 JSON 第 {i} 条无有效 id/sequence')
            ids.append(rid)
            sequences.append(seq)
        encoder = model_dict['protein_encoder'].module
        prev = encoder.training
        encoder.eval()
        vectors=[]
        try:
            for seq in sequences:
                features,mask=encoder([seq])
                pooled=masked_mean(features,mask).float()
                vectors.append(pooled.cpu().numpy()[0])
        finally:
            encoder.train(prev)
        emb=np.asarray(vectors,dtype=np.float32)
        h5_path=tmp_dir/f'{args.cluster_eval_name}_epoch_{epoch}.h5'
        with h5py.File(h5_path,'w') as h5:
            if any(len(s.encode('utf8'))>100 for s in ids):
                raise ValueError('评估 ID 太长，旧 HDF5 S100 会截断')
            h5.create_dataset('ids',data=np.asarray([s.encode('utf8') for s in ids],dtype='S100'))
            h5.create_dataset('embeddings',data=emb)
        results=run_test_clustering_analysis(
            h5_path=str(h5_path),gt_csv_path=args.cluster_eval_truth_csv,
            save_path=str(result_dir),epoch=epoch)
        return _flatten_cluster_results(results,epoch,args.cluster_eval_name)
    except Exception as exc:
        return {'epoch':epoch,'dataset':args.cluster_eval_name,
                'status':'failed','note':repr(exc)}


def update_topk_checkpoints(save_dir, epoch, metric_name, metric_value, state, entries, topk):
    if topk <= 0:
        return entries
    subdir = os.path.join(save_dir, f"topk_{metric_name}")
    os.makedirs(subdir, exist_ok=True)

    def _sort_key(x):
        return x["metric"]

    should_save = len(entries) < topk or metric_value > min(entries, key=_sort_key)["metric"]
    if not should_save:
        return sorted(entries, key=_sort_key, reverse=True)

    filename = f"epoch{epoch:03d}_{metric_name}{metric_value:.6f}.pt"
    save_path = os.path.join(subdir, filename)
    save_checkpoint(save_path, state)
    entries.append({"epoch": epoch, "metric": float(metric_value), "path": save_path})
    entries = sorted(entries, key=_sort_key, reverse=True)

    while len(entries) > topk:
        removed = entries.pop(-1)
        try:
            if os.path.exists(removed["path"]):
                os.remove(removed["path"])
        except OSError:
            pass

    manifest = [{"rank": i + 1, "epoch": x["epoch"], metric_name: x["metric"], "path": x["path"]} for i, x in enumerate(entries)]
    with open(os.path.join(subdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return entries


def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    init_ddp(local_rank)
    if local_rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        if (Path(args.save_dir) / "clustering_eval_history.csv").exists():
            raise RuntimeError("输出目录已存在评估日志，请更换 save_dir，禁止覆盖")
        setup_logging(os.path.join(args.save_dir, "train.log"))

    # 保留旧版 BioData2Seq 编码器构造顺序及原始预处理。
    protein_encoder = BioData2Seq("protein-sequence", embed_dim=args.embed_dim).cuda()
    struct_res_encoder = BioData2Seq("structure-residue", embed_dim=args.embed_dim).cuda()
    reaction_encoder = BioData2Seq("reaction", embed_dim=args.embed_dim).cuda()
    mech_head = SequenceOnlyMechanismHead(
        args.embed_dim, args.repr_dim, topk=args.topk_residues,
        selector_mode=args.selector, qkv_heads=args.qkv_heads).cuda()
    loss_fn = InfoNCELoss(init_temp=1.0).cuda()
    fusion = (MultiSlotFusion(args.embed_dim, args.repr_dim,
                              args.num_slots, args.slot_dim, args.slot_heads).cuda()
              if args.experiment == "floor" else None)

    # 按旧版代码冻结反应编码器主干，注意 DDP 包装前设置可训练性。
    for _, p in reaction_encoder.named_parameters():
        p.requires_grad = False
    for name, p in reaction_encoder.named_parameters():
        if any(tag in name.lower() for tag in
               ["transformer", "position", "fusion", "linear", "projection", "aggreg"]):
            p.requires_grad = True

    model_dict = {
        "protein_encoder": nn.parallel.DistributedDataParallel(protein_encoder, device_ids=[local_rank]),
        "struct_res_encoder": nn.parallel.DistributedDataParallel(struct_res_encoder, device_ids=[local_rank]),
        "reaction_encoder": nn.parallel.DistributedDataParallel(reaction_encoder, device_ids=[local_rank], find_unused_parameters=True),
        "mech_head": nn.parallel.DistributedDataParallel(mech_head, device_ids=[local_rank]),
    }
    if fusion is not None:
        model_dict["fusion"] = nn.parallel.DistributedDataParallel(fusion, device_ids=[local_rank])

    groups = [
        {"params": filter(lambda p: p.requires_grad, model_dict["protein_encoder"].parameters()), "lr": args.base_lr},
        {"params": filter(lambda p: p.requires_grad, model_dict["struct_res_encoder"].parameters()), "lr": args.base_lr},
        {"params": filter(lambda p: p.requires_grad, model_dict["reaction_encoder"].parameters()), "lr": args.reaction_lr},
        {"params": model_dict["mech_head"].parameters(), "lr": args.base_lr},
        {"params": [loss_fn.temperature], "lr": args.temperature_lr},
    ]
    if fusion is not None:
        groups.append({"params": model_dict["fusion"].parameters(), "lr": args.base_lr})
    optimizer = optim.AdamW(groups)

    dataset = BioDataset(args.json_file, args.rhea_smiles_file,
                         args.graph_embedding_path, args.residue_embedding_path)
    selected, fingerprint = get_fixed_20k_subset(dataset, args.subset_file, args.subset_seed)
    # 默认 val_ratio=0：与现在的 20k M0/M3D 一样全部 20k 用于训练。
    if args.val_ratio > 0:
        train_subset, val_subset = split_dataset(Subset(dataset, selected), args.val_ratio, args.seed)
    else:
        train_subset, val_subset = Subset(dataset, selected), None
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_subset, shuffle=True, drop_last=True)
    generator = get_ddp_generator(args.seed)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              sampler=train_sampler, pin_memory=True, generator=generator,
                              collate_fn=custom_collate_fn, drop_last=True)
    val_loader = None
    if val_subset is not None:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, sampler=val_sampler,
                                pin_memory=True, collate_fn=custom_collate_fn)

    if local_rank == 0:
        cfg = dict(vars(args))
        cfg.update(subset_sha256=hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
                   dataset_fingerprint=fingerprint, train_rows=len(train_subset),
                   val_rows=0 if val_subset is None else len(val_subset),
                   world_size=dist.get_world_size())
        Path(args.save_dir, "config.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
        print(f"LEGACY CONTROL selector={args.selector} experiment={args.experiment} "
              f"rows={len(train_subset)} val={len(val_subset) if val_subset else 0} "
              f"world={dist.get_world_size()} fingerprint={fingerprint}", flush=True)
    best_ari = -float('inf')
    ari_topk = []
    start = time.time()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model_dict, train_loader, optimizer, loss_fn, args)
        val_stats = (validate_one_epoch(model_dict, val_loader, loss_fn, args)
                     if val_loader is not None else None)
        if local_rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} training="
                  +json.dumps(train_stats, ensure_ascii=False)
                  +f" temp={loss_fn.temperature.item():.6f}", flush=True)
            logging.info(f"epoch={epoch+1} training={train_stats} val={val_stats}")
            # 推理兼容的增量键（encoder.*），用相同 ProtT5 底座重建。
            seq_delta = sequence_delta(model_dict['protein_encoder'].module.state_dict())
            torch.save(seq_delta, Path(args.save_dir) / f"seq_only_epoch_{epoch+1:03d}.pth")
            row = {"epoch":epoch+1,"variant":args.experiment,"selector":args.selector,
                   "subset_size":len(selected),"train_rows":len(train_subset),
                   "val_rows":0 if val_subset is None else len(val_subset),
                   "temperature": float(loss_fn.temperature.item()),
                   **{f"train_{k}":v for k,v in train_stats.items()}}
            if val_stats is not None:
                row.update({f"val_{k}":v for k,v in val_stats.items()})
            if args.enable_cluster_eval and (epoch+1) % args.cluster_eval_every == 0:
                # 原始 cluster.py / prepare_embeddings 评估协议（按原先方式）。
                eval_row = run_cluster_eval_epoch(model_dict, epoch+1, args)
                row.update(eval_row)
                if eval_row.get('status') != 'ok':
                    raise RuntimeError(f"epoch {epoch+1} 聚类评估失败：{eval_row.get('note')}")
                ari = float(eval_row['ari'])
                if ari > best_ari:
                    best_ari = ari
                    save_checkpoint(Path(args.save_dir)/"best_ari_seq_only.pth", seq_delta)
                # 保存 top-2 的增量权重，避免原始全部 T5 checkpoint 的巨大 I/O。
                if ari > -float('inf'):
                    ari_topk.append((ari, epoch+1))
                    ari_topk.sort(reverse=True)
                    ari_topk = ari_topk[:args.save_top_k_ari]
                if args.save_full_checkpoints:
                    full = build_checkpoint_state(epoch+1, model_dict, loss_fn)
                    if 'fusion' in model_dict:
                        full['fusion'] = model_dict['fusion'].module.state_dict()
                    save_checkpoint(Path(args.save_dir)/f"full_epoch_{epoch+1:03d}.pt", full)
            append_cluster_eval_csv(os.path.join(args.save_dir,"clustering_eval_history.csv"),row)
            print("Top-ARI",ari_topk, "best",best_ari, flush=True)
            print("EPOCH RESULT",json.dumps(row, ensure_ascii=False,default=str),flush=True)
        # 评估只运行 rank0，其他 rank 等待，确保不同 rank 在相同 epoch 进行下一轮。
        dist.barrier()
    if local_rank == 0:
        print(f"Finished {args.experiment}/{args.selector}: best_ARI={best_ari:.6f}, "
              f"elapsed_h={(time.time()-start)/3600:.2f}",flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    args = prepare()
    set_seed(args.seed)
    time_start = time.time()
    main(args)
    if int(os.environ["LOCAL_RANK"]) == 0:
        print(f"time elapsed: {time.time() - time_start:.2f} seconds")