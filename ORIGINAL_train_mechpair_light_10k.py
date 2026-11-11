
import argparse
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

from Data2seq_v2 import BioData2Seq


import sys
# 将项目根目录添加到 sys.path
project_root = "/datapool/home/tanlijun/RAMER" #  /root/RAMMER
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 现在可以导入
from cluster import run_test_clustering_analysis
from zero_shot.step1_get_ram_enzyme_embedding import prepare_embeddings

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
    parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int) # 8
    parser.add_argument("--embed_dim", default=1024, type=int)
    parser.add_argument("--repr_dim", default=512, type=int)
    parser.add_argument("--topk_residues", default=24, type=int)
    parser.add_argument("--save_dir", default="/datapool/home/tanlijun/RAMER/dec-v2/mechpair_light_10k-full")
    parser.add_argument("--val_ratio", default=0.0005, type=float) # 0.05
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
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
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


class SequenceOnlyMechanismHead(nn.Module):
    """
    Final exported protein representation is sequence-only.
    Reaction and structure supervise the sequence representation during training.
    """
    def __init__(self, dim: int, repr_dim: int, topk: int):
        super().__init__()
        self.seq_selector = ReactionGuidedSelector(dim, topk=topk)
        self.struct_selector = ReactionGuidedSelector(dim, topk=topk)

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


def build_step_outputs(batch: Dict, protein_encoder, struct_res_encoder, reaction_encoder, mech_head, loss_fn):
    seq_res, seq_mask = protein_encoder(batch["protein_sequence"])
    rxn = reaction_encoder(batch["reaction_smiles"])
    reaction_global = rxn["global"]

    struct_res, struct_mask = struct_res_encoder(
        batch["gearnet_residue_embedding"].cuda(non_blocking=True),
        batch["gearnet_residue_mask"].cuda(non_blocking=True),
    )
    seq_res, struct_res, shared_mask = align_residue_modalities(seq_res, seq_mask, struct_res, struct_mask)
    out = mech_head(seq_res, struct_res, shared_mask, reaction_global)

    global_loss = symmetric_nce(loss_fn, out["z_func_seq"], out["z_func_rxn"])
    local_loss = symmetric_nce(loss_fn, out["z_local_seq"], out["z_func_rxn"])
    struct_loss = 1.0 - F.cosine_similarity(
        F.normalize(out["z_local_seq"], dim=-1),
        F.normalize(out["z_local_struct"], dim=-1),
        dim=-1,
    ).mean()
    export_loss = symmetric_nce(loss_fn, out["z_protein"], out["z_func_rxn"])
    return {"global_loss": global_loss, "local_loss": local_loss, "struct_loss": struct_loss, "export_loss": export_loss}


def aggregate_losses(step, args):
    total = (
        args.global_weight * step["global_loss"]
        + args.local_weight * step["local_loss"]
        + args.struct_weight * step["struct_loss"]
        + args.export_weight * step["export_loss"]
    )
    return StepOutputs(total, step["global_loss"], step["local_loss"], step["struct_loss"], step["export_loss"])


def train_one_epoch(model_dict, loader, optimizer, loss_fn, args):
    for m in model_dict.values():
        m.train()
    meters = {k: 0.0 for k in ["total", "global", "local", "struct", "export"]}
    for batch in tqdm(loader, disable=dist.get_rank() != 0):
        step = build_step_outputs(batch, model_dict["protein_encoder"], model_dict["struct_res_encoder"], model_dict["reaction_encoder"], model_dict["mech_head"], loss_fn)
        outs = aggregate_losses(step, args)
        optimizer.zero_grad()
        outs.total_loss.backward()
        optimizer.step()

        meters["total"] += outs.total_loss.item()
        meters["global"] += outs.global_loss.item()
        meters["local"] += outs.local_loss.item()
        meters["struct"] += outs.struct_loss.item()
        meters["export"] += outs.export_loss.item()

    num_batches = max(1, len(loader))
    local_tensor = torch.tensor([meters[k] / num_batches for k in ["total", "global", "local", "struct", "export"]], device="cuda")
    dist.reduce(local_tensor, dst=0, op=dist.ReduceOp.SUM)
    if dist.get_rank() == 0:
        local_tensor /= dist.get_world_size()
    return local_tensor


@torch.no_grad()
def validate_one_epoch(model_dict, loader, loss_fn, args):
    for m in model_dict.values():
        m.eval()
    meters = {k: 0.0 for k in ["total", "global", "local", "struct", "export"]}
    for batch in tqdm(loader, disable=True):
        step = build_step_outputs(batch, model_dict["protein_encoder"], model_dict["struct_res_encoder"], model_dict["reaction_encoder"], model_dict["mech_head"], loss_fn)
        outs = aggregate_losses(step, args)
        meters["total"] += outs.total_loss.item()
        meters["global"] += outs.global_loss.item()
        meters["local"] += outs.local_loss.item()
        meters["struct"] += outs.struct_loss.item()
        meters["export"] += outs.export_loss.item()

    num_batches = max(1, len(loader))
    local_tensor = torch.tensor([meters[k] / num_batches for k in ["total", "global", "local", "struct", "export"]], device="cuda")
    dist.reduce(local_tensor, dst=0, op=dist.ReduceOp.SUM)
    if dist.get_rank() == 0:
        local_tensor /= dist.get_world_size()
    return local_tensor


def build_checkpoint_state(epoch, model_dict, loss_fn):
    return {
        "epoch": epoch,
        "protein_encoder": model_dict["protein_encoder"].module.state_dict(),
        "struct_res_encoder": model_dict["struct_res_encoder"].module.state_dict(),
        "reaction_encoder": model_dict["reaction_encoder"].module.state_dict(),
        "mech_head": model_dict["mech_head"].module.state_dict(),
        "temperature": loss_fn.temperature.detach().cpu(),
    }


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


def run_cluster_eval_epoch(model_dict: Dict[str, nn.Module], epoch: int, args):
    # prepare_embeddings, run_test_clustering_analysis = get_cluster_eval_helpers()
    if prepare_embeddings is None or run_test_clustering_analysis is None:
        return {"epoch": epoch, "dataset": args.cluster_eval_name, "status": "skipped", "note": "helpers not found"}

    tmp_dir = os.path.join(args.save_dir, "cluster_eval_tmp")
    result_dir = os.path.join(args.save_dir, "clustering_results")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    seq_ckpt_path = os.path.join(tmp_dir, f"protein_encoder_epoch_{epoch}.pth")
    h5_path = os.path.join(tmp_dir, f"{args.cluster_eval_name}_epoch_{epoch}.h5")
    torch.save(model_dict["protein_encoder"].module.state_dict(), seq_ckpt_path)

    try:
        out_h5 = prepare_embeddings(
            input_path=args.cluster_eval_json,
            output_hdf5=h5_path,
            model_path=seq_ckpt_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        results = run_test_clustering_analysis(
            h5_path=out_h5,
            gt_csv_path=args.cluster_eval_truth_csv,
            save_path=result_dir,
            epoch=epoch,
        )
        row = _flatten_cluster_results(results, epoch, args.cluster_eval_name)
    except Exception as e:
        row = {"epoch": epoch, "dataset": args.cluster_eval_name, "status": "failed", "note": repr(e)}
    # finally:
    #     try:
    #         if os.path.exists(seq_ckpt_path):
    #             os.remove(seq_ckpt_path)
    #     except OSError:
    #         pass
    return row


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
        setup_logging(os.path.join(args.save_dir, "train.log"))

    protein_encoder = BioData2Seq("protein-sequence", embed_dim=args.embed_dim).cuda()
    struct_res_encoder = BioData2Seq("structure-residue", embed_dim=args.embed_dim).cuda()
    reaction_encoder = BioData2Seq("reaction", embed_dim=args.embed_dim).cuda()
    mech_head = SequenceOnlyMechanismHead(args.embed_dim, args.repr_dim, topk=args.topk_residues).cuda()
    loss_fn = InfoNCELoss(init_temp=1.0).cuda()

    model_dict = {
        "protein_encoder": nn.parallel.DistributedDataParallel(protein_encoder, device_ids=[local_rank]),
        "struct_res_encoder": nn.parallel.DistributedDataParallel(struct_res_encoder, device_ids=[local_rank]),
        "reaction_encoder": nn.parallel.DistributedDataParallel(reaction_encoder, device_ids=[local_rank], find_unused_parameters=True),
        "mech_head": nn.parallel.DistributedDataParallel(mech_head, device_ids=[local_rank]),
    }

    # freeze most of reaction encoder for 10k setup
    for _, p in model_dict["reaction_encoder"].named_parameters():
        p.requires_grad = False
    for name, p in model_dict["reaction_encoder"].module.named_parameters():
        lname = name.lower()
        if any(tag in lname for tag in ["transformer", "position", "fusion", "linear", "projection", "aggreg"]):
            p.requires_grad = True

    optimizer = optim.AdamW([
        {"params": filter(lambda p: p.requires_grad, model_dict["protein_encoder"].parameters()), "lr": args.base_lr},
        {"params": filter(lambda p: p.requires_grad, model_dict["struct_res_encoder"].parameters()), "lr": args.base_lr},
        {"params": filter(lambda p: p.requires_grad, model_dict["reaction_encoder"].parameters()), "lr": args.reaction_lr},
        {"params": model_dict["mech_head"].parameters(), "lr": args.base_lr},
        {"params": [loss_fn.temperature], "lr": args.temperature_lr},
    ])

    dataset = BioDataset(
        json_file=args.json_file,
        rhea_smiles_mapping=args.rhea_smiles_file,
        graph_embedding_path=args.graph_embedding_path,
        residue_embedding_path=args.residue_embedding_path,
    )
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    g = get_ddp_generator(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True, generator=g, collate_fn=custom_collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, pin_memory=True, collate_fn=custom_collate_fn)

    print("Starting training with {} samples ({} train, {} val)".format(len(dataset), len(train_dataset), len(val_dataset)))
    best_ari = -1e18
    best_score = -1e18
    ari_topk: List[Dict] = []

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if local_rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}")
        local_rank = int(os.environ["LOCAL_RANK"]) ###
        train_loader_len = len(train_loader)
        print(f"[Rank {local_rank}] train_loader length = {train_loader_len}")
        dist.barrier() ###
        train_metrics = train_one_epoch(model_dict, train_loader, optimizer, loss_fn, args)
        val_metrics = validate_one_epoch(model_dict, val_loader, loss_fn, args)

        if local_rank == 0:
            tr_total, tr_global, tr_local, tr_struct, tr_export = train_metrics.tolist()
            va_total, va_global, va_local, va_struct, va_export = val_metrics.tolist()
            message = (
                f"Epoch {epoch + 1}: train_total={tr_total:.4f}, val_total={va_total:.4f}, "
                f"val_global={va_global:.4f}, val_local={va_local:.4f}, val_struct={va_struct:.4f}, val_export={va_export:.4f}, "
                f"temperature={loss_fn.temperature.item():.4f}"
            )
            print(message)
            logging.info(message)

            state = build_checkpoint_state(epoch + 1, model_dict, loss_fn)
            score = -va_total
            if score > best_score:
                best_score = score
                save_checkpoint(os.path.join(args.save_dir, "best_checkpoint.pt"), state)

            if args.enable_cluster_eval and ((epoch + 1) % args.cluster_eval_every == 0):
                cluster_row = run_cluster_eval_epoch(model_dict, epoch + 1, args)
                append_cluster_eval_csv(os.path.join(args.save_dir, "clustering_eval_history.csv"), cluster_row)
                print("Cluster eval:", cluster_row)
                # logging.info("Cluster eval: " + json.dumps(cluster_row, ensure_ascii=False))
                logging.info("Cluster eval: " + json.dumps(cluster_row, ensure_ascii=False, default=str))
                if cluster_row.get("status") == "ok" and cluster_row.get("ari") is not None:
                    ari = float(cluster_row["ari"])
                    ari_topk = update_topk_checkpoints(args.save_dir, epoch + 1, "ari", ari, state, ari_topk, args.save_top_k_ari)
                    if ari > best_ari:
                        best_ari = ari
                        save_checkpoint(os.path.join(args.save_dir, "best_ari_checkpoint.pt"), state)

            if ari_topk:
                topk_msg = "Top-K(ARI) checkpoints: " + ", ".join([f"#{i+1}: epoch={x['epoch']}, ari={x['metric']:.6f}" for i, x in enumerate(ari_topk)])
                print(topk_msg)
                logging.info(topk_msg)

    if local_rank == 0:
        print(f"Training completed. Best ARI={best_ari:.6f}")
        logging.info(f"Training completed. Best ARI={best_ari:.6f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    args = prepare()
    set_seed(args.seed)
    time_start = time.time()
    main(args)
    if int(os.environ["LOCAL_RANK"]) == 0:
        print(f"time elapsed: {time.time() - time_start:.2f} seconds")