#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAMER 单文件训练：原编码器 + 原训练头 + A 多局部直接对齐。

无类别监督。两个实验使用同一模型主干、20k 固定数据、输出和聚类协议：
  --variant baseline : 原四损失 + 可选部署均值直接对齐
  --variant a        : baseline + K-query 序列/结构融合与反应 InfoNCE
  --variant b        : 无 A 融合；用已知酶-反应关系修正对比学习正负例
  --variant ab       : 同时启用 A 融合和 B 关系感知对比学习
最终 embedding 恒为原 ProtT5 序列残基 masked mean，不使用教师/学生蒸馏。
Diversity loss 在 A/AB 计算并记录；权重默认0，只有设置正权重才参与训练。

本文件在运行时引用原编码器文件，不依赖此前拆分的 ramer/ 包。
快速输入模式直接读取已准备好的 prepared_inputs token IDs（不缓存神经隐藏表示）；
字符串输入模式完全按原编码器 forward，速度可能更慢。
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


# Rank 0 writes a persistent train.log in addition to Slurm's .out/.err files.
_FILE_LOGGER = logging.getLogger("ramer_standalone_training")
_FILE_LOGGER.setLevel(logging.INFO)
_FILE_LOGGER.propagate = False

def setup_file_logger(path):
    """Configure an append-only, timestamped log file on rank 0."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in list(_FILE_LOGGER.handlers):
        _FILE_LOGGER.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _FILE_LOGGER.addHandler(handler)

def log_to_file(message, level=logging.INFO):
    if _FILE_LOGGER.handlers:
        _FILE_LOGGER.log(level, str(message))

def emit_log(message):
    """Print to Slurm stdout and write the same message to train.log."""
    print(message, flush=True)
    log_to_file(message)



def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['baseline', 'a', 'b', 'ab'], default='baseline')
    p.add_argument('--encoder-dir', required=True,
                   help='含 proteinT5_embedding_v2.py 等原编码器文件的目录')
    p.add_argument('--json-file', '--json_file', required=True)
    p.add_argument('--rhea-smiles-file', '--rhea_smiles_file', required=True)
    p.add_argument('--graph-embedding-path', '--graph_embedding_path', required=True)
    p.add_argument('--residue-embedding-path', '--residue_embedding_path', required=True)
    p.add_argument('--protein-model-path', required=True)
    p.add_argument('--reaction-model-path', required=True)
    p.add_argument('--token-cache-dir', default='',
                   help='旧 prepare_data.py 生成的完整 prepared_inputs 目录，建议使用')
    p.add_argument('--node-cache-dir', default='',
                   help='可选：已 pack-nodes 的 prepared_inputs 目录')
    p.add_argument('--save-dir', required=True)
    p.add_argument('--subset-file', default='', help='2万条的共同样本清单，四组须共用同一路径')
    p.add_argument('--subset-size', type=int, default=20000, help='过滤后的抽样总数；0 表示全量')
    p.add_argument('--subset-seed', type=int, default=3408, help='独立于模型初始化的固定抽样种子')
    p.add_argument('--seed', type=int, default=3408)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', '--batch_size', type=int, default=16)
    p.add_argument('--embed-dim', type=int, default=1024)
    p.add_argument('--repr-dim', type=int, default=512)
    p.add_argument('--topk-residues', type=int, default=24)
    p.add_argument('--base-lr', type=float, default=1e-4)
    p.add_argument('--reaction-lr', type=float, default=5e-6)
    p.add_argument('--temperature-lr', type=float, default=2e-5)
    p.add_argument('--global-weight', type=float, default=1.0)
    p.add_argument('--local-weight', type=float, default=0.5)
    p.add_argument('--struct-weight', type=float, default=0.3)
    p.add_argument('--export-weight', type=float, default=0.5)
    p.add_argument('--deploy-weight', type=float, default=0.2,
                   help='两组共同的“真实序列均值↔反应全局向量”InfoNCE；设0可还原旧式四损失')
    p.add_argument('--a-weight', type=float, default=0.5,
                   help='A 多局部融合↔反应对比权重')
    p.add_argument('--diversity-weight', type=float, default=0.0,
                   help='多查询注意力差异正则权重；默认0')
    p.add_argument('--diversity-mode', choices=['fixed','dynamic','floor','dynamic_floor'], default='fixed')
    p.add_argument('--diversity-floor', type=float, default=0.10)
    p.add_argument('--diversity-floor-penalty', type=float, default=5.0)
    p.add_argument('--diversity-warmup-epochs', type=int, default=8)
    p.add_argument('--diversity-decay-epochs', type=int, default=15)
    p.add_argument('--diversity-weight-mid', type=float, default=0.005)
    p.add_argument('--diversity-weight-late', type=float, default=0.002)
    p.add_argument('--attention-export-every', type=int, default=0,
                   help='每隔多少 epoch 导出查询注意力；0=关闭')
    p.add_argument('--attention-export-samples', type=int, default=64)
    p.add_argument('--attention-top-k', type=int, default=20)
    p.add_argument('--relation-pair-sampler', action='store_true',
                   help='B 正例批采样：每个 batch 注入已知共享同一 RHEA 的不同蛋白对')
    p.add_argument('--pairs-per-batch', type=int, default=2,
                   help='每个 16 样本 batch 目标注入的正例蛋白对数，建议 2')
    p.add_argument('--pair-sampler-seed', type=int, default=3408)
    p.add_argument('--b-sequence-weight', type=float, default=0.0,
                   help='可选B扩展：基于共享已知反应的序列-序列弱关系损失；默认0，B核心仅修正对比正负例')
    p.add_argument('--num-slots', type=int, default=4)
    p.add_argument('--slot-dim', type=int, default=256)
    p.add_argument('--slot-heads', type=int, default=4)
    p.add_argument('--compute-val-loss', action='store_true',
                   help='默认不划分验证集；启用后需要 --val-ratio > 0')
    p.add_argument('--val-ratio', type=float, default=0.05,
                   help='仅在 --compute-val-loss 时从同一2w样本保留此比例')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--prefetch-factor', type=int, default=2)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--tf32', action='store_true')
    p.add_argument('--fused-adamw', action='store_true')
    p.add_argument('--cluster-eval', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--cluster-eval-every', type=int, default=1)
    p.add_argument('--cluster-eval-input', default='', help='例如 Price-149.json，JSON列表或FASTA')
    p.add_argument('--cluster-eval-truth-csv', default='', help='仅评估读取，不用于训练')
    p.add_argument('--cluster-eval-mode', choices=['legacy', 'kmeans'], default='legacy')
    p.add_argument('--cluster-eval-module-root', default='', help='原 cluster.py 所在目录')
    p.add_argument('--cluster-eval-n-clusters', type=int, default=0,
                   help='只用于独立 kmeans 协议；不能用测试标签自动确定')
    p.add_argument('--cluster-eval-batch-size', type=int, default=8)
    p.add_argument('--cluster-eval-role', choices=['monitor','development','test'], default='monitor')
    p.add_argument('--select-best-ari', action='store_true',
                   help='只能在标记为 development 的评估集上开启')
    p.add_argument('--save-every', type=int, default=1)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--resume', default='', help='仅可使用本训练文件生成的完整 epoch checkpoint')
    p.add_argument('--max-steps', type=int, default=0, help='冒烟测试上限；不作为完整训练')
    p.add_argument('--save-full-sequence', action='store_true',
                   help='额外保存原 BioData2Seq 兼容的完整序列 state_dict（文件很大）')
    a = p.parse_args()
    if a.batch_size < 2 or a.epochs < 1 or a.num_workers < 0 or a.prefetch_factor < 1:
        p.error('batch-size>=2, epochs>=1, workers>=0, prefetch-factor>=1')
    if a.num_slots < 1 or a.slot_dim % a.slot_heads:
        p.error('num-slots>=1 且 slot-dim 必须能被 slot-heads 整除')
    if a.compute_val_loss and not 0 < a.val_ratio < 1:
        p.error('--compute-val-loss 要求 0 < --val-ratio < 1')
    if a.cluster_eval and (not a.cluster_eval_input or not a.cluster_eval_truth_csv):
        p.error('开启聚类评估时必须提供 --cluster-eval-input 和 --cluster-eval-truth-csv')
    if a.cluster_eval_mode == 'kmeans' and a.cluster_eval and a.cluster_eval_n_clusters < 2:
        p.error('kmeans 需预先固定 --cluster-eval-n-clusters >= 2')
    if a.select_best_ari and (not a.cluster_eval or a.cluster_eval_role != 'development'):
        p.error('只有独立 development 开发集允许选最佳 ARI checkpoint')
    if a.subset_size and not a.subset_file:
        p.error('为保证公平复现实验，--subset-size 非0时必须指定公共 --subset-file')
    if a.save_every < 1 or a.cluster_eval_every < 1:
        p.error('save-every 和 cluster-eval-every 必须>=1')
    if any(x < 0 for x in [a.global_weight,a.local_weight,a.struct_weight,
                            a.export_weight,a.deploy_weight,a.a_weight,a.diversity_weight,a.b_sequence_weight]):
        p.error('损失权重不能为负数')
    if a.diversity_weight > 0 and a.variant not in ('a','ab'):
        p.error('--diversity-weight>0 仅适用于 variant=a/ab')
    if a.relation_pair_sampler and a.variant not in ('b', 'ab'):
        p.error('--relation-pair-sampler 仅允许 variant=b/ab')
    if a.pairs_per_batch < 1 or a.pairs_per_batch*2 >= a.batch_size:
        p.error('需要 1 <= pairs-per-batch 且两倍小于 batch-size，保留部分随机背景')
    if a.b_sequence_weight > 0 and a.variant not in ('b','ab'):
        p.error('--b-sequence-weight>0 仅适用于 variant=b/ab')
    if a.diversity_floor < 0 or a.diversity_floor_penalty <= 0 or a.diversity_warmup_epochs < 1 or a.diversity_decay_epochs < a.diversity_warmup_epochs:
        p.error('diversity floor 或动态 epoch 设置不合法')
    if a.diversity_weight_mid < 0 or a.diversity_weight_late < 0:
        p.error('动态 diversity 权重不能为负数')
    if a.attention_export_every < 0 or a.attention_export_samples < 1 or a.attention_top_k < 1:
        p.error('attention export 参数不合法')
    if a.attention_export_every and a.variant not in ('a','ab'):
        p.error('查询注意力导出仅适用于 variant=a/ab')
    return a


def setup_dist():
    rank = int(os.environ.get('RANK','0'))
    world = int(os.environ.get('WORLD_SIZE','1'))
    local = int(os.environ.get('LOCAL_RANK','0'))
    dev = torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        torch.cuda.set_device(dev)
    if world > 1:
        dist.init_process_group('nccl' if dev.type=='cuda' else 'gloo', device_id=dev if dev.type=='cuda' else None)
    return rank,world,dev


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=json_safe)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def atomic_torch(path, data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(data,tmp); os.replace(tmp,path)


def json_safe(value):
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,(str,int,float,bool)) or value is None: return value
    return str(value)


def record_fingerprint(records):
    """与已交付的 prepare_data.py 所用 fingerprint 计算完全一致。"""
    digest=hashlib.sha256()
    for r in records:
        digest.update(json.dumps([r['primaryAccession'],r['sequence'],r['RHEA'],r['RHEA'],r['smiles']],
            ensure_ascii=False,separators=(',',':')).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def load_training_records(a):
    """优先直接使用已完成的缓存清单，避免在共享存储重复扫描 26 万个路径。"""
    with open(a.rhea_smiles_file,encoding='utf-8') as f: reactions=json.load(f)
    if a.token_cache_dir:
        root=Path(a.token_cache_dir)
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('cache_version')!=1:
            raise ValueError('缓存版本不匹配，请使用此前的 prepare_data.py')
        originals=json.loads((root/'training_records.json').read_text(encoding='utf-8'))
        # 旧缓存只有经过准备时验证的过滤清单；再次核对它确实来自本次指定的训练 JSON。
        # 只读取一个 JSON 大文件，不重复做数十万个小文件的 stat 查询。
        source_rows=json.loads(Path(a.json_file).read_text(encoding='utf-8'))
        for item in originals:
            ix=item['source_index']
            if ix >= len(source_rows) or any(source_rows[ix].get(k)!=item[k]
                        for k in ('primaryAccession','sequence','RHEA')):
                raise ValueError('缓存与本次指定的 --json-file 不一致，请重新 prepare_data.py')
        del source_rows
        records=[]
        for r in originals:
            reaction=r['RHEA']
            if reaction not in reactions: raise ValueError(f'RHEA {reaction} 已不在提供的反应映射中')
            records.append(dict(primaryAccession=r['primaryAccession'],sequence=r['sequence'],
                    RHEA=reaction,smiles=reactions[reaction],source_index=r['source_index'],
                    node_path=str(Path(a.residue_embedding_path)/f"{r['primaryAccession']}_node.pt")))
        fingerprint=record_fingerprint(records)
        if fingerprint!=manifest['dataset_fingerprint']:
            raise ValueError('当前反应映射或缓存记录与 manifest 不一致，拒绝使用失配缓存')
        return records,fingerprint,manifest
    with open(a.json_file,encoding='utf-8') as f: raw=json.load(f)
    graph_names={x.name for x in os.scandir(a.graph_embedding_path) if x.is_file()}
    node_names={x.name for x in os.scandir(a.residue_embedding_path) if x.is_file()}
    records=[]
    for idx,item in enumerate(raw):
        acc=item.get('primaryAccession'); reaction=item.get('RHEA')
        if not acc or f'{acc}_graph.pt' not in graph_names or f'{acc}_node.pt' not in node_names or reaction not in reactions:
            continue
        records.append(dict(primaryAccession=acc,sequence=item['sequence'],RHEA=reaction,
            smiles=reactions[reaction],source_index=idx,node_path=str(Path(a.residue_embedding_path)/f'{acc}_node.pt')))
    return records,record_fingerprint(records),None


def get_subset_indices(count,fp,a,rank,world):
    """公共固定2万条。重复 runs 使用同一清单，不会依赖文件读取顺序或模型 seed。"""
    requested=count if a.subset_size==0 else a.subset_size
    if requested>count: raise ValueError(f'所需 {requested} 条，过滤后只有 {count} 条')
    if a.subset_size==0: return list(range(count))
    path=Path(a.subset_file)
    if rank==0 and not path.exists():
        selected=sorted(random.Random(a.subset_seed).sample(range(count),requested))
        content=dict(dataset_fingerprint=fp,filtered_total=count,
                     subset_seed=a.subset_seed,subset_size=requested,
                     selected_indices=selected)
        # 两个作业同时首次创建公共 subset 文件时，只允许一个发布；
        # 发布采用原子硬链接，其他作业永远看不到尚未写完的 JSON。
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
        try:
            tmp.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            try:os.link(tmp,path)
            except FileExistsError:pass
        finally:
            tmp.unlink(missing_ok=True)
    if world>1: dist.barrier()
    obj=json.loads(path.read_text(encoding='utf-8'))
    if any([obj.get('dataset_fingerprint')!=fp, obj.get('filtered_total')!=count,
            obj.get('subset_seed')!=a.subset_seed,obj.get('subset_size')!=requested]):
        raise ValueError('已有 subset-file 与本次数据、抽样参数不一致；禁止覆盖')
    indices=obj['selected_indices']
    if len(indices)!=requested or len(set(indices))!=requested or min(indices)<0 or max(indices)>=count:
        raise ValueError('subset-file 的数据索引不合法')
    return indices


def divide_subset(indices,a):
    if not a.compute_val_loss: return indices,[]
    shuffled=list(indices)
    random.Random(a.subset_seed+991).shuffle(shuffled)
    num_val=max(a.batch_size,int(len(shuffled)*a.val_ratio))
    if num_val>=len(shuffled): raise ValueError('验证集占比太高')
    return sorted(shuffled[num_val:]),sorted(shuffled[:num_val])


class FlatTokens:
    def __init__(self,folder,metadata):
        self.folder=Path(folder); self.metadata=metadata
        self._open()
    def _open(self):
        self.offsets=np.load(self.folder/'offsets.npy',mmap_mode='r',allow_pickle=False)
        self.ids=np.memmap(self.folder/'ids.int32',mode='r',dtype=np.int32)
        if int(self.offsets[-1])!=len(self.ids) or int(self.offsets[0])!=0:
            raise ValueError('分词缓存 offsets 损坏')
    def __len__(self): return len(self.offsets)-1
    def __getitem__(self,i): return self.ids[int(self.offsets[i]):int(self.offsets[i+1])]
    def __getstate__(self): return dict(folder=self.folder,metadata=self.metadata)
    def __setstate__(self,state):
        self.folder=state['folder'];self.metadata=state['metadata'];self._open()


class PackedNodes:
    def __init__(self,root,manifest):
        folder=Path(root)/'nodes'
        self.meta=manifest['nodes']
        self.mapping=np.load(folder/'row_to_node.npy',mmap_mode='r',allow_pickle=False)
        self.offsets=np.load(folder/'offsets.npy',mmap_mode='r',allow_pickle=False)
        self.flat=np.memmap(folder/'features.float32',mode='r',dtype=np.float32,
            shape=(self.meta['total_residues'],self.meta['feature_dim']))
        self.folder=folder
    def __getitem__(self,i):
        k=int(self.mapping[i]); values=self.flat[int(self.offsets[k]):int(self.offsets[k+1])]
        return torch.from_numpy(np.array(values,dtype=np.float32,copy=True))
    def __getstate__(self): return dict(root=self.folder.parent,manifest={'nodes':self.meta})
    def __setstate__(self,state): self.__init__(state['root'],state['manifest'])


def pad_tokens(tokens,pad_id,side='right'):
    n=max(len(x) for x in tokens)
    ids=torch.full((len(tokens),n),int(pad_id),dtype=torch.long)
    mask=torch.zeros_like(ids)
    for i,x in enumerate(tokens):
        t=torch.as_tensor(np.array(x,dtype=np.int64,copy=True))
        start=0 if side=='right' else n-len(x)
        ids[i,start:start+len(x)]=t
        mask[i,start:start+len(x)]=1
    return {'input_ids':ids,'attention_mask':mask}


class EnzymeDataset(Dataset):
    def __init__(self,records,cache_root='',manifest=None,node_cache_root=''):
        self.records=records; self.protein_cache=None; self.mol_cache=None; self.node_cache=None
        self.known_reactions_by_sequence = {}
        if cache_root:
            root=Path(cache_root)
            if 'protein' in manifest:
                self.protein_cache=FlatTokens(root/'protein',manifest['protein'])
                if len(self.protein_cache)!=len(records): raise ValueError('蛋白 token 缓存记录数量不一致')
            if 'molecule' in manifest:
                self.mol_cache=FlatTokens(root/'molecule',manifest['molecule'])
                molecule_names=json.loads((root/'molecule'/'molecules.json').read_text(encoding='utf-8'))
                self.mol_index={s:i for i,s in enumerate(molecule_names)}
        if node_cache_root:
            nmanifest=json.loads((Path(node_cache_root)/'manifest.json').read_text(encoding='utf-8'))
            if nmanifest.get('dataset_fingerprint')!=record_fingerprint(records):
                raise ValueError('节点缓存和当前数据源不匹配')
            self.node_cache=PackedNodes(node_cache_root,nmanifest)
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        rec=self.records[i]
        if self.node_cache is not None: nodes=self.node_cache[i]
        else:
            nodes=torch.load(rec['node_path'],map_location='cpu',weights_only=True).float()
        if nodes.ndim==1: nodes=nodes.unsqueeze(0)
        if nodes.ndim!=2 or len(nodes)==0:
            raise ValueError(f'无效结构节点形状 {rec["node_path"]}: {tuple(nodes.shape)}')
        reaction_str=rec['smiles']
        react_s,product_s=reaction_str.split('>>')
        react=react_s.split('.') if react_s else []
        prod=product_s.split('.') if product_s else []
        p_tokens=self.protein_cache[i] if self.protein_cache is not None else None
        m_tokens=[self.mol_cache[self.mol_index[s]] for s in react+prod] if self.mol_cache is not None else None
        known = self.known_reactions_by_sequence.get(rec['sequence'], [str(rec['RHEA'])])
        return dict(sequence=rec['sequence'],reactants=react,products=prod,nodes=nodes,
                    protein_ids=p_tokens,molecule_ids=m_tokens,accession=rec['primaryAccession'],
                    reaction_id=str(rec['RHEA']),known_reaction_ids=list(known))


def make_collate(protein_meta=None,molecule_meta=None):
    def collate(batch):
        nodes=[b['nodes'] for b in batch]
        out_nodes=torch.zeros(len(batch),max(len(x) for x in nodes),nodes[0].size(-1),dtype=torch.float32)
        out_mask=torch.zeros(out_nodes.shape[:2],dtype=torch.bool)
        for i,x in enumerate(nodes):
            if x.size(1)!=out_nodes.size(2):raise ValueError('GearNet 节点维度不一致')
            out_nodes[i,:len(x)]=x
            out_mask[i,:len(x)]=True
        result=dict(sequences=[b['sequence'] for b in batch],
                    reactions=[(b['reactants'],b['products']) for b in batch],
                    nodes=out_nodes,node_mask=out_mask,
                    accessions=[b['accession'] for b in batch],
                    reaction_ids=[b['reaction_id'] for b in batch],
                    known_reaction_ids=[b['known_reaction_ids'] for b in batch])
        if protein_meta is not None:
            result['protein_tokens']=pad_tokens([b['protein_ids'] for b in batch],
                    protein_meta['pad_token_id'],protein_meta['padding_side'])
        if molecule_meta is not None:
            result['reaction_tokens']=pad_tokens([x for b in batch for x in b['molecule_ids']],
                    molecule_meta['pad_token_id'],molecule_meta['padding_side'])
            result['reaction_meta']=[(len(b['reactants']),len(b['products'])) for b in batch]
        return result
    return collate


def tokenizer_signature(tokenizer,modality):
    sp=getattr(tokenizer,'sp_model',None)
    sp_hash=hashlib.sha256(sp.serialized_model_proto()).hexdigest() if sp is not None and hasattr(sp,'serialized_model_proto') else None
    cfg=dict(class_=type(tokenizer).__name__,vocab=tokenizer.get_vocab(),
        sentencepiece_sha256=sp_hash,special_tokens_map=tokenizer.special_tokens_map,
        padding_side=tokenizer.padding_side,truncation_side=tokenizer.truncation_side,
        pad_token_id=tokenizer.pad_token_id,legacy=getattr(tokenizer,'legacy',None),
        modality=modality,max_length=2048 if modality=='protein' else 512,
        add_special_tokens=modality=='molecule',
        protein_preprocessing='replace_[UZOB]_with_X_then_space_v1' if modality=='protein' else None)
    # 这个字典的 class 键必须与旧 prepare_data.py 保持一致。
    cfg['class']=cfg.pop('class_')
    raw=json.dumps(cfg,ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str).encode()
    return hashlib.sha256(raw).hexdigest()


class SequenceAdapter(nn.Module):
    """保留原 BioData2Seq 的 encoder.* 参数名称。"""
    def __init__(self,cls,dim,path):
        super().__init__();self.encoder=cls(embed_dim=dim,model_path=path)
    def forward(self,value):
        if isinstance(value,dict):
            device=self.encoder.linear.weight.device
            ids=value['input_ids'].to(device,non_blocking=True)
            mask=value['attention_mask'].to(device,non_blocking=True)
            hidden=self.encoder.model(input_ids=ids,attention_mask=mask).last_hidden_state
            return self.encoder.linear(hidden),mask.bool()
        return self.encoder(value)


class ConfiguredReactionEncoder(nn.Module):
    """用原始 MoleculeEmbedding 和 ReactionTokenEncoder 重建原 ReactionRepresentation，不重写结构。"""
    def __init__(self,mod,dim,model_path,drop_decoder=True):
        super().__init__()
        self.encoder=nn.Module()
        self.encoder.molecule_embedder=mod.MoleculeEmbedding(embed_dim=dim,model_name=model_path)
        self.encoder.token_encoder=mod.ReactionTokenEncoder(embed_dim=dim)
        if drop_decoder:
            for name in ('decoder','lm_head'):
                if hasattr(self.encoder.molecule_embedder.model,name):
                    delattr(self.encoder.molecule_embedder.model,name)
    def forward(self,reactions_or_tokens,reaction_meta=None):
        if isinstance(reactions_or_tokens,dict):
            tokens=reactions_or_tokens
            embedder=self.encoder.molecule_embedder
            device=embedder.linear.weight.device
            ids=tokens['input_ids'].to(device,non_blocking=True)
            mask=tokens['attention_mask'].to(device,non_blocking=True)
            with torch.no_grad():
                hidden=embedder.model.encoder(input_ids=ids,attention_mask=mask).last_hidden_state
            mol_emb=embedder.norm(embedder.linear(hidden))
            mol_mask=mask.bool()
            metadata=reaction_meta
        else:
            molecules=[];metadata=[]
            for react,prod in reactions_or_tokens:
                metadata.append((len(react),len(prod)))
                molecules.extend(react);molecules.extend(prod)
            mol_emb,mol_mask=self.encoder.molecule_embedder(molecules)
        if not metadata or any(a+b==0 for a,b in metadata):
            raise ValueError('反应必须至少有1个分子')
        m=mol_mask.to(mol_emb.dtype).unsqueeze(-1)
        pooled=(mol_emb*m).sum(1)/m.sum(1).clamp_min(1.)
        if len(pooled)!=sum(a+b for a,b in metadata):
            raise ValueError('反应分子数量与token不一致')
        maxmol=max(a+b for a,b in metadata)
        tensor=pooled.new_zeros(len(metadata),maxmol,pooled.size(-1))
        tm=torch.zeros(len(metadata),maxmol,dtype=torch.bool,device=pooled.device)
        side=torch.zeros(len(metadata),maxmol,dtype=torch.long,device=pooled.device)
        offset=0
        for i,(a,b) in enumerate(metadata):
            tensor[i,:a+b]=pooled[offset:offset+a+b]
            tm[i,:a+b]=True
            side[i,a:a+b]=1
            offset+=a+b
        return self.encoder.token_encoder(tensor,tm,side)


def configure_reaction_trainability(module):
    tags=('transformer','position','fusion','linear','projection','aggreg')
    for name,p in module.named_parameters(): p.requires_grad=any(tag in name.lower() for tag in tags)


def masked_mean(x,mask):
    m=mask.to(x.dtype).unsqueeze(-1)
    return (x*m).sum(1)/m.sum(1).clamp_min(1.)


class ReactionGuidedSelector(nn.Module):
    """原反应引导残基注意力；只返回训练真正用到的局部向量。"""
    def __init__(self,dim,topk=24):
        super().__init__()
        self.topk=topk
        self.score=nn.Sequential(nn.Linear(dim*2,dim),nn.GELU(),nn.LayerNorm(dim),nn.Linear(dim,1))
    def forward(self,residue,reaction,mask):
        if not mask.any(dim=-1).all(): raise ValueError('某条样本没有可对齐的序列/结构残基')
        scores=self.score(torch.cat([residue,reaction[:,None,:].expand_as(residue)],dim=-1)).squeeze(-1)
        attn=torch.softmax(scores.float().masked_fill(~mask,-1e9),dim=-1)
        attn=attn*mask.float();attn=attn/attn.sum(-1,keepdim=True).clamp_min(1e-9)
        return (attn.unsqueeze(-1)*residue).sum(1)


def proj(dim,out):return nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.LayerNorm(dim),nn.Linear(dim,out))


class OriginalMechanismHead(nn.Module):
    """保留旧四项损失的头结构；唯一刻意修正：全局使用完整 seq_mask。"""
    def __init__(self,dim,out,topk):
        super().__init__()
        self.seq_selector=ReactionGuidedSelector(dim,topk)
        self.struct_selector=ReactionGuidedSelector(dim,topk)
        self.seq_global_proj=proj(dim,out)
        self.react_global_proj=proj(dim,out)
        self.seq_local_proj=proj(dim,out)
        self.struct_local_proj=proj(dim,out)
        self.export_proj=nn.Sequential(nn.Linear(out*2,dim),nn.GELU(),nn.LayerNorm(dim),nn.Linear(dim,out))
    def forward(self,seq,seq_mask,struct,struct_mask,reaction):
        # 全局直接使用最终导出同一套序列位置；不再用 shared_mask 限制全局。
        seq_mean=masked_mean(seq,seq_mask)
        # local 仍沿用旧代码的“前缀截断+交集”；缺失内部残基必须另提供映射才能严格对齐。
        length=min(seq.size(1),struct.size(1))
        shared=seq_mask[:,:length]&struct_mask[:,:length]
        local_s=self.seq_selector(seq[:,:length],reaction,shared)
        local_g=self.struct_selector(struct[:,:length],reaction,shared)
        z_seq=self.seq_global_proj(seq_mean)
        z_rxn=self.react_global_proj(reaction)
        z_local_seq=self.seq_local_proj(local_s)
        z_local_struct=self.struct_local_proj(local_g)
        z_protein=self.export_proj(torch.cat([z_seq,z_local_seq],-1))
        return dict(global_seq=z_seq,global_rxn=z_rxn,local_seq=z_local_seq,
                    local_struct=z_local_struct,export=z_protein,seq_mean=seq_mean)


class MultiSlotFusion(nn.Module):
    """创新 A：K 个查询并行读取序列/结构残基，逐槽门控；使用融合-反应 InfoNCE。

    查询随机不同初始化，可学习；不输入反应作为查询，以防已知反应捷径。
    不声称 4 个槽=4 个真实活性口袋。diversity loss 保留并记录，默认权重为 0。
    """
    def __init__(self,dim,repr_dim,num_slots,slot_dim,heads):
        super().__init__()
        self.queries=nn.Parameter(torch.randn(num_slots,slot_dim)*0.02)
        self.seq_projection=nn.Sequential(nn.Linear(dim,slot_dim),nn.LayerNorm(slot_dim))
        self.struct_projection=nn.Sequential(nn.Linear(dim,slot_dim),nn.LayerNorm(slot_dim))
        self.seq_attn=nn.MultiheadAttention(slot_dim,heads,dropout=0.,batch_first=True)
        self.struct_attn=nn.MultiheadAttention(slot_dim,heads,dropout=0.,batch_first=True)
        self.gate=nn.Linear(2*slot_dim,slot_dim)
        self.mix=nn.Sequential(nn.Linear(2*slot_dim,slot_dim),nn.GELU(),nn.LayerNorm(slot_dim))
        self.readout=nn.Sequential(nn.Linear(num_slots*slot_dim,repr_dim),nn.GELU(),
                                  nn.LayerNorm(repr_dim),nn.Linear(repr_dim,repr_dim))
    def forward(self,seq,seq_mask,struct,struct_mask,return_attention=False):
        if not seq_mask.any(-1).all() or not struct_mask.any(-1).all():
            raise ValueError('多槽融合需要至少一个有效序列和结构残基')
        q=self.queries.unsqueeze(0).expand(seq.size(0),-1,-1)
        s=self.seq_projection(seq)
        g=self.struct_projection(struct)
        ss,sw=self.seq_attn(q,s,s,key_padding_mask=~seq_mask,need_weights=True)
        gg,gw=self.struct_attn(q,g,g,key_padding_mask=~struct_mask,need_weights=True)
        pair=torch.cat((ss,gg),-1)
        gate=torch.sigmoid(self.gate(pair))
        fused=gate*ss+(1-gate)*gg+self.mix(pair)
        def overlap(attn):
            if attn.size(1)==1: return attn.new_zeros(())
            norm=F.normalize(attn.float(),dim=-1)
            gram=norm@norm.transpose(1,2)
            eye=torch.eye(gram.size(1),device=gram.device,dtype=torch.bool)
            return gram[:,~eye].square().mean()
        fused_repr=self.readout(fused.flatten(1))
        diversity=(overlap(sw)+overlap(gw))*0.5
        if return_attention:
            return fused_repr,diversity,sw,gw
        return fused_repr,diversity


class RAMERModel(nn.Module):
    def __init__(self,a,seq_cls,structure_cls,reaction_module):
        super().__init__()
        self.sequence=SequenceAdapter(seq_cls,a.embed_dim,a.protein_model_path)
        self.structure=nn.Module();self.structure.encoder=structure_cls(a.embed_dim)
        self.reaction=ConfiguredReactionEncoder(reaction_module,a.embed_dim,a.reaction_model_path)
        configure_reaction_trainability(self.reaction)
        self.head=OriginalMechanismHead(a.embed_dim,a.repr_dim,a.topk_residues)
        self.fusion=(MultiSlotFusion(a.embed_dim,a.repr_dim,a.num_slots,a.slot_dim,a.slot_heads)
                     if a.variant in ('a','ab') else None)
        self.temperature=nn.Parameter(torch.tensor(1.,dtype=torch.float32))
    def forward(self,batch,return_attention=False):
        protein=batch.get('protein_tokens') or batch['sequences']
        reactions=batch.get('reaction_tokens') or batch['reactions']
        seq,seq_mask=self.sequence(protein)
        rxn=self.reaction(reactions,batch.get('reaction_meta'))
        device=seq.device
        structure,struct_mask=self.structure.encoder(
            batch['nodes'].to(device,non_blocking=True),
            batch['node_mask'].to(device,non_blocking=True))
        head=self.head(seq,seq_mask,structure,struct_mask,rxn['global'])
        fused=red=seq_attn=struct_attn=None
        if self.fusion is not None:
            if return_attention:
                fused,red,seq_attn,struct_attn=self.fusion(
                    seq,seq_mask,structure,struct_mask,return_attention=True)
            else:
                fused,red=self.fusion(seq,seq_mask,structure,struct_mask)
        return dict(head=head,rxn=rxn['global'],fused=fused,slot_overlap=red,
                    seq_attn=seq_attn,struct_attn=struct_attn,
                    seq_mask=seq_mask if return_attention else None,
                    struct_mask=struct_mask if return_attention else None)


def symmetric_nce(x,y,temp):
    x=F.normalize(x.float(),dim=-1);y=F.normalize(y.float(),dim=-1)
    sim=x@y.T/temp.clamp_min(.3)
    labels=torch.arange(len(x),device=x.device)
    return 0.5*(F.cross_entropy(sim,labels)+F.cross_entropy(sim.T,labels))


def _masked_multi_positive_direction(sim, positive, valid):
    """Multi-positive InfoNCE: denominator contains valid candidates; numerator all known positives."""
    positive = positive.bool() & valid.bool()
    valid = valid.bool()
    row_has_positive = positive.any(dim=1)
    if not row_has_positive.any():
        return sim.new_zeros(())
    neg_inf = torch.finfo(sim.dtype).min
    denominator = torch.logsumexp(sim.masked_fill(~valid, neg_inf), dim=1)
    numerator = torch.logsumexp(sim.masked_fill(~positive, neg_inf), dim=1)
    return (denominator[row_has_positive] - numerator[row_has_positive]).mean()


def relation_aware_nce(x, y, current_reaction_ids, known_reaction_ids, temp, conditional=False):
    """B: use explicit enzyme-reaction annotations to avoid treating known relations as negatives.

    Global mode: all reactions known for an exact protein sequence are positives.
    Conditional mode: only the current reaction is positive; other known reactions for that
    sequence are excluded from the negative denominator because this local representation is
    conditioned on the current reaction. Unannotated relations remain ordinary contrastive
    background; they are not asserted to be biological negatives.
    """
    x=F.normalize(x.float(),dim=-1); y=F.normalize(y.float(),dim=-1)
    sim=x@y.T/temp.clamp_min(.3)
    b=len(current_reaction_ids)
    if len(x)!=b or len(y)!=b or len(known_reaction_ids)!=b:
        raise ValueError('B 关系掩码的 batch 元数据长度不匹配')
    device=sim.device
    positives=torch.zeros((b,b),dtype=torch.bool,device=device)
    valid=torch.ones((b,b),dtype=torch.bool,device=device)
    current=[str(v) for v in current_reaction_ids]
    known=[set(map(str,items)) | {current[i]} for i,items in enumerate(known_reaction_ids)]
    for i in range(b):
        for j in range(b):
            if conditional:
                positives[i,j]=(current[i]==current[j])
                # An alternate known reaction of the anchor enzyme is not a negative for
                # its reaction-conditioned representation; ignore it unless it is current.
                if current[j] in known[i] and current[j] != current[i]:
                    valid[i,j]=False
            else:
                positives[i,j]=(current[j] in known[i])
    if not positives.diagonal().all():
        raise ValueError('B 关系掩码缺少当前配对正例；请检查 RHEA ID 格式')
    loss_lr=_masked_multi_positive_direction(sim,positives,valid)
    loss_rl=_masked_multi_positive_direction(sim.T,positives.T,valid.T)
    return 0.5*(loss_lr+loss_rl), positives.float().sum(), (valid & ~positives).float().sum(), (~valid).float().sum()


def shared_reaction_sequence_loss(seq_vectors, known_reaction_ids, accessions, temperature=0.1):
    """Optional B extension: weakly attract distinct proteins sharing known reactions.

    Disabled by default: sharing a reaction is weak evidence, not proof of the same EC subclass.
    """
    z=F.normalize(seq_vectors.float(),dim=-1)
    b=len(accessions); device=z.device
    if b<2:return z.new_zeros(()),z.new_zeros(()),z.new_zeros(())
    sets=[set(map(str,x)) for x in known_reaction_ids]
    pos=torch.zeros((b,b),dtype=torch.bool,device=device)
    weights=torch.zeros((b,b),dtype=torch.float32,device=device)
    for i in range(b):
        for j in range(b):
            if i==j or accessions[i]==accessions[j]:continue
            inter=sets[i]&sets[j]
            if inter:
                weights[i,j]=len(inter)/max(1,len(sets[i]|sets[j]))
                pos[i,j]=True
    pos.fill_diagonal_(False)
    if not pos.any():return z.new_zeros(()),pos.sum().float(),z.new_zeros(())
    sim=(z@z.T)/max(float(temperature),1e-4)
    valid=~torch.eye(b,dtype=torch.bool,device=device)
    log_den=torch.logsumexp(sim.masked_fill(~valid,torch.finfo(sim.dtype).min),dim=1)
    log_prob=sim-log_den[:,None]
    row_w=weights.sum(1);rows=row_w>0
    loss=-(weights[rows]*log_prob[rows]).sum(1)/row_w[rows].clamp_min(1e-8)
    return loss.mean(),pos.sum().float(),rows.float().sum()


def diversity_weight_for_epoch(epoch,a):
    if a.diversity_mode in ('dynamic','dynamic_floor'):
        if epoch <= a.diversity_warmup_epochs:
            return float(a.diversity_weight)
        if epoch <= a.diversity_decay_epochs:
            return float(a.diversity_weight_mid)
        return float(a.diversity_weight_late)
    return float(a.diversity_weight)


def effective_diversity_loss(raw,a):
    if a.diversity_mode in ('floor','dynamic_floor'):
        above=torch.relu(raw-float(a.diversity_floor))
        below=torch.relu(float(a.diversity_floor)-raw)
        return above + float(a.diversity_floor_penalty)*below.square()
    return raw


def compute_losses(out,temp,a,batch=None,epoch=1):
    h=out['head'];r=out['rxn']
    b_enabled = a.variant in ('b','ab')
    if b_enabled:
        if batch is None or 'reaction_ids' not in batch or 'known_reaction_ids' not in batch:
            raise ValueError('B 方案需要 batch 中的 reaction_ids/known_reaction_ids')
        ids=batch['reaction_ids'];known=batch['known_reaction_ids']
        global_loss,gpos,gneg,gignored=relation_aware_nce(
            h['global_seq'],h['global_rxn'],ids,known,temp,conditional=False)
        local_loss,lpos,lneg,lignored=relation_aware_nce(
            h['local_seq'],h['global_rxn'],ids,known,temp,conditional=True)
        export_loss,epos,eneg,eignored=relation_aware_nce(
            h['export'],h['global_rxn'],ids,known,temp,conditional=True)
        deploy_loss,dpos,dneg,dignored=relation_aware_nce(
            h['seq_mean'],r,ids,known,temp,conditional=False)
        if out['fused'] is not None:
            fusion_loss,fpos,fneg,fignored=relation_aware_nce(
                out['fused'],h['global_rxn'],ids,known,temp,conditional=False)
        else:
            fusion_loss=global_loss.new_zeros(());fpos=gpos*0;fneg=gneg*0;fignored=gignored*0
        n_b_heads = 5.0 if out['fused'] is not None else 4.0
        b_pos=(gpos+lpos+epos+dpos+fpos)/n_b_heads
        b_valid=(gneg+lneg+eneg+dneg+fneg)/n_b_heads
        b_ignored=(gignored+lignored+eignored+dignored+fignored)/n_b_heads
    else:
        global_loss=symmetric_nce(h['global_seq'],h['global_rxn'],temp)
        local_loss=symmetric_nce(h['local_seq'],h['global_rxn'],temp)
        export_loss=symmetric_nce(h['export'],h['global_rxn'],temp)
        deploy_loss=symmetric_nce(h['seq_mean'],r,temp)
        fusion_loss=(symmetric_nce(out['fused'],h['global_rxn'],temp)
                     if out['fused'] is not None else global_loss.new_zeros(()))
        b_pos=global_loss.new_zeros(());b_valid=global_loss.new_zeros(());b_ignored=global_loss.new_zeros(())
    result={
       'global':global_loss,
       'local':local_loss,
       'struct':(1-F.cosine_similarity(F.normalize(h['local_seq'].float(),dim=-1),
                           F.normalize(h['local_struct'].float(),dim=-1),dim=-1)).mean(),
       'export':export_loss,
       'deploy':deploy_loss,
       'fusion':fusion_loss,
       'b_positive_pairs':b_pos,
       'b_valid_background_pairs':b_valid,
       'b_ignored_known_alternatives':b_ignored,
    }
    result['diversity']=(out['slot_overlap'] if out['slot_overlap'] is not None else result['global'].new_zeros(()))
    result['slot_overlap']=result['diversity']
    result['diversity_effective']=effective_diversity_loss(result['diversity'],a)
    current_diversity_weight=diversity_weight_for_epoch(epoch,a)
    result['diversity_weight']=result['diversity'].new_tensor(current_diversity_weight)
    result['diversity_floor_active']=((result['diversity'] > a.diversity_floor).to(result['diversity'].dtype)
        if a.diversity_mode in ('floor','dynamic_floor') else result['diversity'].new_zeros(()))
    if b_enabled and a.b_sequence_weight>0:
        seq_rel,seq_pos,seq_anchors=shared_reaction_sequence_loss(
            h['seq_mean'],batch['known_reaction_ids'],batch['accessions'])
    else:
        seq_rel=result['global'].new_zeros(());seq_pos=result['global'].new_zeros(());seq_anchors=result['global'].new_zeros(())
    result['b_sequence_relation']=seq_rel
    result['b_sequence_positive_pairs']=seq_pos
    result['b_sequence_valid_anchors']=seq_anchors
    result['total']=(a.global_weight*result['global']+a.local_weight*result['local']
        +a.struct_weight*result['struct']+a.export_weight*result['export']
        +a.deploy_weight*result['deploy']+a.a_weight*result['fusion']
        +current_diversity_weight*result['diversity_effective']+a.b_sequence_weight*result['b_sequence_relation'])
    return result


def model_delta(model):
    """不保存冻结的 ProtT5 / MolT5 大底座，但保存所有随机初始化层（即使被冻结）。"""
    state={}
    for k,v in model.state_dict().items():
        if k.startswith('sequence.encoder.model.') and '.lora_' not in k:continue
        if k.startswith('reaction.encoder.molecule_embedder.model.'):continue
        state[k]=v.detach().cpu()
    return state


def load_delta(model,state):
    info=model.load_state_dict(state,strict=False)
    missing=[k for k in info.missing_keys if not (
       (k.startswith('sequence.encoder.model.') and '.lora_' not in k)
       or k.startswith('reaction.encoder.molecule_embedder.model.'))]
    if missing or info.unexpected_keys:
        raise RuntimeError(f'增量checkpoint不匹配：missing={missing[:10]} unexpected={info.unexpected_keys[:10]}')


def sequence_state(model,full=False):
    """BioData2Seq('protein-sequence') 兼容键 encoder.*。"""
    return {k:v.detach().cpu() for k,v in model.sequence.state_dict().items()
        if full or not (k.startswith('encoder.model.') and '.lora_' not in k)}


class RheaPairBatchSampler(torch.utils.data.Sampler):
    """训练子集内部关系感知批采样。配对来自相同已知 RHEA、不同序列。

    用途：提高 B 的 extra positive 触发率。对小类别会过采样，因此同时运行
    同样 4GPU/seed 的随机采样对照；不是模型效果的无偏估计。
    """
    def __init__(self, dataset, train_indices, batch_size, rank, world,
                 pairs_per_batch=2, seed=3408):
        self.dataset = dataset
        self.train_indices = list(train_indices)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world = int(world)
        self.pairs_per_batch = int(pairs_per_batch)
        self.seed = int(seed)
        self.epoch = 0
        self.last_epoch_stats = {}
        # Subset(dataset, train_indices) 的索引是这里的局部索引。
        groups = {}
        for local_i, source_i in enumerate(self.train_indices):
            rec = dataset.records[source_i]
            groups.setdefault(str(rec['RHEA']), []).append(local_i)
        self.groups = {}
        for reaction, members in groups.items():
            seq_seen = {dataset.records[self.train_indices[i]]['sequence'] for i in members}
            if len(seq_seen) >= 2:
                self.groups[reaction] = members
        if not self.groups:
            raise ValueError('固定 20k 训练集没有任何同 RHEA 的不同蛋白对，无法采样')
        self.reaction_keys = sorted(self.groups)
        self.total_groups = len(groups)
        self.steps = len(self.train_indices) // (self.batch_size*self.world)
        if self.steps == 0:
            raise ValueError('当前数据量不足一个全局批次')

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = random.Random(self.seed + 10007*self.epoch + 7919*self.rank)
        # 随机背景部分按 rank 分片；注入的关系正例允许有限重采样。
        perm = list(range(len(self.train_indices)))
        rng.shuffle(perm)
        pool = perm[self.rank::self.world]
        cursor = 0
        emitted = []
        injected_pairs = 0
        def next_random(excluded):
            nonlocal cursor, pool
            while True:
                if cursor >= len(pool):
                    rng.shuffle(pool)
                    cursor = 0
                idx = pool[cursor]
                cursor += 1
                if idx not in excluded:
                    return idx
        for _ in range(self.steps):
            row = []
            used = set()
            for _pair in range(self.pairs_per_batch):
                found = False
                # 同 RHEA 但必须不同蛋白序列；避免把同序列的重复标注当正例。
                for _attempt in range(100):
                    react = rng.choice(self.reaction_keys)
                    cand = self.groups[react]
                    if len(cand) < 2:
                        continue
                    a, b = rng.sample(cand, 2)
                    if a in used or b in used:
                        continue
                    ra = self.dataset.records[self.train_indices[a]]
                    rb = self.dataset.records[self.train_indices[b]]
                    if ra['sequence'] == rb['sequence']:
                        continue
                    row.extend([a,b])
                    used.update([a,b])
                    injected_pairs += 1
                    found = True
                    break
                if not found:
                    # 极稀疏关联数据时保留随机回退；不做假配对。
                    break
            while len(row) < self.batch_size:
                i = next_random(used)
                row.append(i)
                used.add(i)
            rng.shuffle(row)
            emitted.extend(row)
            yield row
        self.last_epoch_stats = {
            'epoch':self.epoch+1,
            'rows_drawn_rank':len(emitted),
            'unique_subset_indices_rank':len(set(emitted)),
            'duplicate_draw_fraction_rank':1.0-len(set(emitted))/max(1,len(emitted)),
            'injected_pairs_rank':injected_pairs,
        }


def build_relation_loader(dataset, indices, rank, world, a):
    subset = Subset(dataset, indices)
    sampler = RheaPairBatchSampler(dataset, indices, a.batch_size, rank, world,
                                   a.pairs_per_batch, a.pair_sampler_seed)
    opt = dict(batch_sampler=sampler, num_workers=a.num_workers,
               pin_memory=torch.cuda.is_available(),
               collate_fn=make_collate(
                   dataset.protein_cache.metadata if dataset.protein_cache else None,
                   dataset.mol_cache.metadata if dataset.mol_cache else None))
    if a.num_workers:
        opt.update(persistent_workers=True, prefetch_factor=a.prefetch_factor)
    return DataLoader(subset, **opt), sampler


def build_loader(dataset,indices,rank,world,a,training):
    subset=Subset(dataset,indices)
    sampler=DistributedSampler(subset,num_replicas=world,rank=rank,
                               shuffle=training,drop_last=training,seed=0)
    opt=dict(batch_size=a.batch_size,sampler=sampler,num_workers=a.num_workers,
             pin_memory=torch.cuda.is_available(),drop_last=training,
             collate_fn=make_collate(
                 dataset.protein_cache.metadata if dataset.protein_cache else None,
                 dataset.mol_cache.metadata if dataset.mol_cache else None))
    if a.num_workers:opt.update(persistent_workers=True,prefetch_factor=a.prefetch_factor)
    loader=DataLoader(subset,**opt)
    return loader,sampler


def load_eval_data(path):
    path=Path(path)
    data=[]
    if path.suffix.lower()=='.json':
        obj=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(obj,list):raise ValueError('评估JSON应为字典对象列表')
        for item in obj:
            identifier=next((item[k] for k in ('primaryAccession','id','protein_id','Entry')
                             if isinstance(item.get(k),str) and item[k]),None)
            seq=next((item[k] for k in ('sequence','protein_sequence')
                      if isinstance(item.get(k),str) and item[k]),None)
            if not identifier or not seq: raise ValueError('评估JSON缺少ID或sequence')
            data.append((identifier,''.join(seq.split())))
    else:
        identifier=None;parts=[]
        for line in path.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line:continue
            if line.startswith('>'):
                if identifier is not None:data.append((identifier,''.join(parts)))
                identifier=line[1:].split()[0];parts=[]
            else:
                if identifier is None:raise ValueError('FASTA序列前缺少header')
                parts.append(line)
        if identifier is not None:data.append((identifier,''.join(parts)))
    if not data or len({r[0] for r in data})!=len(data):
        raise ValueError('评估集为空或者包含重复ID')
    return data


def clustering_metrics(labels,pred,embeddings=None):
    from sklearn.metrics import (adjusted_rand_score,normalized_mutual_info_score,
        adjusted_mutual_info_score,fowlkes_mallows_score,homogeneity_completeness_v_measure,
        silhouette_score)
    from sklearn.metrics.cluster import contingency_matrix
    lab=np.asarray(labels);pred=np.asarray(pred)
    cm=contingency_matrix(lab,pred)
    hom,comp,v=homogeneity_completeness_v_measure(lab,pred)
    out=dict(ari=float(adjusted_rand_score(lab,pred)),nmi=float(normalized_mutual_info_score(lab,pred)),
       ami=float(adjusted_mutual_info_score(lab,pred)),fmi=float(fowlkes_mallows_score(lab,pred)),
       purity=float(cm.max(0).sum()/cm.sum()),homogeneity=float(hom),
       completeness=float(comp),v_measure=float(v),n_clusters=int(len(np.unique(pred))))
    if embeddings is not None and 1<len(np.unique(pred))<len(lab):
        out['silhouette_cosine']=float(silhouette_score(embeddings,pred,metric='cosine'))
    return out


class ClusterEvaluator:
    def __init__(self,a,output):
        self.a=a;self.rows=load_eval_data(a.cluster_eval_input)
        self.output=Path(output)/'cluster_eval';self.output.mkdir(parents=True,exist_ok=True)
        if not Path(a.cluster_eval_truth_csv).is_file(): raise FileNotFoundError(a.cluster_eval_truth_csv)
        if a.cluster_eval_mode=='legacy':
            if a.cluster_eval_module_root:
                sys.path.insert(0,str(Path(a.cluster_eval_module_root).resolve()))
            module=importlib.import_module('cluster')
            self.legacy=module.run_test_clustering_analysis
        else:
            with open(a.cluster_eval_truth_csv,newline='',encoding='utf-8-sig') as f:
                rdr=csv.DictReader(f)
                if not {'id','label'}.issubset(rdr.fieldnames or []):
                    raise ValueError('kmeans 标签CSV必须有 id,label 两列')
                self.truth={r['id']:r['label'] for r in rdr}
            if any(i not in self.truth for i,_ in self.rows):
                raise ValueError('评估中某个ID缺少真实类别标签')
            if a.cluster_eval_n_clusters>len(self.rows):raise ValueError('类簇数不能超过样本数')
    @torch.no_grad()
    def evaluate(self,sequence,epoch):
        was_train=sequence.training;sequence.eval()
        try:
            ids=[];data=[]
            for start in range(0,len(self.rows),self.a.cluster_eval_batch_size):
                block=self.rows[start:start+self.a.cluster_eval_batch_size]
                seq,mask=sequence([x[1] for x in block])
                # 保持旧导出向量格式：原始 masked mean，不做 L2 normalize。
                vec=masked_mean(seq,mask).float()
                if not torch.isfinite(vec).all():raise ValueError('聚类表征存在 NaN/Inf')
                ids.extend([x[0] for x in block])
                data.append(vec.detach().cpu().numpy())
            x=np.concatenate(data,axis=0).astype(np.float32)
        finally:
            sequence.train(was_train)
        import h5py
        path=self.output/f'epoch_{epoch:03d}.h5'
        with h5py.File(path,'w') as f:
            n=max(len(s.encode('utf8')) for s in ids)
            f.create_dataset('ids',data=np.array([s.encode('utf8') for s in ids],dtype=f'S{n}'))
            f.create_dataset('embeddings',data=x,compression='gzip',compression_opts=1)
        if self.a.cluster_eval_mode=='legacy':
            result_dir=self.output/'legacy_results';result_dir.mkdir(exist_ok=True)
            raw=self.legacy(h5_path=str(path),gt_csv_path=self.a.cluster_eval_truth_csv,
                             save_path=str(result_dir),epoch=epoch)
            if not isinstance(raw,dict):raise ValueError('原 cluster.py 必须返回指标字典')
            names={'ARI':'ari','NMI':'nmi','AMI':'ami','FMI':'fmi','Purity':'purity',
                   'Homogeneity':'homogeneity','Completeness':'completeness','V_measure':'v_measure'}
            scores=dict(raw)
            for old,new in names.items():
                if old in scores and new not in scores:scores[new]=scores[old]
            if not isinstance(scores.get('ari'),(float,int)):
                raise ValueError('cluster.py 未返回数值 ari/ARI，请核对旧函数返回值')
        else:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize
            xx=normalize(x)
            assign=KMeans(n_clusters=self.a.cluster_eval_n_clusters,n_init=20,
                          random_state=3408).fit_predict(xx)
            scores=clustering_metrics([self.truth[i] for i in ids],assign,embeddings=xx)
        scores.update(dict(epoch=epoch,n_eval=len(ids),mode=self.a.cluster_eval_mode,
                           embedding_h5=str(path)))
        atomic_json(self.output/f'metrics_epoch_{epoch:03d}.json',scores)
        return scores


@torch.no_grad()
def export_query_attention(model, loader, output_dir, epoch, device, top_k=20):
    """Export per-protein/query top positions only; no loss, backward, or optimizer update."""
    if model.fusion is None:
        return None
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    was_training=model.training; model.eval()
    jsonl_path=output_dir/f'query_attention_epoch_{epoch:03d}.jsonl'
    csv_path=output_dir/f'query_attention_epoch_{epoch:03d}_summary.csv'
    rows=[]
    try:
        for batch in loader:
            protein=batch.get('protein_tokens') or batch['sequences']
            seq,seq_mask=model.sequence(protein)
            structure,struct_mask=model.structure.encoder(
                batch['nodes'].to(device,non_blocking=True), batch['node_mask'].to(device,non_blocking=True))
            _,_,seq_attn,struct_attn=model.fusion(seq,seq_mask,structure,struct_mask,return_attention=True)
            for bi, accession in enumerate(batch['accessions']):
                sequence=''.join(str(batch['sequences'][bi]).split())
                seq_valid=torch.where(seq_mask[bi].bool())[0]
                struct_valid=torch.where(struct_mask[bi].bool())[0]
                map_ok=(len(seq_valid)==len(sequence))
                for qi in range(seq_attn.shape[1]):
                    sw=seq_attn[bi,qi].float()[seq_valid]
                    k=min(int(top_k),int(sw.numel()))
                    sv,si=torch.topk(sw,k=k,sorted=True) if k else (sw,torch.empty(0,dtype=torch.long))
                    token_positions=seq_valid[si].detach().cpu().tolist() if k else []
                    residue_positions=(si+1).detach().cpu().tolist() if k and map_ok else [None]*k
                    seq_items=[]
                    for pos,tok,w in zip(residue_positions,token_positions,sv.detach().cpu().tolist()):
                        aa=sequence[pos-1] if pos is not None and pos<=len(sequence) else None
                        seq_items.append({'residue_position_1based':pos,'token_position_0based':int(tok),
                                          'amino_acid':aa,'weight':float(w)})
                    gw=struct_attn[bi,qi].float()[struct_valid]
                    kg=min(int(top_k),int(gw.numel()))
                    gv,gi=torch.topk(gw,k=kg,sorted=True) if kg else (gw,torch.empty(0,dtype=torch.long))
                    node_positions=struct_valid[gi].detach().cpu().tolist() if kg else []
                    struct_items=[{'gearnet_node_index_1based':int(pos)+1,'weight':float(w)}
                                  for pos,w in zip(node_positions,gv.detach().cpu().tolist())]
                    rows.append({'epoch':int(epoch),'accession':str(accession),'query':int(qi),
                                 'sequence_position_mapping_valid':bool(map_ok),
                                 'sequence_top_positions':seq_items,'structure_top_positions':struct_items})
    finally:
        model.train(was_training)
    with jsonl_path.open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        fields=['epoch','accession','query','sequence_position_mapping_valid','sequence_top_positions','structure_top_positions']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:(json.dumps(row[k],ensure_ascii=False) if isinstance(row[k],(list,dict)) else row[k]) for k in fields})
    return str(jsonl_path)


METRIC_NAMES=['total','global','local','struct','export','deploy','fusion','diversity','slot_overlap','diversity_effective','diversity_weight','diversity_floor_active',
 'b_positive_pairs','b_valid_background_pairs','b_ignored_known_alternatives',
 'b_sequence_relation','b_sequence_positive_pairs','b_sequence_valid_anchors']
HISTORY_FIELDS=['epoch','variant','subset_size','train_rows','val_rows',
  'train_total','train_global','train_local','train_struct','train_export','train_deploy',
  'train_fusion','train_diversity','train_slot_overlap','train_diversity_effective','train_diversity_weight','train_diversity_floor_active','train_b_positive_pairs','train_b_valid_background_pairs',
  'train_b_ignored_known_alternatives','train_b_sequence_relation','train_b_sequence_positive_pairs','train_b_sequence_valid_anchors','val_total','val_global','val_local','val_struct',
  'val_export','val_deploy','val_fusion','val_diversity','val_diversity_effective','val_diversity_weight','val_diversity_floor_active','val_b_positive_pairs','val_b_valid_background_pairs',
  'val_b_ignored_known_alternatives','val_b_sequence_relation','val_b_sequence_positive_pairs','val_b_sequence_valid_anchors',
  'diversity_weight','diversity_mode','diversity_floor','diversity_floor_penalty','b_sequence_weight','attention_export_seconds','ari','nmi','ami','fmi','purity','homogeneity',
  'completeness','v_measure','silhouette_cosine','eval_status','train_seconds','val_seconds',
  'checkpoint_seconds','cluster_eval_seconds','epoch_seconds','temperature']


def append_history(save_dir,row):
    'Append metrics; migrate older CSV headers to include newly added timing columns.'
    path=Path(save_dir)/'clustering_eval_history.csv'
    if path.exists():
        with path.open('r',newline='',encoding='utf-8') as f:
            reader=csv.DictReader(f)
            old_fields=list(reader.fieldnames or [])
            old_rows=list(reader)
        fields=old_fields+[name for name in HISTORY_FIELDS if name not in old_fields]
        if fields != old_fields:
            tmp=path.with_suffix('.csv.tmp')
            with tmp.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
                writer.writeheader();writer.writerows(old_rows)
            os.replace(tmp,path)
    else:
        fields=HISTORY_FIELDS
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        if path.stat().st_size == 0:w.writeheader()
        w.writerow(row)


def top_ari_epochs(save_dir,top_k=3):
    'Return top ARI epochs currently recorded in clustering_eval_history.csv.'
    path=Path(save_dir)/'clustering_eval_history.csv'
    if not path.is_file():return []
    ranked=[]
    with path.open('r',newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                ari=float(row.get('ari',''))
                epoch=int(row.get('epoch',''))
            except (TypeError,ValueError):
                continue
            if not math.isfinite(ari):continue
            if row.get('eval_status') not in (None,'','ok'):continue
            ranked.append({'epoch':epoch,'ari':ari})
    ranked.sort(key=lambda x:(-x['ari'],x['epoch']))
    return ranked[:top_k]


def format_top_ari_message(save_dir,top_k=3):
    ranked=top_ari_epochs(save_dir,top_k)
    if not ranked:return 'Current ARI Top-3 epochs: no valid clustering ARI has been recorded yet.'
    parts=[f"#{i+1} epoch={item['epoch']} ARI={item['ari']:.6f}"
           for i,item in enumerate(ranked)]
    return 'Current ARI Top-3 epochs (all completed evaluations so far): ' + ' | '.join(parts)


def reduce_metrics(sums,count,device,world):
    values=torch.stack([sums[k] for k in METRIC_NAMES]+[torch.tensor(float(count),device=device)])
    if world>1:dist.all_reduce(values)
    count=float(values[-1].item())
    return {k:float(values[i].item())/max(1,count) for i,k in enumerate(METRIC_NAMES)}


def train_epoch(model,loader,sampler,optimizer,epoch,a,rank,world,device,start_step=0):
    model.train();sampler.set_epoch(epoch)
    sums={k:torch.zeros((),device=device) for k in METRIC_NAMES}
    bcount=0;stop=False
    if a.precision=='bf16': cast=torch.autocast(device_type='cuda',dtype=torch.bfloat16)
    else: cast=contextlib.nullcontext()
    for batch in loader:
        with cast:
            result=model(batch)
            loss=compute_losses(result,model.module.temperature if hasattr(model,'module') else model.temperature,a,batch,epoch=epoch+1)
        if not torch.isfinite(loss['total']):raise FloatingPointError('训练损失存在 NaN/Inf')
        optimizer.zero_grad(set_to_none=True)
        loss['total'].backward()
        optimizer.step()
        with torch.no_grad():
            for k in METRIC_NAMES:sums[k]+=loss[k].detach()
        bcount+=1
        if rank==0 and a.log_every and bcount%a.log_every==0:
            emit_log(f'epoch {epoch+1} step {bcount}/{len(loader)} train_total={loss["total"].item():.4f}')
        if a.max_steps and bcount>=a.max_steps:
            stop=(bcount<len(loader));break
    return reduce_metrics(sums,bcount,device,world),stop


@torch.no_grad()
def val_epoch(model,loader,a,device,world,epoch):
    model.eval();sums={k:torch.zeros((),device=device) for k in METRIC_NAMES};steps=0
    for batch in loader:
        if len(batch['sequences'])<2: continue # 单样本 InfoNCE 恒为0；跳过
        cast=(torch.autocast('cuda',dtype=torch.bfloat16)
               if a.precision=='bf16' else contextlib.nullcontext())
        with cast:
            out=model(batch);ls=compute_losses(out,model.module.temperature if hasattr(model,'module') else model.temperature,a,batch,epoch=epoch)
        for k in METRIC_NAMES:sums[k]+=ls[k].detach()
        steps+=1
    return reduce_metrics(sums,steps,device,world)


def model_param_groups(model,a):
    groups=[]
    for module,lr in ((model.sequence,a.base_lr),(model.structure,a.base_lr),
                      (model.reaction,a.reaction_lr),(model.head,a.base_lr)):
        params=[p for p in module.parameters() if p.requires_grad]
        if params:groups.append(dict(params=params,lr=lr))
    if model.fusion is not None:groups.append(dict(params=list(model.fusion.parameters()),lr=a.base_lr))
    groups.append(dict(params=[model.temperature],lr=a.temperature_lr))
    return groups


def base_rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(data):
    random.setstate(data['python']);np.random.set_state(data['numpy']);torch.set_rng_state(data['torch'])
    if data['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state(data['cuda'])


def save_state(model,optimizer,epoch,a,fp,subset_hash,rank,world,step,partial):
    local=base_rng_state()
    all_rng=[None]*world
    if world>1:dist.all_gather_object(all_rng,local)
    else:all_rng=[local]
    if rank==0:
        state=dict(format='ramer-standalone-A-v1',completed_epochs=epoch-(1 if partial else 0),global_step=step,
                   partial_epoch=partial,config=vars(a),dataset_fingerprint=fp,
                   subset_sha256=subset_hash,world_size=world,
                   model_delta=model_delta(model),optimizer=optimizer.state_dict(),rng_by_rank=all_rng)
        atomic_torch(Path(a.save_dir)/('last_partial.pt' if partial else 'last.pt'),state)
        atomic_torch(Path(a.save_dir)/f'sequence_epoch_{epoch:03d}.pt',dict(
            format='ramer-sequence-standalone-A-v1',epoch=epoch,
            state_dict=sequence_state(model),embed_dim=a.embed_dim,
            model_path=a.protein_model_path,
            representation='raw masked mean of projected ProtT5 residue features'))
        # 直接兼容旧 BioData2Seq.load_state_dict(..., strict=False) 的原始权重字典。
        # 加载时仍需实例化相同 ProtT5 基座和相同 LoRA 结构；不含底座的 delta 不是独立模型。
        atomic_torch(Path(a.save_dir)/f'seq_only_epoch_{epoch:03d}.pth',
                     sequence_state(model,full=False))
        if a.save_full_sequence:
            atomic_torch(Path(a.save_dir)/f'protein_encoder_epoch_{epoch:03d}.pth',
                         sequence_state(model,full=True))
    if world>1:dist.barrier()


def main(a):
    rank,world,device=setup_dist()
    seed_all(a.seed)
    if a.precision=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('BF16 仅支持对应CUDA设备')
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=bool(a.tf32)
        torch.backends.cudnn.allow_tf32=bool(a.tf32)
    output=Path(a.save_dir)
    preflight={'ok':True,'error':''}
    if rank==0:
        output.mkdir(parents=True,exist_ok=True)
        if not a.resume and (output/'metrics.jsonl').exists():
            preflight={'ok':False,'error':'输出目录已经存在训练日志；使用新的 --save-dir 或指定 --resume'}
    if world>1:
        buf=[preflight];dist.broadcast_object_list(buf,src=0);preflight=buf[0]
    if not preflight['ok']:raise RuntimeError(preflight['error'])
    if rank==0:
        setup_file_logger(output/'train.log')
        log_to_file('='*100)
        log_to_file(f"RUN START | variant={a.variant} | save_dir={output.resolve()} | world_size={world} | "
                    f"epochs={a.epochs} | batch_size_per_gpu={a.batch_size} | precision={a.precision}")
        log_to_file(f"diversity_mode={a.diversity_mode} | base={a.diversity_weight} | mid={a.diversity_weight_mid} | late={a.diversity_weight_late} | floor={a.diversity_floor} | floor_penalty={a.diversity_floor_penalty} | attention_every={a.attention_export_every} | attention_samples={a.attention_export_samples}")
        log_to_file(f"cluster_eval={a.cluster_eval} | mode={a.cluster_eval_mode} | input={a.cluster_eval_input} | "
                    f"truth_csv={a.cluster_eval_truth_csv} | every={a.cluster_eval_every}")
        if a.cluster_eval_mode=='legacy':
            log_to_file('ARI note: legacy cluster.py returns its best result across configured EC levels; '
                        'the per-epoch ARI Top-3 ranks this returned scalar. Treat it as a monitoring/development metric, not an untouched final-test estimate.')
    records,fp,manifest=load_training_records(a)
    indices=get_subset_indices(len(records),fp,a,rank,world)
    train_indices,val_indices=divide_subset(indices,a)
    subset_hash=hashlib.sha256(json.dumps(indices).encode()).hexdigest()
    dataset=EnzymeDataset(records,a.token_cache_dir,manifest,a.node_cache_dir)
    # Build known reaction sets only from rows in this run's selected training subset.
    # This prevents B from using annotations outside the fixed 20k sample (or held-out val rows).
    known_by_sequence={}
    for idx in train_indices:
        rec=records[idx]
        known_by_sequence.setdefault(rec['sequence'],set()).add(str(rec['RHEA']))
    dataset.known_reactions_by_sequence={k:sorted(v) for k,v in known_by_sequence.items()}

    # 强制从原编码器脚本构造模型，不加载之前拆分版本的 ramer 包。
    original=str(Path(a.encoder_dir).resolve())
    sys.path.insert(0,original)
    from proteinT5_embedding_v2 import ProteinSequenceEmbedding
    from structure_embedding_v2 import StructureResidueEmbedding
    import reaction_molt5_position_embedding_v2 as reaction_module
    core=RAMERModel(a,ProteinSequenceEmbedding,StructureResidueEmbedding,reaction_module).to(device)
    if dataset.protein_cache is not None:
        if tokenizer_signature(core.sequence.encoder.tokenizer,'protein')!=manifest['protein']['tokenizer_signature']:
            raise ValueError('ProtT5 分词器与缓存的token签名不一致')
    if dataset.mol_cache is not None:
        if tokenizer_signature(core.reaction.encoder.molecule_embedder.tokenizer,'molecule')!=manifest['molecule']['tokenizer_signature']:
            raise ValueError('MolT5 分词器与缓存的token签名不一致')
    opt_kwargs=dict(fused=True) if a.fused_adamw else {}
    optimizer=torch.optim.AdamW(model_param_groups(core,a),lr=a.base_lr,**opt_kwargs)
    initial_epoch=0;global_step=0
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        if state.get('format')!='ramer-standalone-A-v1' or state.get('partial_epoch'):
            raise ValueError('仅支持本脚本保存的完整 epoch checkpoint 续训')
        if state['world_size']!=world or state['dataset_fingerprint']!=fp or state['subset_sha256']!=subset_hash:
            raise ValueError('断点续训 GPU 数或数据子集不同')
        mutable={'resume','epochs','save_dir','log_every','save_every','max_steps',
            'num_workers','prefetch_factor','cluster_eval','cluster_eval_every',
            'cluster_eval_input','cluster_eval_truth_csv','cluster_eval_mode',
            'cluster_eval_module_root','cluster_eval_n_clusters','cluster_eval_batch_size',
            'cluster_eval_role','select_best_ari','save_full_sequence'}
        for name,old_value in state['config'].items():
            if name not in mutable and old_value!=getattr(a,name,None):
                raise ValueError(f'断点续训训练设置不能修改: {name}')
        load_delta(core,state['model_delta'])
        optimizer.load_state_dict(state['optimizer'])
        initial_epoch=state['completed_epochs'];global_step=state['global_step']
    if a.relation_pair_sampler:
        loader,sampler=build_relation_loader(dataset,train_indices,rank,world,a)
        if rank==0:
            emit_log(f"RHEA-PAIRED-BATCH enabled: pairs_per_batch={a.pairs_per_batch}; "
                     f"eligible_groups={len(sampler.groups)}/{sampler.total_groups}; "
                     f"steps_per_epoch={len(loader)}; different_sequences_only=True")
    else:
        loader,sampler=build_loader(dataset,train_indices,rank,world,a,True)
    attention_loader=None
    if rank==0 and a.attention_export_every:
        n=min(a.attention_export_samples,len(train_indices))
        selected=sorted(random.Random(a.seed+7919).sample(list(train_indices),n))
        attention_loader,_=build_loader(dataset,selected,0,1,a,False)
    vloader=None
    if a.compute_val_loss:
        vloader,_=build_loader(dataset,val_indices,rank,world,a,False)
    evaluator=None;preflight={'ok':True}
    if rank==0 and a.cluster_eval:
        try:evaluator=ClusterEvaluator(a,output)
        except Exception as exc:preflight={'ok':False,'error':repr(exc)}
    if world>1:
        tmp=[preflight];dist.broadcast_object_list(tmp,src=0);preflight=tmp[0]
    if not preflight['ok']:raise RuntimeError('评估初始化失败：'+preflight['error'])
    model=(nn.parallel.DistributedDataParallel(core,device_ids=[device.index] if device.type=='cuda' else None,
           gradient_as_bucket_view=True,find_unused_parameters=False) if world>1 else core)
    if a.resume:restore_rng(state['rng_by_rank'][rank])
    best_ari=float('-inf')
    if a.resume and (output/'best_ari.json').exists():
        best_ari=float(json.loads((output/'best_ari.json').read_text(encoding='utf-8'))['ari'])
    if rank==0:
        atomic_json(output/'config.json',vars(a))
        atomic_json(output/'subset_summary.json',dict(dataset_fingerprint=fp,
              subset_file=a.subset_file,subset_hash=subset_hash,subset_total=len(indices),
              train_rows=len(train_indices),val_rows=len(val_indices),seed=a.seed))
        emit_log(json.dumps(dict(rank=rank,world=world,variant=a.variant,subsample=len(indices),
              train_rows=len(train_indices),val_rows=len(val_indices),precision=a.precision,
              token_cache=bool(a.token_cache_dir),log_file=str((output/'train.log').resolve())),ensure_ascii=False))
    run_start=time.perf_counter()
    for epoch in range(initial_epoch,a.epochs):
        epoch_start=time.perf_counter()
        if rank==0:
            emit_log(f"Epoch {epoch+1}/{a.epochs} START")

        train_start=time.perf_counter()
        train_log,partial=train_epoch(model,loader,sampler,optimizer,epoch,a,rank,world,device)
        if rank==0 and a.relation_pair_sampler and sampler.last_epoch_stats:
            emit_log('RHEA_PAIR_SAMPLER ' + json.dumps(sampler.last_epoch_stats,ensure_ascii=False))
        if device.type=='cuda':torch.cuda.synchronize()
        train_seconds=time.perf_counter()-train_start
        global_step+= min(a.max_steps if a.max_steps else len(loader),len(loader))

        val_seconds=0.0
        val_log=None
        if vloader is not None and not partial:
            val_start=time.perf_counter()
            val_log=val_epoch(model,vloader,a,device,world,epoch+1)
            if device.type=='cuda':torch.cuda.synchronize()
            val_seconds=time.perf_counter()-val_start

        checkpoint_start=time.perf_counter()
        # 先保存每轮 checkpoint，再进行聚类评估：即使私有 cluster.py 报错也不会损失本轮模型。
        save_state(core,optimizer,epoch+1,a,fp,subset_hash,rank,world,global_step,partial)
        if device.type=='cuda':torch.cuda.synchronize()
        checkpoint_seconds=time.perf_counter()-checkpoint_start

        cluster_log=None;error=None;cluster_eval_seconds=0.0
        if a.cluster_eval and (epoch+1)%a.cluster_eval_every==0 and not partial and rank==0:
            cluster_start=time.perf_counter()
            try:
                cluster_log=evaluator.evaluate(core.sequence,epoch+1)
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
            cluster_eval_seconds=time.perf_counter()-cluster_start
        attention_export_seconds=0.0
        attention_path=None
        if (rank==0 and attention_loader is not None and not partial
                and (epoch+1)%a.attention_export_every==0):
            attn_start=time.perf_counter()
            try:
                attention_path=export_query_attention(
                    core,attention_loader,output/'query_attention',epoch+1,device,a.attention_top_k)
                emit_log(f'QUERY ATTENTION EXPORTED | epoch={epoch+1} | file={attention_path}')
            except Exception as exc:
                emit_log(f'WARNING: query attention export failed at epoch {epoch+1}: {type(exc).__name__}: {exc}')
            if device.type=='cuda': torch.cuda.synchronize()
            attention_export_seconds=time.perf_counter()-attn_start
        # DDP sync: rank0 performs clustering and optional attention export once.
        result={'error':error,'cluster':cluster_log,'cluster_eval_seconds':cluster_eval_seconds,
                'attention_export_seconds':attention_export_seconds,'attention_path':attention_path}
        if world>1:
            one=[result];dist.broadcast_object_list(one,src=0);result=one[0]
        if result['error']:
            if rank==0:
                emit_log('ERROR: checkpoint 已保存，但本轮聚类评估失败：'+result['error'])
            raise RuntimeError('每轮 checkpoint 已保存，但聚类评估失败：'+result['error'])
        cluster_log=result['cluster']
        cluster_eval_seconds=float(result.get('cluster_eval_seconds',0.0))
        attention_export_seconds=float(result.get('attention_export_seconds',0.0))

        if rank==0:
            epoch_seconds=time.perf_counter()-epoch_start
            row=dict(epoch=epoch+1,variant=a.variant,subset_size=len(indices),
                     train_rows=len(train_indices),val_rows=len(val_indices),
                     train_seconds=train_seconds,val_seconds=val_seconds,
                     checkpoint_seconds=checkpoint_seconds,cluster_eval_seconds=cluster_eval_seconds,
                     epoch_seconds=epoch_seconds,attention_export_seconds=attention_export_seconds,
                     temperature=core.temperature.detach().item(),diversity_weight=train_log.get('diversity_weight',a.diversity_weight),
                     diversity_mode=a.diversity_mode,diversity_floor=a.diversity_floor,diversity_floor_penalty=a.diversity_floor_penalty,
                     eval_status='ok' if cluster_log else 'skipped')
            row.update({f'train_{k}':v for k,v in train_log.items()})
            if val_log is not None:row.update({f'val_{k}':v for k,v in val_log.items()})
            if cluster_log:
                row.update({k:cluster_log.get(k) for k in ('ari','nmi','ami','fmi','purity',
                 'homogeneity','completeness','v_measure','silhouette_cosine')})
            append_history(output,row)
            if (a.select_best_ari and cluster_log and
                    isinstance(cluster_log.get('ari'),(int,float,np.number)) and
                    math.isfinite(float(cluster_log['ari'])) and
                    float(cluster_log['ari'])>best_ari):
                best_ari=float(cluster_log['ari'])
                atomic_torch(output/'best_ari_seq_only.pth',sequence_state(core))
                atomic_json(output/'best_ari.json',dict(epoch=epoch+1,ari=best_ari,
                    eval_set=a.cluster_eval_input,role=a.cluster_eval_role))
            with (output/'metrics.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(epoch=epoch+1,train=train_log,val=val_log,
                                  clustering=cluster_log,partial=partial,
                                  timing=dict(train_seconds=train_seconds,val_seconds=val_seconds,
                                      checkpoint_seconds=checkpoint_seconds,
                                      cluster_eval_seconds=cluster_eval_seconds,attention_export_seconds=attention_export_seconds,
                                      epoch_seconds=epoch_seconds)),ensure_ascii=False,default=json_safe)+'\n')
            emit_log('EPOCH SUMMARY ' + json.dumps(row,ensure_ascii=False,default=json_safe))
            if cluster_log and isinstance(cluster_log.get('ari'),(int,float,np.number)):
                emit_log(f"Epoch {epoch+1} clustering: ARI={float(cluster_log['ari']):.6f}, "
                         f"NMI={float(cluster_log.get('nmi',float('nan'))):.6f}, "
                         f"AMI={float(cluster_log.get('ami',float('nan'))):.6f}, "
                         f"FMI={float(cluster_log.get('fmi',float('nan'))):.6f}, "
                         f"Purity={float(cluster_log.get('purity',float('nan'))):.6f}")
            emit_log(format_top_ari_message(output,top_k=3))
            emit_log(f"Epoch {epoch+1} TIME | train={train_seconds:.2f}s | val={val_seconds:.2f}s | "
                     f"checkpoint={checkpoint_seconds:.2f}s | cluster_eval={cluster_eval_seconds:.2f}s | "
                     f"epoch_total={epoch_seconds:.2f}s ({epoch_seconds/60:.2f} min)")
        if partial:break
    if rank==0:
        elapsed=time.perf_counter()-run_start
        emit_log(f"RUN FINISHED | elapsed={elapsed:.2f}s ({elapsed/3600:.2f}h) | output={output.resolve()}")
    if world>1:dist.destroy_process_group()


if __name__=='__main__':
    main(parse_args())