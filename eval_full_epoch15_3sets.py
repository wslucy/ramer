#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用旧 cluster.py 的 run_test_clustering_analysis 评估全量 FLOOR epoch15。

只需序列增量 checkpoint，不需要加载训练中的反应编码器或 GearNet。
严格复用旧聚类实现（EC1-4，方法/K 扫描及取最高 ARI 的 legacy 协议）。
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--encoder-dir', default='/datapool/home/tanlijun/RAMER/dec-v2')
    p.add_argument('--cluster-root', default='/datapool/home/tanlijun/RAMER')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--epoch', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=4)
    for tag in ('price','new392','ram255'):
        p.add_argument(f'--{tag}-json', required=True)
        p.add_argument(f'--{tag}-truth', required=True)
    return p.parse_args()


def load_records(path):
    # 保留旧 JSON 的 id/protein_sequence 优先级；对不同数据集允许等价键名。
    path = Path(path)
    records = []
    if path.suffix.lower() == '.json':
        rows = json.loads(path.read_text(encoding='utf8'))
        if not isinstance(rows, list):
            raise ValueError(f'{path}: JSON 顶层必须是列表')
        for i, row in enumerate(rows):
            rid = next((row[k] for k in ('id','primaryAccession','protein_id','Entry')
                        if isinstance(row.get(k),str) and row[k]),None)
            seq = next((row[k] for k in ('protein_sequence','sequence')
                        if isinstance(row.get(k),str) and row[k]),None)
            if not rid or not seq:
                raise ValueError(f'{path}: row {i} 缺少 id/序列')
            records.append((rid,''.join(seq.split())))
    else:
        # 可选 FASTA，不依赖 BioPython。
        rid = None
        seqparts = []
        for s in path.read_text(encoding='utf8').splitlines():
            s = s.strip()
            if not s: continue
            if s.startswith('>'):
                if rid is not None:
                    records.append((rid,''.join(seqparts)))
                rid = s[1:].split()[0]
                seqparts = []
            else:
                if rid is None:
                    raise ValueError('FASTA 第一条记录必须以 > 开始')
                seqparts.append(s)
        if rid is not None:
            records.append((rid,''.join(seqparts)))
    if not records or len(set(r for r,_ in records)) != len(records):
        raise ValueError(f'{path}: 无数据或 ID 重复')
    return records


def validate_truth(path, ids):
    # 原 cluster.py: pd.read_csv(gt_csv_path, sep="\\t")，字段 Entry/EC number。
    import pandas as pd
    df = pd.read_csv(path, sep='\t')
    if not {'Entry','EC number'}.issubset(df.columns):
        raise ValueError(f'{path} 不符合旧 cluster.py 的 TSV 格式（需要 Entry、EC number 列）。'
                         f' 实际列为 {list(df.columns)}；请先制作与旧协议一致的真值文件。')
    matched = set(ids) & set(df['Entry'].astype(str))
    if not matched:
        raise ValueError(f'{path} 与数据集 ID 完全不匹配')
    return len(matched), len(df)


def unpack_sequence_state(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError('checkpoint 格式不是 dict')
    if 'state_dict' in ckpt and isinstance(ckpt['state_dict'],dict):
        state = ckpt['state_dict']
    elif 'protein_encoder' in ckpt and isinstance(ckpt['protein_encoder'], dict):
        state = ckpt['protein_encoder']
    else:
        state = ckpt
    # 源于原训练脚本的 DDP 参数可能带 module. 前缀。
    state = {k.removeprefix('module.'): v for k,v in state.items()}
    if not any(k.startswith('encoder.') for k in state):
        raise ValueError('序列参数键应以 encoder. 开头，请传 seq_only_epoch_015.pth')
    return state


def validate_loaded_state(loaded, info):
    # 仅允许冻结的底座参数缺失，LoRA 和线性投影必须完整载入。
    missing_required = [k for k in info.missing_keys if
                        not (k.startswith('encoder.model.') and '.lora_' not in k)]
    unexpected = list(info.unexpected_keys)
    if missing_required or unexpected:
        raise RuntimeError(f'checkpoint 与当前 BioData2Seq 不匹配！missing={missing_required[:10]} '
                           f'unexpected={unexpected[:10]}')
    required = [k for k in loaded if k.startswith('encoder.linear.') or '.lora_' in k]
    if not required:
        raise RuntimeError('checkpoint 缺少 LoRA/线性投影增量，拒绝用随机模型评估')


@torch.inference_mode()
def export_embeddings(model, records, path, batch_size):
    ids=[]
    results=[]
    for first in range(0,len(records),batch_size):
        block=records[first:first+batch_size]
        residue,mask=model([s for _,s in block])
        mm=mask.to(residue.dtype).unsqueeze(-1)
        v=((residue*mm).sum(1)/mm.sum(1).clamp_min(1)).float()
        if not torch.isfinite(v).all():
            raise FloatingPointError(f'NaN/Inf at samples {first}:{first+len(block)}')
        ids.extend([x[0] for x in block])
        results.append(v.cpu().numpy())
    vec=np.concatenate(results,0).astype(np.float32)
    length=max(len(s.encode('utf8')) for s in ids)
    path.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(path,'w') as f:
        f.create_dataset('ids', data=np.asarray([x.encode('utf8') for x in ids],dtype=f'S{length}'))
        f.create_dataset('embeddings', data=vec)
    return len(ids)


def main():
    a=parse_args()
    for p in [a.checkpoint,a.price_json,a.price_truth,a.new392_json,a.new392_truth,
              a.ram255_json,a.ram255_truth]:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    paths=[('Price-149',a.price_json,a.price_truth),
           ('NEW-392',a.new392_json,a.new392_truth),
           ('RAM255',a.ram255_json,a.ram255_truth)]
    datasets=[]
    for label,j,t in paths:
        rec=load_records(j)
        match,truth_count=validate_truth(t,[x[0] for x in rec])
        print(f'[DATA] {label}: sequences={len(rec)}, truth_rows={truth_count}, matched_ids={match}',flush=True)
        datasets.append((label,rec,t))
    sys.path.insert(0,a.encoder_dir)
    from Data2seq_v2 import BioData2Seq
    sys.path.insert(0,a.cluster_root)
    from cluster import run_test_clustering_analysis
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=BioData2Seq('protein-sequence',embed_dim=1024).to(device)
    state=unpack_sequence_state(a.checkpoint)
    info=model.load_state_dict(state,strict=False)
    validate_loaded_state(state,info)
    model.eval()
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    summary=[]
    for label,records,truth in datasets:
        out=root/label
        out.mkdir(parents=True,exist_ok=True)
        h5=out/'epoch_015_embeddings.h5'
        n=export_embeddings(model,records,h5,a.batch_size)
        # 直接调用已有 cluster.py 的原函数；不更改 K、缩放、随机种子等任何聚类规则。
        metrics=run_test_clustering_analysis(h5_path=str(h5),gt_csv_path=str(truth),
                                              save_path=str(out/'legacy_results'),epoch=a.epoch)
        if not isinstance(metrics,dict):
            raise RuntimeError(f'{label} cluster.py 没有返回结果')
        row={'dataset':label,'epoch':a.epoch,'num_sequences':n,'checkpoint':a.checkpoint,
             'checkpoint_sha256':sha,**metrics}
        (out/'summary.json').write_text(json.dumps(row,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf8')
        print('[RESULT]',json.dumps(row,ensure_ascii=False,default=str),flush=True)
        summary.append(row)
    names=list(dict.fromkeys(k for row in summary for k in row))
    with (root/'three_dataset_summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,names);w.writeheader();w.writerows(summary)
    print('[DONE]',root/'three_dataset_summary.csv',flush=True)


if __name__=='__main__':
    main()