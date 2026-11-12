#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from Bio import SeqIO
import ast
import csv
import pickle
import warnings
warnings.filterwarnings("ignore", message="unknown class.*will be ignored", category=UserWarning)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score

# ==================== 添加Data2seq模块路径 ====================
sys.path.append("./zero_shot/Data2seq")
from Data2seq import BioData2Seq
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==================== 全局路径配置 ====================
DATA_DIR = "./zero_shot/data"
OUTPUT_DIR = "./zero_shot/output"
MODEL_INIT_PATH = "./zero_shot/model/RAM-protT5.pth"
MODEL_DEC_PATH = "/root/RAMMER/train_model_exp_homo0.5_struct0.1_react0.3_dec0.05_reg-0.01_seed3048/seq_tokenizer_esm_best.pth"
MODEL_INIT10K_PATH = "/root/RAMMER/ckpts/train_model_init/seq_tokenizer_esm_best.pth" # 10k数据的ramer
MODEL_INIT65K_PATH = "/root/RAMMER/ckpts/train_model_init_65k/seq_tokenizer_esm_best.pth" # 65k数据的ramer
MODEL_DEC1_PATH = "/root/RAMMER/train_model_exp_homo0.4_struct0.1_react0.35_dec0.08_reg0.005/seq_tokenizer_esm_best.pth"
MODEL_Dis1_PATH = "/root/RAMMER/zero_shot/model/seq_tokenizer_esm_best.pth"  # dis0.5模型

# 数据集列表（测试集）
DATASETS = [
    {"name": "Price-149", "json": f"{DATA_DIR}/Price-149.json", "truth_csv": f"{DATA_DIR}/price.csv"},
    {"name": "NEW-392",   "json": f"{DATA_DIR}/NEW-392.json",   "truth_csv": f"{DATA_DIR}/new.csv"}
]

# 背景数据库
BACKGROUND_JSON = f"{DATA_DIR}/clean100_set.json"

# 模型标识
MODELS = [
    # {"name": "init", "path": MODEL_INIT_PATH},
    # {"name": "dec",  "path": MODEL_DEC_PATH},
    # {"name": "init10k",  "path": MODEL_INIT10K_PATH}, # 10k数据的ramer
    # {"name": "init65k",  "path": MODEL_INIT65K_PATH},  # 65k数据的ramer
    # {"name": "dec1",  "path": MODEL_DEC1_PATH},  # dec1模型
    {"name": "dis1",  "path": MODEL_Dis1_PATH}  # dis1模型
]

# ==================== Step 1: 生成嵌入HDF5 ====================
def load_json_sequences(json_file):
    ids, sequences = [], []
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        ids.append(item["id"])
        sequences.append(item["protein_sequence"])
    return ids, sequences

def load_fasta_sequences(fasta_file):
    ids, sequences = [], []
    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))
    return ids, sequences

def initialize_hdf5(hdf5_path, embed_dim):
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('ids', (0,), maxshape=(None,), dtype='S100')
        f.create_dataset('embeddings', (0, embed_dim), maxshape=(None, embed_dim), dtype=np.float32)

def append_to_hdf5(hdf5_path, new_ids, new_embeddings, embed_dim):
    with h5py.File(hdf5_path, 'a') as f:
        old_size = f['embeddings'].shape[0]
        new_size = old_size + len(new_embeddings)
        f['ids'].resize((new_size,))
        f['embeddings'].resize((new_size, embed_dim))
        f['ids'][old_size:new_size] = np.array(new_ids, dtype='S')
        f['embeddings'][old_size:new_size] = np.array(new_embeddings, dtype=np.float32)

def generate_embeddings(input_path, output_hdf5, model_path, embed_dim=1024,
                        batch_size=1, save_interval=1000, device="cuda", input_type="json"):
    """生成蛋白质序列的嵌入并保存为HDF5（若文件已存在则跳过）"""
    if os.path.exists(output_hdf5):
        print(f"文件已存在，跳过生成: {output_hdf5}")
        return output_hdf5

    # 加载序列
    if input_type == 'json':
        ids, sequences = load_json_sequences(input_path)
    elif input_type == 'fasta':
        ids, sequences = load_fasta_sequences(input_path)
    else:
        raise ValueError("input_type 必须是 'json' 或 'fasta'")
    print(f"加载 {len(sequences)} 条序列: {input_path}")

    # 加载模型
    device = torch.device(device)
    model = BioData2Seq(modality="protein-sequence", embed_dim=embed_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 初始化HDF5
    initialize_hdf5(output_hdf5, embed_dim)

    all_embeddings = []
    batch_ids = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"生成嵌入 {output_hdf5}"):
            batch_seqs = sequences[i:i+batch_size]
            batch_ids.extend(ids[i:i+batch_size])
            protein_embeddings, protein_mask = model(batch_seqs)
            protein_avg = torch.sum(protein_embeddings * protein_mask.unsqueeze(-1), dim=1) / \
                          torch.sum(protein_mask, dim=1, keepdim=True).clamp(min=1)
            all_embeddings.extend(protein_avg.cpu().numpy())
            if (i // batch_size + 1) % save_interval == 0:
                append_to_hdf5(output_hdf5, batch_ids, all_embeddings, embed_dim)
                all_embeddings = []
                batch_ids = []
    if all_embeddings:
        append_to_hdf5(output_hdf5, batch_ids, all_embeddings, embed_dim)

    print(f"完成: {output_hdf5}")
    return output_hdf5

# ==================== Step 2: 预测函数（maxsep & top1）====================
# ---------- maxsep 核心逻辑 ----------
def dynamic_k_selection(similarities, train_ecs):
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_sims = similarities[sorted_indices]
    ec_set = set()
    valid_range = 0
    for idx in sorted_indices:
        ecs = train_ecs[idx]
        ec_set.update(ecs)
        valid_range += 1
        if len(ec_set) > 10:
            break
    if valid_range <= 1:
        return sorted_indices[:1]
    diffs = sorted_sims[:valid_range-1] - sorted_sims[1:valid_range]
    max_diff_idx = np.argmax(diffs)
    dynamic_k = max_diff_idx + 1
    ec_final_set = set()
    final_k = 0
    for i in range(dynamic_k):
        ecs = train_ecs[sorted_indices[i]]
        ec_final_set.update(ecs)
        final_k += 1
        if len(ec_final_set) > 5:
            final_k -= 1
            break
    return sorted_indices[:final_k]

def run_maxsep_prediction(test_h5, train_h5, train_ec_dict_path, output_csv, batch_size=10000):
    """执行maxsep预测，生成CSV文件"""
    # 加载训练EC字典
    with open(train_ec_dict_path, 'r') as f:
        train_ec_dict = json.load(f)
    # 加载训练嵌入
    with h5py.File(train_h5, 'r') as f:
        train_ids = [tid.decode('utf-8') for tid in f['ids']]
        train_embeddings = np.array(f['embeddings'])
        train_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    train_ecs = [train_ec_dict.get(tid, ['unknown']) for tid in train_ids]

    # 加载测试元数据
    with h5py.File(test_h5, 'r') as f:
        total_test = len(f['ids'])
        test_ids = [tid.decode('utf-8') for tid in f['ids']]

    results = []
    with h5py.File(test_h5, 'r') as f:
        test_embeddings = f['embeddings']
        num_batches = int(np.ceil(total_test / batch_size))
        for batch_idx in tqdm(range(num_batches), desc=f"maxsep预测 {output_csv}"):
            start = batch_idx * batch_size
            end = min((batch_idx+1)*batch_size, total_test)
            batch_emb = test_embeddings[start:end]
            batch_ids = test_ids[start:end]
            batch_norm = batch_emb / np.linalg.norm(batch_emb, axis=1, keepdims=True)
            sim_matrix = np.dot(batch_norm, train_norm.T)
            for i, test_id in enumerate(batch_ids):
                sims = sim_matrix[i]
                top_k_idx = dynamic_k_selection(sims, train_ecs)
                seen_ec = set()
                ordered_ecs = []
                ec_prob = {}
                for idx in top_k_idx:
                    ecs = train_ecs[idx]
                    sim = sims[idx]
                    for ec in ecs:
                        if ec not in seen_ec:
                            seen_ec.add(ec)
                            ordered_ecs.append(ec)
                            ec_prob[ec] = ec_prob.get(ec, []) + [sim]
                ec_prob = {ec: np.mean(probs) for ec, probs in ec_prob.items()}
                results.append({
                    'test_id': test_id,
                    'predicted_ec': ';'.join(ordered_ecs),
                    'selected_k': len(top_k_idx),
                    'ec_probabilities': ';'.join(f"{ec}:{prob:.4f}" for ec, prob in ec_prob.items())
                })
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"maxsep结果保存至: {output_csv}")

# ---------- top1 核心逻辑 ----------
def run_top1_prediction(test_h5, train_h5, train_ec_dict_path, output_csv, top_k=1, batch_size=1000):
    """执行top1预测，生成CSV文件"""
    with open(train_ec_dict_path, 'r') as f:
        train_ec_dict = json.load(f)
    with h5py.File(train_h5, 'r') as f:
        train_ids = [tid.decode('utf-8') for tid in f['ids']]
        train_embeddings = np.array(f['embeddings'])
        train_norm = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    train_ecs = [train_ec_dict.get(tid, ['unknown']) for tid in train_ids]

    with h5py.File(test_h5, 'r') as f:
        test_ids = [tid.decode('utf-8') for tid in f['ids']]
        test_embeddings = np.array(f['embeddings'])

    results = []
    num_batches = int(np.ceil(len(test_ids) / batch_size))
    for batch_idx in tqdm(range(num_batches), desc=f"top1预测 {output_csv}"):
        start = batch_idx * batch_size
        end = min((batch_idx+1)*batch_size, len(test_ids))
        batch_emb = test_embeddings[start:end]
        batch_ids = test_ids[start:end]
        batch_norm = batch_emb / np.linalg.norm(batch_emb, axis=1, keepdims=True)
        batch_sim = np.dot(batch_norm, train_norm.T)
        top_k_indices = np.argsort(-batch_sim, axis=1)[:, :top_k]
        for i, test_id in enumerate(batch_ids):
            k_indices = top_k_indices[i]
            top_ecs = set()
            matched_ids = []
            for idx in k_indices:
                matched_ids.append(train_ids[idx])
                ecs = train_ecs[idx]
                if isinstance(ecs, list):
                    top_ecs.update(ecs)
                else:
                    top_ecs.add(ecs)
            top1_sim = batch_sim[i, k_indices[0]]
            results.append({
                'test_id': test_id,
                'predicted_ec': str(sorted(list(top_ecs))),
                'top1_similarity': top1_sim,
                'matched_train_ids': ";".join(matched_ids)
            })
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"top1结果保存至: {output_csv}")

# ==================== Step 3: 评估函数 ====================
def get_pred_labels_R_maxsep(pred_csv_path):
    """读取maxsep格式预测标签"""
    pred_label = []
    with open(pred_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_ec_str = row['predicted_ec']
            if not pred_ec_str or pd.isna(pred_ec_str):
                pred_label.append([])
            else:
                if pred_ec_str.strip().startswith('['):
                    try:
                        ec_list = ast.literal_eval(pred_ec_str)
                        if isinstance(ec_list, list):
                            pred_label.append(ec_list)
                            continue
                    except:
                        pass
                ec_list = [ec.strip() for ec in pred_ec_str.split(';') if ec.strip()]
                pred_label.append(ec_list)
    return pred_label

def get_pred_probs_R_maxsep(pred_csv_path):
    """读取maxsep格式概率并应用变换"""
    df = pd.read_csv(pred_csv_path)
    pred_probs = []
    for _, row in df.iterrows():
        prob_str = row['ec_probabilities']
        if pd.isna(prob_str) or prob_str == '':
            pred_probs.append(torch.tensor([]))
            continue
        transformed_vals = []
        for item in prob_str.split(';'):
            if ':' not in item:
                continue
            _, val_str = item.split(':', 1)
            try:
                x = float(val_str)
                if x <= 0:
                    x = 1e-6
                exp_neg = torch.exp(torch.tensor(-1.0 / x))
                t = (1 - exp_neg) / (1 + exp_neg)
                transformed_vals.append(t.item())
            except:
                continue
        if not transformed_vals:
            pred_probs.append(torch.tensor([]))
            continue
        probs = torch.tensor(transformed_vals)
        probs = probs / probs.sum()
        pred_probs.append(probs)
    return pred_probs

def get_pred_labels_R_top1(pred_csv_path):
    """读取top1格式预测标签"""
    pred_labels = []
    with open(pred_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_ec_str = row['predicted_ec']
            try:
                ec_list = ast.literal_eval(pred_ec_str)
                if not isinstance(ec_list, list):
                    ec_list = [ec_list]
            except:
                ec_list = [pred_ec_str] if pred_ec_str else []
            pred_labels.append(ec_list)
    return pred_labels

def get_pred_probs_R_top1(pred_csv_path):
    """读取top1格式概率并应用变换"""
    pred_probs = []
    with open(pred_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_ec_str = row['predicted_ec']
            try:
                ec_list = ast.literal_eval(pred_ec_str)
                if not isinstance(ec_list, list):
                    ec_list = [ec_list]
            except:
                ec_list = [pred_ec_str] if pred_ec_str else []
            n_ec = len(ec_list)
            if n_ec == 0:
                pred_probs.append(torch.tensor([]))
                continue
            similarity = float(row['top1_similarity'])
            if similarity <= 0:
                similarity = 1e-6
            exp_neg = torch.exp(torch.tensor(-1.0 / similarity))
            transformed = (1 - exp_neg) / (1 + exp_neg)
            probs = torch.full((n_ec,), transformed.item())
            probs = probs / probs.sum()
            pred_probs.append(probs)
    return pred_probs

def get_true_labels_from_csv(csv_path):
    """读取真实标签CSV（格式：id, EC，分隔符为逗号，第一行为表头）"""
    true_label_dict = {}
    all_label = set()
    print(f"读取真实标签: {csv_path}")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['id']
            ec_str = row['EC']
            ec_list = ec_str.split(';') if ec_str else []
            true_label_dict[pid] = ec_list
            for ec in ec_list:
                all_label.add(ec)
    true_label = [true_label_dict[pid] for pid in true_label_dict]
    return true_label, all_label

def get_true_labels(csv_path):
    result = open(csv_path, 'r')
    csvreader = csv.reader(result, delimiter='\t')
    all_label = set()
    true_label_dict = {}
    header = True
    count = 0
    for row in csvreader:
        # don't read the header
        if header is False:
            count += 1
            true_ec_lst = row[1].split(';')
            true_label_dict[row[0]] = true_ec_lst
            for ec in true_ec_lst:
                all_label.add(ec)
        if header:
            header = False
    true_label = [true_label_dict[i] for i in true_label_dict.keys()]
    return true_label, all_label

def get_ec_pos_dict(mlb, true_label, pred_label):
    ec_list = []
    pos_list = []
    for i in range(len(true_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([true_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([true_label[i]]))[1])
    for i in range(len(pred_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([pred_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([pred_label[i]]))[1])
    label_pos_dict = {}
    for i in range(len(ec_list)):
        ec, pos = ec_list[i], pos_list[i]
        label_pos_dict[ec] = pos
    return label_pos_dict

def get_eval_metrics(pred_label, pred_probs, true_label, all_label):
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    n_test = len(pred_label)
    pred_m = np.zeros((n_test, len(mlb.classes_)))
    true_m = np.zeros((n_test, len(mlb.classes_)))
    pred_m_auc = np.zeros((n_test, len(mlb.classes_)))
    label_pos_dict = get_ec_pos_dict(mlb, true_label, pred_label)
    for i in range(n_test):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
        labels, probs = pred_label[i], pred_probs[i]
        for label, prob in zip(labels, probs):
            if label in all_label:
                pos = label_pos_dict[label]
                pred_m_auc[i, pos] = prob
    pre = precision_score(true_m, pred_m, average='weighted', zero_division=0)
    rec = recall_score(true_m, pred_m, average='weighted')
    f1 = f1_score(true_m, pred_m, average='weighted')
    roc = roc_auc_score(true_m, pred_m_auc, average='weighted')
    acc = accuracy_score(true_m, pred_m)
    return pre, rec, f1, roc, acc

def evaluate_single_prediction(pred_csv_path, true_csv_path, pred_type):
    """评估单个预测文件，输出指标"""
    if pred_type == '_maxsep':
        pred_label = get_pred_labels_R_maxsep(pred_csv_path)
        pred_probs = get_pred_probs_R_maxsep(pred_csv_path)
    elif pred_type == '_top1':
        pred_label = get_pred_labels_R_top1(pred_csv_path)
        pred_probs = get_pred_probs_R_top1(pred_csv_path)
    else:
        raise ValueError("pred_type must be '_maxsep' or '_top1'")
    true_label, all_label = get_true_labels(true_csv_path)
    # true_label, all_label = get_true_labels_from_csv(true_csv_path)
    pre, rec, f1, roc, acc = get_eval_metrics(pred_label, pred_probs, true_label, all_label)
    return {'precision': pre, 'recall': rec, 'f1': f1, 'auc': roc, 'accuracy': acc}

# ==================== 主流程 ====================
def main():
    # 1. 生成背景数据库嵌入（两个模型）
    print("\n========== Step 1: 生成背景数据库嵌入 ==========")
    for model in MODELS:
        output_h5 = f"{DATA_DIR}/clean100_set_{model['name']}.h5"
        generate_embeddings(
            input_path=BACKGROUND_JSON,
            output_hdf5=output_h5,
            model_path=model['path'],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    # 2. 生成测试集嵌入（每个数据集 × 每个模型）
    print("\n========== Step 1: 生成测试集嵌入 ==========")
    for dataset in DATASETS:
        ds_name = dataset['name']
        for model in MODELS:
            output_h5 = f"{DATA_DIR}/{ds_name}_{model['name']}.h5"
            generate_embeddings(
                input_path=dataset['json'],
                output_hdf5=output_h5,
                model_path=model['path'],
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

    # 3. 预测（maxsep & top1）
    print("\n========== Step 2: 运行预测 ==========")
    for model in MODELS:
        model_name = model['name']
        train_h5 = f"{DATA_DIR}/clean100_set_{model_name}.h5"
        train_ec_dict = f"{DATA_DIR}/clean100_set_dict.json"   # 公共字典
        
        # 检查训练集嵌入文件是否存在
        if not os.path.exists(train_h5):
            print(f"警告: 训练嵌入文件 {train_h5} 不存在，跳过模型 {model_name}")
            continue
            
        for dataset in DATASETS:
            ds_name = dataset['name']
            test_h5 = f"{DATA_DIR}/{ds_name}_{model_name}.h5"
            
            # 检查测试集嵌入文件是否存在
            if not os.path.exists(test_h5):
                print(f"警告: 测试嵌入文件 {test_h5} 不存在，跳过 {ds_name} 模型 {model_name}")
                continue
            
            # maxsep 预测
            output_maxsep = f"{OUTPUT_DIR}/{ds_name}_{model_name}_maxsep.csv"
            if os.path.exists(output_maxsep):
                print(f"预测文件已存在，跳过: {output_maxsep}")
            else:
                run_maxsep_prediction(test_h5, train_h5, train_ec_dict, output_maxsep, batch_size=10000)
            
            # top1 预测
            output_top1 = f"{OUTPUT_DIR}/{ds_name}_{model_name}_top1.csv"
            if os.path.exists(output_top1):
                print(f"预测文件已存在，跳过: {output_top1}")
            else:
                run_top1_prediction(test_h5, train_h5, train_ec_dict, output_top1, top_k=1, batch_size=1000)

    # 4. 评估所有预测结果
    print("\n========== Step 3: 评估预测结果 ==========")
    all_metrics = []
    for dataset in DATASETS:
        ds_name = dataset['name']
        true_csv = dataset['truth_csv']
        for model in MODELS:
            model_name = model['name']
            for method in ['maxsep', 'top1']:
                pred_csv = f"{OUTPUT_DIR}/{ds_name}_{model_name}_{method}.csv"
                if not os.path.exists(pred_csv):
                    print(f"警告: 预测文件不存在 {pred_csv}")
                    continue
                pred_type = f"_{method}"
                metrics = evaluate_single_prediction(pred_csv, true_csv, pred_type)
                metrics['dataset'] = ds_name
                metrics['model'] = model_name
                metrics['method'] = method
                all_metrics.append(metrics)
                print(f"\n{ds_name} | {model_name} | {method}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1:        {metrics['f1']:.4f}")
                print(f"  AUC:       {metrics['auc']:.4f}")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")

    # 可选：保存所有指标到CSV
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv(f"{OUTPUT_DIR}/all_metrics_dis1.csv", index=False)
        print(f"\n所有指标已保存至 {OUTPUT_DIR}/all_metrics_dis1.csv")

if __name__ == "__main__":
    main()