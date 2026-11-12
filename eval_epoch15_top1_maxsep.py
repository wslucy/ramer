#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAMER Epoch 15: original Top-1/MaxSep EC retrieval, evaluated on three datasets.

Prediction rules come verbatim from legacy_inference_original.py. The historical
metric code had a dangerous assumption that truth/prediction rows had identical
order. Here weighted Precision/Recall/F1 are computed after joining by test ID.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from eval_full_epoch15_3sets import (
    load_records, unpack_sequence_state, validate_loaded_state, export_embeddings
)

DATASETS = [
    ("Price-149", "Price-149.json", "price.csv"),
    ("NEW-392", "NEW-392.json", "new.csv"),
    ("RAM255", "ram255.json", "ram255.csv"),
]

# Extract (rather than rewrite) original prediction functions; importing the
# entire old inference.py would execute its hard-coded Data2seq path and CUDA mask.
LEGACY_FUNCTIONS = {
    "dynamic_k_selection", "run_maxsep_prediction", "run_top1_prediction",
}

def get_original_predictors(source: Path):
    src = source.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(source))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in LEGACY_FUNCTIONS]
    if {n.name for n in selected} != LEGACY_FUNCTIONS:
        raise RuntimeError("Original inference.py is missing a required prediction function")
    ns = dict(np=np, pd=pd, torch=torch, json=json, h5py=h5py, tqdm=tqdm)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), ns)
    return ns["run_top1_prediction"], ns["run_maxsep_prediction"]

def choose_reference(explicit, filename, roots):
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Reference file not found: {p}")
        return p
    tried = []
    for root in roots:
        candidate = Path(root) / filename
        tried.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Cannot find " + filename + "; tried:\n" +
                            "\n".join(tried) + "\nPass the relevant --background-json or --background-dict.")

def labels_from_truth(path):
    # Original data typically has TSV columns Entry and EC number; accept an
    # equivalent comma-separated id/EC file without changing EC semantics.
    rows = list(csv.reader(path.open("r", encoding="utf-8-sig", newline=""),
                           delimiter="\t" if "\t" in path.read_text(encoding="utf-8-sig").splitlines()[0] else ","))
    if not rows:
        raise ValueError(f"Empty truth file {path}")
    header = [x.strip().lower() for x in rows[0]]
    id_col = next((header.index(s) for s in ("entry","id","primaryaccession","protein_id") if s in header), None)
    ec_col = next((header.index(s) for s in ("ec number","ec","ec_number") if s in header), None)
    if id_col is None or ec_col is None:
        raise ValueError(f"{path}: expected Entry/EC number or id/EC columns; got {rows[0]}")
    out = {}
    for row in rows[1:]:
        if max(id_col, ec_col) >= len(row): continue
        key = row[id_col].strip()
        if not key: continue
        ecs = [v.strip() for v in row[ec_col].split(";") if v.strip() and v.strip().lower() != "nan"]
        if key in out:
            raise ValueError(f"{path}: duplicate ID {key}")
        out[key] = ecs
    if not out: raise ValueError(f"{path}: no ground-truth annotations")
    return out

def labels_from_predictions(path, method):
    import ast as _ast
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row["test_id"].strip()
            raw = (row.get("predicted_ec") or "").strip()
            if method == "top1":
                try:
                    result = _ast.literal_eval(raw)
                    items = result if isinstance(result, list) else [result]
                except (ValueError, SyntaxError):
                    items = [raw] if raw else []
            else:
                if raw.startswith("["):
                    try:
                        parsed = _ast.literal_eval(raw)
                        items = parsed if isinstance(parsed, list) else [parsed]
                    except (ValueError, SyntaxError):
                        items = raw.split(";")
                else:
                    items = raw.split(";")
            if key in out: raise ValueError(f"{path}: duplicate prediction ID {key}")
            out[key] = [str(x).strip() for x in items if str(x).strip()]
    return out

def evaluate(pred_csv, truth_csv, method, test_ids):
    gt = labels_from_truth(truth_csv)
    pred = labels_from_predictions(pred_csv, method)
    # Preserve the benchmark input order and explicitly enforce one-to-one
    # matching instead of relying on positional CSV row alignment.
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Duplicate test IDs")
    absent_gt = [x for x in test_ids if x not in gt]
    absent_pred = [x for x in test_ids if x not in pred]
    if absent_gt or absent_pred:
        raise ValueError(
            f"{truth_csv}: missing truth IDs={absent_gt[:8]}, missing prediction IDs={absent_pred[:8]}"
        )
    true = [gt[x] for x in test_ids]
    predicted = [pred[x] for x in test_ids]
    known = sorted({ec for labels in true for ec in labels})
    if not known: raise ValueError(f"{truth_csv}: no EC labels in test set")
    mlb = MultiLabelBinarizer(classes=known)
    mlb.fit([known])
    with torch.no_grad():
        y = mlb.transform(true)
        yp = mlb.transform([[ec for ec in lab if ec in known] for lab in predicted])
    return {
        "precision": float(precision_score(y, yp, average="weighted", zero_division=0)),
        "recall": float(recall_score(y, yp, average="weighted", zero_division=0)),
        "f1": float(f1_score(y, yp, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y, yp)),
        "n": len(test_ids),
        "predicted_ec_outside_truth_label_space": sum(len([e for e in x if e not in known]) for x in predicted),
    }

def export_if_needed(model, records, path, batch_size, force=False):
    if path.exists() and not force:
        with h5py.File(path, "r") as f:
            ids = [x.decode("utf-8") if isinstance(x,bytes) else str(x) for x in f["ids"][:]]
        expected = [r[0] for r in records]
        if ids != expected:
            raise ValueError(f"{path} has stale/mismatched IDs. Use a distinct --output-dir or --force.")
        print(f"[CACHE] {path}: {len(ids)} embeddings", flush=True)
        return
    if path.exists(): path.unlink()
    export_embeddings(model, records, path, batch_size)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--encoder-dir", default="/datapool/home/tanlijun/RAMER/dec-v2")
    ap.add_argument("--test-dir", default="/datapool/home/tanlijun/RAMER/data/test_data")
    ap.add_argument("--background-json", default="")
    ap.add_argument("--background-dict", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="Regenerate embeddings/predictions")
    args = ap.parse_args()
    if args.batch_size < 1: ap.error("--batch-size must be positive")
    ckpt=Path(args.checkpoint)
    src=Path(__file__).with_name("legacy_inference_original.py")
    if not ckpt.is_file(): raise FileNotFoundError(ckpt)
    if not src.is_file(): raise FileNotFoundError(src)
    test_dir=Path(args.test_dir)
    roots=[test_dir, test_dir.parent, test_dir.parent.parent/"zero_shot"/"data",
           Path.cwd()/"zero_shot"/"data"]
    bg_json=choose_reference(args.background_json, "clean100_set.json", roots)
    bg_dict=choose_reference(args.background_dict, "clean100_set_dict.json", roots)
    reference_map=json.loads(bg_dict.read_text(encoding="utf-8"))
    background=load_records(bg_json)
    if len(background) == 0: raise ValueError("Empty background reference")
    missing_annot=[x for x,_ in background if x not in reference_map]
    if missing_annot:
        raise ValueError(f"Missing EC annotations for {len(missing_annot)} background entries: {missing_annot[:8]}")
    datasets=[]
    for name,json_name,truth_name in DATASETS:
        jp=test_dir/json_name;tp=test_dir/truth_name
        if not jp.is_file(): raise FileNotFoundError(jp)
        if not tp.is_file(): raise FileNotFoundError(tp)
        rec=load_records(jp)
        overlap=set(x for x,_ in rec)&set(x for x,_ in background)
        if overlap:
            raise RuntimeError(f"{name}: {len(overlap)} shared IDs with reference library; potential retrieval leakage: {sorted(overlap)[:8]}")
        # Verify annotations before allocating GPU/model memory.
        truth=labels_from_truth(tp)
        lacking=[x for x,_ in rec if x not in truth]
        if lacking: raise ValueError(f"{tp}: missing test labels: {lacking[:8]}")
        datasets.append((name,rec,tp))
    from importlib import import_module
    sys.path.insert(0,args.encoder_dir)
    BioData2Seq=import_module("Data2seq_v2").BioData2Seq
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=BioData2Seq("protein-sequence",embed_dim=1024).to(device)
    state=unpack_sequence_state(ckpt)
    loaded=model.load_state_dict(state,strict=False)
    validate_loaded_state(state,loaded)
    model.eval()
    run_top1,run_maxsep=get_original_predictors(src)
    root=Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=True)
    print(f"[EVAL] checkpoint={ckpt} sha256={hashlib.sha256(ckpt.read_bytes()).hexdigest()}",flush=True)
    print(f"[REFERENCE] json={bg_json} labels={bg_dict} n={len(background)}",flush=True)
    bg_h5=root/"clean100_set_epoch015.h5"
    export_if_needed(model,background,bg_h5,args.batch_size,args.force)
    summary=[]
    for name,rec,truth in datasets:
        test_h5=root/f"{name}_epoch015.h5"
        export_if_needed(model,rec,test_h5,args.batch_size,args.force)
        for method,predictor in (("top1",run_top1),("maxsep",run_maxsep)):
            pred_path=root/f"{name}_{method}_epoch015.csv"
            if not pred_path.exists() or args.force:
                if method=="top1":
                    predictor(str(test_h5),str(bg_h5),str(bg_dict),str(pred_path),top_k=1,batch_size=1000)
                else:
                    predictor(str(test_h5),str(bg_h5),str(bg_dict),str(pred_path),batch_size=10000)
            metrics=evaluate(pred_path,truth,method,[r[0] for r in rec])
            row=dict(dataset=name,method=method,**metrics)
            summary.append(row)
            print("[RESULT]",json.dumps(row,ensure_ascii=False),flush=True)
    output=root/"top1_maxsep_weighted_prf.csv"
    with output.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    print("[DONE]",output,flush=True)

if __name__=="__main__":
    main()
