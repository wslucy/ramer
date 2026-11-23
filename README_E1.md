# RAMER E1：基于原始 MLP_BASE 20k 的位置×通道路由

## 已交付文件
- train_mechpair_light_E1_20k.py：完整训练、逐轮聚类评估、序列表征 H5 导出
- run_E1_20k.sbatch：独立 Slurm 训练作业
- test_E1_cpu.py：无需下载 ProtT5 的张量/梯度冒烟测试

## E1 的定义
保留原 MLP_BASE 的 BioDataset、固定 20k 数据指纹校验、四个 Base 损失和原始 MLP 注意力。H[B,L,1024] 经过可学习投影并分为功能候选 512 维和互补 512 维。序列自身预测残基 soft gate；部署表示为原始序列 masked mean 加路由功能残差。残差映射零初始化，确保第一步与原 Base masked mean 相同。加一项序列路由到反应投影的 InfoNCE（权重默认 0.1，反应 target detached）。

没有引入 A/B/Floor、额外 PDB 坐标或 E2 结构教师。特别注意：互补 512 维未受几何监督，因此 E1 不能声称已经学到结构-功能的语义解耦。E2 必须验证 PDB 序列-坐标映射。

## 拉取并检查
    cd /datapool/home/tanlijun/RAMER/dec-v2/RAMER_unsupervised_AB
    git -C ramer_experiments pull --ff-only origin main
    cp ramer_experiments/train_mechpair_light_E1_20k.py .
    cp ramer_experiments/run_E1_20k.sbatch .
    cp ramer_experiments/test_E1_cpu.py .
    python -m py_compile train_mechpair_light_E1_20k.py
    bash -n run_E1_20k.sbatch
    python test_E1_cpu.py

## 与 4 卡 MLP_BASE 严格匹配
    mkdir -p slurm_logs
    CODE_DIR="$PWD" SEED=3408 EPOCHS=50 RUN_TAG=matched4 \
    sbatch --partition=a100 --nodes=1 --ntasks=1 --gres=gpu:4 \
      --cpus-per-task=32 --mem=192G --time=0 --export=ALL \
      --job-name=E1_matched4 \
      --output="$PWD/slurm_logs/E1_matched4_%j.out" \
      --error="$PWD/slurm_logs/E1_matched4_%j.err" run_E1_20k.sbatch

默认 batch size 16/卡，global batch 64。相同 seed 与数据 fingerprint。如果资源不足而改成 2/3 卡，必须为 MLP_BASE 重跑相同卡数的对照。

## 参数消融
- ROUTER_MODE=gated（主实验）
- ROUTER_MODE=uniform（同样参数量基本一致，仅取消残基区别加权）
- ROUTE_WEIGHT=0.1（辅助损失）
- ROUTE_SCALE=0.25（固定残差系数）
建议相同 4-GPU、相同 seed、不同 RUN_TAG，比较 gated 与 uniform。

## 结果和推理
默认输出目录：legacy_20k_controls/E1_gated_seed3408_matched4_g4/
- 每轮历史：clustering_eval_history.csv，新增 train_route、train_gate_spread、train_support_fraction
- 完整部署格式：e1_deploy_epoch_XXX.pt、best_ari_e1_deploy.pt
- seq_only_epoch_XXX.pth **没有路由器**，不得作为 E1 模型用于 Top-1/MaxSep。

导出：
    python train_mechpair_light_E1_20k.py \
      --export_checkpoint "$PWD/legacy_20k_controls/E1_gated_seed3408_matched4_g4/e1_deploy_epoch_010.pt" \
      --export_json /datapool/home/tanlijun/RAMER/data/test_data/Price-149.json \
      --export_h5 "$PWD/e1_price_epoch010.h5"

输出 H5 的 ids/embeddings 与旧评估接口兼容，但 clean100_set 等检索参考库也必须用**同一 E1 checkpoint**重新编码，不可混用旧 Base 参考库。

## E2 不在本次交付范围
需验证 PDB 根目录/文件命名、PDB 与 UniProt 的残基对应、链、缺失位置及 GearNet 节点排序后，才能安全构造 Cα 接触标签，不应按数组下标直接对齐。
