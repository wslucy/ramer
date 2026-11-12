# RAMER：旧版代码 M3D-FLOOR、MLP↔QKV 对照、B 关系正例采样、全量 Epoch 15 三数据集评估

## 0. 结论与边界

本包是在用户上传的 `train_mechpair_light_10k.py` (675 行) 和实际使用的 `train_mechpair_AB_standalone.py` 基础上改造的。

**旧版实验保留：** `BioData2Seq` (旧 ProtT5/GearNet/MolT5)、ReactionGuidedSelector 原始 MLP 算法、`SequenceOnlyMechanismHead` 的共享 mask 截断/全局池化、原四个损失的计算和权重、FP32、原 `cluster.py` 聚类规则。新增配置会分别叠加 `Ldeploy(0.2)+LFusion(0.5)+Floor(0.01,m=0.1,beta=5)`，并可把两个旧 MLP selector 换成标准 `nn.MultiheadAttention` 的 QKV selector。

**公平 20k：** 严格读取现有 `subsets/fixed_20k_seed3408.json`，根据与现有训练相同的训练数据 fingerprint 验证过滤顺序与反应映射。若指纹失配直接拒绝运行。默认 20k 全部作为训练集 (`val_ratio=0`)，与目前 20k M0/M3D 的训练行数相同。**4GPU × batch16 的全局 batch64，与过去 8GPU × batch16 的全局 batch128 不同**；因此新旧 GPU 数目之间不能直接当作单变量严格比较，四个新旧代码内对照则完全一致。

**不能仅凭这个实验把旧 0.823 与新 0.805 的差归因于缓存/加速**。旧版和新版还有 `shared_mask` 全局池化、反应编码器具体组件/冻结、DDP 温度参数等实现差异，旧 0.823 的原始数据和评估选择设置也尚未在本环境内核实。需要用这里的 MLP_BASE 作为同环境对照，而不是假设它必然再次达到 0.823。

**评估：** 继续调用原 `cluster.py -> run_test_clustering_analysis`，输出 legacy 指标。为了避免每一轮把第二份 ProtT5 加载进 rank0 显存，旧版比较脚本直接用**当前已经加载的序列编码器**导出原始 masked mean + HDF5，这不会改动训练损失/参数更新。保存每一轮 `seq_only_epoch_XXX.pth` 增量权重；必须结合相同 ProtT5 底座重载。50 Epoch 的全量 checkpoint 不默认每轮保存，因为过于庞大。

> 旧版 `InfoNCELoss.temperature` 参数原本就在 DDP 外，未做梯度 AllReduce；这里为保持原始实现不改动。这个差异也提醒我们“完全排除旧/新训练差异”需要更精细的实现审计。

## 1. 文件说明

- `train_mechpair_light_MLP_20k.py`：完整的旧版派生文件，默认 selector=mlp，可跑 MLP_BASE / MLP_FLOOR。
- `train_mechpair_light_QKV_20k.py`：同样完整的旧版派生文件，默认 selector=qkv，可跑 QKV_BASE / QKV_FLOOR。
- `run_legacy_attention_4gpu.sbatch` + `submit_4gpu.sh`：4GPU 固定20k、50Epoch，一次提交一个 Slurm 作业。
- `train_mechpair_AB_pair_sampler.py`：完整的当前 AB standalone 训练文件，仅额外加入可选同 RHEA 的正例配对 BatchSampler。
- `run_B_sampler_4gpu.sbatch` + `submit_4gpu.sh`：在 4GPU 下**同条件**比较 B_RANDOM 与 B_PAIRED。
- `eval_full_epoch15_3sets.py` + `eval_full_epoch15_3sets.sbatch`：用已有 `cluster.py` 评估全量 M3D-FLOOR 的 Epoch 15 在 Price-149、NEW-392 和 RAM255 上的聚类。
- `test_components.py`：CPU 冒烟测试，无需大模型。

## 2. 上传并安装

```bash
cd /datapool/home/tanlijun/RAMER/dec-v2/RAMER_unsupervised_AB
unzip RAMER_legacy4gpu_attention_B_eval.zip
cp RAMER_legacy4gpu_attention_B_eval/*.py .
cp RAMER_legacy4gpu_attention_B_eval/*.sbatch .
cp RAMER_legacy4gpu_attention_B_eval/submit_4gpu.sh .
mkdir -p slurm_logs
python -m py_compile train_mechpair_light_MLP_20k.py train_mechpair_light_QKV_20k.py train_mechpair_AB_pair_sampler.py eval_full_epoch15_3sets.py
python test_components.py
```

注意：`test_components.py` 仅测试核心张量/采样，不会替代 4GPU 真实训练环境验证。需要 `/datapool/home/tanlijun/RAMER/dec-v2/Data2seq_v2.py` 以及原始各个编码器脚本，旧代码也必须能在当前 `ramer` 环境运行。

## 3. 4GPU，严格同一 20k：旧版+策略 A 与 QKV 对照

首先跑 **MLP_BASE**（原始注意力/原四项损失）：

```bash
CODE_DIR="$PWD" SEED=3408 EPOCHS=50 RUN_TAG=s3408 bash submit_4gpu.sh MLP_BASE
```

再跑 **MLP_FLOOR**（原始注意力 + M3D-FLOOR）：

```bash
CODE_DIR="$PWD" SEED=3408 EPOCHS=50 RUN_TAG=s3408 bash submit_4gpu.sh MLP_FLOOR
```

把旧残基注意力替换成 Transformer QKV，但不加 A/Floor：

```bash
CODE_DIR="$PWD" SEED=3408 EPOCHS=50 RUN_TAG=s3408 bash submit_4gpu.sh QKV_BASE
```

替换成 QKV 且加 A/Floor：

```bash
CODE_DIR="$PWD" SEED=3408 EPOCHS=50 RUN_TAG=s3408 bash submit_4gpu.sh QKV_FLOOR
```

如果 4 张 A100 也难排队，可在下一条命令前设置 `AFTER_JOB_ID=上一个作业实际ID`，此时依赖 `afterok` 串行执行，无需抢多组 4 卡。例如：

```bash
CODE_DIR="$PWD" AFTER_JOB_ID=345678 SEED=3408 EPOCHS=50 RUN_TAG=s3408 bash submit_4gpu.sh MLP_FLOOR
```

保存到 `legacy_20k_controls/<RUN_ID>_seed3408_s3408/`；内有 `clustering_eval_history.csv` 和每轮 `seq_only_epoch_XXX.pth`。保存的序列权重是 **ProtT5 底座 + LoRA/线性层增量**，不是独立完整模型。

建议分析 2×2 矩阵：

| | 原始 MLP | 反应 QKV |
|---|---|---|
| 原四项损失 | MLP_BASE | QKV_BASE |
| 加 deploy+fusion+Floor | MLP_FLOOR | QKV_FLOOR |

- `MLP_FLOOR − MLP_BASE`：旧版训练代码中加入 M3D-FLOOR 是否有效。
- `QKV_BASE − MLP_BASE`：把原始局部注意力换成 QKV 的影响。
- `QKV_FLOOR − MLP_FLOOR`：在 M3D-FLOOR 下切换局部注意力的影响。
- `QKV_FLOOR − QKV_BASE`：QKV 条件下加 M3D-FLOOR 是否有效。

注意 QKV 层的可训练参数量通常高于原 MLP 层，若显著领先，还需要参数量匹配消融。`num_slots=4` 是**策略 A 的独立可学习 Query**，与局部 selector 的反应 Query 不同。

## 4. 反应驱动局部注意力：MLP 与 Transformer QKV 的区别

### 4.1 原 MLP

对于第 i 个残基特征 `h_i` 和反应向量 `r`，旧版先拼接 `[h_i ; r]`，用网络计算标量：

`score_i = Linear_2(LayerNorm(GELU(Linear_1([h_i ; r]))))`。

`alpha_i = softmax_i(score_i)`；`local = sum_i alpha_i * h_i`。

**没有独立的 Wq/Wk/Wv**；依靠非线性 MLP 直接对“残基+反应”的组合评分，值向量就是未投影的 `h_i`。这属于拼接/加性式注意力家族，并非标准 Transformer 点积注意力。

### 4.2 新 QKV（8 heads）

`nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)`：

- `Q = W_Q · r`，Q 只来自当前**反应全局向量**，形状 `[B,1,1024]`；
- `K = W_K · h_i`，K 是每个序列或结构**残基的投影特征**，形状 `[B,L,1024]`；
- `V = W_V · h_i`，V 同样来自每个残基，但**与 K 采用不同的可训练投影**；
- 每头 `d_head=1024/8=128`，注意力 `softmax(QK^T / sqrt(128) + padding_mask)`；
- 每个头对 V 加权求和，8 个头拼接后通过 `W_O` 得到局部向量 `[B,1024]`；
- softmax 在有效残基位置计算；输出平均各头的注意力供 Top-k 诊断。

结构残基和序列残基使用**两个独立参数的 QKV selector**，但共同使用反应向量作为 Q，与旧 MLP 的输入来源一致。**策略 A 的 4 个 learned Query 依然独立存在**，并不接收反应作为 Q；两部分的监督作用不同。

## 5. 在现有 B-L0 框架中加入关系正例批采样

原随机 batch=16，`train_b_positive_pairs` 通常接近 16，绝大部分是对角正例。给每个 batch 固定插入 **2 对同 RHEA、不同序列**，再填 12 个随机样本，能在训练时直接构造可用额外正例。

对照必须都使用**相同 4GPU、batch16、20k、seed3408、50Epoch**，不要拿先前 8GPU 的 B-L0 直接比较，否则全局 batch 不同。

```bash
CODE_DIR="$PWD" SEED=3408 EPOCHS=50 RUN_TAG=compare bash submit_4gpu.sh B_RANDOM
CODE_DIR="$PWD" SEED=3408 EPOCHS=50 RUN_TAG=compare PAIRS_PER_BATCH=2 bash submit_4gpu.sh B_PAIRED
```

保存到 `b_sampler_4gpu`。重点比较 `train_b_positive_pairs`、`train_b_valid_background_pairs`、`train_total`、ARI 的最佳值/相同 epoch 均值。在当前批大小16下，2对不同蛋白同 RHEA 预计每个 batch 比纯对角至少多出 **4 个方向性正例配对**（具体 `train_b_positive_pairs` 是多个损失头的平均值，不等于单一方向实际数量）。

**偏差提醒**：同 RHEA 的酶不必然属于同一 EC 子类。新采样是在你的已知 RHEA 关系下增加正例，不应把 RHEA 同反应直接当作相同 EC 标签。配对采样还会提高重复 RHEA 组的训练频率，因此需监控受采样影响的类别覆盖与潜在过拟合；这是一个需要实证的候选，而不是已保证有效。

## 6. 全量 M3D-FLOOR Epoch 15：评估 Price-149、NEW-392、RAM255

这个任务**不需要重新训练，也不需要 4/8 张 GPU**，只需 **1 GPU**。用你现有的 `seq_only_epoch_015.pth`：

```bash
cd /datapool/home/tanlijun/RAMER/dec-v2/RAMER_unsupervised_AB
find /datapool/home/tanlijun/RAMER -type f \( -iname '*new*392*.json' -o -iname '*ram*255*.json' -o -iname '*ram*255*.csv' -o -iname 'new.csv' \)
```

我们能从之前 `inference.py` 核实 NEW-392 的候选文件名为 `NEW-392.json` 和 `new.csv`，但**没有在已提供文件中核实 RAM255 的绝对路径或真值格式**，因此请将实际路径传给提交命令：

```bash
RAM255_JSON=/实际/路径/RAM255.json \
RAM255_TRUTH=/实际/路径/RAM255_labels.tsv \
CODE_DIR="$PWD" \
CHECKPOINT="$PWD/experiments_next_full/FLOOR_FULL_seed3408_fp32_full50_v1/seq_only_epoch_015.pth" \
sbatch --export=ALL \
  --output="$PWD/slurm_logs/eval_floor15_%j.out" \
  --error="$PWD/slurm_logs/eval_floor15_%j.err" \
  eval_full_epoch15_3sets.sbatch
```

如 NEW-392 不在候选目录，显式提供 `NEW392_JSON=/实际/路径/NEW-392.json NEW392_TRUTH=/实际/路径/new.csv`。原 `cluster.py` 期望标签文件**虽名为 `.csv`，实际使用制表符 (`\t`) 分割，必须有 `Entry`、`EC number` 两列**。如果 RAM255 标签不是这种格式，需要先按实际字段制作 TSV 映射再使用，不能直接拿任意 CSV 硬套。没有真实映射就无法诚实报告 RAM255 ARI。

结果保存到 `eval_full_floor_epoch15_three_sets/`，包括各数据集的 `epoch_015_embeddings.h5`、`legacy_results/clustering_metrics_epoch15.csv`（EC 层级、聚类方法、簇数的详细指标）、`summary.json` 和 `three_dataset_summary.csv`。

这里完全沿用旧 `cluster.py` 的比较协议，即每个测试集内可能选择**EC 层级/聚类方法/聚类数中的最高 ARI**；它适合与你之前 legacy 结果做内部一致性比较，不等同于未经参数选择的独立测试集估计。建议同时报告固定层级、固定聚类规则的结果，并先检查训练集与三个测试集是否重叠。

## 7. 自检

```bash
bash -n run_legacy_attention_4gpu.sbatch run_B_sampler_4gpu.sbatch eval_full_epoch15_3sets.sbatch submit_4gpu.sh
python -m py_compile train_mechpair_light_MLP_20k.py train_mechpair_light_QKV_20k.py train_mechpair_AB_pair_sampler.py eval_full_epoch15_3sets.py
python test_components.py
```

这是**代码语法、算法张量和采样器 CPU 测试**；无法在本地重现服务器完整 ProtT5/MolT5 和 4GPU Slurm 环境，运行前请按文件检查和日志确认。

## 更新（2026-11-11）：无需改 Slurm 文件，提交时自由选择 GPU 数和节点

现在使用统一入口 `submit_4gpu.sh`（历史文件名保留；已支持任意 1–8 卡，须符合节点和分区实际资源）。

- `GPUS=1`：申请 1 张 GPU；`GPUS=2`、`GPUS=3` 同理。
- `GPU_NODE=auto`：不绑定节点，Slurm 自动找满足申请条件的节点。
- `GPU_NODE=node105`：只在 node105 上排队，注意该节点必须有足够空闲 GPU。
- `GPU_PARTITION=a100`：默认 A100 分区。其他分区需确认节点、显卡类型和显存需求。
- `TIME_LIMIT=0`：请求无时限运行，须遵守分区/QOS 限制。
- 训练 torchrun 进程数自动检测分配的 GPU（`--nproc_per_node=gpu`）；**不需手改为 1/2/3/4**。
- 默认每张 GPU 申请 8 CPU 核心、48GB 内存，最低 64GB 整机内存；可通过 `CPUS_PER_GPU` 和 `MEM_PER_GPU_GB` 调整。需确认集群节点允许。
- 评估模式统一为 `EVAL15`，只支持 `GPUS=1`。

示例：

```bash
GPUS=1 GPU_NODE=auto bash submit_4gpu.sh MLP_BASE
GPUS=2 GPU_NODE=auto RUN_TAG=g2_floor bash submit_4gpu.sh MLP_FLOOR
GPUS=3 GPU_NODE=node105 RUN_TAG=g3_qkv bash submit_4gpu.sh QKV_BASE
GPUS=1 GPU_NODE=auto bash submit_4gpu.sh EVAL15
GPUS=2 GPU_NODE=auto RUN_TAG=g2_bpair PAIRS_PER_BATCH=2 bash submit_4gpu.sh B_PAIRED
```

注意：GPU 数变化会改变全局 Batch Size（默认每卡 16），不能直接把不同 GPU 数的最优 ARI 视为严格的单变量对照。做 MLP 与 QKV 对照、B RANDOM 与 B PAIRED 对照时，请使用相同 GPU 数与每卡 Batch Size。

修改后需要重新 `git pull` 并复制更新的 `submit_4gpu.sh`、`run_legacy_attention_4gpu.sbatch`、`run_B_sampler_4gpu.sbatch`、`eval_full_epoch15_3sets.sbatch` 到项目根目录。


## 2026-11-12 更正：Epoch 15 的三数据集评估应是 Top-1/MaxSep 的 P/R/F1

**不要使用** `eval_full_epoch15_3sets.sbatch`：它运行的是 ARI/NMI 聚类评估，不是用户所需的 EC 检索评估。

现在改为：
- 原始预测代码备份：`legacy_inference_original.py`（原上传的 `inference.py`）。
- 三数据集评估主程序：`eval_epoch15_top1_maxsep.py`。
- 单卡 Slurm：`eval_epoch15_top1_maxsep.sbatch`。
- 输出：`eval_floor15_top1_maxsep/top1_maxsep_weighted_prf.csv`。

原方法保留 `clean100_set.json` / `clean100_set_dict.json` 参考库、余弦相似度、Top-1、MaxSep 动态 K 选择算法和 sklearn weighted precision/recall/F1。相比历史代码，指标计算改为**按测试蛋白 ID 匹配真实标签**，以避免仅靠 CSV 行顺序造成错位。作为安全检查，如果测试 ID 出现在背景库中会报错。默认不计算可能因类别分布而失败的 AUC。

请先确认参考库文件的真实位置：

```bash
find /datapool/home/tanlijun/RAMER -name clean100_set.json -o -name clean100_set_dict.json
```

然后执行（如默认路径能找到参考库，不需指定 BACKGROUND_JSON 和 BACKGROUND_DICT）：

```bash
cd /datapool/home/tanlijun/RAMER/dec-v2/RAMER_unsupervised_AB
mkdir -p slurm_logs
CODE_DIR="$PWD" sbatch --output="$PWD/slurm_logs/eval_floor15_prf_%j.out" \
  --error="$PWD/slurm_logs/eval_floor15_prf_%j.err" \
  --export=ALL eval_epoch15_top1_maxsep.sbatch
```

若参考库在自定义目录：

```bash
CODE_DIR="$PWD" BACKGROUND_JSON=/actual/path/clean100_set.json \
BACKGROUND_DICT=/actual/path/clean100_set_dict.json \
sbatch --export=ALL eval_epoch15_top1_maxsep.sbatch
```

Price-149、NEW-392、RAM255 的真实标签文件使用 `test_data/price.csv`、`test_data/new.csv`、`test_data/ram255.csv`。指标均为多标签、weighted 平均。
