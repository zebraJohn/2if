# newidea 组合模型 ETTh1 测试实验设计

日期：2026-08-12
状态：已获用户确认（架构 + 实验配置）

## 1. 背景与目标

`NewIdea.docx` 描述了一个面向电力设备局放（PD）预测的模型：以 **AE/UHF 双通道时频融合** + **设备自适应** 为核心。完整模型为四段流水线：

```
Times2D（多尺度周期特征挖掘）
  → FreEformer（时频融合 + 通道相关性建模）
  → Device Adapter（设备个性化，窄两层 MLP）
  → iTransformer（表征提纯，输出预测）
```

本实验目标：**在 ETTh1 数据集上测试 newidea 组合模型（主干部分）**。基础模型（Times2D / FreEformer / iTransformer）已单独测试过，本次只测组合模型，产出 MSE/MAE 基准结果，验证组合架构有效性。

## 2. 已确认的决策

| 决策点 | 选择 |
|---|---|
| 运行位置 | 在 Pre2if 建统一实验台 |
| 测试对象 | 仅 newidea 组合模型（不含基础模型单独测试） |
| Adapter 处理 | 主干优先，adapter **预埋接口**，本次实验单设备 → identity，不参与训练 |
| 实验规模 | 标准长周期基准：seq_len=96，pred_len∈{96,192,336,720}，features=M，MSE/MAE |
| 环境 | `times2d` conda env（torch 2.11.0+cu128，CUDA 可用） |

## 3. 组合模型架构（v1）

```
输入 x [B, L=96, C=7]
   │
   ▼
① Times2D Backbone（复用 model/Times2D.py 内部模块）
   多尺度周期折叠 + 二维卷积(conv2D) + TSTiEncoder
   输出: 多尺度周期特征 z_t2d [B, C, L]     （保留时间维）
   │
   ▼
② FreEformer 时频增强（复用 fre_trans_real/imag）
   对 z_t2d 逐通道 FFT → 频域 self-attention（real/imag 双支）→ irfft
   输出: 时频增强特征 z_fre [B, C, L]
   │
   ▼
③ Device Adapter（新建，预埋，本实验 identity）
   窄两层 MLP（Linear-GELU-Linear），按 device_id 选择
   本实验恒为单设备（device_id=0），输出恒等于输入
   │
   ▼
④ iTransformer 提纯+预测（复用 DataEmbedding_inverted + Encoder + projector）
   z_fre 倒置: 通道作为 token [B, C, E]
   自注意力学习通道间关联 → 线性投影 → 预测 [B, pred_len, C]
```

### 设计要点

1. **特征级联，保留时间维**：docx 中"压缩成一维表征向量 z"更偏向分类任务；ETTh1 是预测任务，v1 保留时间维度 `[B,C,L]` 做逐层特征增强。三个模型的维度接口天然兼容：Times2D 中间特征 `[B,N,T]` → FreEformer `[B,N,L,1]`（扩维）→ iTransformer 倒置 token。
2. **复用优先，不重写模型**：
   - ① 复用 Times2D 的 `conv2D` 多尺度周期卷积、`TSTiEncoder` backbone、`RevIN`
   - ② 复用 FreEformer 的 `Fre_Trans` / `fre_trans_real` / `fre_trans_imag` 频域注意力（将其输入从原始信号改为 Times2D 增强特征）
   - ③ DeviceAdapter 为新建轻量模块
   - ④ 复用 iTransformer 的 `DataEmbedding_inverted`、`Encoder`、`projector`
3. **Adapter 预埋**：`DeviceAdapter` 类带 `device_id` 接口，本次实验恒为单设备（输出=identity），代码路径保留，后续真实局放数据直接启用。
4. **归一化**：Times2D 内部已有 RevIN；iTransformer 端保留 `use_norm`，保证整链稳定。

### 输入输出约定

- 模型入口统一为 TSLib 约定 `forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)`，返回 `[B, pred_len, C]`
- 与统一实验台的 `experiments/exp_forecast.py` 兼容（`model = model_dict[args.model].Model(args)`）

## 4. 统一实验台结构（Pre2if）

```
Pre2if/
├── model/
│   ├── newidea.py            # 【新建】组合模型 NewIdeaModel（主干 + 预埋 adapter）
│   ├── Times2D.py            # 已有，复用
│   ├── FrePatchTST3_fre_all.py  # 已有，复用
│   └── iTransformer.py       # 已有，复用
├── dataset/ETT-small/ETTh1.csv   # 已有
├── layers/                   # 【合并】（实现调整）
│   ├── Embed.py, SelfAttention_Family.py, Transformer_EncDec.py   ← iTransformer（标准 TSLib）
│   └── RevIN.py, Conv_Blocks.py, encoders.py, All_layers.py, attention.py  ← Times2D
│   ※ 实现时改用 iTransformer 标准 layers 而非 FreEformer 超集：
│      FreEformer 的 layers 依赖自定义 utils（CKA/plot_mat）与 reformer_pytorch，风险高；
│      频域增强用标准 FullAttention 实现，同样忠实于 FreEformer 的频域注意力架构。
├── data_provider/            # 来自 FreEformer（data_factory.py, data_loader.py）
├── utils/                    # 来自 FreEformer（metrics.py, timefeatures.py, tools.py, losses.py ...）
├── experiments/              # 来自 FreEformer（exp_basic.py, exp_forecast.py）
│   └── exp_basic.py          # 【改】model_dict 注册 'NewIdea'
├── run.py                    # 来自 FreEformer
│                             # 【改】补齐 Times2D 所需参数（add/affine/serial_conv/wo_conv/subtract_last）
├── scripts/ETTh1/            # 新建：newidea 模型 × 4 个 pred_len
└── results/                  # 输出 ETTh1_newidea_comparison.md
```

## 5. 实验矩阵

统一配置：`features=M, enc_in=7, dec_in=7, c_out=7, 标准 12/4/4 月划分, MSE loss, MSE/MAE 指标`

| seq_len | pred_len | 说明 |
|---|---|---|
| 96 | 96 | 短预测 |
| 96 | 192 | 中预测 |
| 96 | 336 | 长预测 |
| 96 | 720 | 超长预测 |

超参初值：`d_model=512, e_layers=2, batch_size=32, lr=1e-4, dropout=0.2, train_epochs=50, patience=10`（综合三个源仓库 ETTh1 配置），训练中可调。

## 6. 执行流程

1. **搭建实验台**：拷贝 FreEformer 脚手架（layers/data_provider/utils/experiments/run.py）→ 补 Times2D 三个 layer 文件 → 改 `exp_basic.py`（注册 NewIdea）→ 补齐 `run.py` 参数
2. **实现组合模型**：编写 `model/newidea.py`，复用三个模型内部模块，按 §3 架构级联；Adapter 类预埋
3. **冒烟测试**：seq_len=96, pred_len=96，跑 1 epoch 训练 + 1 次测试，验证 layers 合并无冲突、维度对齐、loss 下降、CUDA 正常
4. **正式实验**：4 个 pred_len 后台顺序跑，日志写入 `logs/`
5. **汇总结果**：解析日志生成 `results/ETTh1_newidea_comparison.md`（4 行 MSE/MAE）
6. **交付**：结果表 + checkpoint + 与基础模型结果对比（用户已有基础模型结果）

## 7. 成功标准

- 4 个配置全部跑通（训练收敛、测试完成）
- 输出 MSE/MAE 对比表
- 与用户已有的三个基础模型结果对比，给出组合模型有效性分析（供后续迭代）

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| layers 合并版本冲突 | 冒烟测试前置，逐模型 import 验证 |
| 组合模型维度不匹配 | 每一级复用模块单独单测（shape assert） |
| 训练不稳定/不收敛 | RevIN + use_norm 保稳定；超参沿用源仓库；必要时调 lr/dropout |
| 时长较长（4 次训练） | 后台顺序跑，日志监控，失败单配置重跑 |
| Times2D 周期/补丁参数按 seq_len=720 设计（period_list=[720,360,110,96,48], patch_len=[48,32,16,6,3]） | 组合模型用 seq_len=96 时，需在实现中复核 period_len/patch_len 有效性，必要时缩小 period_list/patch_len 到适配 96 的尺度 |

## 9. 后续扩展（本次不做）

- Device Adapter 启用：多设备数据 + 冻结主干微调
- AE/UHF 双模态输入适配
- 分类任务头
