#!/bin/bash
# newidea 组合模型 ETTh1 完整基准实验
# seq_len=96, pred_len ∈ {96, 192, 336, 720}, features=M, MSE/MAE
# 用法: bash scripts/ETTh1/NewIdea_ETTh1.sh
# 说明: 默认使用 PATH 中的 python，跨环境可移植（Kaggle/Linux/Windows Git Bash 均可运行）。
#       如需指定解释器，用环境变量 PYTHON 覆盖，例如本机 conda 环境:
#       PYTHON=/d/Anaconda/envs/times2d/python bash scripts/ETTh1/NewIdea_ETTh1.sh
# 注意: 2026-08-13 修复 — 原 lradj=type1(每 epoch 减半)+train_epochs=50 会把 LR 衰减到 ~1e-16，
#       训练实际在第 ~10 epoch 后死亡（测试损失冻结在基线）。改用 type3(每 5 epoch 减半)+30 epochs。
#       若仍塌缩到"预测均值"基线(MSE≈1.0)，下一步再调大 learning_rate（如 1e-3）或降 dropout。
#       保持 LF 行尾；不要在 Windows 编辑器里以 CRLF 保存（会导致 do\r 语法错误）。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON=${PYTHON:-python}

model_name=NewIdea
root_path=./dataset/ETT-small/
data_path=ETTh1.csv
data_name=ETTh1

mkdir -p logs/NewIdea

seq_len=96

# Times2D 周期/补丁参数（按 seq_len=96 调优）
period_list="48 24 12 6"
patch_len="16 16 8 4"

for pred_len in 96 192 336 720
do
  echo "========== Running ${data_name} seq_len=${seq_len} pred_len=${pred_len} =========="
  ${PYTHON} -u run.py \
    --is_training 1 \
    --model_id ${data_name}_NewIdea_${seq_len}_${pred_len} \
    --model $model_name \
    --data $data_name \
    --root_path $root_path \
    --data_path $data_path \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --d_model 512 --n_heads 8 --e_layers 2 --d_ff 512 \
    --dropout 0.2 --activation gelu \
    --period_list $period_list \
    --patch_len $patch_len \
    --embed_size 1 \
    --use_adapter 0 --n_devices 1 \
    --use_norm 1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --lradj type3 \
    --train_epochs 30 --patience 10 \
    --des 'Exp' \
    --use_gpu 1 --gpu 0 \
    > logs/NewIdea/${model_name}_${data_name}_${seq_len}_${pred_len}.log 2>&1
  echo "---------- finished pred_len=${pred_len}, tail of log: ----------"
  tail -3 logs/NewIdea/${model_name}_${data_name}_${seq_len}_${pred_len}.log
done
echo "========== All done. Summary: =========="
grep -H "mse:" logs/NewIdea/*.log
