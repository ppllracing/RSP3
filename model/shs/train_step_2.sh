# !/bin/bash

# 串行
# python "3_Codes/1216-E2E/op_model_dl.py" --mode train --current_step 2 --path_ckpt "3_Codes/1216-E2E/ckpts/step_1/rsu/epoch=9.ckpt" --lr_pre_trained "0.001"
# python "3_Codes/1216-E2E/op_model_dl.py" --mode train --current_step 2 --path_ckpt "3_Codes/1216-E2E/ckpts/step_1/rsu/epoch=9.ckpt" --lr_pre_trained "0.0001"
# python "3_Codes/1216-E2E/op_model_dl.py" --mode train --current_step 2 --path_ckpt "3_Codes/1216-E2E/ckpts/step_1/rsu/epoch=9.ckpt" --lr_pre_trained "0.00001"
# python "3_Codes/1216-E2E/op_model_dl.py" --mode train --current_step 2 --path_ckpt "3_Codes/1216-E2E/ckpts/step_1/rsu/epoch=9.ckpt" --lr_pre_trained "0.0"
# python "3_Codes/1216-E2E/op_model_dl.py" --mode train --current_step 2

# 并行
# 手动定义你要用的 GPU 编号（比如 4 张卡：0, 1, 2, 3）
GPUS=(0 1 2 3 4)

# 定义学习率列表
LRS=("0.001" "0.0001" "0.00001" "0.0" "")

for i in "${!LRS[@]}"; do
  GPU=${GPUS[$((i % ${#GPUS[@]}))]}  # 自动轮换分配 GPU
  LR="${LRS[$i]}"

  # 构造命令
  if [ -z "$LR" ]; then
    CMD="CUDA_VISIBLE_DEVICES=$GPU python 3_Codes/1216-E2E/op_model_dl.py --mode train --current_step 2"
  else
    CMD="CUDA_VISIBLE_DEVICES=$GPU python 3_Codes/1216-E2E/op_model_dl.py --mode train --current_step 2 --path_ckpt 3_Codes/1216-E2E/ckpts/step_1/rsu/last.ckpt --lr_pre_trained $LR"
  fi

  echo "[INFO] Launching on GPU $GPU: $CMD"
  eval "$CMD &"
  sleep 60  # 等待一会儿，防止同时启动太多进程
done

wait
echo "All processes completed."