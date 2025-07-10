import os
import torch
torch.set_float32_matmul_precision('medium')
import lightning as L
L.seed_everything(2025)
import argparse

from apis.agents.for_model_compare import AgentModelCompare

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == '__main__':
    # 获取变动的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths_ckpt_pd', type=str, nargs='+', help='path of ckpt for pathplanning on PD')
    parser.add_argument('--path_ckpt_pp', type=str, help='path of ckpt for pathplanning on PP')
    parser.add_argument('--path_ckpt_obu', type=str, help='path of ckpt for obu perception')
    args = parser.parse_args()

    # 初始化agent
    agent_model_compare = AgentModelCompare({
        'pd': args.paths_ckpt_pd,
        'pp': args.path_ckpt_pp,
        'obu': args.path_ckpt_obu
    })

    # 开始测试
    agent_model_compare.run()
