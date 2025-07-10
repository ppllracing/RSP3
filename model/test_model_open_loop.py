import os
import torch
torch.set_float32_matmul_precision('medium')
import lightning as L
L.seed_everything(2025)
import argparse

from apis.agents.for_model_test import AgentModelTest

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == '__main__':
    # 获取变动的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='perception', help='perception or pathplanning')
    parser.add_argument('--path_ckpt', type=str, help='path of ckpt')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:0 or cuda:1 or ...')
    args = parser.parse_args()

    # 初始化agent
    agent_model_test = AgentModelTest(args.model_type, args.path_ckpt, device=args.device)

    # 开始测试
    agent_model_test.run()
