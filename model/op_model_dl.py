import os
import torch
torch.set_float32_matmul_precision('medium')
import lightning as L
L.seed_everything(2025)
import argparse

from apis.tools.config import Configuration
from apis.agents.for_model_dl import AgentModelDL

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':
    # 获取变动的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--current_step', type=int, default=1, help='1 or 2')
    parser.add_argument('--path_ckpt', type=str, default=None, help='path of ckpt')
    parser.add_argument('--lr_pre_trained', type=float, default=None, help='lr of pre-trained')
    # parser.add_argument('--use_heuristic', type=int, default=None, help='use heuristic')
    # parser.add_argument('--use_risk_assessment', type=int, default=None, help='use risk assessment')
    parser.add_argument('--device_num', type=int, default=1, help='1, 2, ...')
    args = parser.parse_args()

    # 获取基本的cfg
    path_cfg = '/'.join(os.path.abspath(__file__).split('/')[:-1] + ['configs', 'config.yaml'])
    cfg = Configuration(path_cfg)

    # 判断参数合法性，并根据此修改cfg
    current_step = args.current_step
    cfg.current_step = current_step
    cfg.device_num = args.device_num
    path_ckpt = None if args.path_ckpt is None or args.path_ckpt.lower() == 'none' else args.path_ckpt
    if args.mode == 'train':
        if current_step == 1:
            # 当前是step_1，不应该有预训练模型
            assert path_ckpt is None, 'Please set path_ckpt to None when training step 1.'
            # 不需要设置use_heuristic和use_risk_assessment
            # assert args.use_heuristic is None, 'Please set use_heuristic to None when training step 1.'
            # assert args.use_risk_assessment is None, 'Please set use_risk_assessment to None when training step 1.'
            # cfg.use_heuristic = args.use_heuristic
            # cfg.use_risk_assessment = args.use_risk_assessment
        else:
            # # 当前是step_2或step_3或之后的，需要预训练模型
            # assert path_ckpt is not None, 'Please set path_ckpt to a valid path when training other step.'
            # # 需要设置use_heuristic和use_risk_assessment
            # assert args.use_heuristic is not None, 'Please set use_heuristic to a valid value when training other step.'
            # assert args.use_risk_assessment is not None, 'Please set use_risk_assessment to a valid value when training other step.'
            last_step = f'step_{cfg.current_step - 1}'
            cfg.ckpts[last_step] = path_ckpt
            cfg.train['lr_pre_trained'] = args.lr_pre_trained
            # cfg.use_heuristic = bool(args.use_heuristic)
            # cfg.use_risk_assessment = bool(args.use_risk_assessment)
    else:
        assert path_ckpt is not None, 'Please set path_ckpt to a valid path when testing.'
        # assert cfg.use_heuristic is None, 'Please set use_heuristic to None when testing.'
        # assert cfg.use_risk_assessment is None, 'Please set use_risk_assessment to None when testing.'

    # 跑模型
    agent_model_dl = AgentModelDL(cfg, flag_train=args.mode=='train')
    agent_model_dl.run()
