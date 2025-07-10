import os
import torch
torch.set_float32_matmul_precision('medium')
import lightning as L
L.seed_everything(2025)
import argparse
import json
from tqdm import tqdm

from apis.agents.for_carla import AgentCarla
from apis.agents.for_model_test import AgentModelTest
from apis.agents.for_condition_setting import AgentConditionSetting
from apis.agents.for_parking_plot import AgentParkingPlot
from apis.agents.for_map import AgentMap
from apis.agents.for_planner import AgentPlanner
from apis.agents.for_get_datas_from_carla import AgentGetDatasFromCarla
from apis.tools.util import read_datas_from_disk, init_logger
from apis.tools.config import Configuration
from apis.agents.for_close_loop import (
    AgentCloseLoopOurModel, 
    # AgentCloseLoopE2EParkingCARLA
)

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == '__main__':
    # 获取变动的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ckpt', type=str, help='path of ckpt')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:0 or cuda:1 or ...')
    parser.add_argument('--model_name', action='append', help='model tested by close-loop')
    args = parser.parse_args()

    # 获取基本的cfg
    path_cfg = '/'.join(os.path.abspath(__file__).split('/')[:-1] + ['configs', 'config.yaml'])
    cfg = Configuration(path_cfg)

    # 初始化agent
    agent_model_test = AgentModelTest('pathplanning', args.path_ckpt, device=args.device)
    # cfg = agent_model_test.cfg  # 用模型中的cfg更新
    agent_parking_plot = AgentParkingPlot(cfg)
    agent_map = AgentMap(cfg)
    agent_map.init_from_seleted_parking_plot(agent_parking_plot)
    agent_carla = AgentCarla(
        cfg, cfg.carla_client,
        agent_map=agent_map,
        agent_parking_plot=agent_parking_plot
    )
    agent_planner = AgentPlanner(
        cfg,
        agent_map.bev_points[0, 0, 0, :2], 
        agent_carla.actors_dict['vehicle']['ego']['base_params']
    )
    agent_condition_setting = AgentConditionSetting(cfg)
    agent_get_datas_from_carla = AgentGetDatasFromCarla(
        cfg, 
        {'carla': agent_carla, 'parking_plot': agent_parking_plot, 'map': agent_map, 'planner': agent_planner}
    )

    # 初始化logger
    logger  = init_logger(cfg,  'TestModelCloseLoop')

    # 获取测试用的模型
    model_our = agent_model_test.get_model()

    # 设定闭环测试的agents
    _agents = {
        'Our': AgentCloseLoopOurModel(cfg, model_our, agent_carla, agent_map, agent_get_datas_from_carla),
        # 'E2EParkingCARLA': AgentCloseLoopE2EParkingCARLA(cfg, agent_carla, agent_map, agent_get_datas_from_carla)
    }
    agents_close_loop = {name: _agents[name] for name in args.model_name}

    # 加载完整数据集，并从完整数据集中获取路径规划失败的数据
    dataset = read_datas_from_disk(agent_model_test.cfg.path_dataset, 'dataset', 'pkl')
    setting_ids = []
    for datas in dataset:
        if not datas['global_path']['success']:
            setting_ids.append(datas['setting_id'])
    # setting_ids = setting_ids[:5]
    logger.info(f'There are {len(setting_ids)} settings with path planning failed.')

    # 准备好文件夹
    path_results_close_loop = os.path.join(cfg.path_results, 'close_loop')
    os.makedirs(path_results_close_loop, exist_ok=True)
    for name in args.model_name:
        os.makedirs(os.path.join(path_results_close_loop, name), exist_ok=True)

    # 开始测试
    logger.info('Start testing...')
    infos_model = {name: [] for name in args.model_name}
    for setting_id in tqdm(setting_ids, desc='Setting', leave=False):
        setting = agent_condition_setting.get_setting_by_id(setting_id)

        # 遍历要进行闭环测试的模型
        for name in infos_model:
            # 重置环境
            agent_condition_setting.set_agents_from_setting(
                setting,
                agent_carla=agent_carla,
                agent_parking_plot=agent_parking_plot
            )
            
            # 从carla中获取数据集
            datas = agent_get_datas_from_carla.get_datas(
                datas_stamp=True, datas_camera=True, datas_vehicle=True, 
                datas_parking_plot=True, datas_bev=True, datas_aim=True, 
                # datas_path=True, is_global_path=True
            )
            # assert not datas['path']['success'], 'The global path is not success.'
            datas = {
                'stamp': agent_get_datas_from_carla.trans_for_stamp(datas['stamp'], datas['stamp']),
                'camera': agent_get_datas_from_carla.trans_for_datas_camera(datas['camera'], datas['stamp']),
                'bev': agent_get_datas_from_carla.trans_for_datas_bev(datas['bev']),
                'aim': agent_get_datas_from_carla.trans_for_datas_aim(datas['aim'])
            }

            infos_ = agents_close_loop[name].run(
                datas,
                os.path.join(path_results_close_loop, name, str(setting_id))
            )
            infos_disp = {'flags': infos_['flags'], 'errors': infos_['errors']}
            logger.debug(f'Infos of {name} on setting {setting_id}: {json.dumps(infos_disp, indent=4)}')
            infos_model[name].append(infos_)
    logger.info('Testing finished.')

    # 结果分析
    logger.info('Start analyzing...')
    for name in args.model_name:
        logger.info(f'Analyzing [{name}]...')
        infos_list = infos_model[name]

        flags = None
        errors = None
        for infos in infos_list:
            # 规整flag
            flags_= infos['flags']
            if flags is None:
                flags = {k: [] for k in flags_}
            for k in flags:
                flags[k].append(flags_[k])
            
            # 规整error
            errors_= infos['errors']
            if errors is None:
                errors = {k: [] for k in errors_}
            for k in errors:
                errors[k].append(errors_[k])

        # 展示结果
        logger.info('Flags:')
        for k in flags:
            if k == 'plan':
                logger.info(f'\t{k:-<15}: {sum(flags[k])} / {len(flags[k])} = {sum(flags[k]) / len(flags[k])*100:.2f}%')
            else:
                logger.info(f"\t{k:-<15}: {sum(flags[k])} / {sum(flags['plan'])} = {sum(flags[k]) / sum(flags['plan'])*100:.2f}%")
        logger.info('Errors:')
        for k in errors:
            errors_ = [e for f, e in zip(flags['plan'], errors[k]) if f]
            logger.info(f'\t{k:-<15}: [{min(errors_):>7.2f}, {sum(errors_) / len(errors_):>7.2f}, {max(errors_):>7.2f}]')
    logger.info('Analyzing finished.')