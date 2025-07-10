import os
import torch
torch.set_float32_matmul_precision('medium')
import lightning as L
L.seed_everything(2025)
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from apis.tools.config import Configuration
from apis.agents.for_carla import AgentCarla
from apis.agents.for_planner import AgentPlanner
from apis.agents.for_get_datas_from_carla import AgentGetDatasFromCarla
from apis.agents.for_condition_setting import AgentConditionSetting

def get_images_from_setting_id(setting_id):
    # 读取某个condition
    condition_setting = agent_condition_setting.get_setting_by_id(setting_id)
    agent_condition_setting.set_agents_from_setting(
        condition_setting,
        agent_carla=agent_carla,
        agent_parking_plot=agent_carla.agent_parking_plot
    )

    # 获取当前的所有图像
    datas_camera = agent_get_datas_from_carla.datas_camera_once()
    return datas_camera

def draw_and_save(datas_camera, save_path):
    # 提取NPC的图像
    cam_npc = datas_camera['npc']['image'].transpose(1, 2, 0)
    # 提取rsu的图像
    cam_rsu = datas_camera['rsu_rgb']['image'].transpose(1, 2, 0)
    # 提取obu的图像
    cam_obu_front = datas_camera['obu_front_rgb']['image'].transpose(1, 2, 0)
    cam_obu_left = datas_camera['obu_left_rgb']['image'].transpose(1, 2, 0)
    cam_obu_right = datas_camera['obu_right_rgb']['image'].transpose(1, 2, 0)
    cam_obu_rear = datas_camera['obu_rear_rgb']['image'].transpose(1, 2, 0)

    # 绘图
    fig = plt.figure()
    plt.subplot(3, 2, 1)
    plt.imshow(cam_npc)
    plt.title('NPC')
    plt.subplot(3, 2, 2)
    plt.imshow(cam_rsu)
    plt.title('RSU')
    plt.subplot(3, 2, 3)
    plt.imshow(cam_obu_front)
    plt.title('OBU_FRONT')
    plt.subplot(3, 2, 4)
    plt.imshow(cam_obu_left)
    plt.title('OBU_LEFT')
    plt.subplot(3, 2, 5)
    plt.imshow(cam_obu_right)
    plt.title('OBU_RIGHT')
    plt.subplot(3, 2, 6)
    plt.imshow(cam_obu_rear)
    plt.title('OBU_REAR')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == '__main__':
    # 获取基本的cfg
    path_cfg = '/'.join(os.path.abspath(__file__).split('/')[:-1] + ['configs', 'config.yaml'])
    cfg = Configuration(path_cfg)

    agent_carla = AgentCarla(cfg)
    agent_planner = AgentPlanner(cfg, agent_carla.agent_map.get_origin_xy(), cfg.ego['base_params'])
    agent_get_datas_from_carla = AgentGetDatasFromCarla(
        cfg=cfg,
        agents={
            'carla': agent_carla,
            'parking_plot': agent_carla.agent_parking_plot,
            'map': agent_carla.agent_map,
            'planner': agent_planner
        }
    )
    agent_condition_setting = AgentConditionSetting(cfg)
    
    path_folder_compare_perception = os.path.join(cfg.path_results, 'compare_perception')
    os.makedirs(path_folder_compare_perception, exist_ok=True)
    for i in tqdm(range(len(agent_condition_setting))):
        # 获取某个condition的图像
        datas_camera = get_images_from_setting_id(i)
        # 绘图并保存
        save_path = os.path.join(path_folder_compare_perception, f'condition_{i}.png')
        draw_and_save(datas_camera, save_path)
