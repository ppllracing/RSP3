import os
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from tqdm import tqdm

from apis.tools.config import Configuration
from apis.tools.util import read_datas_from_disk, save_datas_to_disk, cal_risk_degrees_for_path_points
from apis.agents.for_analyse_in_generator import AgentAnalyseInGenerator

def process_data(datas):
    global_path = datas['global_path']
    global_aim = datas['global_aim']
    global_path['path_points_rear'][0, :2] = global_aim['start_id']
    if global_path['success']:
        global_path['path_points_rear'][-1, :2] = global_aim['end_id']
    datas['global_path'] = global_path
    datas['global_aim'] = global_aim

    sequence = datas['sequence']
    for i in range(len(sequence)):
        _aim = sequence[i]['aim']
        _path = sequence[i]['path']
        _path['path_points_rear'][0, :2] = _aim['start_id']
        if _path['success']:
            _path['path_points_rear'][-1, :2] = _aim['end_id']
        sequence[i]['path'] = _path
        sequence[i]['aim'] = _aim
    datas['sequence'] = sequence
    return datas

def store_dataset(dataset, cfg: Configuration):
    dataset_normal = dataset[:-cfg.collect['num_folder_free']]
    dataset_free = dataset[-cfg.collect['num_folder_free']:]

    # 保存数据集
    save_datas_to_disk(dataset_normal, cfg.path_datas, 'dataset_normal', 'pkl')
    print(f'Save dataset_normal to {os.path.join(cfg.path_datas, "dataset_normal.pkl")}')
    save_datas_to_disk(dataset_free, cfg.path_datas, 'dataset_free', 'pkl')
    print(f'Save dataset_free to {os.path.join(cfg.path_datas, "dataset_free.pkl")}')

    # 计算文件大小
    print(f'Normal dataset size: {os.path.getsize(os.path.join(cfg.path_datas, "dataset_normal.pkl")) / 1024 ** 3:.2f} GB')
    print(f'Free dataset size: {os.path.getsize(os.path.join(cfg.path_datas, "dataset_free.pkl")) / 1024 ** 3:.2f} GB')

    return dataset_normal, dataset_free

if __name__ == '__main__':
    # 获取基本的cfg
    path_cfg = '/'.join(os.path.abspath(__file__).split('/')[:-1] + ['configs', 'config.yaml'])
    cfg = Configuration(path_cfg)
    dtype = cfg.dtype_model
    agent_analyse = AgentAnalyseInGenerator(cfg)

    # 遍历pkl_files
    pkl_files = []
    for i in range(cfg.collect['num_folder_max']):
        pkl_files.append(os.path.join(cfg.path_datas, 'pkl_files', f'Setting_{i}.pkl'))
        assert os.path.exists(pkl_files[-1]), f'Not found {pkl_files[-1]}'

    dataset = []
    for pkl_file in tqdm(pkl_files, desc='Reading pkl files'):
        datas = read_datas_from_disk(path_datas=pkl_file, mode='pkl')
        dataset.append(
            process_data(datas)
        )

    agent_analyse.logger.info('Start to store origin data to dataset.pkl and dataset_free.pkl!')
    dataset_normal, dataset_free = store_dataset(dataset, cfg)

    # 对数据进行分析
    agent_analyse.logger.info('Start to analyse normal dataset!')
    agent_analyse(dataset_normal, os.path.join(cfg.path_datas, 'dataset_normal'))
    agent_analyse.logger.info('Start to analyse free dataset!')
    agent_analyse(dataset_free, os.path.join(cfg.path_datas, 'dataset_free'))
