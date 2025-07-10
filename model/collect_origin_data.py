import multiprocessing
import sys
import shutil
import time
import os
import lightning as L
L.seed_everything(2025)
from tqdm import tqdm

from apis.tools.config import Configuration
from apis.tools.util import save_datas_to_disk, read_datas_from_disk
from apis.datas.collector import Collector
from apis.agents.for_analyse_in_generator import AgentAnalyseInGenerator

def step_1(c: Collector):
    path_pkl_files = os.path.join(c.path_datas, 'pkl_files')
    if not os.path.exists(path_pkl_files):
        os.makedirs(path_pkl_files)

    datas_paths = []
    # 限定生成数据的数量
    for folder_id in tqdm(
        range(c.num_folder_max), total=c.num_folder_max,
        desc='Step-1', unit='folder',
        leave=False
    ):
        flag_collect = False
        # 判断当前condition对应的数据在不在
        if not os.path.exists(os.path.join(c.path_datas, f'Setting_{folder_id}')):
            flag_collect = True
            # 获取当前setting
            setting = c.agent_condition_setting.get_setting_by_id(folder_id)

            def try_collect_data(setting, mode_change_ego_pose):
                # 初始化所有变量
                c.reset(setting)

                # 在生成每一个folder时候的进度条
                tbar_current = tqdm(
                    initial=0, total=1000, desc='', unit='step', leave=False
                )

                # 根据初始状态进行路径规划
                tbar_current.set_description('Planning')
                c.datas['global_aim'], c.datas['global_path'], c.datas['parking_plot'] = c.global_path_planning()
                tbar_current.update(int(0.1 * tbar_current.total))

                # 开始循迹，并存储部分数据
                # 采集完数据就是完成一半
                tbar_current.set_description('Collecting')
                c.collect_origin_data(tbar_current, mode_change_ego_pose)
                tbar_current.set_postfix({'Seq-Full': len(c.datas['stamp'])})
                tbar_current.update(int(tbar_current.total / 2 - tbar_current.n))
                return tbar_current

            # 尝试在当前setting下采集数据，通过两种方式去循迹
            modes = ['place_directly', 'place_directly']
            for i in range(len(modes)):
                tbar_current = try_collect_data(setting, modes[i])
                if len(c.datas['stamp']) < len(c.datas['global_path']['path_points_rear']) and i < len(modes) - 1:
                    # 当前采集的数据量不足，大概率是循迹的时候出现了bug，
                    # 要么是发生碰撞了，要么是没跟上某个点，重新来一次
                    c.logger.debug(f'Recollecting data in setting_{folder_id}!')
                    tbar_current.close()
                else:
                    # 采集的数据量足够，可以进行后续的处理
                    break

            # 生成耗时的伴生数据
            tbar_current.set_description('Generating')
            c.generate_associated_data()
            tbar_current.update(int(0.1 * tbar_current.total))

            # 检测数据的时序是否对的上
            assert all(len(c.datas[k1]) == len(c.datas[k2]) for k1, k2 in zip(list(c.datas.keys())[4:-1], list(c.datas.keys())[5:])), 'Sequence is not aligned!'

            # 根据时间戳生成指定文件夹
            tbar_current.set_description('Create Folder')
            folder_name, path_datas_folder = c.create_folder(f'Setting_{folder_id}')
            tbar_current.set_postfix({'Folder': folder_name})
            tbar_current.update(int(0.1 * tbar_current.total))

            # 将datas内部数据进行转换
            tbar_current.set_description('Transforming')
            datas = c.transform_origin_datas(path_datas_folder, folder_id)
            tbar_current.update(int(0.1 * tbar_current.total))

            # 将数据保存，并保存可视化数据用于后期查看
            tbar_current.set_description('Saving')
            process0 = multiprocessing.Process(target=save_datas_to_disk, args=(datas, path_datas_folder, 'datas', 'pkl'))
            process0.start()
            process1 = multiprocessing.Process(target=save_datas_to_disk, args=(datas, path_pkl_files, f'Setting_{i}', 'pkl'))
            process1.start()
            save_datas_to_disk(setting.data, path_datas_folder,'setting', 'json')
            c.show_save_datas(path_datas_folder, flag_show=False, flag_save=c.flag_save)
            tbar_current.update(int(0.1 * tbar_current.total))
            process0.join()
            process1.join()
            tbar_current.update(int(0.1 * tbar_current.total))

            assert tbar_current.n == tbar_current.total, 'tbar_current is not completed!'
            tbar_current.close()

        # 判断在不在path_pkl_files内
        path_aim = os.path.join(path_pkl_files, f'Setting_{folder_id}.pkl')
        assert not (flag_collect and os.path.exists(path_aim)), f'There is a history data in {path_pkl_files} for Setting_{folder_id}'
        if not os.path.exists(path_aim):
            # 把这个数据复制过来
            path_source = os.path.join(c.path_datas, f'Setting_{folder_id}', 'datas.pkl')
            shutil.copy(path_source, path_aim)
        c.logger.debug(f'Setting_{folder_id} is already collected!')
        datas_paths.append(path_aim)

    return datas_paths

def step_2(c: Collector, datas_paths):
    # 分块
    assert len(datas_paths) == c.num_folder_max, 'Dataset is not complete!'
    dataset_normal = datas_paths[:-cfg.collect['num_folder_free']]
    dataset_free = datas_paths[-cfg.collect['num_folder_free']:]

    for i in tqdm(
        range(len(dataset_normal)),
        desc='Loading Datas for Normal', leave=False
    ):
        path_ = dataset_normal[i]
        datas = read_datas_from_disk(path_datas=path_, mode='pkl')
        dataset_normal[i] = datas
    for i in tqdm(
        range(len(dataset_free)),
        desc='Loading Datas for Free', leave=False
    ):
        path_ = dataset_free[i]
        datas = read_datas_from_disk(path_datas=path_, mode='pkl')
        dataset_free[i] = datas

    # 保存数据集
    save_datas_to_disk(dataset_normal, cfg.path_datas, 'dataset_normal', 'pkl')
    c.logger.info(f'Save dataset_normal to {os.path.join(cfg.path_datas, "dataset_normal.pkl")}')
    save_datas_to_disk(dataset_free, cfg.path_datas, 'dataset_free', 'pkl')
    c.logger.info(f'Save dataset_free to {os.path.join(cfg.path_datas, "dataset_free.pkl")}')

    # 计算文件大小
    c.logger.info(f'Normal dataset size: {os.path.getsize(os.path.join(cfg.path_datas, "dataset_normal.pkl")) / 1024 ** 3:.2f} GB')
    c.logger.info(f'Free dataset size: {os.path.getsize(os.path.join(cfg.path_datas, "dataset_free.pkl")) / 1024 ** 3:.2f} GB')

    return dataset_normal, dataset_free

if __name__ == '__main__':
    path_cfg = '/'.join(os.path.abspath(__file__).split('/')[:-1] + ['configs', 'config.yaml'])
    cfg = Configuration(path_cfg)
    c = Collector(cfg)
    analyse = AgentAnalyseInGenerator(cfg)

    c.logger.info('Start to collect origin data from CARLA!')
    datas_paths = step_1(c)
    
    c.logger.info('Start to store origin data to dataset.pkl and dataset_free.pkl!')
    dataset_normal, dataset_free = step_2(c, datas_paths)

    # 对数据进行分析
    c.logger.info('Start to analyse normal dataset!')
    analyse(dataset_normal, os.path.join(cfg.path_datas, 'dataset_normal'))
    c.logger.info('Start to analyse free dataset!')
    analyse(dataset_free, os.path.join(cfg.path_datas, 'dataset_free'))
