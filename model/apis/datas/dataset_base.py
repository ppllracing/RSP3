import random
from tqdm import tqdm
from torch.utils.data import Dataset as D

from ..tools.config import Configuration
from ..tools.util import timefunc, init_logger

# 定义路径规划的数据集
class Dataset_Base(D):
    def __init__(self, cfg: Configuration, mode='normal'):
        assert mode in ['normal', 'free'], 'Unsupported mode'
        self.mode = mode
        self.cfg = cfg
        self.logger = init_logger(cfg, self.__class__.__name__)
        duration, self.dataset = timefunc(self.calDataset)
        self.logger.info(f'{self.__class__.__name__} loaded, time cost: {duration:.2f}s')
        self.logger.info(f'{self.__class__.__name__} origin size: {len(next(iter(self.dataset.values())))}')
    
    def calDataset(self):
        # 这个函数一定要被子类实现
        assert False, 'Not implemented'
        # dataset = None

        # # 从pkl中加载原始数据
        # if self.mode == 'normal':
        #     datas_folders = read_datas_from_disk(self.cfg.path_datas, 'dataset_normal', 'pkl')
        #     assert len(datas_folders) == (self.cfg.collect['num_folder_max'] - self.cfg.collect['num_folder_free']), 'Dataset not enough, please check your data collection.'
        # else:
        #     datas_folders = read_datas_from_disk(self.cfg.path_datas, 'dataset_free', 'pkl')
        #     assert len(datas_folders) == self.cfg.collect['num_folder_free'], 'Dataset not enough, please check your data collection.'
        
        # # 反方向遍历所有数据
        # for i in tqdm(
        #     range(len(datas_folders) - 1, -1, -1), 
        #     desc=self.__class__.__name__, unit='folder', 
        #     leave=False
        # ):
        #     datas_all = datas_folders[i]
        #     datas_seq = datas_all['sequence']

        #     # 对于每一个sequence，提取数据
        #     for datas in tqdm(
        #         datas_seq,
        #         desc='Sequence', unit='frame', 
        #         leave=False
        #     ):
        #         # 将提取的数据记入dataset
        #         dataset = self.extract_data_from_datas(datas, dataset)

        #     # 删除加载过的数据
        #     del datas_folders[i]
        # return dataset
    
    # def extract_data_from_datas(self, datas, dataset):
    #     # 这个函数一定要被子类实现
    #     assert False, 'Not implemented'

    def __len__(self):
        return len(self.dataset['stamp'])
    
    def __getitem__(self, index):
        return {
            key: val[index] for key, val in self.dataset.items()
        }