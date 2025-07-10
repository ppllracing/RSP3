from .dataset_base import Dataset_Base as D
from ..tools.config import Configuration
from ..tools.util import extract_data_for_perception, timefunc

# 定义语义分割数据集
class Dataset_Perception(D):
    def __init__(self, cfg: Configuration, mode='normal'):
        super().__init__(cfg, mode='normal')
    
    def calDataset(self):
        cfgs = {
            'num_folder_max': self.cfg.collect['num_folder_max'],
            'num_folder_free': self.cfg.collect['num_folder_free'],
            'dtype_model': self.cfg.dtype_model,
            'is_rsu': self.cfg.is_rsu,
            'path_datas': self.cfg.path_datas
        }
        dataset = extract_data_for_perception(self.mode, cfgs)
        return dataset
