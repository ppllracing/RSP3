from .dataset_base import Dataset_Base as D
from ..tools.config import Configuration
from ..tools.util import extract_data_for_path_planning, timefunc

# 定义路径规划的数据集
class Dataset_PathPlanning(D):
    def __init__(self, cfg: Configuration, mode='normal'):
        super().__init__(cfg, mode)

    def calDataset(self):
        cfgs = {
            'num_folder_max': self.cfg.collect['num_folder_max'],
            'num_folder_free': self.cfg.collect['num_folder_free'],
            'dtype_model': self.cfg.dtype_model,
            'is_rsu': self.cfg.is_rsu,
            'path_datas': self.cfg.path_datas,
            'max_num_for_path': self.cfg.max_num_for_path,
            'end_value_for_path_point_token': self.cfg.end_value_for_path_point_token,
            'pad_value_for_path_point_token': self.cfg.pad_value_for_path_point_token,
            'start_value_for_path_point_token': self.cfg.start_value_for_path_point_token
        }
        dataset = extract_data_for_path_planning(self.mode, cfgs)
        return dataset

    # def extract_data_from_datas(self, datas, dataset):
    #     if self.mode == 'free' or datas['path']['success']:
    #         # if self.mode == 'free':
    #         #     datas['path']['success'] = False  # 在自由模式下，路径规划一律当成不成功
    #         return extract_data_for_path_planning_from_datas(datas, self.cfg, dataset)
    #     else:
    #         return dataset