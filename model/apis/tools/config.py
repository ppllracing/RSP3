import os
import torch
import yaml
import threading
import pprint

camera_lock = threading.Lock()

class Configuration:
    def __init__(self, path_cfg=None, *args, **kwargs):
        if path_cfg is None:
            path_cfg = os.path.join(os.path.join(*(['/'] + os.path.abspath(__file__).split('/')[:-3] + ['configs', 'config.yaml'])))
        with open(path_cfg, 'r', encoding='utf-8') as file_cfg:
            dict_cfg = yaml.safe_load(file_cfg)['cfg']
        
        self.dtype_carla = dict_cfg['dtype_carla']
        self.dtype_model = dict_cfg['dtype_model']
        if self.dtype_model == 'float16':
            self.dtype_model_torch = torch.float16
        elif self.dtype_model == 'float32':
            self.dtype_model_torch = torch.float32
        elif self.dtype_model == 'float64':
            self.dtype_model_torch = torch.float64
        else:
            assert False, 'Dtype Error!'
        self.device = dict_cfg['device']
        self.logger_level = dict_cfg['logger_level']
        if self.logger_level == 'DEBUG':
            self.logger_level = 10
        elif self.logger_level == 'INFO':
            self.logger_level = 20
        elif self.logger_level == 'WARN':
            self.logger_level = 30
        elif self.logger_level == 'ERROR':
            self.logger_level = 40
        else:
            assert False, 'LEVEL Error!'
        self.fps = dict_cfg['fps']
        self.device_num = dict_cfg['device_num']
        if self.device_num == 'multi':
            self.device_num = torch.cuda.device_count()
        self.current_step = dict_cfg['current_step']

        self.collect = dict_cfg['for_collect']
        self.train = dict_cfg['for_train']
        self.valid = dict_cfg['for_valid']
        self.test = dict_cfg['for_test']

        self.carla_client = dict_cfg['carla_client']

        paths = dict_cfg['paths']
        self.path_logs = '/'.join(os.path.abspath(__file__).split('/')[:-3] + [paths['logs']])
        if not os.path.exists(self.path_logs):
            os.makedirs(self.path_logs, exist_ok=True)
        self.path_ckpts = '/'.join(os.path.abspath(__file__).split('/')[:-3] + [paths['ckpts']])
        if not os.path.exists(self.path_ckpts):
            os.makedirs(self.path_ckpts, exist_ok=True)
        self.path_datas = '/'.join(os.path.abspath(__file__).split('/')[:-3] + [paths['datas']])
        if not os.path.exists(self.path_datas):
            os.makedirs(self.path_datas, exist_ok=True)
        self.path_condition_settings = os.path.join(self.path_datas, 'condition_settings.pkl')
        self.path_dataset = os.path.join(self.path_datas, 'dataset.pkl')
        self.path_results = '/'.join(os.path.abspath(__file__).split('/')[:-3] + [paths['results']])
        if not os.path.exists(self.path_results):
            os.makedirs(self.path_results, exist_ok=True)

        self.cameras = dict_cfg['cameras']
        self.ego = dict_cfg['ego']
        self.map_bev = dict_cfg['map_bev']
        self.parking_plot = dict_cfg['parking_plot']

        # 补充一些参数，用于后续方便使用
        cal_range_local = lambda resolution, final_dim: [-(final_dim / 2 - 0.5) * resolution, (final_dim / 2 - 0.5) * resolution]
        self.map_bev['x_range_local'] = cal_range_local(self.map_bev['resolution'][0], self.map_bev['final_dim'][0])
        self.map_bev['y_range_local'] = cal_range_local(self.map_bev['resolution'][1], self.map_bev['final_dim'][1])
        self.map_bev['z_range_local'] = cal_range_local(self.map_bev['resolution'][2], self.map_bev['final_dim'][2])

        Path = dict_cfg['Path']
        self.max_num_for_path = Path['max_num']
        self.start_value_for_path_point_token = Path['start_value']
        self.pad_value_for_path_point_token = Path['pad_value']
        self.end_value_for_path_point_token = Path['end_value']

        model = dict_cfg['model']
        # self.use_heuristic = model['use_heuristic']
        # self.use_risk_assessment = model['use_risk_assessment']
        self.is_rsu = model['is_rsu']
        self.bev_model_params = model['BEV_Model']
        self.bev_encoder_params = model['BEV_Encoder']
        self.feature_fusion_params = model['Feature_Fusion']
        self.segmentation_head_params = model['Segmentation_Head']
        self.path_planning_head_params = model['Path_Planning_Head']
        # self.risk_assessment_head_params = model['Risk_Assessment_Head']
        # self.orientation_head_params = model['Orientation_Head']

        self.ckpts = dict_cfg['ckpts']
        self.test_case = dict_cfg['test_case']

    def __str__(self):
        return pprint.pformat(self.__dict__)