import math
import carla
import numpy as np
import random

from .for_base import AgentBase
from ..tools.config import Configuration

class AgentParkingPlot(AgentBase):
    def __init__(self, cfg: Configuration):
        super().__init__(cfg)
        self.dtype = cfg.dtype_carla

        self.pp_id = self.cfg.cameras['npc']['pp_id']  # 默认车位id
        self.pp_infos = None  # 每个停车位的信息，包含了id, usable、location、roation和box
        self.pp_vehicle_infos = None  # 已在车位内的车辆的信息
        self.current_pp_id = None  # 选中的车位id

        self.init_all()
    
    def init_all(self):
        paras_parking_plot = self.cfg.parking_plot

        # 为每个车位生成一个info
        self.pp_infos = []
        self.pp_vehicle_infos = []
        for range_row in paras_parking_plot['range']:
            # 生成所有车位的位置
            x, y0, y1, z, pitch, yaw, roll = range_row
            ys = np.linspace(y0, y1, paras_parking_plot['num_plot_per_row'], dtype=self.dtype)
            xs = np.ones_like(ys, dtype=self.dtype) * x
            zs = np.ones_like(ys, dtype=self.dtype) * z
            pitchs = np.ones_like(ys, dtype=self.dtype) * pitch
            yaws = np.ones_like(ys, dtype=self.dtype) * yaw
            rolls = np.ones_like(ys, dtype=self.dtype) * roll
            xyzPYRs = np.stack(
                [
                    xs, ys, zs,
                    pitchs, yaws, rolls
                ], 
                axis=-1
            )
            
            # 生成infos
            _pp_infos = []
            for i, xyzPYR in enumerate(xyzPYRs):
                # 记录当前车位数据
                id_pp = i + len(self.pp_infos)
                self.logger.debug(f'Parking Plot {id_pp} is at {xyzPYR}')
                flag_usable = True
                _pp_infos.append({
                    'id': id_pp,
                    'usable': flag_usable,  # 车位是否可用
                    'xyzPYR': xyzPYR.tolist(),  # 中心坐标
                    'xyzPYR_aim': [  # 停车位坐标
                        xyzPYR[0] - 1.3 * math.cos(xyzPYR[4]),
                        xyzPYR[1],
                        xyzPYR[2],
                        xyzPYR[3],
                        xyzPYR[4],
                        xyzPYR[5]
                    ],
                    'location': carla.Location(*xyzPYR[0:3]), 
                    'rotation': carla.Rotation(*xyzPYR[3:]), 
                    'transform': carla.Transform(
                        carla.Location(*xyzPYR[0:3]), 
                        carla.Rotation(*xyzPYR[3:])
                    ), 
                    'box': carla.BoundingBox(
                        carla.Location(), 
                        carla.Vector3D(2.5, 1.7, 1.0)
                    ),
                    'box_show': carla.BoundingBox(
                        carla.Location(*xyzPYR[0:3]), 
                        carla.Vector3D(2.5, 1.7, 1.0)
                    )
                })

            self.pp_infos.extend(_pp_infos)

        # 选一个车位
        self.reset_selecet_a_new_pp_id()

        super().init_all()

    def reset_pp_vehicle_infos(self):
        self.pp_vehicle_infos.clear()

    def reset_pp_infos_usable(self):
        for info in self.pp_infos:
            info['usable'] = True
    
    def reset_selecet_a_new_pp_id(self, pp_id=None):
        self.current_pp_id = self.get_pp_id_randomly() if pp_id is None else pp_id
        pp_info = self.get_pp_info_from_id(self.current_pp_id)
        assert pp_info['usable'] == True, 'The selected parking plot is not usable.'

    def get_pp_id_randomly(self):
        return random.choice(self.pp_id['optional'])

    def get_pp_info_from_id(self, pp_id=None):
        if pp_id is None:
            # 直接从cfg中获取rsu默认的车位
            pp_id = self.pp_id['coordinate']
        for pp_info in self.pp_infos:
            if pp_info['id'] == pp_id:
                return pp_info
        raise ValueError(f'Parking Plot [{pp_id}] is not found in the parking plot infos.')

    def transform_xyzPYR_from_pp(self, xyzPYR_relative, pp_id=None):
        pp_info = self.get_pp_info_from_id(pp_id)
        xyzPYR_pp = pp_info['xyzPYR']
        
        xyzPYR = [
            xyzPYR_pp[j] + xyzPYR_relative[j] for j in range(6)
        ]

        return xyzPYR
    
