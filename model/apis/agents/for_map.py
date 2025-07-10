import itertools
import numpy as np
import carla
import matplotlib.pyplot as plt

from .for_base import AgentBase
from .for_parking_plot import AgentParkingPlot
from ..tools.config import Configuration
from ..tools.util import init_logger, select_bev_points

class AgentMap(AgentBase):
    def __init__(self, cfg: Configuration, *args, **kwargs):
        super().__init__(cfg)
        self.dtype = cfg.dtype_carla

        self.bev_center = None  # 中心点的全局坐标
        self.bev_range_local = None  # map_bev的xyz边界（局部）
        self.bev_range_global = None  # map_bev的xyz边界（全局）
        self.bev_resolution = None  # bev_point在xyz方向上的分辨率，其中xy使用于map_bev
        self.bev_dimension = None  # bev_point和bev_point_bboxs在xyz方向上的数量，其中xy使用于map_bev
        self.map_bev = None  # map_bev本身，[channnel, height, width]
        self.bev_channel_name = None  # map_bev每个channel的名字
        self.bev_points = None  # 每个方格的中心点坐标
        self.bev_infos = None  # 每个方格的信息，包含了location、roation和box
        self.bev_map_info = None  # 整个map_bev的信息，包含了location、rotation和box
        self.bev_area_plot = None  # 停车位的位置

    def init_from_seleted_parking_plot(self, agent_pp: AgentParkingPlot):
        paras_map_bev = self.cfg.map_bev
        pp_info = agent_pp.get_pp_info_from_id(agent_pp.current_pp_id)

        assert pp_info['usable'], 'The selected parking plot is not usable'
        map_bev_xyzPYR = agent_pp.transform_xyzPYR_from_pp(paras_map_bev['xyzPYR_relative'])
        map_bev_location = carla.Location(*map_bev_xyzPYR[:3])
        map_bev_rotation = carla.Rotation(*map_bev_xyzPYR[3:])

        self.bev_center = np.array([*map_bev_xyzPYR[:3]], dtype=self.dtype)
        self.bev_range_local = np.array([
            paras_map_bev['x_range_local'], 
            paras_map_bev['y_range_local'], 
            paras_map_bev['z_range_local']
        ], dtype=self.dtype)
        self.bev_range_global = self.bev_range_local + self.bev_center.reshape(3, 1)
        self.bev_resolution = np.array(paras_map_bev['resolution'], dtype=self.dtype)
        self.bev_dimension = paras_map_bev['final_dim']
        self.map_bev = np.zeros([paras_map_bev['num_channel'], *self.bev_dimension[:2]], dtype=self.dtype)
        self.bev_channel_name = paras_map_bev['channel_name']

        # 生成整体的bbox
        self.bev_map_info = {
            'location': map_bev_location, 
            'rotation': map_bev_rotation, 
            'transform': carla.Transform(map_bev_location, map_bev_rotation),
            'box': carla.BoundingBox(
                carla.Location(), 
                carla.Vector3D(*((self.bev_range_local[:, 1] - self.bev_range_local[:, 0]) / 2))
            ),
            'box_show': carla.BoundingBox(
                map_bev_location, 
                carla.Vector3D(*((self.bev_range_local[:, 1] - self.bev_range_local[:, 0]) / 2))
            )
        }

        # 计算每个方格的中心点坐标
        xyzs_local = [
            np.flip(np.linspace(*self.bev_range_local[0, :2], self.bev_dimension[0], dtype=self.dtype)).tolist(), # map_bev是从左上角算起的，所以x要反序
            np.linspace(*self.bev_range_local[1, :2], self.bev_dimension[1], dtype=self.dtype).tolist(),
            np.linspace(*self.bev_range_local[2, :2], self.bev_dimension[2], dtype=self.dtype).tolist()
        ]
        xyzs_global = [xyz_local + self.bev_center for xyz_local in itertools.product(*xyzs_local)]
        self.bev_points = np.stack(xyzs_global).reshape(*self.bev_dimension, -1)
        
        # 每个方格生成一个bbox
        infos = []
        for xyz in xyzs_global:
            infos.append({
                'location': carla.Location(*xyz), 
                'rotation': map_bev_rotation,
                'transform': carla.Transform(carla.Location(*xyz), map_bev_rotation),
                'box': carla.BoundingBox(
                    carla.Location(), 
                    carla.Vector3D(*(self.bev_resolution / 2))
                ),
                'box_show': carla.BoundingBox(
                    carla.Location(*xyz), 
                    carla.Vector3D(*(self.bev_resolution / 2))
                )
            })
        self.bev_point_infos = np.array(infos).reshape(*self.bev_dimension)

        # 找出Free Space和Obstacle Space
        # 因为没有障碍物，所以self.map_bev全是0
        assert (self.map_bev == 0.0).all(), 'Obstacle Space should be empty'
        self.map_bev[0] = 1.0 - self.map_bev[1]  # 剩余的空间为Free Space

        # 绘制自车的位置
        # 由于在初始化阶段并不知道自车的位置，所以这里暂时不绘制自车的位置

        # 绘制停车位的位置，但其实没必要
        # self.map_bev[3] = select_bev_points(np.zeros_like(self.map_bev[3]), self.bev_point_infos, pp_info)

        # self.show_map_bev(self.map_bev)

        assert self.check_map_bev(self.map_bev), 'Map-BEV is illegal'

        super().init_all()

    def show_map_bev(self, map_bev):
        fig = plt.figure(figsize=(10, 5))
        for i in range(len(self.bev_channel_name)):
            plt.subplot(1, len(self.bev_channel_name), i+1) 
            plt.imshow(map_bev[i])
            plt.title(self.bev_channel_name[i])
        plt.pause(0.1)
        plt.close(fig)

    def check_map_bev(self, map_bev):
        if not np.min(map_bev) == 0.0:
            return False
        if not np.max(map_bev) == 1.0:
            return False
        if not (map_bev[0:3].sum(axis=0) == 1.0).all():
            # 只检测前三个channel，因为第四个channel是车位占据的，无论车位内是否有车，都要被检测出来
            return False
        return True

    def get_vertices_of_global(self):
        # 返回map_bev的四个角点
        vertices = [
            [self.bev_range_global[0, 1], self.bev_range_global[1, 0], 0],
            [self.bev_range_global[0, 1], self.bev_range_global[1, 1], 0],
            [self.bev_range_global[0, 0], self.bev_range_global[1, 1], 0],
            [self.bev_range_global[0, 0], self.bev_range_global[1, 0], 0]
        ]
        return vertices

    def get_origin_xy(self):
        return self.bev_points[0, 0, 0, :2]
