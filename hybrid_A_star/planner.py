import os
import argparse
import math
import numpy as np
import pandas as pd

from map.init_from_map_bev import Init_From_Map_BEV
from solution_test.plt_np import oup_new_datas_np
from path_plan import path_planner
from animation.animation import ploter, plt
from animation.record_solution import DataRecorder
from animation.record_solution_split import DataRecorder_Split
from animation.curve_plot import CurvePloter
from map import costmap_for_bev
from velocity_plan import velocity_planner
from interpolation import path_interpolation
from optimization import path_optimazition, ocp_optimization
from config import read_config
from tqdm import tqdm
from scipy.interpolate import CubicSpline, CubicHermiteSpline, interp1d

class Planner:
    def __init__(self, discrete, vehicle_params, dtype):
        self.config = read_config.read_config(config_name='config')
        self.ego_vehicle = costmap_for_bev.Vehicle(vehicle_params)
        self.discrete = discrete
        self.map_bev_shape = None
        self.dtype = dtype
        self.datas_init = None

    def cal_datas_init(self, map_bev, start_xyt, end_xyt, cfg_map_bev, add_wall=True):
        map_bev = np.array(map_bev, dtype=self.dtype)
        self.map_bev_shape = map_bev.shape[-2:]

        # 从map_bev中提取障碍物
        map_obs = map_bev[1]
        if add_wall:
            # 在map_bev的周围设定一圈障碍物
            map_obs[:, 0] = 1
            map_obs[:, -1] = 1
            map_obs[0, :] = 1
            map_obs[-1, :] = 1
        self.datas_init = {
            "start_xyt": start_xyt,
            "end_xyt": end_xyt,
            "bound_xy": [0, map_obs.shape[0] * self.discrete[0], 0, map_obs.shape[1] * self.discrete[1]],
            "obs": [[x_idx * self.discrete[0], y_idx * self.discrete[1]] for x_idx, y_idx in zip(*np.where(map_obs == 1))],
            "cfg_map_bev": cfg_map_bev
        }
        return self.datas_init

    def get_handle(self):
        # 得到整体地图框图点以及里面的障碍物点，并重新离散化
        park_map = costmap_for_bev.Map(self.datas_init, self.config['map_discrete_size'])

        # create path planner
        handle = path_planner.PathPlanner(
            config=self.config,
            map=park_map,
            vehicle=self.ego_vehicle
        )
        return handle

    def run(self, map_bev, start_xyt, end_xyt, cfg_map_bev):
        self.cal_datas_init(map_bev, start_xyt, end_xyt, cfg_map_bev)
        handle = self.get_handle()
        path_points_list = handle.path_planning()

        return path_points_list