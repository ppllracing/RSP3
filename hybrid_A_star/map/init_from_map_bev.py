import json
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
# Specify the file path

# 展示
class Init_From_Map_BEV:
    def __init__(self, map_bev, discrete, dtype):
        self.map_bev = map_bev
        self.discrete_x = discrete[0]
        self.discrete_y = discrete[1]
        self.dtype = dtype

    def run(self, start_xyt, end_xyt):
        # 各层
        map_able_area, map_obs, map_ego, map_pp = self.map_bev[0:4]

        # 将数据写入dict
        data_dict = {
            "start_xyt": start_xyt,
            "end_xyt": end_xyt,
            "bound_xy": [0, map_obs.shape[0] * self.discrete_x, 0, map_obs.shape[1] * self.discrete_y],
            "obs": [[x_idx * self.discrete_x, y_idx * self.discrete_y] for x_idx, y_idx in zip(*np.where(map_obs == 1))],
        }

        return data_dict
