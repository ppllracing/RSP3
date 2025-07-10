import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import json


class oup_new_datas_np:
    def __init__(self, pre_json_path, oup_json_path):
 
        self.oup_json_path = oup_json_path
        self.img_flag = False
        self.pre_json_path = pre_json_path
        self.discrete_x = 0.059
        self.discrete_y = 0.125

    def oup_trajectories_np_datas(self):

        with open(self.pre_json_path, 'r', encoding='utf-8') as f:
            data_pre_json = json.load(f)

        #增加每个时段的轨迹图
        for id_T in range(len(data_pre_json)):
            # 读取csv文件
            inp_csv_path = f'solution_test/Solution_test{id_T}.csv'
            data = pd.read_csv(inp_csv_path, sep="\t")  # 读取带有制表符分隔的文件
            # 提取 x 和 y 列
            x = np.array(data['x'])
            y = np.array(data['y'])
            x, y = self.interpolate_values(x, y)
            # 将x,y转为网格图中的坐标
            x, y = x / self.discrete_x, y / self.discrete_y
            np_traj = np.zeros(np.array(data_pre_json[id_T]['bev']['map_bev'][0]).shape)

            for i in range(len(x)):
                x_id = math.ceil(x[i])
                y_id = math.ceil(y[i])
                if x_id >= np_traj.shape[0] or x_id < 0 or y_id >= np_traj.shape[1] or y_id < 0:
                    continue
                np_traj[x_id, y_id] = 1

            data_pre_json[id_T]['bev']['map_bev'].append(np_traj.tolist())
        
        # 保存轨迹图
        with open(self.oup_json_path, 'w', encoding='utf-8') as f:
            json.dump(data_pre_json, f, ensure_ascii=False)
        #画图
        if self.img_flag:
            self.oup_trajectories_np_img(np_traj)

    def oup_trajectories_np_img(self, np_traj):
        # 绘制轨迹图
        plt.imshow(np_traj, cmap='gray')
        plt.show()
    
    def interpolate_values(self, x_values, y_values, max_dist=0.1):
        result_x = [x_values[0]]
        result_y = [y_values[0]]
        
        for i in range(1, len(x_values)):
            prev_x = x_values[i - 1]
            curr_x = x_values[i]
            prev_y = y_values[i - 1]
            curr_y = y_values[i]
            
            # 计算当前点与前一个点的距离
            while np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) > max_dist:
                # 计算插值的比例
                ratio = max_dist / np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                
                # 计算插值点
                new_x = prev_x + ratio * (curr_x - prev_x)
                new_y = prev_y + ratio * (curr_y - prev_y)
                
                # 添加插值点
                result_x.append(new_x)
                result_y.append(new_y)
                
                # 更新 prev_x 和 prev_y
                prev_x = new_x
                prev_y = new_y
            
            # 添加当前值到结果列表
            result_x.append(curr_x)
            result_y.append(curr_y)
        
        return np.array(result_x), np.array(result_y)
if __name__ == '__main__':

    pre_json_path = '/mnt/d/parking/datas_2.json'

    oup_json_path = 'new_datas/datas_2.json'

    oup_new_datas_np(pre_json_path, oup_json_path).oup_trajectories_np_datas()