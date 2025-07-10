import json
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
# Specify the file path

# 展示
class FromBev_get_start_end_point:
    def __init__(self, data_file_path, oup_csv_path, case_name):
        self.data_file_path = data_file_path
        self.show_img_flag = False
        self.oup_csv_path = oup_csv_path
        self.case_name = case_name
        self.discrete_x = 0.059
        self.discrete_y = 0.125

    def get_theta(self, np_ego):
        rows, cols = np.where(np_ego == 1)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        #取中间线上的有的y坐标
        middle_row = (min_row + max_row) // 2
        y_duiying = np.where(np_ego[middle_row, :] == 1)
        x_left = []
        x_left.append(np.where(np_ego[:, y_duiying[0][5]] == 1)[0][0])
        x_left.append(np.where(np_ego[:, y_duiying[0][-5]] == 1)[0][0])

        theta = math.atan2((y_duiying[0][-5] - y_duiying[0][5]) * self.discrete_y, (x_left[1] - x_left[0]) * self.discrete_x) - math.pi

        return theta

    #获得ego的四个顶点
    def get__ego_four_points(self, np_ego, theta):

        indices = np.where(np_ego == 1)

        ymax = indices[1].max()
        ymin = indices[1].min()
        xmax = indices[0].max()
        xmin = indices[0].min()

        if theta < -math.pi/2:
            point_ymax = (indices[0][np.where(indices[1] == ymax)].min(), ymax)
            point_ymin = (indices[0][np.where(indices[1] == ymin)].max(), ymin)
            point_xmax = (xmax, indices[1][np.where(indices[0] == xmax)].max())
            point_xmin = (xmin, indices[1][np.where(indices[0] == xmin)].min())
        else:
            point_ymax = (indices[0][np.where(indices[1] == ymax)].max(), ymax)
            point_ymin = (indices[0][np.where(indices[1] == ymin)].min(), ymin)
            point_xmax = (xmax, indices[1][np.where(indices[0] == xmax)].min())
            point_xmin = (xmin, indices[1][np.where(indices[0] == xmin)].max())

        return point_ymax, point_ymin, point_xmax, point_xmin
    def get_Parking_area_point(self, np_PP):
        indices = np.where(np_PP == 1)
        ymax = indices[1].max()
        ymin = indices[1].min()
        xmin = indices[0].min()
        xmax = indices[0].max()
        lr = 0.929 / self.discrete_x
        if xmin < 25:
            theta_end = 0
            end_rear_center_y = (ymax + ymin) // 2
            end_rear_center_x = math.ceil(xmin + lr * math.cos(theta_end))
        else:
            theta_end = -math.pi
            end_rear_center_y = (ymax + ymin) // 2
            end_rear_center_x = math.ceil(xmax + lr * math.cos(theta_end))
        return end_rear_center_x, end_rear_center_y, theta_end
    #获得后轴中心
    def get_start_rear_center(self, point_ymax, point_ymin, point_xmax, point_xmin, theta):
        if theta < -math.pi/2:
            rear_center_x = (point_xmax[0] + point_ymax[0]) / 2
            rear_center_y = (point_xmax[1] + point_ymax[1]) / 2
        else:
            rear_center_x = (point_xmin[0] + point_ymax[0]) / 2
            rear_center_y = (point_xmin[1] + point_ymax[1]) / 2
        #后轴中心到车辆最后面的距离
        lr = 0.929


        start_rear_center_x = round(rear_center_x + lr * math.cos(theta) / self.discrete_x)


        start_rear_center_y = round(rear_center_y + lr * math.sin(theta) / self.discrete_y)

        return start_rear_center_x, start_rear_center_y
    # Load and display the JSON data
    def show_img(self, point_ymax, point_ymin, point_xmax, point_xmin, start_rear_center_x, start_rear_center_y, end_rear_center_x, end_rear_center_y, np_PP, np_able_PP, np_obs, np_ego):
        #将所求点放在这张图上
        array_1 = np.zeros(np_PP.shape)
        #画出车辆的四个顶点 + 后轴中心 
        array_1[point_ymax[0], point_ymax[1]] = 1
        array_1[point_ymin[0], point_ymin[1]] = 1
        array_1[point_xmax[0], point_xmax[1]] = 1
        array_1[point_xmin[0], point_xmin[1]] = 1
        array_1[start_rear_center_x, start_rear_center_y] = 1
        #画上停车点的后轴中心点
        array_1[end_rear_center_x, end_rear_center_y] = 1
        y_indices, x_indices = np.where(array_1 == 1)

        #画图
        fig, axs = plt.subplots(1,5)
        axs.flat[0].imshow(np_PP, cmap='gray')
        axs.flat[1].imshow(np_able_PP, cmap='gray')
        axs.flat[2].imshow(np_obs, cmap='gray')
        axs.flat[3].imshow(np_ego, cmap='gray')
        axs.flat[4].imshow(array_1, cmap='gray')
        plt.scatter(x_indices, y_indices, color='red', marker='o')  # 用红色圆点表示
        plt.show()

    def calculate_start_end_point_then_print_csv(self):
        with open(self.data_file_path, 'r') as file:
            json_data = json.load(file)
        #获取所有参数
        for id_T in range(len(json_data)):
            #获取三层bev值
            # id_T = 16
            np_PP = np.array(json_data[id_T]['bev']['map_bev'][0])
            np_able_PP = np.array(json_data[id_T]['bev']['map_bev'][1])
            np_obs = np.array(json_data[id_T]['bev']['map_bev'][2])
            np_ego = np.array(json_data[id_T]['bev']['map_bev'][3])
            #获得车辆的角度
            theta_start = self.get_theta(np_ego)
            print("车辆的角度:", theta_start)
            #得到车辆的后轴中心
            #先获得车辆的四个顶点
            point_ymax, point_ymin, point_xmax, point_xmin = self.get__ego_four_points(np_ego, theta_start)
            #然后获得后轴中心
            start_rear_center_x, start_rear_center_y = self.get_start_rear_center(point_ymax, point_ymin, point_xmax, point_xmin, theta_start)
            #接下来获得停车位的坐标，来求出停车点的位置
            end_rear_center_x, end_rear_center_y, theta_end = self.get_Parking_area_point(np_PP)
            #画图
            if self.show_img_flag:
                self.show_img(point_ymax, point_ymin, point_xmax, point_xmin, start_rear_center_x, start_rear_center_y, end_rear_center_x, end_rear_center_y, np_PP, np_able_PP, np_obs, np_ego)
            print("T:", id_T)
            print(f"start_point: x:{start_rear_center_x}-y:{start_rear_center_y}-theta:{theta_start}")
            print(f"end_point: x:{end_rear_center_x}-y:{end_rear_center_y}-theta:{theta_end}")

            oup_csv_path_now = self.oup_csv_path + '/' + self.case_name + str(id_T) + '.csv'
            #将数据写入csv文件
            self.write_csv(oup_csv_path_now, start_rear_center_x, start_rear_center_y, theta_start, end_rear_center_x, end_rear_center_y, theta_end, np_obs)

        return len(json_data)
    
    def write_csv(self, filename, start_rear_center_x, start_rear_center_y, theta_start, end_rear_center_x, end_rear_center_y, theta_end, np_obs):
        #先得到起点终点的信息
        initial_data = [start_rear_center_x * self.discrete_x, start_rear_center_y * self.discrete_y, theta_start, end_rear_center_x * self.discrete_x,
                            end_rear_center_y * self.discrete_y, theta_end, 0, np_obs.shape[0] * self.discrete_x, 0, np_obs.shape[1] * self.discrete_y]
        
        combined_data = initial_data.copy()

        xy_obs = np.where(np_obs == 1)
        nums = len(xy_obs[0])
        for i in range(nums):
            combined_data.extend([xy_obs[0][i] * self.discrete_x, xy_obs[1][i] * self.discrete_y])

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(combined_data)
    
if __name__ == '__main__':
    data_file_path = '/mnt/d/parking/datas_2.json'
    oup_csv_path = 'TestCases'
    get_start_end_point = FromBev_get_start_end_point(data_file_path, oup_csv_path, 'test')
    get_start_end_point.show_img_flag = False
    get_start_end_point.calculate_start_end_point_then_print_csv()

