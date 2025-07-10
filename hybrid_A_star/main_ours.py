'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-2021 1140349586@qq.com
LastEditTime: 2023-07-12 19:39:18
FilePath: /Automated Valet Parking/main.py
Description: the main file of the hybrid a star algorithm for parking

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


from path_plan import path_planner
from animation.animation import ploter, plt
from animation.record_solution import DataRecorder
from animation.curve_plot import CurvePloter
from map import costmap_for_bev
from velocity_plan import velocity_planner
from interpolation import path_interpolation
from optimization import path_optimazition, ocp_optimization
from config import read_config
from tqdm import tqdm

import os

import argparse
#for test 
from map.read_data_obs import FromBev_get_start_end_point
from solution_test.plt_np import oup_new_datas_np

def main(file, config, id_T):#这里的file是case文件，config是参数信息
    # create the park map
    #得到整体地图框图点以及里面的障碍物点
    park_map = costmap_for_bev.Map(
        file=file, discrete_size=config['map_discrete_size'])#这是地图离散化大小

    # create vehicle
    ego_vehicle = costmap_for_bev.Vehicle()

    # create path planner
    planner = path_planner.PathPlanner(config=config,
                                       map=park_map,
                                       vehicle=ego_vehicle)

    # create path optimizer
    path_optimizer = path_optimazition.path_opti(park_map, ego_vehicle, config)

    # create path interpolation
    interplotor = path_interpolation.interpolation(
        config=config, map=park_map, vehicle=ego_vehicle)

    # create velocity planner
    v_planner = velocity_planner.VelocityPlanner(vehicle=ego_vehicle,
                                                 velocity_func_type=config['velocity_func_type'])

    # create path optimization planner
    ocp_planner = ocp_optimization.ocp_optimization(
        park_map=park_map, vehicle=ego_vehicle, config=config)

    # rapare memory to store path
    final_opt_path = []  # store the optimization path
    final_insert_path = []  # store the interpolation path
    final_ocp_path = []  # store ocp path

    # path planning
    optimal_tf = 0
    pre_tf = 0
    t = 0
    optimal_time_info = []
    original_path, path_info, split_path = planner.path_planning()
    final_pre_opt_path = []
    for path_i in split_path:
        # optimize path
        opti_path, forward = path_optimizer.get_result(path_i)

        # cubic fitting
        path_arc_length, path_i_info = interplotor.cubic_fitting(opti_path)

        # velocity planning
        v_acc_func, terminiate_time = v_planner.solve_nlp(
            arc_length=path_arc_length)

        # insert points
        insert_path = interplotor.cubic_interpolation(
            path=opti_path, path_i_info=path_i_info, v_a_func=v_acc_func, forward=forward, terminate_t=terminiate_time, path_arc_length=path_arc_length)

        # ocp problem solve
        ocp_traj, optimal_ti, optimal_dt = ocp_planner.solution(
            path=insert_path)
        pre_tf += insert_path[-1][-1]
        optimal_time_info.append([optimal_ti, optimal_dt])
        final_pre_opt_path.extend(insert_path)
        # add time information
        for ocp_i in ocp_traj:
            t += optimal_dt
            ocp_i.append(t)
        optimal_tf += optimal_ti

        final_opt_path.extend(opti_path)
        final_insert_path.extend(insert_path)
        final_ocp_path.extend(ocp_traj)

    # print time
    print('trajectory_time:', optimal_tf)
    print('pre_optimization_time:', pre_tf)

    # save traj into a csv file
    DataRecorder.record(save_path=config['save_path'],id_T=id_T,
                        save_name=case_name, trajectory=final_insert_path)
    # DataRecorder.record(save_path=config['save_path'] + '_preopt',
    #                     save_name=case_name, trajectory=final_insert_path)

    # animation
    # ploter.plot_obstacles(map=park_map)
    # park_map.visual_cost_map()
    # ploter.plot_final_path(path=original_path, label='Hybrid A*',
    #                        color='green', show_car=False)
    # ploter.plot_final_path(path=final_opt_path, label='Optimized Path',
    #                        color='blue', show_car=False)
    # ploter.plot_final_path(path=final_insert_path, label='Interpolation Traj',
    #                        color='red', show_car=False)
    # ploter.plot_final_path(path=final_ocp_path, label='Optimized Traj',
    #                        color='gray', show_car=True)
    # plt.legend()
    # fig_name = args.case_name + '.png'
    # fig_path = os.path.join(config['pic_path'], args.case_name)
    # if not os.path.exists(fig_path):
    #     os.makedirs(fig_path)
    # save_fig = os.path.join(fig_path, fig_name)
    # plt.savefig(save_fig, dpi=600)
    # plt.close()
    # gif_name = args.case_name + '.gif'
    # save_gif_name = os.path.join(fig_path, gif_name)
    # ploter.save_gif(path=final_ocp_path, color='gray', map=park_map,
    #                 show_car=True, save_gif_name=save_gif_name)
    print(f'solved_{id_T}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hybridAstar')
    parser.add_argument("--config_name", type=str, default="config")
    parser.add_argument("--case_name", type=str, default="test")
    parser.add_argument("--data_file_path", type=str, default=None)
    parser.add_argument("--oup_json_path", type=str, default=None)
    parser.add_argument("--Benchmark_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--pic_path", type=str, default=None)
    parser.add_argument("--mode", type=int, default=0,
                        help='0: solve this scenario, 1: load result and plot figure')
    args = parser.parse_args()

    # initial
    # load configure file to a dict
    config = read_config.read_config(config_name=args.config_name)
    case_name = args.case_name

    # 通过超参数设定相关参数
    config['data_file_path'] = args.data_file_path if not args.data_file_path is None else config['data_file_path']
    config['oup_json_path'] = args.oup_json_path if not args.oup_json_path is None else config['oup_json_path']
    config['Benchmark_path'] = args.Benchmark_path if not args.Benchmark_path is None else config['Benchmark_path']
    config['save_path'] = args.save_path if not args.save_path is None else config['save_path']
    config['pic_path'] = args.pic_path if not args.pic_path is None else config['pic_path']

    #利用json数据构建.csv文件
    get_start_end_point = FromBev_get_start_end_point(config['data_file_path'], config['Benchmark_path'], case_name)
    T_nums = get_start_end_point.calculate_start_end_point_then_print_csv()

    # read benchmark case

    # file = os.path.join(config['Benchmark_path'], case_name)
    for id_T in tqdm(range(T_nums)):
    # id_T = 23
        file = os.path.join(config['Benchmark_path'], case_name + str(id_T) + '.csv')

        if (args.mode == 0):
            main(file=file, config=config, id_T=id_T)


        elif (args.mode == 1):
            data_save_name = 'Solution_' + case_name
            data_save_path = config['save_path']

            save_fig_path = os.path.join(config['pic_path'], args.case_name)

            CurvePloter.plot_curve(data_save_path = data_save_path,
                                data_save_name = data_save_name,
                                save_fig_path = save_fig_path)
        else:
            raise TypeError('wrong mode, please make sure the mode number is 0 or 1')
    
    print('--------------------erery_T_trajectories_np_datas_cal_done---------------------')
    print("--------------------we start to oup_new_datas_np-------------------------------")
    oup_new_datas_np(config['data_file_path'], config['oup_json_path']).oup_trajectories_np_datas()
    print("--------------------all_the_work_is_done---------------------------------------")
    
    
