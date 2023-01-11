import math
import numpy as np

from Camera import Camera
from landmark import Landmark
from map import Map
from agent import Agent, Estmation_Agent, Mcl, KalmanFilter
from IdealCamera import IdealCamera
from IdealRobot import IdealRobot
from Robot import Robot
from world import World

if __name__ == '__main__':   ###name_indent
    world = World(30, 0.1, debug=False) 

    ### 地図を生成して3つランドマークを追加 ###
    m = Map()                                  
    m.append_landmark(Landmark(2,-2))
    m.append_landmark(Landmark(-1,-3))
    m.append_landmark(Landmark(-2,-2))
    m.append_landmark(Landmark(3,3))
    m.append_landmark(Landmark(-3,3))
    world.append(m)          

    ### ロボットを作る ###
    # straight = Agent(0.2, 0.0)    
    # circling = Agent(0.2, 10.0/180*math.pi)  
    # robot1 = IdealRobot( np.array([ 2, 3, math.pi/6]).T,    sensor=IdealCamera(m), agent=straight )             # 引数にcameraを追加、整理
    # robot2 = IdealRobot( np.array([-2, -1, math.pi/5*6]).T, sensor=IdealCamera(m), agent=circling, color="red")  # robot3は消しました
    # world.append(robot1)
    # world.append(robot2)

    # for i in range(1):
    #     initial_pose = np.array([0,0,0]).T
    #     estimator = Mcl(m, initial_pose, 100, motion_noise_stds={"nn":0.19,"no":0.001,"on":0.13,"oo":0.2}, systematic_resampling=False, expon_motion_noise=True)
    #     circling = Estmation_Agent(0.1, 0.2, 10.0/180*math.pi, estimator)
    #     # r = Robot(np.array([0,0,0]).T, sensor=Camera(m), agent=circling,color="gray",
    #     #         noise_per_meter=5, noise_std=math.pi/60,
    #     #         bias_rate_stds = (0.1,0.1),
    #     #         expected_stuck_time = 1,  expected_escape_time = 1,
    #     #         expected_kidnap_time = 1e100, kidnap_range_x = (-5.0,5.0), kidnap_range_y = (-5.0,5.0))
    #     r = Robot(np.array([0,0,0]).T, sensor=Camera(m), agent=circling,color="gray",
    #             noise_per_meter=5, noise_std=math.pi/60,
    #             bias_rate_stds = (0.1,0.1))
    #     world.append(r)

    # circling = Agent(0.2, 10.0/180*math.pi)
    # nobias_robot = IdealRobot(np.array([0,0,0]).T, sensor=Camera(m), agent=circling, color="red")
    # world.append(nobias_robot)
    # bias_robot = Robot(np.array([0,0,0]).T, sensor=None, agent=circling, color="red", noise_per_meter=0, bias_rate_stds=(0.2,0.2))
    # world.append(bias_robot)

    # initial_pose = np.array([0,0,0]).T
    # estimator = Mcl(m, initial_pose, 100, motion_noise_stds={"nn":0.19,"no":0.001,"on":0.13,"oo":0.2}, systematic_resampling=False, expon_motion_noise=False, arrow_color="blue")
    # circling = Estmation_Agent(0.1, 0.2, 10.0/180*math.pi, estimator)
    # r = Robot(np.array([0,0,0]).T, sensor=Camera(m), agent=circling,color="gray",
    #         noise_per_meter=5, noise_std=math.pi/60,
    #         bias_rate_stds = (0.1,0.1))
    # world.append(r)
    # estimator = Mcl(m, initial_pose, 100, motion_noise_stds={"nn":0.19,"no":0.001,"on":0.001,"oo":0.2}, systematic_resampling=False, expon_motion_noise=True, arrow_color="red")
    # circling = Estmation_Agent(0.1, 0.2, 10.0/180*math.pi, estimator)
    # r = Robot(np.array([0,0,0]).T, sensor=Camera(m), agent=circling,color="gray",
    #         noise_per_meter=5, noise_std=math.pi/60,
    #         bias_rate_stds = (0.1,0.1))
    # world.append(r)

    initial_pose = np.array([0,0,0]).T
    kf = KalmanFilter(m, initial_pose, motion_noise_stds={"nn":0.19,"no":0.001,"on":0.13,"oo":0.2})
    circling = Estmation_Agent(0.1, 0.2, 10.0/180*math.pi, kf)
    r = Robot(np.array([0,0,0]).T, sensor=Camera(m), agent=circling,color="gray",
            noise_per_meter=5, noise_std=math.pi/60,
            bias_rate_stds = (0.1,0.1))
    world.append(r)
    ### アニメーション実行 ###
    world.draw()