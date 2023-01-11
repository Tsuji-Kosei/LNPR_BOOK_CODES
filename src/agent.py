import copy
import math
from matplotlib.patches import Ellipse
import numpy as np
import random
from scipy.stats import multivariate_normal, expon, norm

from IdealCamera import IdealCamera
from IdealRobot import IdealRobot

class Agent: 
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega
        
    def decision(self, observation=None):
        return self.nu, self.omega

class Particle:
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight
    
    def expon_motion_noise(self, pose, nu, omega, time_interval, expon_motion_noise_pdf, theta_noise):
        self.distance_until_noise = expon_motion_noise_pdf.rvs()
        self.distance_until_noise -= abs(nu)*time_interval
        if self.distance_until_noise<=0.0:
            self.distance_until_noise +=expon_motion_noise_pdf.rvs()
            pose[2] += theta_noise.rvs()
        return pose

    def motion_update(self, nu, omega, time, noise_rate_pdf, expon_motion_noise, expon_motion_noise_pdf, theta_noise):
        # 実際のロボットのノイズに合わせて計算した入力のノイズを載せる
        ns = noise_rate_pdf.rvs()
        noised_nu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        noised_omega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)
        if expon_motion_noise:
            self.pose = self.expon_motion_noise(self.pose, nu, omega, time, expon_motion_noise_pdf, theta_noise)

    
    def observation_update(self, observation, envmap, distance_dev_rate, direction_dev):
        for d in observation:
            obs_pos = d[0]
            obs_id = d[1]

            # パーティクルの位置と地図から、ランドマークの距離と方角を算出
            pos_on_map = envmap.landmarks[obs_id].pos
            particle_suggest_pos = IdealCamera.observation_function(self.pose, pos_on_map)

            # 尤度の計算 距離のばらつき(標準偏差)は測定した距離に比例すると仮定している
            distance_dev = distance_dev_rate*particle_suggest_pos[0]
            cov = np.diag(np.array([distance_dev**2,direction_dev]))
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos)


# estimator    
class Mcl:
    def __init__(self, env_map, init_pose, num, motion_noise_stds={"nn":0.19,"no":0.001,"on":0.13,"oo":0.2},
                 distance_dev_rate = 0.14, direction_dev = 0.05,
                 noise_per_meter = 5, theta_noise_std=math.pi/60,
                 systematic_resampling = False, expon_motion_noise = False,
                 arrow_color = "blue"):
        """
        motion_noise_stds, distance_dev_rate, direction_devは実験によって取得
        systematic_resampling: 系統サンプリングを採用
        expon_motion_noise: パーティクルの角度をたまに変化させる(小石を踏むのに対応)
        """
        
        self.particles = [Particle(init_pose, 1.0/num) for i in range(num)]
        self.map = env_map
        self.distance_dev_rate = distance_dev_rate # センサーのノイズ
        self.direction_dev = direction_dev     # センサーのノイズ

        v = motion_noise_stds
        c = np.diag([v["nn"]**2,v["no"]**2,v["on"]**2,v["oo"]**2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)

        self.expon_motion_noise_pdf = expon(scale=1.0/(1e-100+noise_per_meter))
        self.theta_noise = norm(scale=theta_noise_std)
        self.expon_motion_noise = expon_motion_noise

        self.systematic_resampling = systematic_resampling

        self.ml = self.particles[0]
        self.pose = self.ml.pose

        self.arrow_color = arrow_color
    
    def set_ml(self):
        i = np.argmax([p.weight for p in self.particles])
        self.ml = self.particles[i]
        self.pose = self.ml.pose
    
    def motion_update(self, nu, omega, time):
        for p in self.particles:
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf, self.expon_motion_noise, self.expon_motion_noise_pdf, self.theta_noise)
    
    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        self.set_ml() #　一番正しいと思われる姿勢を取得
        if self.systematic_sampling:
            self.systematic_sampling()
        else:
            self.resampling()
    
    def resampling(self):
        ws = [e.weight for e in self.particles]
        if sum(ws)<1e-100: ws = [e+1e-100 for e in ws]
        ps = random.choices(self.particles, weights=ws, k=len(self.particles)) # wsの要素に比例した確率でパーティクルをnum個選ぶ
        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles:
            p.weight = 1.0/len(self.particles)

    def systematic_sampling(self):
        ws = np.cumsum([e.weight for e in self.particles]) #累積和
        if ws[-1]<1e-100: ws = [e+1e-100 for e in ws]
        
        step = ws[-1]/len(self.particles)
        r = np.random.uniform(0.0,step)
        cur_pos = 0
        ps = []

        while len(ps)<len(self.particles):
            if r<ws[cur_pos]:
                ps.append(self.particles[cur_pos])
                r += step
            else:
                cur_pos += 1

        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles:
            p.weight = 1.0/len(self.particles)

    def draw(self, ax, elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        vys = [math.sin(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        # 矢印の描画
        elems.append(ax.quiver(xs,ys,vxs,vys,
                                angles='xy',scale_units='xy', scale=1.5,color=self.arrow_color,alpha=0.5))

def sigma_ellipse(p, cov, n):
    eig_vals, eig_vec = np.linalg.eig(cov)
    ang = math.atan2(eig_vec[:,0][1],eig_vec[:,0][0])/math.pi*180
    return Ellipse(p,width=2*n*math.sqrt(eig_vals[0]), height=2*n*math.sqrt(eig_vals[1]),angle=ang, fill=False, color="blue", alpha=0.5)

class KalmanFilter:
    def __init__(self, env_map, init_pose, motion_noise_stds={"nn":0.19,"no":0.001,"on":0.13,"oo":0.2}):
        self.belief = multivariate_normal(mean=np.array([0.0,0.0,math.pi/4]), cov = np.diag([0.1,0.2,0.01]))
        self.pose = self.belief.mean

    def motion_update(self, nu, omega, time):
        pass
    
    def observation_update(self, observation):
        pass
    
    def draw(self, ax, elems):
        # xy平面上の誤差の3σ範囲
        e = sigma_ellipse(self.belief.mean[0:2], self.belief.cov[0:2,0:2],3)
        elems.append(ax.add_patch(e))

        # θ方向の誤差の3σ範囲
        x,y,c = self.belief.mean
        sigma3 = math.sqrt(self.belief.cov[2,2])*3
        xs = [x+math.cos(c-sigma3),x,x+math.cos(c+sigma3)]
        ys = [y+math.sin(c-sigma3),y,y+math.sin(c+sigma3)]
        elems += ax.plot(xs,ys,color="blue", alpha=0.5)

class Estmation_Agent(Agent):
    def __init__(self, time_interval, nu, omega, estimator):
        super().__init__(nu, omega)
        self.estimator = estimator
        self.time_interval = time_interval

        self.prev_nu = 0.0
        self.prev_omega = 0.0
    
    def decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega,self.time_interval)
        self.estimator.observation_update(observation)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        return self.nu, self.omega
    
    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)
        x,y,theta = self.estimator.pose
        s = "({:.2f},{:.2f},{})".format(x,y,int(theta*180/math.pi)%360)
        elems.append(ax.text(x,y+0.1,s,fontsize=8))
