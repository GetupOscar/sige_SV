import numpy as np
import math
import matplotlib.pyplot as plt
import random

class Network:
    def __init__(self):
        #basic parameters
        self.mec = 8
        self.user = 8
        self.x_min = 0
        self.y_min = 0
        self.x_max = 75
        self.y_max = 75
        self.mec_x = [0, 25, 75, 100, 0, 25, 75, 100]
        self.mec_y = [0, 25, 25, 0, 100, 75, 75, 100]
        #self.mec_x = [250, 250, 750, 750, 500]#random.sample(range(0, self.x_max), self.mec)
        #self.mec_y = [250, 750, 250, 750, 500]#random.sample(range(0, self.y_max), self.mec)
        self.user_x = random.sample(range(0, self.x_max), self.user)
        self.user_y = random.sample(range(0, self.y_max), self.user)
        self.mec_mec_distance = np.zeros((self.mec, self.mec), dtype=float)
        self.mec_user_distance = np.zeros((self.mec, self.user), dtype=float)
        self.mec_user_distance_new = np.zeros((self.mec, self.user), dtype=float)
        self.user_mec_distance = np.zeros((self.user, self.mec), dtype=float)
        self.user_mec_distance_new = np.zeros((self.user, self.mec), dtype=float)
        self.sigma_dBm = -110  # channel noise dbm
        self.sigma = 10 ** (self.sigma_dBm / (1000 * 10))
        self.a = 3
        self.episode = 2
        #user mobility parameters
        self.user_mobility_direction = ['up', 'down', 'left', 'right', 'static']
        self.user_mobility_speed = 10
        self.user_mobility_static = 0
        self.direction_index_label = np.zeros((self.user, len(self.user_mobility_direction)), dtype=int)
        #VR video frame requirement parameters
        self.user_requirement_scenario = np.zeros(self.user, dtype=int)
        self.fov_index = []
        self.fov_index_new = []
        self.user_fov = np.zeros(self.user, dtype=int)
        self.fov_number = 8
        #self.fov_number = 6
        self.scenario = 1
        self.cluster_index = []

        #brownian motion
        self.fov_requirement = np.zeros(self.user, dtype=int)
        self.fov_centre_x = np.zeros(self.user, dtype=float)
        self.fov_centre_y = np.zeros(self.user, dtype=float)
        self.cube_length = 12
        #self.cube_length = 9
        self.cube_width = 6
        self.fov_number_row = 3
        self.fov_x = [1.5, 4.5, 7.5, 10.5, 1.5, 4.5, 7.5, 10.5]
        self.fov_y = [1.5, 1.5, 1.5, 1.5, 4.5, 4.5, 4.5, 4.5]
        #self.fov_x = [1.5, 4.5, 7.5, 1.5, 4.5, 7.5]
        #self.fov_y = [1.5, 1.5, 1.5, 4.5, 4.5, 4.5]
        self.delta_t = 3
        self.fov_point_change_sigma = math.sqrt(2 * self.delta_t)
        #self.fov_point_change_sigma = (2 * math.sqrt(self.delta_t))
        self.distance_brownian = np.zeros((self.user, self.fov_number), dtype=float)
        self.direction_flag_up = []
        self.direction_flag_down = []
        self.direction_flag_left = []
        self.direction_flag_right = []
        #QoE of transmission parameters
        self.I = np.zeros(self.user, dtype=int) + 1
        self.D = np.zeros(self.user, dtype=int)
        self.delta = 1
        self.MSK = np.zeros(self.user, dtype=int)
        self.V_PSNR = np.zeros(self.user, dtype=int)
        self.sum_V_PSNR = 0
        #self.reward = np.zeros(self.user, dtype=int)
        #VR rendering
        self.compress = 300
        self.spherical_frame = 0.0
        self.original_frame = 0.0
        self.original_frame_bit = 0.0
        self.fov_frame = 0.0
        self.fov_frame_cycle = 0.0
        self.spherical_frame_cycle = 0.0
        self.original_frame_cycle = 0.0
        self.resolution = 1080
        self.horizon = 150
        self.vertical = 210
        self.viewport = 2
        self.f_mec = 100
        self.f_vr = self.f_mec / 10
        self.F_mec = 100
        self.F_vr = self.F_mec / 10
        self.process_mec = 0
        self.process_vr = 0
        self.M_mec = 0
        self.C_mec = 0
        self.M_vr = 0
        self.C_vr = 0
        self.T_mec = 0
        self.T_vr = 0
        self.S_1 = 0
        self.S_2 = 0
        self.pixel = 8
        self.RGB = 3
        self.fiber_rate = 3 * 100000000
        self.cycles = 1000
        self.MEC_process_ability = [4, 5, 5, 4, 4, 5, 5, 4]
        #self.MEC_process_ability = [4, 4, 4, 5, 6]
        self.fov_frame_bit = 0.0
        self.GPU = 35
        self.thread = 16
        self.VR_process_ability = 2

        #uplink transmission
        self.M = 6
        self.I_M = np.identity(self.M)
        self.U = np.zeros((self.mec, self.user, self.M), dtype=complex)
        self.H_up = np.zeros((self.mec, self.user, self.M), dtype=complex)
        self.temp = np.zeros((self.mec, self.user, self.M), dtype=complex)
        self.I_interference_uplink = np.zeros((self.mec, self.user, self.M), dtype=complex)
        self.R_up = np.zeros((self.mec, self.user), dtype=float)
        self.R_up_change = np.zeros((self.user, self.mec), dtype=float)
        self.B = 1  # bandwidth
        self.p_user = 1
        self.uplink_threshold = 2
        self.uplink_success_index = np.zeros(self.user, dtype=int)
        self.uplink_fail_index = []
        self.track_information = 1
        #downlink tranmission
        self.H_down = np.zeros((self.mec, self.user, self.M), dtype=complex)
        self.V = []
        self.R_down = np.zeros(self.user, dtype=float)
        self.I_M_down = 1
        self.p_mec = 2
        self.downlink_threshold = 0.11

    def calculate_mec_mec_distance(self):
        for i in range(self.mec):
            for j in range(self.mec):
                length = math.sqrt((self.mec_x[i] - self.mec_x[j]) ** 2 + (self.mec_y[i] - self.mec_y[j]) ** 2)
                self.mec_mec_distance[i, j] = length

    def calculate_mec_user_distance(self):
        self.mec_user_distance = np.zeros((self.mec, self.user), dtype=float)
        self.mec_user_distance_new = np.zeros((self.mec, self.user), dtype=float)
        for i in range(self.mec):
            for j in range(self.user):
                length = math.sqrt((self.mec_x[i] - self.user_x[j]) ** 2 + (self.mec_y[i] - self.user_y[j]) ** 2)
                self.mec_user_distance[i, j] = length
        #print('mec_user_distance: ', self.mec_user_distance)
        self.mec_user_distance_new = self.mec_user_distance_new + 1
        for i in range(self.mec):
            for j in range(self.user):
                if self.mec_user_distance[i, j] / 50 > 1:
                    self.mec_user_distance_new[i, j] = math.sqrt((self.mec_user_distance[i, j] / 50) ** (-self.a))

    def calculate_user_mec_distance(self):
        self.user_mec_distance = np.zeros((self.user, self.mec), dtype=float)
        #self.user_mec_distance_new = np.zeros((self.user, self.mec), dtype=float)#
        for i in range(self.user):
            for j in range(self.mec):
                length = math.sqrt((self.user_x[i] - self.mec_x[j]) ** 2 + (self.user_y[i] - self.mec_y[j]) ** 2)
                self.user_mec_distance[i, j] = length

    #用户移动模型，当用户移动到区域边缘的时候，如果左侧在区域边缘，那么下次运动只能往右，上，下或者不动
    def user_mobility(self):
        for i in range(self.user):
            user_mobility_direction_new = self.user_mobility_direction.copy()
            itemindex = np.argwhere(self.direction_index_label[i] == 1)
            itemindex = itemindex.flatten()
            for j in range(len(itemindex)):
                self.direction_index_label[i, itemindex[j]] = 0
                user_mobility_direction_new.remove(self.user_mobility_direction[itemindex[j]])
            direction_index = random.randint(0, len(user_mobility_direction_new) - 1)
            if user_mobility_direction_new[direction_index] == 'up':
                self.user_y[i] += self.user_mobility_speed
                if self.user_y[i] > self.y_max:
                    self.user_y[i] = self.y_max
                    self.direction_index_label[i, 0] = 1
            elif user_mobility_direction_new[direction_index] == 'down':
                self.user_y[i] -= self.user_mobility_speed
                if self.user_y[i] < self.y_min:
                    self.user_y[i] = self.y_min
                    self.direction_index_label[i, 1] = 1
            elif user_mobility_direction_new[direction_index] == 'left':
                self.user_x[i] -= self.user_mobility_speed
                if self.user_x[i] < self.x_min:
                    self.user_x[i] = self.x_min
                    self.direction_index_label[i, 2] = 1
            elif user_mobility_direction_new[direction_index] == 'right':
                self.user_x[i] += self.user_mobility_speed
                if self.user_x[i] > self.x_max:
                    self.user_x[i] = self.x_max
                    self.direction_index_label[i, 3] = 1
            elif user_mobility_direction_new[direction_index] == 'static':
                self.user_x[i] += self.user_mobility_static
                self.user_y[i] += self.user_mobility_static

    def show_positions(self):
        print('user_x: ', self.user_x)
        print('user_y: ', self.user_y)
        plt.scatter(self.mec_x, self.mec_y, s=75, alpha=0.5)
        plt.scatter(self.user_x, self.user_y, s=50, alpha=0.5)
        plt.title('Location of MECs and VR users', fontsize=14)
        plt.xlabel('X(m)', fontsize=12)
        plt.ylabel('Y(m)', fontsize=12)
        plt.axis("equal")
        plt.show()

    def brownian_motion(self):
        for i in range(self.user):
            self.direction_flag_up.append(False)
            self.direction_flag_down.append(False)
            self.direction_flag_left.append(False)
            self.direction_flag_right.append(False)
        for i in range(self.user):
            self.fov_centre_x[i] = random.uniform(0, self.cube_length)
            self.fov_centre_y[i] = random.uniform(0, self.cube_width)
        for i in range(self.user):
            if self.direction_flag_left[i] == True:
                while self.direction_flag_left[i]:
                    brown_motion_x = np.random.normal(0, self.fov_point_change_sigma)
                    self.fov_centre_x[i] = self.fov_centre_x[i] + brown_motion_x
                    if self.fov_centre_x[i] > 0:
                        self.direction_flag_left[i] = False
            elif self.direction_flag_right[i] == True:
                while self.direction_flag_right[i]:
                    brown_motion_x = np.random.normal(0, self.fov_point_change_sigma)
                    self.fov_centre_x[i] = self.fov_centre_x[i] + brown_motion_x
                    if self.fov_centre_x[i] < self.cube_length:
                        self.direction_flag_right[i] = False
            elif self.direction_flag_up[i] == True:
                while self.direction_flag_up[i]:
                    brown_motion_y = np.random.normal(0, self.fov_point_change_sigma)
                    self.fov_centre_y[i] = self.fov_centre_y[i] + brown_motion_y
                    if self.fov_centre_y[i] < self.cube_width:
                        self.direction_flag_up[i] = False
            elif self.direction_flag_down[i] == True:
                while self.direction_flag_down[i]:
                    brown_motion_y = np.random.normal(0, self.fov_point_change_sigma)
                    self.fov_centre_y[i] = self.fov_centre_y[i] + brown_motion_y
                    if self.fov_centre_y[i] > 0:
                        self.direction_flag_down[i] = False
            else:
                brown_motion_x = np.random.normal(0, self.fov_point_change_sigma)
                brown_motion_y = np.random.normal(0, self.fov_point_change_sigma)
                self.fov_centre_x[i] = self.fov_centre_x[i] + brown_motion_x
                self.fov_centre_y[i] = self.fov_centre_y[i] + brown_motion_y
                if self.fov_centre_x[i] > self.cube_length:
                    self.fov_centre_x[i] = self.cube_length
                    self.direction_flag_right[i] = True
                if self.fov_centre_x[i] < 0:
                    self.fov_centre_x[i] = 0
                    self.direction_flag_left[i] = True
                if self.fov_centre_y[i] > self.cube_width:
                    self.fov_centre_y[i] = self.cube_width
                    self.direction_flag_up[i] = True
                if self.fov_centre_y[i] < 0:
                    self.fov_centre_y[i] = 0
                    self.direction_flag_down[i] = True
        for i in range(self.user):
            for j in range(self.fov_number):
                self.distance_brownian[i, j] = math.sqrt((self.fov_centre_x[i] - self.fov_x[j]) ** 2 + (self.fov_centre_y[i] - self.fov_y[j]) ** 2)
            self.fov_requirement[i] = np.argmin(self.distance_brownian[i])

    def VR_requirement(self):
        self.brownian_motion()
        for i in range(self.user):
            self.user_requirement_scenario[i] = np.random.randint(0, self.scenario)
        self.user_fov = self.fov_requirement
        user_requirement_scenario_new = np.unique(self.user_requirement_scenario)
        scenario_index = []
        fov_index1 = []
        for i in range(len(user_requirement_scenario_new)):
            scenario_index.append([])
            fov_index1.append([])
        for i in range(len(user_requirement_scenario_new)):
            scenario_index[i] = []
            fov_index1[i] = []
            for j in range(len(self.user_requirement_scenario)):
                if user_requirement_scenario_new[i] == self.user_requirement_scenario[j]:
                    scenario_index[i].append(j)
                    fov_index1[i].append(self.user_fov[j])
        fov_index_new_new = []
        for i in range(len(fov_index1)):
            fov_index_new_new.append([])
            fov_index_new_new[i] = np.unique(fov_index1[i])
        cluster = []
        for i in range(len(user_requirement_scenario_new)):
            cluster.append([])
            for j in range(len(fov_index_new_new[i])):
                cluster[i].append([])
        for i in range(len(fov_index1)):
            for j in range(len(fov_index_new_new[i])):
                for k in range(len(fov_index1[i])):
                    if fov_index1[i][k] == fov_index_new_new[i][j]:
                        cluster[i][j].append(scenario_index[i][k])
        self.fov_index = []
        for i in range(len(fov_index_new_new)):
            for j in range(len(fov_index_new_new[i])):
                self.fov_index.append([])
        count = 0
        self.cluster_index = cluster
        for i in range(len(cluster)):
            for j in range(len(cluster[i])):
                for k in range(len(cluster[i][j])):
                    self.fov_index[count].append(cluster[i][j][k])
                count = count + 1
        count_cluster = len(self.fov_index)

    #添加对下行链路传输是否成功的判断！！！
    def QoE(self):
        for i in range(self.user):
            self.D[i] = 0
            self.MSK[i] = 0
            self.V_PSNR[i] = 0
            if self.R_down[i] >= self.downlink_threshold:
                self.D[i] = 1
            self.MSK[i] = (self.I[i] - self.D[i]) ** 2
            self.V_PSNR[i] = 10 * np.log10((1 + self.delta) / (self.MSK[i] + self.delta))
        self.sum_V_PSNR = np.sum(self.V_PSNR)
        #print(self.sum_V_PSNR)

    def VR_spherical_rendering(self):
        self.fov_frame = self.resolution * self.resolution * self.RGB * self.pixel * self.viewport / (1024 * 1024 * self.compress)
        self.fov_frame_bit = self.resolution * self.resolution * self.RGB * self.pixel * self.viewport
        self.fov_frame_cycle = self.fov_frame_bit * self.cycles
        #self.spherical_frame = self.fov_frame_bit * 5
        self.original_frame = self.fov_frame * 4 / 3
        self.original_frame_bit = self.fov_frame_bit * 4 / 3
        self.original_frame_cycle = self.original_frame_bit * self.cycles
        # self.spherical_frame_cycle = self.spherical_frame * self.cycles
        #self.S_1 = 4 * math.pi * (self.resolution ** 2)
        #self.S_2 = self.horizon * self.vertical * (math.pi * self.resolution / 180) ** 2
        #self.process_mec = self.viewport * (self.S_1 + self.S_2)
        #self.M_mec = self.process_mec * self.pixel * self.RGB
        #self.C_mec = self.viewport * self.horizon * self.vertical * self.pixel * self.RGB * (math.pi * self.resolution / 180) ** 2
        #self.process_vr = self.viewport * self.S_2
        #self.M_vr = self.process_vr * self.pixel * self.RGB

    def uplink_transmission(self):
        #print(self.R_up)
        for number in range(self.episode):
            U_new = np.zeros((self.user, self.M), dtype=complex)
            for i in range(self.user):
                for j in range(self.M):
                    U_new[i][j] = complex(random.random(), random.random())
            self.U = U_new
            # create channel matrix
            for i in range(self.mec):
                for j in range(self.user):
                    for k in range(self.M):
                        self.H_up[i, j, k] = complex(random.gauss(0, 1), random.gauss(0, 1)) * \
                                             self.mec_user_distance_new[i, j]
            for i in range(self.mec):
                for j in range(self.user):
                    temp = 0
                    for k in range(self.user):
                        if k != j:
                            temp += self.p_user * np.dot(
                                np.dot(self.U[k].reshape(-1, 1), self.H_up[i, k].reshape(1, -1)),
                                np.dot(self.H_up[i, k].reshape(-1, 1), self.U[k].reshape(1, -1)))
                    x_up = self.I_M + self.p_user * np.dot(
                        np.dot(self.U[j].reshape(-1, 1), self.H_up[i, j].reshape(1, -1)),
                        np.dot(self.H_up[i, j].reshape(-1, 1), self.U[j].reshape(1, -1))) / (self.sigma)
                    x_up = np.linalg.det(x_up)
                    self.R_up[i, j] += self.B * math.log(abs(x_up), 2)
        self.R_up /= self.episode
        for i in range(self.user):
            for j in range(self.mec):
                self.R_up_change[i][j] = self.R_up[j][i]

        for i in range(self.user):
            if np.max(self.R_up[:, i] >= self.uplink_threshold):
                self.uplink_success_index[i] = 1
            else:
                self.uplink_fail_index.append(i)
        #print(self.R_up)

    #def downlink_channel_and_precoding_matrix(self):

    def downlink_transmission_for_q_learning_new(self, select_action, fov_index_new_new):
        #self.user_mobility()
        #self.calculate_mec_user_distance()
        self.fov_index = fov_index_new_new
        self.R_down = np.zeros(self.user, dtype=float)
        iterations = 100
        for iteration in range(iterations):
            self.H_down = np.zeros((self.mec, self.user, self.M), dtype=complex)
            self.V = []
            mec_index = []
            self.fov_index_new = []
            #self.R_down = np.zeros(self.user, dtype=float)
            x_down = np.zeros(self.user, dtype=float)
            Interference_down = np.zeros(self.user, dtype=float)
            for i in range(len(select_action)):
                mec_index.append(select_action[i])
                self.fov_index_new.append(self.fov_index[i])
            vr_requirement_index = len(self.fov_index_new)
            V_new = np.zeros((vr_requirement_index, self.M), dtype=complex)
            for i in range(vr_requirement_index):
                for j in range(self.M):
                    V_new[i][j] = complex(random.random(), random.random())
            self.V = V_new
            for i in range(self.mec):
                for j in range(self.user):
                    for k in range(self.M):
                        self.H_down[i, j, k] = complex(random.gauss(0, 1), random.gauss(0, 1)) * \
                                               self.mec_user_distance_new[i, j]
            for i in range(len(self.fov_index_new)):
                for j in range(len(self.fov_index_new[i])):
                    for k in range(len(mec_index)):
                        if k != i:
                            Interference_down[self.fov_index_new[i][j]] += self.p_mec * abs(
                                self.H_down[mec_index[k], self.fov_index_new[i][j]].dot(self.V[k].reshape(-1, 1))) ** 2
            #print('Interference_down: ', Interference_down)
            #for i in range(self.user):
                #Interference_down[i] = Interference_down[i] * (10 ** (-2))
                #Interference_down[i] = 10 ** (-10)
                #Interference_down[i] = Interference_down[i] #* (10 ** (-5))
            #print('Interference_down: ', Interference_down)
            # print(self.fov_index_new)
            # print(self.V)
            for i in range(len(self.fov_index_new)):
                for j in range(len(self.fov_index_new[i])):
                    x_down[self.fov_index_new[i][j]] = self.I_M_down + self.p_mec * abs(self.H_down[mec_index[i], self.fov_index_new[i][j]].dot(self.V[i].reshape(-1, 1))) ** 2 / (Interference_down[self.fov_index_new[i][j]] + self.sigma)
            #print('x_down: ', x_down)
            for i in range(len(x_down)):
                if x_down[i] == 0:
                    x_down[i] = 1
            for i in range(self.user):
                self.R_down[i] += self.B * math.log(x_down[i], 2)
            #print('R_down: ', self.R_down)
        self.R_down /= iterations
