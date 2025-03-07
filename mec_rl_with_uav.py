import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import scipy.io as sio
import random
import datetime
import os
import imageio
import glob

# 计算二维直角坐标系下的两个点的距离
def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

# 离散化采样圆形区域，用于无人机随机动作的选取
# 以无人机的1个epoch内可以移动的距离为半径（即 6），找出圆内所有整数坐标点
# 总共有73个，为固定值：(73, {0: array([0, 0]), 1: array([1, 0]), 2: array([1, -1]), 3: array([1, 1]),......})
#【注意】目前是将移动距离修改成了60
def discrete_circle_sample_count(n):
    count = 0       # 统计生成的坐标点的数量
    move_dict = {}  # 存储所有整数坐标点
    for x in range(-n, n + 1):
        y_l = int(np.floor(np.sqrt(n**2 - x**2)))   #y 值的取值范围
        for y in range(-y_l, y_l + 1):
            move_dict[count] = np.array([y, x])
            count += 1
    return (count), move_dict

# 目标网络用于指导当前网络的更新
# 1. 获取当前网络和目标网络的权重参数
# 2. 线性插值，将两者的权重参数向量按比例混合
# 3. 将更新后的权重参数向量设置回目标网络
def update_target_net(model, target, tau=0.8):
    weights = model.get_weights()
    target_weights = target.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = (1 - tau) * weights[i]  + tau * target_weights[i]
    target.set_weights(target_weights)

# 多智能体系统中的参数合并算法，即：利用所有智能体的参数进行加权平均，并将结果返回给每个智能体，达到参数共享的目的
# nets 神经网络列表 nets[agent_no] 神经网络模型  omega 表示当前模型对自身参数的重要程度
def merge_fl(nets, omega=0.5):
    for agent_no in range(len(nets)):
        # 遍历所有的神经网络模型，获取每个模型当前的参数值 target_params
        target_params = nets[agent_no].get_weights()
        # 定义一个空的列表 other_params 用于存储其他神经网络模型的参数
        other_params = []
        for i, net in enumerate(nets):
            # 去除当前模型
            if i == agent_no:
                continue
            # 将当前模型之外的所有模型的参数都添加进other_params列表中。
            other_params.append(net.get_weights())
        for i in range(len(target_params)):
            # 将其他模型的相应参数取平均值，并乘以超参数(1 - omega)，再加上当前模型的相应参数乘以 omega
            others = np.sum([w[i] for w in other_params], axis=0) / len(other_params)
            target_params[i] = omega * target_params[i] + (1 - omega) * others
            # print([others.shape, target_params[i].shape])
        # 将新模型的参数值设置给当前模型
        nets[agent_no].set_weights(target_params)

# 输出最大值处的位置坐标
# 在移动半径为 6 的圆形范围内，找到 move_dist 中的最大值所对应的位置。
# 找到所有最大值的位置
# 计算这些位置与圆心的距离
# 获得距离最近的位置坐标
# 在给定的移动距离矩阵 move_dist 中，找到具有最大值的位置。
# 在多个最大值位置中，选择一个与无人机当前位置最近的坐标，从而为无人机选择下一步的行动位置。这种选择可以帮助无人机在特定环境中优化其移动路径，达到更有效的任务执行。
def circle_argmax(move_dist, uav_move_r):       #移动距离的矩阵 move_dist，无人机的移动半径 uav_move_r
    max_pos = np.argwhere(tf.squeeze(move_dist, axis=-1) == np.max(move_dist))
    pos_dist = np.linalg.norm(max_pos - np.array([uav_move_r, uav_move_r]), axis=1)
    return max_pos[np.argmin(pos_dist)]

# 无人机的actor和critic网络
# 观察图可以被看作是 2 通道图像，其中第三维指的是由边缘设备感知的数据包的聚合大小和累计数据包的个数。
# 整个网络的目标是：用来提取具有较高任务需求的区域，
# 通过平均池化将（None，121，121，2）观察图投射到（None，121，121，1）运动图上，以此来确定移动终端和无人机的轨迹动作，得到（None，13,13,1）类型的数据
def uav_actor(input_dim_list, cnn_kernel_size, uav_move_r):
    # state_map 用于输入状态图
    state_map = keras.Input(shape=input_dim_list[0])
    # 进行卷积操作，映射成一个输出通道为2的张量。即：卷积操作后，会生成2个通道的特征图。 过滤器为：3*3的矩阵，这样设置可以更好地捕捉局部特征
    cnn_message = layers.Conv2D(2, cnn_kernel_size, activation='relu', padding='same')(state_map)
    # 平均池化的窗口大小为：9*9，表示每个池化区域的大小为9个像素点
    cnn_message = layers.AveragePooling2D(pool_size=int(input_dim_list[0][0] / (2 * uav_move_r + 1)))(cnn_message)
    # 以20%的概率随机失活张量中的神经元
    cnn_message = layers.AlphaDropout(0.2)(cnn_message)
    # 最后一维输出一个标量值，表示模型对输入数据的预测结果
    move_out = layers.Dense(1, activation='relu')(cnn_message)  #一种移动动作
    model = keras.Model(inputs=[state_map], outputs=move_out)
    return model

def uav_critic(input_dim_list, cnn_kernel_size):
    #state_map 用于输入状态图，move_map 用于输入动作图
    state_map = keras.Input(shape=input_dim_list[0])
    move_map =  keras.Input(shape=input_dim_list[1])

    # 合并了最后一个维度，由原先的 2 通道，变成了 1 通道，所以卷积核也设置为 1
    cnn_message = layers.Dense(1, activation='relu')(state_map)
    cnn_message = layers.Conv2D(1, kernel_size=cnn_kernel_size, activation='relu', padding='same')(cnn_message)
    cnn_message = layers.AveragePooling2D(pool_size=cnn_kernel_size * 2)(cnn_message)
    cnn_message = layers.AlphaDropout(0.2)(cnn_message)
    cnn_message = layers.Flatten()(cnn_message)
    cnn_message = layers.Dense(2, activation='relu')(cnn_message)

    #将move_map展平，然后通过全连接层输出一个值
    move_mlp = layers.Flatten()(move_map)
    move_mlp = layers.Dense(1, activation='relu')(move_mlp)
    #将状态图处理的结果 cnn_message 和动作图处理的结果 move_mlp 合并为一个张量
    all_mlp = layers.concatenate([cnn_message, move_mlp], axis=-1)
    reward_out = layers.Dense(1, activation='relu')(all_mlp)

    model = keras.Model(inputs=[state_map, move_map], outputs=reward_out)
    return model

# 传感器的actor和critic网络
def center_actor(input_dim_list):
    #device_data_amount_k: 表示设备的数据量输入。
    ##device_compute_k: 表示设备的计算能力输入。
    #device_transfer_k: 表示设备的传输能力输入。
    device_data_amount_k  = keras.Input(shape=input_dim_list[0])
    device_compute_k  = keras.Input(shape=input_dim_list[1])
    device_transfer_k  = keras.Input(shape=input_dim_list[2])

    # 将所有输入连接在一起
    # concatenated = layers.Concatenate()([device_distance_k, device_data_amount_k, device_compute_k, device_transfer_k])
    #axis=-1表示在最后一个维度上进行连接
    concatenated = layers.concatenate([device_data_amount_k, device_compute_k, device_transfer_k], axis=-1)
    # concatenated = tf.squeeze(concatenated, axis=-1)
    # concatenated = layers.GlobalAveragePooling1D()(concatenated)
    # mlp_message = layers.Dense(64, activation='relu')(concatenated)
    mlp_message = layers.Dense(32, activation='relu')(concatenated)
    off_who = layers.Dense(10, activation='softmax')(mlp_message)        # 9种卸载动作 + 1种D2D动作 = 10种动作
    model = keras.Model(inputs=[device_data_amount_k, device_compute_k, device_transfer_k], outputs=[off_who])
    return model

def center_critic(input_dim_list, op):
    device_data_amount_k  = keras.Input(shape=input_dim_list[0])
    device_compute_k  = keras.Input(shape=input_dim_list[1])
    device_transfer_k  = keras.Input(shape=input_dim_list[2])
    #表示执行操作的输入
    execute_op  = keras.Input(shape=op)

    # 将所有输入连接在一起
    concatenated = layers.Concatenate()([device_data_amount_k, device_compute_k, device_transfer_k])
    mlp_message = layers.Dense(16, activation='relu')(concatenated)
    mlp_message = layers.Dense(8, activation='relu')(mlp_message)
    off_who = layers.Dense(10, activation='relu')(mlp_message)  # 9种卸载动作 + 1种D2D动作 = 10种动作

    # off_who = tf.expand_dims(off_who, axis=-1)
    # execute_op = tf.expand_dims(execute_op, axis=-1)
    # execute_op_mlp = layers.Dense(1, activation='relu')(execute_op)
    all_mlp = layers.concatenate([off_who, execute_op], axis=-1)
    reward_out = layers.Dense(1, activation='relu')(all_mlp)
    # reward_out = tf.squeeze(reward_out, axis=-1)
    # reward_out = layers.Dense(1, activation='relu')(reward_out)

    model = keras.Model(inputs=[device_data_amount_k, device_compute_k, device_transfer_k, execute_op], outputs=reward_out)
    return model

class MEC_RL_With_Uav(object):
    def __init__(self, env, tau, gamma, lr_ua, lr_uc, lr_ca, lr_cc, batch, epsilon=0.2):
        # 环境
        self.env = env

        # 各设备
        self.uavs = self.env.uavs
        self.sensors = self.env.sensors
        self.servers = self.env.servers
        self.uav_num = self.env.uav_num
        self.server_num = self.env.server_num
        self.sensor_num = self.env.sensor_num
        self.device_num = self.env.server_num + self.env.uav_num + 1

        # 观测半径和移动半径
        self.uav_obs_r = self.env.uav_obs_r
        self.uav_collect_r = self.env.uav_collect_r
        self.server_collect_r = self.env.server_collect_r
        self.uav_move_r = self.env.uav_move_r
        self.sensor_move_r = self.env.sensor_move_r

        # 无人机的状态、动作输出形状
        self.state_map_shape = (self.env.uav_obs_r*2 + 1, self.env.uav_obs_r*2 + 1, 2)
        self.move_map_shape = ( self.env.uav_move_r*2 + 1, self.env.uav_move_r*2 + 1)

        # 传感器的状态、动作输出形状
        self.last_sensor_no =  []
        self.device_distance =  (self.uav_num + self.server_num + 1)
        self.device_data_amount = (self.uav_num + self.server_num + 1)
        self.device_compute = (self.uav_num + self.server_num + 1)
        self.device_transfer = (self.uav_num + self.server_num + 1)

        self.execute_op_shape = (self.uav_num + self.server_num + 2)

        # 无人机移动动作的随机选取
        self.move_count, self.move_dict = discrete_circle_sample_count(self.env.uav_move_r)

        # 一些超参数
        self.tau = tau
        self.cnn_kernel_size = 3
        self.gamma = gamma
        self.lr_uc = lr_uc
        self.lr_ua = lr_ua
        self.lr_cc = lr_cc
        self.lr_ca = lr_ca
        self.batch_size = batch
        self.epsilon = epsilon

        # 经验
        self.uav_memory = {}
        self.uav_softmax_memory = {}
        self.center_memory = []
        self.sensor_softmax_memory = {}
        self.sample_prop = 1 / 4

        # 设备网络与优化器相关设置
        self.uav_actors = []
        self.uav_critics = []
        self.target_uav_actors = []
        self.target_uav_critics = []
        # 分别存储与无人机 actor 和 critic 模型的优化器
        self.uav_actor_opt = []
        self.uav_critic_opt = []
        #初始化sensor两个优化器, Adam 优化器是由于其在深度学习中表现出的有效性
        self.center_actor_opt = keras.optimizers.Adam(learning_rate=lr_ca)
        self.center_critic_opt = keras.optimizers.Adam(learning_rate=lr_cc)

        self.summaries = {}

        for _ in range(self.uav_num):
            self.uav_critic_opt.append(keras.optimizers.Adam(learning_rate=lr_uc))
            self.uav_actor_opt.append(keras.optimizers.Adam(learning_rate=lr_ua))
            # 初始化所有的actor网络
            new_uav_actor = uav_actor([self.state_map_shape], self.cnn_kernel_size, self.uav_move_r)
            target_uav_actor = uav_actor([self.state_map_shape], self.cnn_kernel_size, self.uav_move_r)
            update_target_net(new_uav_actor, target_uav_actor, tau=0)
            self.uav_actors.append(new_uav_actor)
            self.target_uav_actors.append(target_uav_actor)
            # 初始化所有的critic网络
            new_uav_critic = uav_critic([self.state_map_shape, self.move_map_shape], self.cnn_kernel_size)
            target_uav_critic = uav_critic([self.state_map_shape, self.move_map_shape], self.cnn_kernel_size)
            update_target_net(new_uav_critic, target_uav_critic, tau=0)
            self.uav_critics.append(new_uav_critic)
            self.target_uav_critics.append(target_uav_critic)

        self.center_actor = center_actor([self.device_data_amount, self.device_compute, self.device_transfer])
        self.center_critic = center_critic([self.device_data_amount, self.device_compute, self.device_transfer], self.execute_op_shape)
        self.target_center_actor = center_actor([self.device_data_amount, self.device_compute, self.device_transfer])
        self.target_center_critic = center_critic([self.device_data_amount, self.device_compute, self.device_transfer], self.execute_op_shape)
        update_target_net(self.center_actor, self.target_center_actor, tau=0)
        update_target_net(self.center_critic, self.target_center_critic, tau=0)
        os.makedirs('new_logs/model_figs', exist_ok=True)
        keras.utils.plot_model(self.center_actor, 'new_logs/model_figs/new_center_actor.png', show_shapes=True)
        keras.utils.plot_model(self.center_critic, 'new_logs/model_figs/new_center_critic.png', show_shapes=True)
    #在给定的环境中生成并执行无人机和传感器的决策, epoch当前的训练周期或轮次
    def actor_act(self, epoch):
        tmp = random.random()   # 随机生成一个0-1之间的小数
        if tmp >= self.epsilon and epoch >= 16:
             # 存储神经网络输出的动作、soft_max形式、当前状态
            uav_act_list = []
            uav_softmax_list = []
            uav_cur_state_list = []

            sensor_act_list = []
            sensor_softmax_list = []
            sensor_cur_state_list = []
            new_sensor_cur_state_list = []

            #【 第一步： 无人机生成移动决策 】
            for i, uav in enumerate(self.uavs):
                #从环境中获取当前无人机 uav 的观察状态（即感知到的周围信息），并使用 tf.expand_dims 将其增加一个维度
                state_map = tf.expand_dims(self.env.get_uav_obs(uav), axis=0)
                assemble_state = [state_map]
                uav_cur_state_list.append(assemble_state)

                # action_output的输出shape为：（1，13，13，1），move_dist的shape为 （13，13，1）
                # 对神经网络的输出结果进行处理，计算move，move_ori的shape为 （2，）
                # 将新坐标系原点与无人机的移动位置对齐，并且先取的 y，后取的 x
                # 计算move_softmax，即：在shape为（1，13，13，1）的全零数据中心，只有 move_ori 这一点的值为 1
                # 拓展一个维度，move_softmax的shape变为（1，13，13，1）
                #调用该无人机的 actor 网络进行预测，action_output 是预测结果，move_dist 是网络输出的移动距离矩阵
                action_output = self.uav_actors[i].predict(assemble_state)
                move_dist = action_output[0]
                #使用自定义函数 circle_argmax 获取在 move_dist 中最大的值的位置（即无人机的理想移动方向），并根据该位置计算无人机的实际移动坐标
                move_ori = circle_argmax(move_dist, self.uav_move_r)
                move = [move_ori[1] - self.uav_move_r, move_ori[0] - self.uav_move_r]
                # 创建一个与 move_dist 相同形状的全零数组，
                # 将理想移动方向 move_ori 位置的值设为 1，形成一个概率分布（softmax形式），
                # 最后增加一个维度以适应后续处理
                move_softmax = np.zeros(move_dist.shape)
                move_softmax[move_ori] = 1
                move_softmax = tf.expand_dims(move_softmax, axis=0)
                # 存储，均是以数组的形式进行的存储
                #将计算得到的移动决策和对应的概率分布存储到uav_act_list和 uav_softmax_list中
                uav_act_list.append([move])
                uav_softmax_list.append([move_softmax])

                # 将一些数据信息保存到 MATLAB 格式文件中
                # 将当前无人机的状态和移动距离数据保存到一个 MATLAB 格式的文件 debug.mat 中
                sio.savemat('debug.mat', {'state': self.env.get_uav_obs(uav), 'move': move_dist})
            print(uav_act_list)

            # 直接执行无人机的移动，获取新的无人机的位置信息
            for i, uav in enumerate(self.uavs):
                self.uav_move(uav_act_list[i], uav)
                if(epoch <= 2000):
                    uav.position_x.append(uav.position[0])
                    uav.position_y.append(uav.position[1])
                if(epoch >= 8000):
                    uav.position_x_last.append(uav.position[0])
                    uav.position_y_last.append(uav.position[1])

            #【 第二步：传感器生成卸载决策 】
            self.last_sensor_no = []
            for n, sensor in enumerate(self.sensors):
                # 参数 False 用于区分 网络生成决策（False），还是随机生成决策（True）
                # if_covered 是表示是否被设备覆盖，若没有被覆盖直接(True)，则直接跳过此情况，不进行网络的生成
                if_covered, amount, compute, transfer = self.env.get_sensor_obs(sensor, False)

                if(if_covered):
                    continue

                self.last_sensor_no.append(sensor.no)
                device_data_amount = tf.expand_dims(amount, axis=0)
                device_compute = tf.expand_dims(compute, axis=0)
                device_transfer = tf.expand_dims(transfer, axis=0)
                sensor_cur_state_list.append([device_data_amount, device_compute, device_transfer])
                # 调用sensor的演员网络（self.center_actor）来预测传感器的卸载操作
                action_output = self.center_actor.predict([device_data_amount, device_compute, device_transfer])
                execute_op_dist = action_output[0]
                # 初始化一个长度为10的列表execute，并将所有值设为0
                execute = [0] * 10
                # 找出execute_op_dist中最大值的索引，并将execute列表中该索引位置的值设置为1
                execute[np.argmax(execute_op_dist)] = 1

                # 创建了一个形状为 self.execute_op_shape 的全零数组
                execute_op_softmax = np.zeros(self.execute_op_shape)
                # 选取最大值的索引并设置对应位置为1
                execute_op_softmax[np.argmax(execute_op_dist)] = 1
                # 增加数组的维度
                execute_op_softmax = tf.expand_dims(execute_op_softmax, axis=0)
                # 将选择的卸载执行操作添加到列表中
                sensor_act_list.append(execute)
                # 将构建好的 execute_op_softmax（表示卸载操作的概率分布）添加到 sensor_softmax_list 列表中
                sensor_softmax_list.append([execute_op_softmax])
            print(sensor_act_list)

            #【 第三步：具体执行两种决策 】
            new_state_maps, uav_rewards, sensor_rewards = self.env.step(uav_act_list, sensor_act_list, False)

            #【 第四步：获取新的执行后的状态， 并将一整条数据存储到memory中 】
            for i, uav in enumerate(self.uavs):
                state_map = tf.expand_dims(new_state_maps[i], axis=0)
                new_states = [state_map]
                # 判断无人机是否在经验记忆中
                if uav.no in self.uav_memory.keys():
                    self.uav_memory[uav.no].append([uav_cur_state_list[i], uav_softmax_list[i], uav_rewards[i], new_states])
                else:
                    self.uav_memory[uav.no] = [[uav_cur_state_list[i], uav_softmax_list[i], uav_rewards[i], new_states]]

            # 这段代码的主要功能是遍历所有传感器，判断哪些传感器是最近被处理的，
            # 并且获得它们的当前状态（包括数据量、计算能力和传输能力）。
            # 将这些状态信息以及相关的动作和奖励信息记录到一个中心内存列表中，以支持后续的学习和决策过程。
            # 这一过程确保相关数据能够被充分利用，从而提高模型的训练效果
            count_device_distance = 0
            for n, sensor in enumerate(self.sensors):
                #检查传感器是否在最后一个被处理的传感器列表中,确保只处理最近被覆盖的传感器
                if n in self.last_sensor_no:
                    # 获取传感器的状态信息
                    if_covered, amount, compute, transfer = self.env.get_sensor_obs(sensor, True)

                    device_data_amount = tf.expand_dims(amount, axis=0)
                    device_compute = tf.expand_dims(compute, axis=0)
                    device_transfer = tf.expand_dims(transfer, axis=0)
                    new_sensor_cur_state_list.append([device_data_amount, device_compute, device_transfer])
                    
                    self.center_memory.append([sensor_cur_state_list[count_device_distance], sensor_softmax_list[count_device_distance], sensor_rewards[count_device_distance], new_sensor_cur_state_list[count_device_distance]])
                    count_device_distance += 1
        else:
            # 随机移动决策，没有保存数据到经验池中
            uav_act_list = []
            sensor_act_list = []
            # 为无人机生成随机移动决策
            for i, uav in enumerate(self.uavs):
                # 从 self.move_dict.values() 中随机选择一个移动坐标，并将其添加到 uav_act_list 中
                move = random.sample(list(self.move_dict.values()), 1)[0]
                # 调用 tolist() 方法的作用是将这个数组转换为标准的 Python 列表
                uav_act_list.append([move.tolist()])

            # 执行无人机的移动，获取新的无人机的位置信息
            for i, uav in enumerate(self.uavs):
                self.uav_move(uav_act_list[i], uav)
                if(epoch <= 2000):
                    uav.position_x.append(uav.position[0])
                    uav.position_y.append(uav.position[1])
                if(epoch >= 8000):
                    uav.position_x_last.append(uav.position[0])
                    uav.position_y_last.append(uav.position[1])
            
            # 随机卸载决策
            for i, sensor in enumerate(self.sensors):
                execute = [0] * (self.device_num + 1)
                # 随机选择一个索引，将其位置的值设置为1
                execute[np.random.randint(self.device_num + 1)] = 1    # 增加D2D选项
                sensor_act_list.append(execute)
            new_state_maps, uav_rewards, sensor_rewards = self.env.step(uav_act_list, sensor_act_list, True)

        return uav_rewards, sensor_rewards
            
    def replay(self):
        #【 第一步:无人机部分 】
        for no, uav_memory in self.uav_memory.items():
            # 当经验记忆的大小大于 batch_size=128 的时候对经验进行采样，进行训练
            if len(uav_memory) < self.batch_size:
                continue
            # 从经验记忆中进行采样，选取 32 个最近的数据，从最近的256个数据中，随机采样 96 个数据。 
            # 最终得到的数组大小为 128 个
            samples = uav_memory[-int(self.batch_size * self.sample_prop):] + random.sample(uav_memory[-self.batch_size * 2:], int(self.batch_size * (1 - self.sample_prop)))

            # state_map的shape为 （128，121，121，2）
            # move_softmax的shape为 （128，13，13，1）
            # a_reward的shape为 （128，1）
            # new_state_map的shape为（128，121，121，2）
            state_map = np.vstack([sample[0][0] for sample in samples]) # sample[0]表示当前状态样本，sample[0][0]表示该状态的具体数据
            move_softmax = np.vstack([sample[1][0] for sample in samples]) # sample[1] 表示存储的动作，sample[1][0] 指的是该动作的具体概率分布
            a_reward = tf.expand_dims([sample[2] for sample in samples], axis=-1) # sample[2] 表示当前样本的奖励,并使用 tf.expand_dims() 将这个一维数组增加一个维度
            new_state_map = np.vstack([sample[3][0] for sample in samples]) # sample[3] 表示下一个状态，sample[3][0] 则是提取该状态的具体数据

            # 目标网络输出的动作的shape为（128，13，13，1）
            # 目标网络输出的奖励shape为 （128，1） 即：未来的累计回报值
            # 目标qs的shape为 （128，1）即：未来计算当前状态下某个动作的价值，将 当前状态下选择动作带来的即时回报 与 未来的会报纸 组合起来。
            # 求得总体价值 target_qs
            new_actions = self.target_uav_actors[no].predict([new_state_map])
            q_future = self.target_uav_critics[no].predict([new_state_map, new_actions])
            target_qs = a_reward + q_future * self.gamma

            # 通过优化神经网络模型提升智能体的决策策略。
            # 使用 TensorFlow的自动微分工具 tf.GradientTape() 计算损失函数对神经网络模型权重参数的梯度，根据梯度更新神经网络模型的参数
            with tf.GradientTape() as tape:
                # q_values的shape为 （128，1）即：当前神经网络模型对于当前状态和动作的价值估计，预测价值
                # uc_error的shape为 （128，1）即：将 预测价值 与 目标价值 的差值，表示为误差
                # 进行平方得到损失函数
                # 通过调用当前无人机的critic网络,传入当前状态（state_map）和对应的动作概率分布（move_softmax），以预测其对应的q值（即价值估计）
                q_values = self.uav_critics[no]([state_map, move_softmax])
                uc_error = q_values - tf.cast(target_qs, dtype=tf.float32)
                uc_loss = tf.reduce_mean(tf.math.square(uc_error)) # 使用均方误差（MSE）来计算损失,uc_loss为当前批次中所有样本的平均误差平方
            # uc_grad中的每个元素表示损失函数uc_loss对应的参数的梯度值
            # 使用 tape.gradient() 方法计算损失函数 对于神经网络模型权重参数的梯度 uc_grad。
            # self.uav_critics[no].trainable_variables 表示所有可训练参数（即权重和偏置）
            uc_grad = tape.gradient(uc_loss, self.uav_critics[no].trainable_variables)

            # 将梯度 uc_grad 应用于神经网络模型的权重参数上，更新神经网络模型的参数
            # self.uav_critic_opt[no]是优化器对象，可以根据梯度更新参数
            # zip(uc_grad, self.uav_critics[no].trainable_variables) 将每一个参数的梯度和其对应的参数拼成一个元组，传递给优化器进行更新
            self.uav_critic_opt[no].apply_gradients(zip(uc_grad, self.uav_critics[no].trainable_variables))

            # 实现深度强化学习中Actor模型的训练过程
            with tf.GradientTape() as tape:
                # 显式声明需要监视的变量，即：Actor模型的可训练参数，以便在计算梯度时跟踪其变化
                tape.watch(self.uav_actors[no].trainable_variables)
                # actions的shape为（128，13，13，1）
                # 计算当前状态下选择某个动作的概率分布actions，即：当前神经网络模型对于当前状态的动作概率估计
                actions = self.uav_actors[no]([state_map])
                # new_r的shape为（128，1）
                # 计算当前状态下选择动作actions后的累计回报值new_r，即：Critic模型的输出结果
                new_r = self.uav_critics[no]([state_map, actions])
                # 计算当前状态下的累计回报的均值作为Actor的损失函数ua_loss
                ua_loss = tf.reduce_mean(new_r)  # 计算输入张量的平均值
            # ua_grad中的每个元素表示损失函数ua_loss对应的参数的梯度值
############为什么actor参数更新不是梯度上升呢？
            ua_grad = tape.gradient(ua_loss, self.uav_actors[no].trainable_variables)
            # 使用优化器（self.uav_actor_opt[no]）来更新Actor模型的参数
            self.uav_actor_opt[no].apply_gradients(zip(ua_grad, self.uav_actors[no].trainable_variables))

            self.summaries['uav%s-critic_loss' % no] = uc_loss
            self.summaries['uav%s-actor_loss' % no] = ua_loss
        
        #【 第二步:传感器部分 】
        if len(self.center_memory) < self.batch_size:
            return
        samples = self.center_memory[-int(self.batch_size * self.sample_prop):] + random.sample(self.center_memory[-self.batch_size * 2:], int(self.batch_size * (1 - self.sample_prop)))
        device_data_amount = np.vstack([sample[0][0] for sample in samples]) # sample[0][0] 表示当前样本状态的设备数据量部分
        device_compute = np.vstack([sample[0][1] for sample in samples]) # ample[0][1] 表示当前样本状态的设备计算能力部分
        device_transfer = np.vstack([sample[0][2] for sample in samples]) # sample[0][2] 表示当前样本状态的设备传输能力部分
        execute_op_softmax = np.vstack([sample[1][0] for sample in samples]) # sample[1][0] 表示当前样本中存储的动作选择的概率分布
        c_reward = tf.expand_dims([sample[2] for sample in samples], axis=-1) # sample[2] 表示当前样本的奖励信息

        new_device_data_amount = np.vstack([sample[3][0] for sample in samples]) # sample[3][0] 代表新的设备数据量
        new_device_compute = np.vstack([sample[3][1] for sample in samples]) # sample[3][1] 代表新的设备计算能力
        new_device_transfer = np.vstack([sample[3][2] for sample in samples]) # sample[3][2] 代表新的设备传输能力
        new_actions = self.target_center_actor.predict([new_device_data_amount, new_device_compute, new_device_transfer])
        cq_future = self.target_center_critic.predict([[new_device_data_amount, new_device_compute, new_device_transfer], new_actions])
        c_target_qs = c_reward + cq_future * self.gamma

        with tf.GradientTape() as tape:   
            tape.watch(self.center_critic.trainable_variables)
            cq_values = self.center_critic([[device_data_amount, device_compute, device_transfer], execute_op_softmax])
            cc_error = cq_values - tf.cast(c_target_qs, dtype=tf.float32)
            cc_loss = tf.reduce_mean(tf.math.square(cc_error))
        cc_grad = tape.gradient(cc_loss, self.center_critic.trainable_variables)
        self.center_critic_opt.apply_gradients(zip(cc_grad, self.center_critic.trainable_variables))
        with tf.GradientTape() as tape:
            tape.watch(self.center_actor.trainable_variables)
            c_act = self.center_actor([device_data_amount, device_compute, device_transfer])
            ca_loss = tf.reduce_mean(self.center_critic([[device_data_amount, device_compute, device_transfer], c_act]))
############为什么actor参数更新不是梯度上升呢？
        ca_grad = tape.gradient(ca_loss, self.center_actor.trainable_variables)
        self.center_actor_opt.apply_gradients(zip(ca_grad, self.center_actor.trainable_variables))

        self.summaries['center-critic_loss'] = cc_loss
        self.summaries['center-actor_loss'] = ca_loss

    # 智能体完成一定训练轮数episode后，保存智能体的模型
    def save_model(self, episode, time_str):
        for i in range(self.uav_num):
            self.uav_actors[i].save('new_logs/models/{}/uav-actor-{}_episode{}.h5'.format(time_str, i, episode))
            self.uav_critics[i].save('new_logs/models/{}/uav-critic-{}_episode{}.h5'.format(time_str, i, episode))
        self.center_actor.save('new_logs/models/{}/center-actor_episode{}.h5'.format(time_str, episode))
        self.center_critic.save('new_logs/models/{}/center-critic_episode{}.h5'.format(time_str, episode))

    # 开始训练
    def train(self, max_epochs=2000, max_step=500, up_freq=8, render=False, render_freq=1, FL=False, FL_omega=0.5, anomaly_edge=False):
        cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_log_dir = 'new_logs/fit/' + cur_time
        env_log_dir = 'new_logs/picture/picture' + cur_time
        record_dir = 'new_logs/records/' + cur_time
        os.makedirs(env_log_dir, exist_ok=True)
        os.makedirs(record_dir, exist_ok=True)
        os.makedirs('new_logs/models/' + cur_time)
        #创建一个TensorFlow记录器，用于在训练期间记录和保存模型的训练日志和指标
        summary_writer = tf.summary.create_file_writer(train_log_dir)
        episode, steps, epoch, total_reward = 0, 0, 0, 0
        finish_length = []
        finish_size = []

        while epoch < max_epochs:
            print('epoch%s' % epoch)
            
            if render and (epoch % 100 == 1):
                self.env.render(env_log_dir, epoch, True)
            # self.env.render(env_log_dir, epoch, True)
            
            # max_step取的是200，表示为episode，在一个eposide中最多执行200步
            # 当执行200步后，即为一个episode，开始进行一次训练，并进行一次经验回放
            if steps >= max_step:
                episode += 1
                # 删除最后面的-256个数据记忆
                #self.uav_memory 是一个字典，用于存储每个无人机（UAV）的记忆或经验。每个键 m 代表一架无人机的编号
                #清除比较旧的记忆，以便只保留最新的经验，通常是最新的 batch_size 个经验供训练使用
                for m in self.uav_memory.keys():
                    del self.uav_memory[m][0:-self.batch_size * 2]
                del self.center_memory[0:-self.batch_size * 2]
                # for n in self.center_memory.keys():
                #     del self.center_memory[n][0:-self.batch_size * 2]
                print('episode {}: {} total reward, {} steps, {} epochs'.format(episode, total_reward / steps, steps, epoch))
                #summary_writer将训练过程中的各类指标记录下来。
                with summary_writer.as_default():
                    tf.summary.scalar('Main/episode_reward', total_reward, step=episode)
                    tf.summary.scalar('Main/episode_steps', steps, step=episode)
                #这一行确保所有的记录数据都被写入文件
                summary_writer.flush()
                
                self.save_model(episode, cur_time)
                steps = 0
                total_reward = 0

            # 开始行动，获得奖励
            cur_uav_rewards, cur_sensor_rewards = self.actor_act(epoch)

            # 经验回放
            self.replay()

            for i, uav in enumerate(self.uavs):
                if(epoch == 1999):
                    file_path1 = 'new_logs/array_x.npy' + str(i)
                    np.save(file_path1, uav.position_x)
                    file_path2 = 'new_logs/array_y.npy' + str(i)
                    np.save(file_path2, uav.position_y)
                if(epoch == 9999):
                    file_path3 = 'new_logs/array_x_last.npy' + str(i)
                    np.save(file_path3, uav.position_x_last)
                    file_path4 = 'new_logs/array_y_last.npy' + str(i)
                    np.save(file_path4, uav.position_y_last)

            #判断是用来确定是否到了更新网络的时机，up_freq 是一个用于控制目标网络更新的频率的参数
            if epoch % up_freq == 1:
                if FL:  #如果设置了FL（Federated Learning，即联邦学习），则会进行参数的聚合
                    merge_fl(self.uav_actors, FL_omega)
                    merge_fl(self.uav_critics, FL_omega)
                for i in range(self.uav_num):
                    update_target_net(self.uav_actors[i], self.target_uav_actors[i], self.tau)
                    update_target_net(self.uav_critics[i], self.target_uav_critics[i], self.tau)
                update_target_net(self.center_actor, self.target_center_actor, self.tau)
                update_target_net(self.center_critic, self.target_center_critic, self.tau)
            
            total_reward += np.sum(cur_uav_rewards) + np.sum(cur_sensor_rewards)
            steps += 1
            epoch += 1

            with summary_writer.as_default():
                if self.uav_memory:
                    #记录损失
                    if len(self.uav_memory[0]) > self.batch_size:
                        tf.summary.scalar('Center/center_actor_loss', self.summaries['center-actor_loss'], step=epoch)
                        tf.summary.scalar('Center/center_critic_loss', self.summaries['center-critic_loss'], step=epoch)
                        for uav_count in range(self.uav_num):
                            tf.summary.scalar('Uav/uav%s_actor_loss' % uav_count, self.summaries['uav%s-actor_loss' % uav_count], step=epoch)
                            tf.summary.scalar('Uav/uav%s_critic_loss' % uav_count, self.summaries['uav%s-critic_loss' % uav_count], step=epoch)
                tf.summary.scalar('Main/cur_uav_rewards', np.sum(cur_uav_rewards), step=epoch)
                tf.summary.scalar('Main/cur_sensor_rewards', np.sum(cur_sensor_rewards), step=epoch)

            summary_writer.flush()

        # 保存最终模型
        self.save_model(episode, cur_time)

        # 渲染成动图
        self.env.render(env_log_dir, epoch, True)
        img_paths = glob.glob(env_log_dir + '/*.png')
        img_paths.sort(key=lambda x: int(x.split('.')[0].split('\\')[-1]))
        gif_images = []
        for path in img_paths:
            gif_images.append(imageio.imread(path))
        imageio.mimsave(env_log_dir + '/all.gif', gif_images, fps=15)

    # 该代码的主要功能是控制无人机的移动。
    # 首先，初始化移动向量并根据最大移动半径调整其大小。
    # 然后，根据随机生成的方向改变无人机的移动方向，确保移动向量不会超出设定边界，并更新无人机的位置。
    # 这种移动策略不仅支持计划航向，还考虑了随机行为和边界碰撞的处理，以确保无人机在模拟环境中移动的合理性和稳定性。
    def uav_move(self, uav_act, uav):
        #初始化 uav 的 action.move 属性为一个零数组，表示无人机开始时不移动
        uav.action.move = np.zeros(2)
        #检查无人机的计划移动向量 uav_act[0] 的大小（即模）是否超过其最大移动半径 uav.uav_move_r
        if np.linalg.norm(uav_act[0]) > uav.uav_move_r:
            #按比例缩小移动向量，使其大小等于 uav.uav_move_r
            uav_act[0] = [int(uav_act[0][0] * uav.uav_move_r / np.linalg.norm(uav_act[0])), int(uav_act[0][1] * uav.uav_move_r / np.linalg.norm(uav_act[0]))]
        #检查 uav_act[0] 是否为零向量（即没有朝任何方向移动），并以 50% 的概率决定是否生成一个随机移动
        if not np.count_nonzero(uav_act[0]) and np.random.rand() > 0.5:
            mod_x = np.random.normal(loc=0, scale=1)
            mod_y = np.random.normal(loc=0, scale=1)
            #将随机数限制在 -1 到 1 之间，并缩放到最大移动半径的一半，以限制随机移动的范围。
            mod_x = int(min(max(-1, mod_x), 1) * uav.uav_move_r / 2)
            mod_y = int(min(max(-1, mod_y), 1) * uav.uav_move_r / 2)
            uav_act[0] = [mod_x, mod_y]
        uav.action.move = np.array(uav_act[0])
        new_x = uav.position[0] + uav.action.move[0]
        new_y = uav.position[1] + uav.action.move[1]
        if new_x < 0 or new_x > 200 - 1:
            uav.action.move[0] = -uav.action.move[0]
        if new_y < 0 or new_y > 200 - 1:
            uav.action.move[1] = -uav.action.move[1]
        uav.position += uav.action.move

        
