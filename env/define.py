import numpy as np
import random
import logging
from random import randint
from env import traffic

logging.basicConfig(level=logging.WARNING)

#【 第一步：动作状态的声明 】
# 移动终端智能体的卸载策略动作 sensor.action.offload
# 边缘设备无人机的移动决策动作 uav.action.move
class Action(object):
    def __init__(self):
        self.move = None
        self.offload = []
# 智能体的状态，position，目前没使用，设计到的有些多，比较复杂，就先注释掉了
# class State(object):

#【 第二步：城市场景下的传输和收集速率的计算 】
collecting_channel_param = {'suburban': (4.88, 0.43, 0.1, 21),          #郊区
                            'urban': (9.61, 0.16, 1, 20),               #城市
                            'dense-urban': (12.08, 0.11, 1.6, 23),      #密集城市
                            'high-rise-urban': (27.23, 0.08, 2.3, 34)}  #高层建筑
collecting_params = collecting_channel_param['urban']
a = collecting_params[0]        #a 和 b 通常是用于信道模型中的特征参数，这些参数可能与信道的衰减状态、环境影响等有关。
b = collecting_params[1]
yita0 = collecting_params[2]    #yita0 和 yita1 通常代表系统中的某些效率或损耗参数，比如信道的增益或噪声参数等。
yita1 = collecting_params[3]
carrier_f = 2.5e9                   #载波频率G2.5Hz

#【 第三步：移动终端、服务器、无人机 各设备的参数声明 】
# 移动终端智能体（传感器），1. 数据生成 2. 任务卸载 3. 本地执行
class Sensor(object):
    sensor_count = 0
    def __init__(self, position, sensor_move_r):
        # 设备编号、位置、奖励、计算速率、移动半径(速度)、动作 sensor.action.offload
        self.no = Sensor.sensor_count
        Sensor.sensor_count += 1
        self.position = position
        self.computing_rate = 100
        self.sensor_move_r = sensor_move_r
        self.action = Action()

        # 数据生成参数：生成率、生成阈值、泊松分布参数、最大数据大小
        self.sensor_data_gen_rate = 1       #生成数据的速率，在每个时间步长能够生成平均1个单位的数据
        self.gen_threshold = 0.3            #生成数据的阈值，只有当生成的随机数小于阈值，传感器才会生成数据
        self.total_data = {}
        self.lam = 1e3                      #在每个时隙中由泊松分布生成的数据量期望为1000。这通常适用于高数据生成速率的场景
        self.sensor_max_data_size = 2e3     #传感器生成数据的最大大小或者容量，设定为2000。
        
        # 传输速率参数：
        self.ptr = 0.2
        self.h = 5
        self.noise = 1e-13
        # 自己设置的传感器的带宽，默认传输带宽相同
        self.sensor_bandwidth = 1e3
        self.noise_power = 1e-13 * self.sensor_bandwidth

# 边缘设备MEC服务器 1. 本地执行
class EdgeServer(object):
    server_count = 0
    def __init__(self, pos, server_collect_r):
        # 设备编号、位置、计算速率
        self.no = EdgeServer.server_count
        EdgeServer.server_count += 1
        self.position = pos
        self.computing_rate = 1000

        # 设备的收集半径
        self.server_collect_r = server_collect_r

        # 目前边缘设备没有使用到 存储数据的字段 total_data，
######### 【 假设 】默认无人机和服务器能量无限，只要符合范围要求，能够传递过去的任务都会被执行完，产生时延。
        # self.total_data = {}

# 边缘设备无人机，1. 任务收集 2. 本地执行
class EdgeUav(object):
    edge_count = 0
    def __init__(self, pos, uav_obs_r, uav_collect_r, uav_move_r):
        # 设备编号、位置、奖励、计算速率、动作 uav.action.move 
        self.no = EdgeUav.edge_count
        EdgeUav.edge_count += 1
        self.position = pos
        self.computing_rate = 500
        self.action = Action()

        # 设备的移动半径(速度)、观察和收集半径
        self.uav_obs_r = uav_obs_r
        self.uav_collect_r = uav_collect_r
        self.uav_move_r = uav_move_r

        self.position_x = []
        self.position_y = []
        self.position_x_last = []
        self.position_y_last = []
        
        # 收集速率参数： 
        self.h = 5
        self.ptr_col = 0.2

        # self.total_data = {}
        # self.state = AgentState()
    
# 【 第四步： 生成设备以及初始化设备位置、step()执行卸载决策 】
class MEC_world(object):
    def __init__(self, map_size, uav_num, server_num, sensor_num, uav_obs_r, uav_collect_r, server_collect_r, uav_move_r, sensor_move_r):
        # 场景大小、地图场景状态
        self.map_size = map_size
        # 使用三维数组表示地图的信息。
        # 为之后作为 State 输入到网络中做准备，第一维和第二维用于表示位置信息，第三维用于表示该设备的具体情况
        self.DS_state = np.ones([map_size, map_size, 2])

        # 各设备的定义 以及 数量
        self.uavs = []
        self.servers = []
        self.sensors = []
        self.sensor_count = sensor_num
        self.server_count = server_num
        self.uav_count = uav_num

        # 各设备的观察半径，收集半径、移动半径(速度)的设置
        self.uav_obs_r = uav_obs_r
        self.uav_collect_r = uav_collect_r
        self.server_collect_r = server_collect_r
        self.uav_move_r = uav_move_r
        self.sensor_move_r = sensor_move_r

        # 用于存放每一个传感器的时延
        #【 假设 】以前是每一个传感器都会有自己的卸载决策，现在是只有被设备覆盖的传感器，才能有自己的卸载决策
        # self.sensor_delay = [0] * self.sensor_count
        self.sensor_delay = []

        # 设备创建，随机生成它们的位置
        #传感器的位置。这里创建了两个列表，分别表示传感器的 x 坐标和 y 坐标。位置的范围被限制在地图大小的 10% 到 90% 之间，以确保传感器不会出现在地图的边缘。
        self.sensor_position = [random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))], k=sensor_num), random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))], k=sensor_num)]
        # 创建 Sensor 实例，并将其添加到 self.sensors 列表中,传入初始位置x,y以及移动半径sensor_move_r
        for i in range(sensor_num):
            self.sensors.append(Sensor(np.array([self.sensor_position[0][i], self.sensor_position[1][i]]), self.sensor_move_r))

        #这行代码定义了两个服务器的初始位置。位置的设置使得服务器位于地图的四分之一处，从而确保它们分布在不同的区域来优化对传感器的覆盖率。
        self.server_position = [(map_size / 4, map_size / 4, 200 - map_size / 4, 200 - map_size / 4),(map_size / 4, 200 - map_size / 4, map_size / 4, 200 - map_size / 4)]
        #创建指定数量的服务器实例，并将其添加到 self.servers 列表中。服务器的位置也是用 np.array 表示，传入其收集半径 server_collect_r。
        for i in range(server_num):
            self.servers.append(EdgeServer(np.array([int(self.server_position[0][i]), int(self.server_position[1][i])]), server_collect_r))
        # 无人机的位置。这里创建了两个列表，分别表示无人机的 x 坐标和 y 坐标。位置的范围被限制在地图大小的 15% 到 85% 之间，以确保无人机不会出现在地图的边缘。
        self.uav_position = [random.sample([i for i in range(int(0.15 * self.map_size), int(0.85 * self.map_size))], uav_num), random.sample([i for i in range(int(0.15 * self.map_size), int(0.85 * self.map_size))], uav_num)]
        #设置该无人机的x,y坐标，观察半径、收集半径和移动半径
        for i in range(uav_num):
            self.uavs.append(EdgeUav(np.array([self.uav_position[0][i], self.uav_position[1][i]]), self.uav_obs_r, uav_collect_r, self.uav_move_r))

    # 执行卸载决策
    def step(self):
        #【 第一步：执行卸载决策，获得 sensor_delay ，作为卸载决策的奖励 】
        # 重置传感器的时延为空
        self.sensor_delay = []
        for i, sensor in enumerate(self.sensors):   #通过 enumerate 函数遍历所有传感器，获取每个传感器及其索引
            # 判断传感器有没有被设备覆盖。如果没有，返回 True，直接跳过此次遍历
            # if self.if_sensor_within_range(sensor):
            #     continue

            #  判断是否存在卸载决策，并且传感器自身是否存在待卸载的数据
            if (sum(sensor.action.offload) and sensor.total_data != {}):
                # 重置数据为0： 数据量大小、卸载位置
                data_size = 0
                position_offload = [0,0]

                # 计算该传感器设备的总数据量
                for g in sensor.total_data:
                    data_size += sensor.total_data[g]
                
                # 求得神经网络动作的输出的索引值，将索引值进行匹配：
                # 第一个表示自身执行，中间四个表示服务器执行，最后四个表示无人机执行
                position_index = sensor.action.offload.index(1)

                # 索引为 0 表示自身执行，【 假设 】 无论处理多久，直接奖励设置为 0
                if position_index == 0: 
                    self.sensor_delay.append(data_size / sensor.computing_rate)
                    sensor.total_data = {}
                    continue
                # 索引为 1、2、3、4，表示卸载到服务器执行
                elif position_index == 1: 
                    position_offload = [50,50]
                    # 判断神经网络输出的结果是否在 此设备的覆盖范围内
                    if self.is_point_in_circle(sensor.position, position_offload, self.server_collect_r):
                        self.server_transmit_and_process(sensor, data_size, position_offload, 0)
                        continue
                elif position_index == 2:
                    position_offload = [50,150]
                    if self.is_point_in_circle(sensor.position, position_offload, self.server_collect_r):
                        self.server_transmit_and_process(sensor, data_size, position_offload, 1)
                        continue
                elif position_index == 3:
                    position_offload = [150,50]
                    if self.is_point_in_circle(sensor.position, position_offload, self.server_collect_r):
                        self.server_transmit_and_process(sensor, data_size, position_offload, 2)
                        continue
                elif position_index == 4:
                    position_offload = [150,150]
                    if self.is_point_in_circle(sensor.position, position_offload, self.server_collect_r):
                        self.server_transmit_and_process(sensor, data_size, position_offload, 3)
                        continue
                elif position_index == 5:
                    position_offload = self.uavs[0].position
                    if self.is_point_in_circle(sensor.position, position_offload, self.uav_collect_r) and sensor.total_data:
                        self.uav_collect_and_process(sensor, data_size, position_offload, 0)
                        continue
                elif position_index == 6:
                    position_offload = self.uavs[1].position
                    if self.is_point_in_circle(sensor.position, position_offload, self.uav_collect_r) and sensor.total_data:
                        self.uav_collect_and_process(sensor, data_size, position_offload, 1)
                        continue
                elif position_index == 7:
                    position_offload = self.uavs[2].position
                    if self.is_point_in_circle(sensor.position, position_offload, self.uav_collect_r) and sensor.total_data:
                        self.uav_collect_and_process(sensor, data_size, position_offload, 2)
                        continue
                elif position_index == 8:
                    position_offload = self.uavs[3].position
                    if self.is_point_in_circle(sensor.position, position_offload, self.uav_collect_r) and sensor.total_data:
                        self.uav_collect_and_process(sensor, data_size, position_offload, 3)
                        continue

                # 如果没有走上面的 continue，则代表网络输出的卸载决策，不在设备的覆盖范围内
                # 是一个错误的卸载决策，所以直接将奖励设置为 0
                self.sensor_delay.append(0)

        #【 第二步：对时延进行处理，转化为卸载决策的奖励 】
        # 卸载决策的奖励: 延迟分支1，时延越小，奖励越大，时延为0，则奖励为0 
        # 并结合 PF 参数（可以修改最终的奖励）

        #map函数应用于 self.sensor_delay 列表中的每一个元素
        #对于每个时延值 x，如果 x 不为0，则使用公式 1 / (PF * x) 计算奖励；如果 x 为0，则奖励为0
        #结果存储回self.sensor_delay，此时self.sensor_delay变成了奖励值的列表
        PF = 1 / 2
        self.sensor_delay = list(map(lambda x: 1 / (PF*x) if x != 0 else 0, self.sensor_delay))
        for i in range(len(self.sensor_delay)):
            # 如果具体的值大于 1，则直接设置为 1
            if self.sensor_delay[i] > 1:
                self.sensor_delay[i] = 1
            #将当前元素的值四舍五入到小数点后 3 位
            self.sensor_delay[i] = round(self.sensor_delay[i], 3)

        #【 第三步：传感器生成数据、规范移动、获得设备状态 DS_state 】
        # 先重置 DS_state
        #将设备状态数组 DS_state 重置为全1数组，后续将被更新为具体的设备状态
        self.DS_state = np.ones([self.map_size, self.map_size, 2])
        
        for i, sensor in enumerate(self.sensors):
            new_data = sensor.sensor_data_gen_rate * np.random.poisson(sensor.lam)
            new_data = min(new_data, sensor.sensor_max_data_size)

            #【 假设 】为了更好的测试，现在取消掉了生成数据的限制，现在是每一个传感器在每个时隙都会生成数据
            # if new_data >= self.sensor_max_data_size or random.random() >= self.gen_threshold:
            #     return
            if new_data:
                # 累加未被处理的数据
                #检查 sensor.total_data 字典是否已有数据
                if sensor.total_data:
                    #这行代码通过将字典的键转换为列表，并获取最后一个键，从而找出当前数据的最高索引。
                    last_key = list(sensor.total_data.keys())[-1]
                    sensor.total_data[last_key+1] = new_data
                else:
                    sensor.total_data[0] = new_data

            # 传感器的随机移动
            count = 0
            move_dict = {}
            #移动范围，遍历所有可能的y值
            for x in range(-self.sensor_move_r, self.sensor_move_r + 1):
                #计算对应的y的范围
                y_l = int(np.floor(np.sqrt(self.sensor_move_r**2 - x**2)))
                #遍历所有可能的y值
                for y in range(-y_l, y_l + 1):
                    move_dict[count] = np.array([y, x])
                    count += 1
            #随机选择一个移动方向
            move = random.sample(list(move_dict.values()), 1)[0]

            # 传感器遵守交通规则的移动
            if traffic.traffic_move(sensor, move, i) :
                #sensor没被任何设备覆盖
                if self.if_sensor_within_range(sensor):
                    #【问题】关于这个位置和矩阵这里，感觉没有想明白
                    # 这里二维坐标系下的（x,y）与 矩阵中的（i,j）,是相反的
                    # self.DS_state[sensor.position[1], sensor.position[0]] = [1, sum(sensor.total_data.values())]

                    #self.DS_state被更新为一个包含两个值的列表，第一个值是传感器生成的所有数据的总和，第二个值是传感器生成的数据的数量
                    #DS_state表示这个位置有一个sensor，他有多少数据总量和多少个数据包
                    self.DS_state[sensor.position[1], sensor.position[0]] = [sum(sensor.total_data.values()), len(sensor.total_data)]
                continue
        # # 【2023年7月10日改】将其他设备的位置信息，也添加进无人机的输入网络中
        # for server in self.servers:
        #     self.DS_state[server.position[1], server.position[0]] = [2,0]
        # for uav in self.uavs:
        #     self.DS_state[uav.position[1], uav.position[0]] = [3,0]
    
    # 判断传感器是否在处理设备的覆盖范围内。以此来确定是否能够进行卸载
    def is_point_in_circle(self, point, circle_center, radius):
        distance = np.sqrt((circle_center[0] - point[0])**2 + (circle_center[1] - point[1])**2)
        return distance <= radius

    def server_transmit_and_process(self, sensor, data_size, position_offload, server_num):
        # 如果处理的数据，包含有累计的（上一次的卸载决策错了或者上一次不在设备的覆盖范围内），则直接将奖励定为最大值 1
        if(data_size >1800):
            self.sensor_delay.append(2)   #传感器处理的数据量超过1800时，给予固定的延迟奖励（值为2,函数调用地方还会变成1的）
        else :
            # 计算传输和处理延迟
            dist = np.linalg.norm(np.array(sensor.position) - np.array(position_offload)) #距离
            transmit_or_collect_delay = data_size / self.transmit_rate(dist, sensor)         #传输延迟
            server_or_uav_process_delay = data_size / self.servers[server_num].computing_rate #处理延迟
            self.sensor_delay.append(transmit_or_collect_delay + server_or_uav_process_delay) #总延迟
        sensor.total_data = {}

    def uav_collect_and_process(self, sensor, data_size, position_offload, uav_num):
        if(data_size >1800):
            self.sensor_delay.append(2)   #传感器处理的数据量超过1800时，给予固定的延迟奖励（值为2,函数调用地方还会变成1的）
        else:
            dist = np.linalg.norm(np.array(sensor.position) - np.array(position_offload))
            #【 注意 】 目前这里没有使用 collect_rate，无人机的收集速率使用的还是 数据的传输速率
            transmit_or_collect_delay = data_size / self.transmit_rate(dist, sensor)
            server_or_uav_process_delay = data_size / self.uavs[uav_num].computing_rate
            self.sensor_delay.append(transmit_or_collect_delay + server_or_uav_process_delay)
        sensor.total_data = {}

    #判断给定传感器是否在可处理设备的范围内，传感器没有被任何设备覆盖返回true
    def if_sensor_within_range(self, sensor): 
        points = [
            sensor.position,
            self.servers[0].position,
            self.servers[1].position,
            self.servers[2].position,
            self.servers[3].position,
            self.uavs[0].position,
            self.uavs[1].position,
            self.uavs[2].position,
            self.uavs[3].position,
        ]
        distances = []
        # 计算第一个点到其他点之间的距离
        for point in points:
            distance = np.linalg.norm(np.array(points[0]) - np.array(point))
            distances.append(distance)

        # 针对没有被任何设备覆盖的传感器，直接跳过
        if (
            distances[1] > 30 and
            distances[2] > 30 and
            distances[3] > 30 and
            distances[4] > 30 and
            distances[5] > 40 and
            distances[6] > 40 and
            distances[7] > 40 and
            distances[8] > 40
        ):
            return True
        return False
    
    # 数据的传输速率
    def transmit_rate(self, dist, sensor):
        # 防止位置重叠，dist=0
        if (dist == 0):
            dist = 1
        W = 1e6 * sensor.sensor_bandwidth
        Pl = 1 / (1 + a * np.exp(-b * (np.arctan(sensor.h / dist) - a)))
        fspl = (4 * np.pi * carrier_f * dist / (3e8))**2
        L = Pl * fspl * 10**(yita0 / 20) + 10**(yita1 / 20) * fspl * (1 - Pl)
        transmit_rate = W * np.log2(1 + sensor.ptr / (L * sensor.noise * W))
        return transmit_rate/100 
    # 无人机的收集速率（目前使用的是上面的传输速率）
    # def collect_rate(dist, sensor, uav):
    #     Pl = 1 / (1 + a * np.exp(-b * (np.arctan(uav.h / dist) - a)))
    #     L = Pl * yita0 + yita1 * (1 - Pl)
    #     gamma = uav.ptr_col / (L * sensor.noise_power**2)
    #     collect_rate = sensor.sensor_bandwidth * np.log2(1 + gamma)
    #     collect_rate = 8000
    #     return collect_rate
