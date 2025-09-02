import math
import random
import time
import torch
import torch.nn.functional as F
import quadsim_cuda


# =============== 梯度衰减自定义函数 ===============
# 解决可微分长时序仿真中的梯度爆炸问题
class GDecay(torch.autograd.Function):
    """
    梯度衰减函数：前向传播保持数值不变，反向传播时应用指数衰减
    用途：在长时序反向传播中防止梯度指数级增长
    原理：通过乘以小于1的系数逐步衰减早期时间步的梯度影响
    """
    @staticmethod
    def forward(ctx, x, alpha):
        """
        前向传播：透明传递输入值
        Args:
            x: 输入张量
            alpha: 梯度衰减系数 (0,1)，越小衰减越强
        """
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：对输入梯度应用衰减
        Args:
            grad_output: 来自上游的梯度
        Returns:
            衰减后的梯度，None（alpha不需要梯度）
        """
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply  # 创建函数别名，简化调用


# =============== 可微分四旋翼动力学封装 ===============
# 将CUDA实现的物理仿真核心包装为PyTorch自动微分函数
class RunFunction(torch.autograd.Function):
    """
    可微分四旋翼动力学仿真函数
    
    功能：执行一个时间步的完整四旋翼物理仿真，包括：
    1. 控制系统：一阶延迟滤波器模拟执行器延迟
    2. 阻力模型：线性+二次阻力，分别考虑机体各轴
    3. 扰动建模：重力场不均匀性、风场扰动
    4. 运动积分：基于Verlet积分的位置和速度更新
    5. 梯度衰减：防止长时序反向传播中的梯度爆炸
    """
    @staticmethod
    def forward(ctx, R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt, airmode):
        """
        前向传播：执行一步物理仿真
        Args:
            R: 机体姿态矩阵 [B,3,3] - 当前机体到世界坐标系的旋转
            dg: 重力扰动 [B,3] - 模拟重力场不均匀性
            z_drag_coef: Z轴阻力系数 [B,1] - 上升/下降时的额外阻力
            drag_2: 阻力系数 [B,2] - [二次项,一次项]空气阻力参数
            pitch_ctl_delay: 俯仰控制延迟 [B,1] - 模拟电机响应延迟
            act_pred: 预测动作 [B,3] - 网络输出的期望加速度
            act: 当前动作 [B,3] - 上一时刻的实际输出
            p: 位置 [B,3] - 当前世界坐标位置
            v: 速度 [B,3] - 当前世界坐标速度
            v_wind: 风速 [B,3] - 环境风场速度
            a: 加速度 [B,3] - 当前加速度
            grad_decay: 梯度衰减因子 - 控制长时序稳定性
            ctl_dt: 控制时间步长 - 物理仿真时间间隔
            airmode: 空气模式参数 - 高速飞行时的角速度阻尼
        Returns:
            (act_next, p_next, v_next, a_next): 下一时刻状态
        """
        # 调用CUDA核心执行高效并行物理仿真
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, airmode)
        
        # 保存反向传播需要的中间变量
        ctx.save_for_backward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next)
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        return act_next, p_next, v_next, a_next

    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        """
        反向传播：计算物理仿真对输入参数的梯度
        Args:
            d_act_next, d_p_next, d_v_next, d_a_next: 来自损失函数的输出梯度
        Returns:
            输入参数的梯度：与forward参数一一对应，None表示不需要梯度
        """
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next = ctx.saved_tensors
        # 调用CUDA核心执行反向传播梯度计算
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next,
            ctx.grad_decay, ctx.ctl_dt)
        # 返回梯度：与forward参数顺序一致，None表示该参数不参与梯度计算
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None

run = RunFunction.apply  # 创建函数别名，简化调用


# =============== 四旋翼仿真环境主类 ===============
class Env:
    """
    集成式四旋翼仿真环境
    
    核心功能：
    1. 随机场景生成：多样化障碍物分布、起终点配置
    2. 物理仿真：高精度四旋翼动力学模型
    3. 视觉渲染：实时深度图生成，支持多种几何体
    4. 碰撞检测：精确的最近点查询算法
    5. 编队仿真：支持多无人机协同训练
    """
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10) -> None:
        """
        初始化四旋翼仿真环境
        Args:
            batch_size: 并行仿真的无人机数量（影响GPU内存和计算效率）
            width, height: 深度图分辨率（通常64x48，4x下采样到16x12输入网络）
            grad_decay: 梯度衰减因子 ∈ (0,1)，控制长时序梯度稳定性
            device: 计算设备 ('cpu' | 'cuda')
            fov_x_half_tan: 水平视场角半角正切值，决定观察范围
            single: 是否启用单机模式（vs多机编队避障）
            gate: 是否生成门型障碍物（穿越任务）
            ground_voxels: 是否生成地面体素障碍（地形建模）
            scaffold: 是否生成脚手架结构（结构化环境）
            speed_mtp: 速度倍率，影响最大飞行速度和任务难度
            random_rotation: 是否随机旋转环境（增强泛化）
            cam_angle: 相机俯仰角度（度），模拟真实安装角度
        """
        # =============== 环境基础配置 ===============
        self.device = device
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.grad_decay = grad_decay
        
        # =============== 障碍物几何参数配置 ===============
        # 定义各种障碍物的空间分布和尺寸范围，支持随机生成多样化场景
        
        # 球体障碍物参数：[x范围, y范围, z范围, 半径范围]
        self.ball_w = torch.tensor([8., 18, 6, 0.2], device=device)      # 分布宽度：8m×18m×6m空间，半径变化0.2m
        self.ball_b = torch.tensor([0., -9, -1, 0.4], device=device)     # 分布中心：x=0, y=-9, z=-1，基础半径0.4m
        
        # 长方体体素障碍物参数：[x范围, y范围, z范围, x尺寸范围, y尺寸范围, z尺寸范围]
        self.voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device)  # 位置分布宽度 + 尺寸变化范围
        self.voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device)  # 位置分布中心 + 基础尺寸
        
        # 地面体素参数：用于构建地面地形和建筑
        self.ground_voxel_w = torch.tensor([8., 18,  0, 2.9, 2.9, 1.9], device=device)  # z范围为0（贴地），大尺寸
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)  # z=-1（地面以下）
        
        # 垂直圆柱体参数：[x范围, y范围, 半径范围]（z轴无限高）
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device)         # 8m×18m分布，半径变化0.35m
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)         # 中心位置，基础半径0.05m
        
        # 水平圆柱体参数：[x范围, z范围, 半径范围]（y轴无限长）
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device)         # 8m×6m分布，半径变化0.1m
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)        # 中心位置，基础半径0.05m
        
        # 门型障碍物参数：[x范围, y范围, z范围, 孔径范围]
        self.gate_w = torch.tensor([2.,  2,  1.0, 0.5], device=device)   # 2m×2m×1m分布，孔径变化0.5m
        self.gate_b = torch.tensor([3., -1,  0.0, 0.5], device=device)   # 位置偏移，基础孔径0.5m
        
        # =============== 环境参数 ===============
        self.v_wind_w = torch.tensor([1,  1,  0.2], device=device)       # 风速范围：x,y,z方向
        self.g_std = torch.tensor([0., 0, -9.80665], device=device)      # 标准重力加速度
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)  # 屋顶模式的位置偏移
        
        # 子时间步长：用于最近点查询的时间插值
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)
        # =============== 起终点配置 ===============
        # 8个预设的起点位置（用于多机编队）
        self.p_init = torch.as_tensor([
            [-1.5, -3.,  1],  # 左下
            [ 9.5, -3.,  1],  # 右下
            [-0.5,  1.,  1],  # 左中下
            [ 8.5,  1.,  1],  # 右中下
            [ 0.0,  3.,  1],  # 左中上
            [ 8.0,  3.,  1],  # 右中上
            [-1.0, -1.,  1],  # 左中
            [ 9.0, -1.,  1],  # 右中
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        
        # 8个预设的终点位置（与起点对应）
        self.p_end = torch.as_tensor([
            [8.,  3.,  1],    # 右上
            [0.,  3.,  1],    # 左上
            [8., -1.,  1],    # 右中
            [0., -1.,  1],    # 左中
            [8., -3.,  1],    # 右下
            [0., -3.,  1],    # 左下
            [8.,  1.,  1],    # 右中上
            [0.,  1.,  1],    # 左中上
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        
        # 光流图（当前未使用）
        self.flow = torch.empty((batch_size, 0, height, width), device=device)
        
        # =============== 环境配置标志 ===============
        self.single = single              # 单机模式
        self.gate = gate                  # 门型障碍物
        self.ground_voxels = ground_voxels # 地面体素
        self.scaffold = scaffold          # 脚手架结构
        self.speed_mtp = speed_mtp        # 速度倍数
        self.random_rotation = random_rotation  # 随机旋转
        self.cam_angle = cam_angle        # 相机角度
        self.fov_x_half_tan = fov_x_half_tan    # 视场角
        
        # 初始化环境状态
        self.reset()
        # self.obj_avoid_grad_mtp = torch.tensor([0.5, 2., 1.], device=device)

    def reset(self):
        """
        重置环境状态：生成新的随机场景和无人机初始状态
        """
        B = self.batch_size
        device = self.device

        # =============== 相机姿态初始化 ===============
        # 为每个batch生成随机的相机俯仰角（在基础角度上添加噪声）
        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        # 构造相机旋转矩阵：绕X轴旋转（俯仰角）
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),  # 第一行
            zeros, ones, zeros,                                  # 第二行
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),  # 第三行
        ], -1).reshape(B, 3, 3)

        # =============== 障碍物生成 ===============
        # 生成30个球体障碍物：位置(x,y,z) + 半径
        self.balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b
        # 生成30个体素障碍物：位置(x,y,z) + 尺寸(rx,ry,rz)
        self.voxels = torch.rand((B, 30, 6), device=device) * self.voxel_w + self.voxel_b
        # 生成30个垂直圆柱体：位置(x,y) + 半径
        self.cyl = torch.rand((B, 30, 3), device=device) * self.cyl_w + self.cyl_b
        # 生成2个水平圆柱体：位置(x,z) + 半径
        self.cyl_h = torch.rand((B, 2, 3), device=device) * self.cyl_h_w + self.cyl_h_b

        # =============== 渲染和编队参数 ===============
        # 视场角随机抖动（±5%）
        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        # 每组无人机数量：随机选择4或8架
        self.n_drones_per_group = random.choice([4, 8])
        # 无人机半径：0.1-0.15m随机
        self.drone_radius = random.uniform(0.1, 0.15)
        if self.single:
            self.n_drones_per_group = 1  # 单机模式

        # =============== 速度和尺度参数 ===============
        # 为每组无人机生成相同的最大速度（0.75-3.25 m/s）
        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp
        # 根据速度计算环境缩放因子
        scale = (self.max_speed - 0.5).clamp_min(1)

        # 推力估计误差：±1%随机误差
        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        # =============== 屋顶模式障碍物 ===============
        # 50%概率启用屋顶模式：将部分障碍物移到高处
        roof = torch.rand((B,)) < 0.5
        # 非屋顶模式：将前15个球体和体素替换为圆柱体
        self.balls[~roof, :15, :2] = self.cyl[~roof, :15, :2]
        self.voxels[~roof, :15, :2] = self.cyl[~roof, 15:, :2]
        # 屋顶模式：将前15个障碍物移到高处
        self.balls[~roof, :15] = self.balls[~roof, :15] + self.roof_add[:4]
        self.voxels[~roof, :15] = self.voxels[~roof, :15] + self.roof_add
        
        # =============== 障碍物边界约束 ===============
        # 确保障碍物不会超出环境边界（考虑安全距离）
        self.balls[..., 0] = torch.minimum(torch.maximum(self.balls[..., 0], self.balls[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.balls[..., 3])
        self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.voxels[..., 3])
        self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl[..., 2])
        self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl_h[..., 2])
        
        # 屋顶模式：将第一个体素设为"天花板"（z=201，尺寸200）
        self.voxels[roof, 0, 2] = self.voxels[roof, 0, 2] * 0.5 + 201
        self.voxels[roof, 0, 3:] = 200

        # =============== 地面体素模式 ===============
        if self.ground_voxels:
            # 生成地面球体：用于创建地形起伏
            ground_balls_r = 8 + torch.rand((B, 2), device=device) * 6      # 球体半径8-14m
            ground_balls_r_ground = 2 + torch.rand((B, 2), device=device) * 4  # 地面半径2-6m
            # 计算球体高度：根据几何关系 ground_balls_h = ground_balls_r - sqrt(ground_balls_r^2 - ground_balls_r_ground^2)
            ground_balls_h = ground_balls_r - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            # 几何关系图：
            # |   ground_balls_h
            # ----- ground_balls_r_ground
            # |  /
            # | / ground_balls_r
            # |/
            
            # 将前两个球体设为地面球体
            self.balls[:, :2, 3] = ground_balls_r
            self.balls[:, :2, 2] = ground_balls_h - ground_balls_r - 1

            # 添加10个地面体素障碍物
            ground_voxels = torch.rand((B, 10, 6), device=device) * self.ground_voxel_w + self.ground_voxel_b
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1  # 确保在地面以下
            self.voxels = torch.cat([self.voxels, ground_voxels], 1)

        # =============== 障碍物Y坐标缩放 ===============
        # 根据速度调整障碍物的Y坐标分布，使任务难度与速度匹配
        self.voxels[:, :, 1] *= (self.max_speed + 4) / scale
        self.balls[:, :, 1] *= (self.max_speed + 4) / scale
        self.cyl[:, :, 1] *= (self.max_speed + 4) / scale

        # =============== 门型障碍物 ===============
        if self.gate:
            # 生成门的位置和半径
            gate = torch.rand((B, 4), device=device) * self.gate_w + self.gate_b  # [x, y, z, r]
            p = gate[None, :, :3]  # 门的位置
            
            # 检查门是否与其他障碍物太近
            nearest_pt = torch.empty_like(p)
            quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, 1)
            gate_x, gate_y, gate_z, gate_r = gate.unbind(-1)
            # 如果门太靠近其他障碍物，将其移到远处
            gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50
            
            # 构造门的4个支柱（上下左右）
            ones = torch.ones_like(gate_x)
            gate = torch.stack([
                torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),  # 右柱
                torch.stack([gate_x, gate_y, gate_z + gate_r + 5, ones * 0.05, ones * 5, ones * 5], -1),  # 上柱
                torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),  # 左柱
                torch.stack([gate_x, gate_y, gate_z - gate_r - 5, ones * 0.05, ones * 5, ones * 5], -1),  # 下柱
            ], 1)

            self.voxels = torch.cat([self.voxels, gate], 1)
        # =============== 障碍物X坐标缩放 ===============
        # 根据速度缩放所有障碍物的X坐标
        self.voxels[..., 0] *= scale
        self.balls[..., 0] *= scale
        self.cyl[..., 0] *= scale
        self.cyl_h[..., 0] *= scale
        
        # 地面体素模式：调整地面球体的X坐标约束
        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3), scale * 8 - 0.3 - ground_balls_r_ground)

        # =============== 无人机参数初始化 ===============
        # 控制延迟参数：模拟真实控制系统的延迟
        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)  # 俯仰控制延迟
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)      # 偏航控制延迟

        # 无人机位置缩放：为每组无人机生成相同的缩放因子
        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        scale = torch.cat([
            scale,                    # X轴缩放（基于速度）
            rd + 0.5,                # Y轴缩放（0.5-1.5）
            torch.rand_like(scale) - 0.5  # Z轴缩放（-0.5-0.5）
        ], -1)
        
        # 设置无人机起始位置和目标位置（添加随机噪声）
        self.p = self.p_init * scale + torch.randn_like(scale) * 0.1
        self.p_target = self.p_end * scale + torch.randn_like(scale) * 0.1

        # =============== 随机旋转环境 ===============
        if self.random_rotation:
            # 为每组无人机生成相同的偏航角偏差（-0.75到0.75弧度）
            yaw_bias = torch.rand(B//self.n_drones_per_group, device=device).repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75
            c = torch.cos(yaw_bias)
            s = torch.sin(yaw_bias)
            l = torch.ones_like(yaw_bias)
            o = torch.zeros_like(yaw_bias)
            # 构造绕Z轴的旋转矩阵
            R = torch.stack([c,-s, o, s, c, o, o, o, l], -1).reshape(B, 3, 3)
            
            # 旋转无人机位置和目标位置
            self.p = torch.squeeze(R @ self.p[..., None], -1)
            self.p_target = torch.squeeze(R @ self.p_target[..., None], -1)
            
            # 旋转所有障碍物的位置
            self.voxels[..., :3] = (R @ self.voxels[..., :3].transpose(1, 2)).transpose(1, 2)
            self.balls[..., :3] = (R @ self.balls[..., :3].transpose(1, 2)).transpose(1, 2)
            self.cyl[..., :3] = (R @ self.cyl[..., :3].transpose(1, 2)).transpose(1, 2)

        # =============== 脚手架结构 ===============
        if self.scaffold and random.random() < 0.5:
            # 50%概率添加脚手架结构
            x = torch.arange(1, 6, dtype=torch.float, device=device)      # X坐标：1-5
            y = torch.arange(-3, 4, dtype=torch.float, device=device)     # Y坐标：-3到3
            z = torch.arange(1, 4, dtype=torch.float, device=device)      # Z坐标：1-3
            
            # 生成垂直支柱网格
            _x, _y = torch.meshgrid(x, y)
            scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            
            # 添加随机偏移和缩放
            x_bias = torch.rand_like(self.max_speed) * self.max_speed
            scale = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scale + torch.stack([
                x_bias,                                    # X偏移
                torch.randn_like(self.max_speed),          # Y偏移
                torch.rand_like(self.max_speed) * 0.01    # Z偏移
            ], -1)
            self.cyl = torch.cat([self.cyl, scaf_v], 1)   # 添加到垂直圆柱体
            
            # 生成水平支柱网格
            _x, _z = torch.meshgrid(x, z)
            scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            scaf_h = scaf_h * scale + torch.stack([
                x_bias,                                    # X偏移
                torch.randn_like(self.max_speed) * 0.1,   # Y偏移
                torch.rand_like(self.max_speed) * 0.01    # Z偏移
            ], -1)
            self.cyl_h = torch.cat([self.cyl_h, scaf_h], 1)  # 添加到水平圆柱体

        # =============== 无人机状态初始化 ===============
        # 初始速度：随机小速度
        self.v = torch.randn((B, 3), device=device) * 0.2
        # 风速：随机风场
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        # 初始动作：随机小动作
        self.act = torch.randn_like(self.v) * 0.1
        self.a = self.act  # 初始加速度等于动作
        # 重力扰动：模拟重力场的不均匀性
        self.dg = torch.randn((B, 3), device=device) * 0.2

        # =============== 姿态初始化 ===============
        # 初始化姿态矩阵
        R = torch.zeros((B, 3, 3), device=device)
        # 使用CUDA核计算初始姿态：基于当前动作和目标方向
        self.R = quadsim_cuda.update_state_vec(R, self.act, torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()  # 保存上一帧姿态
        self.p_old = self.p          # 保存上一帧位置
        
        # 安全边际：每个无人机0.1-0.3m随机
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1

        # =============== 阻力系数 ===============
        # 阻力系数：线性+二次阻力
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3  # [0.3-0.45, 0.3-0.45]
        self.drag_2[:, 0] = 0  # 线性阻力设为0
        # Z轴阻力系数：通常为1
        self.z_drag_coef = torch.ones((B, 1), device=device)

    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5):
        """
        更新无人机姿态矩阵（静态方法，用于姿态控制）
        参数：
        - R: 当前姿态矩阵 [B, 3, 3]
        - a_thr: 推力向量 [B, 3]
        - v_pred: 预测速度方向 [B, 3]
        - alpha: 姿态更新系数（0-1）
        - yaw_inertia: 偏航惯性系数
        返回：新的姿态矩阵 [B, 3, 3]
        """
        # 当前前向向量
        self_forward_vec = R[..., 0]
        # 标准重力
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        # 计算净推力（减去重力）
        a_thr = a_thr - g_std
        # 推力大小
        thrust = torch.norm(a_thr, 2, -1, True)
        # 上向量（推力方向）
        self_up_vec = a_thr / thrust
        
        # 计算新的前向向量：结合当前前向和预测速度
        forward_vec = self_forward_vec * yaw_inertia + v_pred
        # 平滑更新：alpha权重当前方向，1-alpha权重新方向
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)
        # 约束前向向量的Z分量：确保与上向量垂直
        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[2]
        # 归一化前向向量
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        # 计算左向量：上向量×前向向量
        self_left_vec = torch.cross(self_up_vec, self_forward_vec)
        
        # 返回新的姿态矩阵：[前向, 左向, 上向]
        return torch.stack([
            self_forward_vec,
            self_left_vec,
            self_up_vec,
        ], -1)

    def render(self, ctl_dt):
        """
        渲染深度图
        参数：
        - ctl_dt: 控制时间步长（未使用，保持接口一致）
        返回：
        - canvas: 深度图 [B, H, W]
        - None: 光流图（当前未实现）
        """
        # 创建深度图画布
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        
        # 调用CUDA渲染核
        # 参数说明：
        # - canvas: 输出深度图
        # - self.flow: 光流图（未使用）
        # - self.balls: 球体障碍物
        # - self.cyl: 垂直圆柱体障碍物
        # - self.cyl_h: 水平圆柱体障碍物
        # - self.voxels: 体素障碍物
        # - self.R @ self.R_cam: 相机姿态矩阵（机体姿态 × 相机旋转）
        # - self.R_old: 上一帧机体姿态
        # - self.p: 当前位置
        # - self.p_old: 上一帧位置
        # - self.drone_radius: 无人机半径
        # - self.n_drones_per_group: 每组无人机数量
        # - self._fov_x_half_tan: 视场角
        quadsim_cuda.render(canvas, self.flow, self.balls, self.cyl, self.cyl_h,
                            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
                            self.p_old, self.drone_radius, self.n_drones_per_group,
                            self._fov_x_half_tan)
        return canvas, None

    def find_vec_to_nearest_pt(self):
        """
        计算到最近障碍点的向量（用于避障损失计算）
        返回：
        - 向量：从当前位置到最近障碍点的方向向量 [B, 10, 3]
        """
        # 预测未来10个时间步的位置（用于前瞻性避障）
        p = self.p + self.v * self.sub_div  # [10, B, 3]
        
        # 调用CUDA核计算最近点
        nearest_pt = torch.empty_like(p)
        quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, self.n_drones_per_group)
        
        # 返回从当前位置到最近点的向量
        return nearest_pt - p

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        """
        执行一步物理仿真
        参数：
        - act_pred: 预测动作 [B, 3]
        - ctl_dt: 控制时间步长（默认1/15秒）
        - v_pred: 预测速度方向 [B, 3]
        """
        # 更新重力扰动：添加随机噪声
        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        
        # 保存上一帧位置
        self.p_old = self.p
        
        # 调用可微分动力学函数执行一步仿真
        self.act, self.p, self.v, self.a = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
        
        # 更新姿态
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)  # 姿态更新系数
        self.R_old = self.R.clone()  # 保存旧姿态
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)  # 更新新姿态

    def _run(self, act_pred, ctl_dt=1/15, v_pred=None):
        """
        纯Python版本的物理仿真（用于调试，当前未使用）
        这是CUDA版本的参考实现
        """
        # 动作平滑更新
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        
        # 重力扰动更新
        self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        
        # Z轴阻力计算
        z_drag = 0
        if self.z_drag_coef is not None:
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]  # 速度的上分量
            v_prep = self.v - v_up  # 速度的垂直分量
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()  # 电机转速
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07  # Z轴阻力
        
        # 线性阻力
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        
        # 计算下一时刻加速度
        a_next = self.act + self.dg - z_drag - drag
        
        # 位置和速度积分（使用梯度衰减）
        self.p_old = self.p
        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.a = a_next

        # 姿态更新
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

