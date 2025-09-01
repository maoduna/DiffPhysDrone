from collections import defaultdict
import math
from random import normalvariate
from matplotlib import pyplot as plt
from env_cuda import Env
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
from model import Model


# 命令行参数解析 - 控制训练的所有超参数
parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None, help='从检查点恢复训练的路径')
parser.add_argument('--batch_size', type=int, default=64, help='批次大小，同时训练的无人机数量')
parser.add_argument('--num_iters', type=int, default=50000, help='总训练迭代次数')

# 损失函数系数 - 用于平衡不同训练目标
parser.add_argument('--coef_v', type=float, default=1.0, help='速度跟踪损失权重：滑窗均速与目标速度的SmoothL1')
parser.add_argument('--coef_speed', type=float, default=0.0, help='遗留参数，通常为0')
parser.add_argument('--coef_v_pred', type=float, default=2.0, help='速度估计损失权重：网络预测速度vs真实速度的MSE')
parser.add_argument('--coef_collide', type=float, default=2.0, help='碰撞惩罚权重：距离障碍物过近时的softplus损失')
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='避障距离权重：二次屏障函数，鼓励保持安全距离')
parser.add_argument('--coef_d_acc', type=float, default=0.01, help='加速度正则化权重：||控制输出||^2，平滑控制')
parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='加速度变化率正则化权重：||Δ控制||^2，平滑控制变化')
parser.add_argument('--coef_d_snap', type=float, default=0.0, help='遗留参数：加速度二阶变化率正则化')
parser.add_argument('--coef_ground_affinity', type=float, default=0., help='遗留参数：地面亲和性')
parser.add_argument('--coef_bias', type=float, default=0.0, help='遗留参数：速度方向偏置损失')

# 优化器参数
parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
parser.add_argument('--grad_decay', type=float, default=0.4, help='梯度衰减因子，用于长时序梯度稳定')

# 环境参数
parser.add_argument('--speed_mtp', type=float, default=1.0, help='速度倍数，影响最大飞行速度')
parser.add_argument('--fov_x_half_tan', type=float, default=0.53, help='相机水平视场角的半角正切值')
parser.add_argument('--timesteps', type=int, default=150, help='每个训练序列的时间步数')
parser.add_argument('--cam_angle', type=int, default=10, help='相机俯仰角度（度）')

# 环境配置开关
parser.add_argument('--single', default=False, action='store_true', help='单机模式vs多机编队')
parser.add_argument('--gate', default=False, action='store_true', help='是否添加门型障碍物')
parser.add_argument('--ground_voxels', default=False, action='store_true', help='是否添加地面体素障碍')
parser.add_argument('--scaffold', default=False, action='store_true', help='是否添加脚手架结构')
parser.add_argument('--random_rotation', default=False, action='store_true', help='是否随机旋转环境')
parser.add_argument('--yaw_drift', default=False, action='store_true', help='是否添加偏航漂移扰动')
parser.add_argument('--no_odom', default=False, action='store_true', help='是否禁用里程计输入（仅视觉）')

args = parser.parse_args()
writer = SummaryWriter()  # 创建TensorBoard写入器
print(args)

# 设备配置
device = torch.device('cuda')

# 创建可微分物理仿真环境
# 参数说明：batch_size, width=64, height=48, grad_decay, device, 其他环境配置...
env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle)

# 创建策略网络模型
# 输入维度取决于是否使用里程计：
# - 无里程计：7维 [目标速度(3) + 机体朝向(3) + 安全边际(1)]
# - 有里程计：10维 [当前速度(3) + 目标速度(3) + 机体朝向(3) + 安全边际(1)]
# 输出维度：6维 [将被重塑为(3,2)，分别表示期望加速度和速度预测]
if args.no_odom:
    model = Model(7, 6)     # 仅视觉输入
else:
    model = Model(7+3, 6)   # 视觉 + 里程计输入
model = model.to(device)

# 从检查点恢复模型（可选）
if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)

# 配置优化器和学习率调度器
optim = AdamW(model.parameters(), args.lr)
# 余弦退火调度器：从初始学习率衰减到0.01倍
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

# 控制时间步长（秒），对应15Hz控制频率
ctl_dt = 1 / 15


# 用于累积标量日志，便于平滑显示
scaler_q = defaultdict(list)
def smooth_dict(ori_dict):
    """将标量字典添加到累积队列中，用于TensorBoard平滑显示"""
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

def barrier(x: torch.Tensor, v_to_pt):
    """屏障函数：当距离x小于1时产生二次惩罚，v_to_pt为速度权重"""
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

def is_save_iter(i):
    """判断是否保存可视化：前2000次迭代每250次保存，之后每1000次保存"""
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0

# 创建训练进度条
pbar = tqdm(range(args.num_iters), ncols=80)
B = args.batch_size  # 批次大小缩写

# =============== 主训练循环 ===============
for i in pbar:
    # 重置环境和模型状态，开始新的训练序列
    env.reset()          # 重新生成随机环境（障碍物、起点终点等）
    model.reset()        # 重置模型状态（主要是隐状态）
    
    # 初始化历史记录列表，用于存储整个序列的轨迹数据
    p_history = []           # 位置历史 [T, B, 3]
    v_history = []           # 速度历史 [T, B, 3]
    target_v_history = []    # 目标速度历史 [T, B, 3]
    vec_to_pt_history = []   # 到最近障碍点的向量历史 [T, B, 3]
    act_diff_history = []    # 动作差分历史（未使用）
    v_preds = []             # 网络预测的速度 [T, B, 3]
    vid = []                 # 视频帧（用于可视化）
    v_net_feats = []         # 网络特征（未使用）
    h = None                 # GRU隐状态

    # 动作延迟缓冲区，模拟真实控制系统的延迟
    act_lag = 1
    act_buffer = [env.act] * (act_lag + 1)  # 初始化为当前动作
    
    # 计算初始目标速度向量（从当前位置指向目标位置）
    target_v_raw = env.p_target - env.p
    
    # 可选：添加偏航漂移扰动，模拟真实飞行中的姿态漂移
    if args.yaw_drift:
        # 生成随机偏航角速度（约5度/秒 / 15Hz）
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        # 构造绕Z轴的旋转矩阵
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3)


    # =============== 时序展开循环（模拟飞行过程）===============
    for t in range(args.timesteps):
        # 添加控制频率的随机抖动，模拟真实系统的时间不确定性
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)  # 15Hz ± 10%
        
        # 渲染深度图和光流（当前仅使用深度图）
        depth, flow = env.render(ctl_dt)
        
        # 记录当前状态
        p_history.append(env.p)                          # 记录位置
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())  # 记录到最近障碍点的向量

        # 保存可视化帧（仅在特定迭代）
        if is_save_iter(i):
            vid.append(depth[4])  # 保存第5个batch的深度图用于可视化

        # 更新目标速度向量
        if args.yaw_drift:
            # 应用偏航漂移：旋转目标速度向量
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            # 正常模式：重新计算目标速度向量
            target_v_raw = env.p_target - env.p.detach()
            
        # 运行一步物理仿真：更新位置、速度、加速度
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        # =============== 构造机体坐标系和状态向量 ===============
        R = env.R  # 当前机体姿态矩阵
        
        # 构造水平化的机体坐标系（用于状态表示）
        fwd = env.R[:, :, 0].clone()  # 当前前向向量
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0                 # 投影到水平面
        up[:, 2] = 1                  # 上向量固定为世界Z轴
        fwd = F.normalize(fwd, 2, -1) # 归一化前向向量
        # 构造右手坐标系：[前向, 左向, 上向]
        R = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)

        # 计算目标速度（限制到最大速度）
        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
        
        # 构造状态向量（在机体坐标系下）
        state = [
            torch.squeeze(target_v[:, None] @ R, 1),  # 目标速度（机体系）[3]
            env.R[:, 2],                              # 真实机体上向量（世界系）[3]  
            env.margin[:, None]                       # 安全边际 [1]
        ]
        
        # 当前速度（机体坐标系）
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        
        # 根据配置决定是否包含里程计信息
        if not args.no_odom:
            state.insert(0, local_v)  # 添加当前速度到状态向量开头
            
        state = torch.cat(state, -1)  # 拼接所有状态分量

        # =============== 深度图预处理 ===============
        # 深度图归一化和增强：转换为逆深度 + 噪声增强
        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        x = F.max_pool2d(x[:, None], 4, 4)  # 下采样 4x：48x64 -> 12x16
        
        # =============== 策略网络前向推理 ===============
        act, values, h = model(x, state, h)  # 输出：动作[B,6], 值函数, 隐状态

        # =============== 动作解码 ===============
        # 将6维动作重塑为(B,3,2)，然后通过旋转矩阵转换到世界坐标系
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred)  # 记录速度预测用于损失计算
        
        # 构造实际控制输出：期望加速度 - 速度预测 - 重力 + 推力估计误差 + 重力
        # 这相当于：期望加速度 + 推力估计误差
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)  # 添加到动作缓冲区
        
        # 记录网络特征（未使用）
        v_net_feats.append(torch.cat([act, local_v, h], -1))

        # 记录历史数据
        v_history.append(env.v)              # 真实速度
        target_v_history.append(target_v)    # 目标速度

    # =============== 时序数据整理和损失计算 ===============
    p_history = torch.stack(p_history)                    # [T, B, 3]
    act_buffer = torch.stack(act_buffer)                   # [T+1, B, 3]
    v_history = torch.stack(v_history)                     # [T, B, 3]
    target_v_history = torch.stack(target_v_history)       # [T, B, 3]
    v_preds = torch.stack(v_preds)                         # [T, B, 3]
    
    # =============== 损失1：地面亲和性（遗留，通常权重为0）===============
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()

    # =============== 损失2：速度跟踪 ===============
    # 计算滑动窗口平均速度（30步）
    v_history_cum = v_history.cumsum(0)
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30  # [T-30, B, 3]
    T, B, _ = v_history.shape
    # 计算平均速度与目标速度的差异
    delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    # =============== 损失3：速度估计（用于无里程计训练）===============
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())

    # =============== 损失4：速度方向偏置（遗留，通常权重为0）===============
    target_v_history_norm = torch.norm(target_v_history, 2, -1)
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1)  # 前向速度分量
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3

    # =============== 损失5-7：控制正则化 ===============
    # 计算加速度变化率（jerk）和二阶变化率（snap）
    jerk_history = act_buffer.diff(1, 0).mul(15)      # 一阶差分 * 频率
    snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)  # 二阶差分
    
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()      # 加速度幅值正则化
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()   # 加速度变化率正则化  
    loss_d_snap = snap_history.pow(2).sum(-1).mean()   # 加速度二阶变化率正则化

    # =============== 损失8-9：避障和碰撞 ===============
    vec_to_pt_history = torch.stack(vec_to_pt_history)        # [T, B, 3]
    distance = torch.norm(vec_to_pt_history, 2, -1)           # 到最近障碍的距离
    distance = distance - env.margin                          # 减去安全边际
    
    # 计算速度权重：基于距离变化率，用于强调快速接近障碍的情况
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)  # 接近速度权重
    
    # 避障距离损失：当距离小于1时的二次屏障函数
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
    
    # 碰撞惩罚：距离为负（碰撞）时的指数惩罚
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

    # =============== 损失10：速度大小跟踪（遗留）===============
    speed_history = v_history.norm(2, -1)                     # 速度大小历史
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)  # 前向速度vs目标速度大小

    # =============== 总损失：各项损失的加权和 ===============
    loss = args.coef_v * loss_v + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_bias * loss_bias + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk + \
        args.coef_d_snap * loss_d_snap + \
        args.coef_speed * loss_speed + \
        args.coef_v_pred * loss_v_pred + \
        args.coef_collide * loss_collide + \
        args.coef_ground_affinity * loss_ground_affinity

    # =============== 训练步骤 ===============
    # 检查数值稳定性
    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    # 更新进度条显示
    pbar.set_description_str(f'loss: {loss:.3f}')
    
    # 反向传播和参数更新
    optim.zero_grad()      # 清零梯度
    loss.backward()        # 反向传播
    optim.step()           # 更新参数
    sched.step()           # 更新学习率


    # =============== 指标计算和日志记录 ===============
    with torch.no_grad():
        # 计算性能指标
        avg_speed = speed_history.mean(0)                    # 每个智能体的平均速度 [B]
        success = torch.all(distance.flatten(0, 1) > 0, 0)   # 每个智能体是否避免碰撞 [B]
        _success = success.sum() / B                         # 成功率（避免碰撞的比例）
        
        # 收集所有损失和指标用于TensorBoard
        smooth_dict({
            'loss': loss,                                    # 总损失
            'loss_v': loss_v,                               # 速度跟踪损失
            'loss_v_pred': loss_v_pred,                     # 速度估计损失
            'loss_obj_avoidance': loss_obj_avoidance,       # 避障损失
            'loss_d_acc': loss_d_acc,                       # 加速度正则化
            'loss_d_jerk': loss_d_jerk,                     # 加速度变化率正则化
            'loss_d_snap': loss_d_snap,                     # 加速度二阶变化率正则化
            'loss_bias': loss_bias,                         # 速度方向偏置损失
            'loss_speed': loss_speed,                       # 速度大小损失
            'loss_collide': loss_collide,                   # 碰撞惩罚
            'loss_ground_affinity': loss_ground_affinity,   # 地面亲和性损失
            'success': _success,                            # 成功率（无碰撞）
            'max_speed': speed_history.max(0).values.mean(), # 最大速度
            'avg_speed': avg_speed.mean(),                  # 平均速度
            'ar': (success * avg_speed).mean()              # 成功智能体的平均速度
        })
        # =============== 可视化和模型保存 ===============
        if is_save_iter(i):
            # 生成轨迹图（使用第5个batch的数据作为代表）
            # 位置轨迹图
            fig_p, ax = plt.subplots()
            p_history_sample = p_history[:, 4].cpu()  # 第5个batch
            ax.plot(p_history_sample[:, 0], label='x')
            ax.plot(p_history_sample[:, 1], label='y')
            ax.plot(p_history_sample[:, 2], label='z')
            ax.legend()
            ax.set_title('Position History')
            
            # 速度轨迹图
            fig_v, ax = plt.subplots()
            v_history_sample = v_history[:, 4].cpu()
            ax.plot(v_history_sample[:, 0], label='x')
            ax.plot(v_history_sample[:, 1], label='y')
            ax.plot(v_history_sample[:, 2], label='z')
            ax.legend()
            ax.set_title('Velocity History')
            
            # 控制输出轨迹图
            fig_a, ax = plt.subplots()
            act_buffer_sample = act_buffer[:, 4].cpu()
            ax.plot(act_buffer_sample[:, 0], label='x')
            ax.plot(act_buffer_sample[:, 1], label='y')
            ax.plot(act_buffer_sample[:, 2], label='z')
            ax.legend()
            ax.set_title('Control Actions')
            
            # 保存图到TensorBoard
            writer.add_figure('p_history', fig_p, i + 1)
            writer.add_figure('v_history', fig_v, i + 1)
            writer.add_figure('a_reals', fig_a, i + 1)
            
        # 每10000次迭代保存一次模型检查点
        if (i + 1) % 10000 == 0:
            torch.save(model.state_dict(), f'checkpoint{i//10000:04d}.pth')
            
        # 每25次迭代记录一次平滑后的标量到TensorBoard
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                writer.add_scalar(k, sum(v) / len(v), i + 1)  # 平均值
            scaler_q.clear()  # 清空累积队列
