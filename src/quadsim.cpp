#include <torch/extension.h>

#include <vector>

// =============== CUDA核心函数前向声明 ===============
// 以下函数在对应的.cu文件中实现，此处仅声明接口

// =============== 高性能深度图渲染接口 ===============
// 功能：实时渲染无人机第一人称视角深度图，支持复杂几何场景
// 
// 渲染特性：
// - 支持多种几何体：球体、圆柱体（垂直/水平）、轴对齐包围盒
// - 编队感知：渲染其他无人机（使用椭圆体模型模拟机体形状）
// - 射线追踪：精确计算像素级深度值
// - 批量并行：同时处理多个视角
// 
// 参数详解：
// - canvas: [B,H,W] 输出深度图缓冲区，存储每像素到最近表面的距离
// - flow: [B,2,H,W] 光流图缓冲区（当前版本未实现）
// - balls: [B,N,4] 球体障碍物 (中心x,y,z + 半径)
// - cylinders: [B,N,3] 垂直圆柱体 (底面中心x,y + 半径，z轴无限延伸)
// - cylinders_h: [B,N,3] 水平圆柱体 (轴心x,z + 半径，y轴无限延伸)
// - voxels: [B,N,6] 长方体体素 (中心x,y,z + 半尺寸rx,ry,rz)
// - R: [B,3,3] 当前相机姿态矩阵（机体→世界坐标变换）
// - R_old: [B,3,3] 上一帧姿态矩阵（用于运动模糊，当前未使用）
// - pos: [B,3] 当前无人机世界坐标位置
// - pos_old: [B,3] 上一帧位置（用于运动模糊，当前未使用）
// - drone_radius: 无人机碰撞半径（用于编队避障检测）
// - n_drones_per_group: 编队规模（同组无人机会相互感知）
// - fov_x_half_tan: 水平视场角半角正切值（决定观察范围宽度）
void render_cuda(
    torch::Tensor canvas,
    torch::Tensor flow,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor R_old,
    torch::Tensor pos,
    torch::Tensor pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan);

// =============== 可微分深度图重渲染反向传播接口 ===============
// 功能：实现深度图渲染的反向传播，支持端到端训练中的梯度计算
// 
// 用途：在某些需要对渲染过程求导的场景中使用，例如：
// - 基于渲染损失的直接优化
// - 视觉伺服控制中的梯度计算
// - 形状优化或相机位姿估计
// 
// 算法：通过有限差分近似计算深度图对相机参数的雅可比矩阵
// 
// - depth: [B,1,H,W] 输入深度图（经4x4下采样）
// - dddp: [B,3,H/2,W/2] 输出位置梯度 (∂depth/∂position)
// - fov_x_half_tan: 相机内参，用于像素到射线的坐标变换
void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan);

// =============== 最近障碍点查询接口 ===============
// 功能：高效计算每个无人机到最近障碍物表面的距离和方向
// 
// 核心算法：
// - 球体：解析计算点到球面的最短距离
// - 圆柱体：2D/3D距离场计算
// - 体素：AABB最近点查询
// - 编队：椭圆体距离函数（考虑机体朝向）
// 
// 应用场景：
// 1. 避障损失函数：基于距离的软约束
// 2. 碰撞检测：实时安全监控
// 3. 路径规划：梯度信息用于局部路径优化
// 
// - nearest_pt: [T,B,3] 输出最近点坐标（T个预测时间步）
// - balls: [B,M,4] 场景中的球体障碍物
// - cylinders: [B,M,3] 垂直圆柱体障碍物
// - cylinders_h: [B,M,3] 水平圆柱体障碍物  
// - voxels: [B,M,6] 长方体体素障碍物
// - pos: [T,B,3] 无人机轨迹位置序列
// - drone_radius: 无人机安全半径（碰撞检测用）
// - n_drones_per_group: 编队内无人机数量（相互感知范围）
void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group);

// =============== 四旋翼姿态控制接口 ===============
// 功能：基于推力矢量和飞行意图更新机体姿态
// 
// 控制原理：
// 1. 推力方向决定机体上轴（Z轴）
// 2. 飞行意图与偏航惯性结合决定前轴（X轴）
// 3. 右手定则确定左轴（Y轴）
// 4. 平滑插值避免姿态突变
// 
// 物理意义：
// - 模拟四旋翼通过调节各电机转速实现姿态控制
// - 考虑机体转动惯量差异（偏航轴 vs 俯仰/滚转轴）
// - 支持期望航向与当前航向的平滑过渡
// 
// - R: [B,3,3] 当前机体姿态矩阵（机体→世界）
// - a_thr: [B,3] 期望推力矢量（世界坐标，不含重力）
// - v_pred: [B,3] 期望飞行方向矢量
// - alpha: [B,1] 姿态响应速率 ∈ [0,1]，越大响应越慢
// - yaw_inertia: 偏航惯性系数，模拟偏航轴转动惯量差异
// 返回：[B,3,3] 更新后的姿态矩阵
torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia);

// =============== 高精度四旋翼动力学仿真接口 ===============
// 功能：执行一步完整的六自由度刚体动力学仿真
// 
// 物理模型包含：
// 1. 控制系统：一阶延迟模拟电机响应特性
// 2. 阻力模型：分轴空气阻力（线性+二次项）
// 3. 环境扰动：重力场不均匀性、风场干扰
// 4. 空气动力学：高速飞行时的角速度阻尼效应
// 5. 运动积分：基于Verlet积分的高精度数值求解
// 
// 可微分特性：
// - 支持PyTorch自动微分框架
// - 梯度衰减机制防止长时序训练中的梯度爆炸
// - 所有物理参数均可学习和优化
// 
// - R: [B,3,3] 机体姿态矩阵
// - dg: [B,3] 重力场扰动（模拟地球重力场不均匀性）
// - z_drag_coef: [B,1] 垂直轴阻力系数（下洗流效应）
// - drag_2: [B,2] 阻力参数 [二次项系数, 一次项系数]
// - pitch_ctl_delay: [B,1] 控制延迟时间常数
// - act_pred: [B,3] 网络预测的期望加速度
// - act: [B,3] 当前时刻的实际控制输出
// - p: [B,3] 当前位置
// - v: [B,3] 当前速度
// - v_wind: [B,3] 环境风速
// - a: [B,3] 当前加速度
// - ctl_dt: 仿真时间步长
// - airmode_av2a: 空气模式系数（角速度→附加加速度）
// 返回：[act_next, p_next, v_next, a_next] 下一时刻完整状态
std::vector<torch::Tensor> run_forward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor act_pred,
    torch::Tensor act,
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor a,
    float ctl_dt,
    float airmode_av2a);

// =============== 可微分物理反向传播接口 ===============
// 功能：计算动力学仿真的完整雅可比矩阵，支持端到端梯度反向传播
// 
// 计算原理：
// 1. 基于链式法则计算各物理参数的敏感性
// 2. 考虑控制延迟、阻力模型等的梯度贡献
// 3. 应用梯度衰减避免长时序优化中的数值不稳定
// 4. 高效并行计算，保持与前向传播一致的性能
// 
// 梯度流向：
// 损失函数 → 下一时刻状态 → 物理参数 → 网络输出
// 
// 应用场景：
// - 端到端强化学习：策略梯度直接传播到动作输出
// - 物理参数辨识：通过观测数据学习无人机物理特性
// - 鲁棒控制：对环境扰动的敏感性分析
// 
// - R: [B,3,3] 机体姿态矩阵（前向传播中保存的中间值）
// - dg: [B,3] 重力扰动向量
// - z_drag_coef: [B,1] Z轴阻力系数
// - drag_2: [B,2] 阻力参数
// - pitch_ctl_delay: [B,1] 控制延迟常数
// - v: [B,3] 当前速度
// - v_wind: [B,3] 风速
// - act_next: [B,3] 下一时刻动作（前向计算结果）
// - _d_act_next: [B,3] 动作输出梯度（来自损失函数）
// - d_p_next: [B,3] 位置输出梯度
// - d_v_next: [B,3] 速度输出梯度
// - _d_a_next: [B,3] 加速度输出梯度
// - grad_decay: 梯度衰减因子（防止梯度爆炸）
// - ctl_dt: 时间步长
// 返回：[d_act_pred, d_act, d_p, d_v, d_a] 各输入参数的梯度
std::vector<torch::Tensor> run_backward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor act_next,
    torch::Tensor _d_act_next,
    torch::Tensor d_p_next,
    torch::Tensor d_v_next,
    torch::Tensor _d_a_next,
    float grad_decay,
    float ctl_dt);

// =============== 可选的输入验证宏（当前未启用）===============
// 用于调试和开发阶段的输入检查，可以检测常见的张量错误
// 在生产环境中通常关闭以提高性能
// 
// #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// =============== Python扩展模块接口绑定 ===============
// 使用PyBind11框架将C++/CUDA函数暴露给Python环境
// 编译时TORCH_EXTENSION_NAME被设置为'quadsim_cuda'
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 模块文档字符串
  m.doc() = "四旋翼无人机物理仿真CUDA加速模块 - 提供高性能可微分动力学仿真";
  
  // =============== 视觉感知模块 ===============
  m.def("render", &render_cuda, 
        "实时深度图渲染：从无人机视角生成障碍物深度图，支持多种几何体和编队感知");
  m.def("rerender_backward", &rerender_backward_cuda, 
        "深度图梯度计算：实现可微分渲染的反向传播，用于视觉伺服控制");
  
  // =============== 空间查询模块 ===============
  m.def("find_nearest_pt", &find_nearest_pt_cuda, 
        "最近障碍点查询：高效计算无人机到最近障碍物的距离和方向，用于避障");
  
  // =============== 动力学仿真模块 ===============
  m.def("update_state_vec", &update_state_vec_cuda, 
        "姿态控制更新：基于推力矢量和飞行意图计算机体姿态变化");
  m.def("run_forward", &run_forward_cuda, 
        "前向动力学仿真：执行一步完整的六自由度刚体动力学计算");
  m.def("run_backward", &run_backward_cuda, 
        "反向梯度传播：计算动力学仿真的完整雅可比矩阵，支持端到端训练");
}
