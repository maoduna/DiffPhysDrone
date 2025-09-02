#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

// =============== 深度图渲染函数 ===============
// 功能：从无人机视角渲染深度图，模拟相机感知
// 参数说明：
// - canvas: 输出深度图 [B, H, W]
// - flow: 光流图（当前未使用）
// - balls: 球体障碍物 [B, N, 4] (x,y,z,radius)
// - cylinders: 垂直圆柱体障碍物 [B, N, 3] (x,y,radius)
// - cylinders_h: 水平圆柱体障碍物 [B, N, 3] (x,z,radius)
// - voxels: 体素障碍物 [B, N, 6] (x,y,z,rx,ry,rz)
// - R: 当前相机姿态矩阵 [B, 3, 3]
// - R_old: 上一帧相机姿态矩阵 [B, 3, 3]
// - pos: 当前无人机位置 [B, 3]
// - pos_old: 上一帧无人机位置 [B, 3]
// - drone_radius: 无人机半径
// - n_drones_per_group: 每组无人机数量
// - fov_x_half_tan: 水平视场角的半角正切值
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

// =============== 渲染反向传播函数 ===============
// 功能：计算深度图对相机姿态的梯度，实现可微分渲染
// 参数说明：
// - depth: 深度图 [B, 1, H, W]
// - dddp: 输出梯度 [B, 3, H, W] (对相机位置的梯度)
// - fov_x_half_tan: 水平视场角的半角正切值
void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan);

// =============== 最近点查找函数 ===============
// 功能：计算无人机到最近障碍点的向量，用于避障损失计算
// 参数说明：
// - nearest_pt: 输出最近点位置 [B, N, 3] (N为时间步数)
// - balls: 球体障碍物 [B, M, 4]
// - cylinders: 垂直圆柱体障碍物 [B, M, 3]
// - cylinders_h: 水平圆柱体障碍物 [B, M, 3]
// - voxels: 体素障碍物 [B, M, 6]
// - pos: 无人机位置序列 [B, N, 3] (N个时间步)
// - drone_radius: 无人机半径
// - n_drones_per_group: 每组无人机数量
void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group);

// =============== 姿态更新函数 ===============
// 功能：根据推力和预测速度更新无人机姿态矩阵
// 参数说明：
// - R: 当前姿态矩阵 [B, 3, 3]
// - a_thr: 推力向量 [B, 3] (已减去重力)
// - v_pred: 预测速度方向 [B, 3]
// - alpha: 姿态更新系数 [B, 1] (0-1之间)
// - yaw_inertia: 偏航惯性系数
// 返回：新的姿态矩阵 [B, 3, 3]
torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia);

// =============== 前向动力学仿真函数 ===============
// 功能：执行一步物理仿真，更新无人机状态
// 参数说明：
// - R: 姿态矩阵 [B, 3, 3]
// - dg: 重力扰动 [B, 3]
// - z_drag_coef: Z轴阻力系数 [B, 1]
// - drag_2: 阻力系数 [B, 2] (线性+二次)
// - pitch_ctl_delay: 俯仰控制延迟 [B, 1]
// - act_pred: 预测动作 [B, 3]
// - act: 当前动作 [B, 3]
// - p: 当前位置 [B, 3]
// - v: 当前速度 [B, 3]
// - v_wind: 风速 [B, 3]
// - a: 当前加速度 [B, 3]
// - ctl_dt: 控制时间步长
// - airmode_av2a: 空气模式参数
// 返回：下一时刻状态 [act_next, p_next, v_next, a_next]
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

// =============== 反向梯度计算函数 ===============
// 功能：计算物理仿真对输入参数的梯度，实现可微分物理
// 参数说明：
// - R: 姿态矩阵 [B, 3, 3]
// - dg: 重力扰动 [B, 3]
// - z_drag_coef: Z轴阻力系数 [B, 1]
// - drag_2: 阻力系数 [B, 2]
// - pitch_ctl_delay: 俯仰控制延迟 [B, 1]
// - v: 速度 [B, 3]
// - v_wind: 风速 [B, 3]
// - act_next: 下一时刻动作 [B, 3]
// - _d_act_next: 动作梯度 [B, 3] (下划线表示输入输出参数)
// - d_p_next: 位置梯度 [B, 3]
// - d_v_next: 速度梯度 [B, 3]
// - _d_a_next: 加速度梯度 [B, 3] (下划线表示输入输出参数)
// - grad_decay: 梯度衰减因子
// - ctl_dt: 控制时间步长
// 返回：输入参数梯度 [d_act_pred, d_act, d_p, d_v, d_a]
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

// =============== C++接口层（已注释） ===============
// 注意：AT_ASSERT在0.4版本后变成了AT_CHECK
// 这些是输入验证的宏定义，当前未使用
// #define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 这是一个示例函数，展示如何添加输入验证
// void render(
//     torch::Tensor canvas,
//     torch::Tensor nearest_pt,
//     torch::Tensor balls,
//     torch::Tensor cylinders,
//     torch::Tensor voxels,
//     torch::Tensor Rt) {
//   CHECK_INPUT(input);
//   CHECK_INPUT(weights);
//   CHECK_INPUT(bias);
//   CHECK_INPUT(old_h);
//   CHECK_INPUT(old_cell);

//   return render_cuda(input, weights, bias, old_h, old_cell);
// }

// =============== Python模块绑定 ===============
// 使用PyBind11将C++/CUDA函数暴露给Python
// TORCH_EXTENSION_NAME 在编译时被定义为 'quadsim_cuda'
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 渲染相关函数
  m.def("render", &render_cuda, "render (CUDA)");                    // 深度图渲染
  m.def("rerender_backward", &rerender_backward_cuda, "rerender_backward_cuda (CUDA)");  // 渲染梯度
  
  // 几何计算函数
  m.def("find_nearest_pt", &find_nearest_pt_cuda, "find_nearest_pt (CUDA)");  // 最近点查找
  
  // 动力学仿真函数
  m.def("update_state_vec", &update_state_vec_cuda, "update_state_vec (CUDA)");  // 姿态更新
  m.def("run_forward", &run_forward_cuda, "run_forward_cuda (CUDA)");           // 前向仿真
  m.def("run_backward", &run_backward_cuda, "run_backward_cuda (CUDA)");        // 反向梯度
}
