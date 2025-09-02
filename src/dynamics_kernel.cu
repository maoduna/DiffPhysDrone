/**
 * @file dynamics_kernel.cu
 * @brief 四旋翼无人机动力学仿真CUDA核心函数
 * @details 包含状态更新、前向传播和反向传播等核心动力学计算功能
 */

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

/**
 * @brief 更新状态向量CUDA核心函数
 * @details 根据推力指令和预测速度更新无人机的旋转矩阵
 * 
 * @param R_new 输出新旋转矩阵 [batch, 3, 3]
 * @param R 当前旋转矩阵 [batch, 3, 3]
 * @param a_thr 推力指令 [batch, 3] (ax, ay, az)
 * @param v_pred 预测速度 [batch, 3] (vx, vy, vz)
 * @param alpha 平滑因子 [batch, 1]
 * @param yaw_inertia 偏航惯性系数
 */
template <typename scalar_t>
__global__ void update_state_vec_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_new,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a_thr,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_pred,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> alpha,
    float yaw_inertia) {
    
    // 计算当前线程对应的无人机索引
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (b >= B) return;
    
    // 获取推力指令并添加重力
    scalar_t ax = a_thr[b][0];
    scalar_t ay = a_thr[b][1];
    scalar_t az = a_thr[b][2] + 9.80665;  // 添加重力加速度
    
    // 计算推力大小
    scalar_t thrust = sqrt(ax*ax+ay*ay+az*az);
    
    // 计算上向量（推力方向）
    scalar_t ux = ax / thrust;
    scalar_t uy = ay / thrust;
    scalar_t uz = az / thrust;
    
    // 计算前向向量（考虑偏航惯性）
    scalar_t fx = R[b][0][0] * yaw_inertia + v_pred[b][0];
    scalar_t fy = R[b][1][0] * yaw_inertia + v_pred[b][1];
    scalar_t fz = R[b][2][0] * yaw_inertia + v_pred[b][2];
    
    // 归一化前向向量
    scalar_t t = sqrt(fx * fx + fy * fy + fz * fz);
    fx = (1 - alpha[b][0]) * (fx / t) + alpha[b][0] * R[b][0][0];
    fy = (1 - alpha[b][0]) * (fy / t) + alpha[b][0] * R[b][1][0];
    fz = (1 - alpha[b][0]) * (fz / t) + alpha[b][0] * R[b][2][0];
    
    // 确保前向向量垂直于上向量（z分量约束）
    fz = (fx * ux + fy * uy) / -uz;
    
    // 重新归一化前向向量
    t = sqrt(fx * fx + fy * fy + fz * fz);
    fx /= t;
    fy /= t;
    fz /= t;
    
    // 计算左向量（叉积：上向量 × 前向向量）
    // 构建新的旋转矩阵 [前向向量, 左向量, 上向量]
    R_new[b][0][0] = fx;                    // 前向向量的x分量
    R_new[b][0][1] = uy * fz - uz * fy;     // 左向量的x分量
    R_new[b][0][2] = ux;                    // 上向量的x分量
    R_new[b][1][0] = fy;                    // 前向向量的y分量
    R_new[b][1][1] = uz * fx - ux * fz;     // 左向量的y分量
    R_new[b][1][2] = uy;                    // 上向量的y分量
    R_new[b][2][0] = fz;                    // 前向向量的z分量
    R_new[b][2][1] = ux * fy - uy * fx;     // 左向量的z分量
    R_new[b][2][2] = uz;                    // 上向量的z分量
}

/**
 * @brief 前向传播CUDA核心函数
 * @details 模拟无人机动力学，计算下一时刻的状态
 * 
 * @param R 旋转矩阵 [batch, 3, 3]
 * @param dg 随机扰动 [batch, 3]
 * @param z_drag_coef z方向阻力系数 [batch, 1]
 * @param drag_2 二次阻力系数 [batch, 2] (二次项系数, 一次项系数)
 * @param pitch_ctl_delay 俯仰控制延迟 [batch, 1]
 * @param act_pred 预测动作 [batch, 3]
 * @param act 当前动作 [batch, 3]
 * @param p 当前位置 [batch, 3]
 * @param v 当前速度 [batch, 3]
 * @param v_wind 风速 [batch, 3]
 * @param a 当前加速度 [batch, 3]
 * @param act_next 输出下一时刻动作 [batch, 3]
 * @param p_next 输出下一时刻位置 [batch, 3]
 * @param v_next 输出下一时刻速度 [batch, 3]
 * @param a_next 输出下一时刻加速度 [batch, 3]
 * @param ctl_dt 控制时间步长
 * @param airmode_av2a 空模式角速度到加速度的转换系数
 */
template <typename scalar_t>
__global__ void run_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dg,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z_drag_coef,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> drag_2,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pitch_ctl_delay,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_pred,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_wind,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a_next,
    float ctl_dt, float airmode_av2a) {
    
    // 计算当前线程对应的无人机索引
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (i >= B) return;
    
    // 计算控制延迟的指数衰减因子
    scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);
    
    // 平滑动作更新：act_next = act_pred * (1-alpha) + act * alpha
    for (int j=0; j<3; j++)
        act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;
    
    // 计算相对于风速的速度
    scalar_t v_rel_wind_x = v[i][0] - v_wind[i][0];
    scalar_t v_rel_wind_y = v[i][1] - v_wind[i][1];
    scalar_t v_rel_wind_z = v[i][2] - v_wind[i][2];
    
    // 将相对速度投影到无人机坐标系
    scalar_t v_up_s = v_rel_wind_x * R[i][0][2] + v_rel_wind_y * R[i][1][2] + v_rel_wind_z * R[i][2][2];      // 上方向分量
    scalar_t v_fwd_s = v_rel_wind_x * R[i][0][0] + v_rel_wind_y * R[i][1][0] + v_rel_wind_z * R[i][2][0];    // 前向分量
    scalar_t v_left_s = v_rel_wind_x * R[i][0][1] + v_rel_wind_y * R[i][1][1] + v_rel_wind_z * R[i][2][1];   // 左向分量
    
    // 计算速度的二次项（用于阻力计算）
    scalar_t v_up_2 = v_up_s * abs(v_up_s);
    scalar_t v_fwd_2 = v_fwd_s * abs(v_fwd_s);
    scalar_t v_left_2 = v_left_s * abs(v_left_s);

    // 计算阻力加速度（二次项和一次项）
    scalar_t a_drag_2[3], a_drag_1[3];
    for (int j=0; j<3; j++){
        a_drag_2[j] = v_up_2 * R[i][j][2] * z_drag_coef[i][0] + v_left_2 * R[i][j][1] + v_fwd_2 * R[i][j][0];
        a_drag_1[j] = v_up_s * R[i][j][2] * z_drag_coef[i][0] + v_left_s * R[i][j][1] + v_fwd_s * R[i][j][0];
    }
    
    // 计算角速度（通过动作向量的点积）
    scalar_t dot = act[i][0] * act_next[i][0] + act[i][1] * act_next[i][1] + (act[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
    scalar_t n1 = act[i][0] * act[i][0] + act[i][1] * act[i][1] + (act[i][2] + 9.80665) * (act[i][2] + 9.80665);
    scalar_t n2 = act_next[i][0] * act_next[i][0] + act_next[i][1] * act_next[i][1] + (act_next[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
    scalar_t av = acos(max(-1., min(1., dot / max(1e-8, sqrt(n1) * sqrt(n2))))) / ctl_dt;

    // 计算空模式加速度（角速度转换为加速度）
    scalar_t ax = act[i][0];
    scalar_t ay = act[i][1];
    scalar_t az = act[i][2] + 9.80665;
    scalar_t thrust = sqrt(ax*ax+ay*ay+az*az);
    scalar_t airmode_a[3] = {
        ax / thrust * av * airmode_av2a,
        ay / thrust * av * airmode_av2a,
        az / thrust * av * airmode_av2a};
    
    // 计算下一时刻加速度：act_next + dg - 阻力 + 空模式加速度
    for (int j=0; j<3; j++)
        a_next[i][j] = act_next[i][j] + dg[i][j] - a_drag_2[j] * drag_2[i][0] - a_drag_1[j] * drag_2[i][1] + airmode_a[j];
    
    // 更新位置：p_next = p + v*dt + 0.5*a*dt^2
    for (int j=0; j<3; j++)
        p_next[i][j] = p[i][j] + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;
    
    // 更新速度：v_next = v + (a + a_next)/2 * dt
    for (int j=0; j<3; j++)
        v_next[i][j] = v[i][j] + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;
}

/**
 * @brief 反向传播CUDA核心函数
 * @details 计算梯度，用于训练时的反向传播
 * 
 * @param R 旋转矩阵 [batch, 3, 3]
 * @param dg 随机扰动 [batch, 3]
 * @param z_drag_coef z方向阻力系数 [batch, 1]
 * @param drag_2 二次阻力系数 [batch, 2]
 * @param pitch_ctl_delay 俯仰控制延迟 [batch, 1]
 * @param v 当前速度 [batch, 3]
 * @param v_wind 风速 [batch, 3]
 * @param act_next 下一时刻动作 [batch, 3]
 * @param _d_act_next 下一时刻动作的梯度 [batch, 3]
 * @param d_act_pred 预测动作的梯度 [batch, 3]
 * @param d_act 当前动作的梯度 [batch, 3]
 * @param d_p 位置的梯度 [batch, 3]
 * @param d_v 速度的梯度 [batch, 3]
 * @param d_a 加速度的梯度 [batch, 3]
 * @param d_p_next 下一时刻位置的梯度 [batch, 3]
 * @param d_v_next 下一时刻速度的梯度 [batch, 3]
 * @param _d_a_next 下一时刻加速度的梯度 [batch, 3]
 * @param grad_decay 梯度衰减系数
 * @param ctl_dt 控制时间步长
 */
template <typename scalar_t>
__global__ void run_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dg,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z_drag_coef,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> drag_2,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pitch_ctl_delay,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_wind,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_act_pred,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_act,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_a,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_act_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_a_next,
    float grad_decay,
    float ctl_dt) {
    
    // 计算当前线程对应的无人机索引
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (i >= B) return;
    
    // 计算控制延迟的指数衰减因子
    scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);
    
    // 获取动作和加速度的梯度
    scalar_t d_act_next[3] = {_d_act_next[i][0], _d_act_next[i][1], _d_act_next[i][2]};
    scalar_t d_a_next[3] = {_d_a_next[i][0], _d_a_next[i][1], _d_a_next[i][2]};
    
    // 反向传播开始
    // 从速度更新方程开始：v_next = v + 0.5*(a + a_next)*dt
    for (int j=0; j<3; j++){
        d_v[i][j] = d_v_next[i][j] * pow(grad_decay, ctl_dt);  // 应用梯度衰减
        d_a[i][j] = 0.5 * ctl_dt * d_v_next[i][j];             // 对当前加速度的梯度
        d_a_next[j] += 0.5 * ctl_dt * d_v_next[i][j];          // 对下一时刻加速度的梯度
    }
    
    // 从位置更新方程：p_next = p + v*dt + 0.5*a*dt^2
    for (int j=0; j<3; j++){
        d_p[i][j] = d_p_next[i][j] * pow(grad_decay, ctl_dt);  // 应用梯度衰减
        d_v[i][j] += ctl_dt * d_p_next[i][j];                   // 对速度的梯度
        d_a[i][j] += 0.5 * ctl_dt * ctl_dt * d_p_next[i][j];  // 对加速度的梯度
    }
    
    // 计算阻力相关的梯度
    scalar_t d_a_drag_2[3];
    scalar_t d_a_drag_1[3];
    for (int j=0; j<3; j++){
        d_act_next[j] += d_a_next[j];                           // 对下一时刻动作的梯度
        d_a_drag_2[j] = -d_a_next[j] * drag_2[i][0];           // 对二次阻力的梯度
        d_a_drag_1[j] = -d_a_next[j] * drag_2[i][1];           // 对一次阻力的梯度
    }

    // 重新计算相对风速（用于梯度计算）
    scalar_t v_rel_wind_x = v[i][0] - v_wind[i][0];
    scalar_t v_rel_wind_y = v[i][1] - v_wind[i][1];
    scalar_t v_rel_wind_z = v[i][2] - v_wind[i][2];
    scalar_t v_fwd_s = v_rel_wind_x * R[i][0][0] + v_rel_wind_y * R[i][1][0] + v_rel_wind_z * R[i][2][0];
    scalar_t v_left_s = v_rel_wind_x * R[i][0][1] + v_rel_wind_y * R[i][1][1] + v_rel_wind_z * R[i][2][1];
    scalar_t v_up_s = v_rel_wind_x * R[i][0][2] + v_rel_wind_y * R[i][1][2] + v_rel_wind_z * R[i][2][2];
    
    // 计算速度分量的梯度
    scalar_t d_v_fwd_s = 0;
    scalar_t d_v_left_s = 0;
    scalar_t d_v_up_s = 0;
    
    // 计算阻力对速度分量的梯度
    for (int j=0; j<3; j++){
        // 二次阻力项：v^2 * R * drag_coef
        d_v_fwd_s += d_a_drag_2[j] * 2 * abs(v_fwd_s) * R[i][j][0];
        d_v_left_s += d_a_drag_2[j] * 2 * abs(v_left_s) * R[i][j][1];
        d_v_up_s += d_a_drag_2[j] * 2 * abs(v_up_s) * R[i][j][2] * z_drag_coef[i][0];
        
        // 一次阻力项：v * R * drag_coef
        d_v_fwd_s += d_a_drag_1[j] * R[i][j][0];
        d_v_left_s += d_a_drag_1[j] * R[i][j][1];
        d_v_up_s += d_a_drag_1[j] * R[i][j][2] * z_drag_coef[i][0];
    }

    // 将速度分量的梯度传播到笛卡尔坐标系的速度梯度
    for (int j=0; j<3; j++){
        d_v[i][j] += R[i][j][0] * d_v_fwd_s;   // 前向分量贡献
        d_v[i][j] += R[i][j][1] * d_v_left_s;  // 左向分量贡献
        d_v[i][j] += R[i][j][2] * d_v_up_s;    // 上向分量贡献
    }
    
    // 计算动作的梯度：act_next = act_pred * (1-alpha) + act * alpha
    for (int j=0; j<3; j++){
        d_act_pred[i][j] = (1 - alpha) * d_act_next[j];  // 对预测动作的梯度
        d_act[i][j] = alpha * d_act_next[j];              // 对当前动作的梯度
    }
}

} // namespace

/**
 * @brief 前向传播CUDA接口函数
 * @details 调用CUDA核心函数进行动力学前向传播
 * 
 * @return 返回包含下一时刻状态的张量向量 [act_next, p_next, v_next, a_next]
 */
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
    float airmode_av2a){

    // 创建输出张量
    torch::Tensor act_next = torch::empty_like(act);
    torch::Tensor p_next = torch::empty_like(p);
    torch::Tensor v_next = torch::empty_like(v);
    torch::Tensor a_next = torch::empty_like(a);

    // 设置CUDA内核参数
    const int threads = R.size(0);  // 每个块使用batch size个线程
    const dim3 blocks(1);           // 只使用一个块
    
    // 根据数据类型分发到对应的模板函数
    AT_DISPATCH_FLOATING_TYPES(R.type(), "run_forward_cuda", ([&] {
        run_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            dg.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z_drag_coef.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drag_2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pitch_ctl_delay.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_wind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            a.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            a_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            ctl_dt, airmode_av2a);
    }));
    
    return {act_next, p_next, v_next, a_next};
}

/**
 * @brief 反向传播CUDA接口函数
 * @details 调用CUDA核心函数进行梯度计算
 * 
 * @return 返回包含梯度的张量向量 [d_act_pred, d_act, d_p, d_v, d_a]
 */
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
    float ctl_dt){

    // 创建输出梯度张量
    torch::Tensor d_act_pred = torch::empty_like(dg);
    torch::Tensor d_act = torch::empty_like(dg);
    torch::Tensor d_p = torch::empty_like(dg);
    torch::Tensor d_v = torch::empty_like(dg);
    torch::Tensor d_a = torch::empty_like(dg);

    // 设置CUDA内核参数
    const int threads = R.size(0);
    const dim3 blocks(1);
    
    // 根据数据类型分发到对应的模板函数
    AT_DISPATCH_FLOATING_TYPES(R.type(), "run_backward_cuda", ([&] {
        run_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            dg.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z_drag_coef.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drag_2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pitch_ctl_delay.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_wind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_act_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_act.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_a.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_a_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_decay, ctl_dt);
    }));
    
    return {d_act_pred, d_act, d_p, d_v, d_a};
}

/**
 * @brief 更新状态向量CUDA接口函数
 * @details 调用CUDA核心函数更新无人机的旋转矩阵
 * 
 * @return 返回更新后的旋转矩阵
 */
torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia) {
    
    const int threads = a_thr.size(0);
    const dim3 blocks(1);
    torch::Tensor R_new = torch::empty_like(R);
    
    // 根据数据类型分发到对应的模板函数
    AT_DISPATCH_FLOATING_TYPES(a_thr.type(), "update_state_vec", ([&] {
        update_state_vec_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R_new.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            a_thr.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            alpha.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            yaw_inertia);
    }));
    
    return R_new;
}
