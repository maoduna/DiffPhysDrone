/**
 * @file quadsim_kernel.cu
 * @brief 四旋翼无人机仿真CUDA核心函数
 * @details 包含渲染、最近点查找和重渲染反向传播等核心功能
 */

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

/**
 * @brief 渲染CUDA核心函数
 * @details 计算每个像素点的深度值，考虑无人机、球体、圆柱体和体素等障碍物
 * 
 * @param canvas 输出画布，存储每个像素的深度值
 * @param flow 光流场（未使用）
 * @param balls 球体障碍物信息 [batch, n_balls, 4] (x, y, z, radius)
 * @param cylinders 垂直圆柱体信息 [batch, n_cylinders, 3] (x, y, radius)
 * @param cylinders_h 水平圆柱体信息 [batch, n_cylinders_h, 3] (x, z, radius)
 * @param voxels 体素障碍物信息 [batch, n_voxels, 6] (x, y, z, rx, ry, rz)
 * @param R 当前旋转矩阵 [batch, 3, 3]
 * @param R_old 上一帧旋转矩阵（未使用）
 * @param pos 当前位置 [batch, 3]
 * @param pos_old 上一帧位置（未使用）
 * @param drone_radius 无人机半径
 * @param n_drones_per_group 每组无人机数量
 * @param fov_x_half_tan 水平视场角的一半正切值
 */
template <typename scalar_t>
__global__ void render_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> canvas,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> flow,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_old,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pos_old,
    float drone_radius,
    int n_drones_per_group,
    float fov_x_half_tan) {

    // 计算当前线程对应的像素索引
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = canvas.size(0);  // batch size
    const int H = canvas.size(1);  // height
    const int W = canvas.size(2);  // width
    if (c >= B * H * W) return;  // 边界检查
    
    // 从线性索引转换为3D索引 (batch, height, width)
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;
    
    // 计算视场角参数
    const scalar_t fov_y_half_tan = fov_x_half_tan / W * H;
    const scalar_t fu = (2 * (u + 0.5) / H - 1) * fov_y_half_tan - 1e-5;
    const scalar_t fv = (2 * (v + 0.5) / W - 1) * fov_x_half_tan - 1e-5;
    
    // 计算射线方向向量 (dx, dy, dz)
    scalar_t dx = R[b][0][0] - fu * R[b][0][2] - fv * R[b][0][1];
    scalar_t dy = R[b][1][0] - fu * R[b][1][2] - fv * R[b][1][1];
    scalar_t dz = R[b][2][0] - fu * R[b][2][2] - fv * R[b][2][1];
    
    // 获取相机位置
    const scalar_t ox = pos[b][0];
    const scalar_t oy = pos[b][1];
    const scalar_t oz = pos[b][2];

    // 初始化最小距离为地面距离
    scalar_t min_dist = 100;
    scalar_t  t = (-1 - oz) / dz;
    if (t > 0) min_dist = t;

    // 检测其他无人机的碰撞
    // 使用椭圆体模型：x^2 + y^2 + 4z^2 = r^2
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[i][0];
        scalar_t cy = pos[i][1];
        scalar_t cz = pos[i][2];
        scalar_t r = 0.15;  // 无人机半径
        
        // 求解二次方程 at^2 + bt + c = 0
        scalar_t a = dx * dx + dy * dy + 4 * dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + 4 * dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        
        if (d >= 0) {
            // 取较小的正根作为碰撞点
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // 检测球体障碍物的碰撞
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        
        // 标准球体碰撞检测
        scalar_t a = dx * dx + dy * dy + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // 检测垂直圆柱体的碰撞（忽略z轴）
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        
        // 2D圆碰撞检测
        scalar_t a = dx * dx + dy * dy;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
        scalar_t d = b * b - 4 * a * c;
        
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }
    
    // 检测水平圆柱体的碰撞（忽略y轴）
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        
        // x-z平面圆碰撞检测
        scalar_t a = dx * dx + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // 检测轴对齐包围盒（AABB）体素的碰撞
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        scalar_t rx = voxels[batch_base][i][3];  // x方向半长度
        scalar_t ry = voxels[batch_base][i][4];  // y方向半长度
        scalar_t rz = voxels[batch_base][i][5];  // z方向半长度
        
        // 计算射线与AABB的交点参数
        scalar_t tx1 = (cx - rx - ox) / dx;
        scalar_t tx2 = (cx + rx - ox) / dx;
        scalar_t tx_min = min(tx1, tx2);
        scalar_t tx_max = max(tx1, tx2);
        
        scalar_t ty1 = (cy - ry - oy) / dy;
        scalar_t ty2 = (cy + ry - oy) / dy;
        scalar_t ty_min = min(ty1, ty2);
        scalar_t ty_max = max(ty1, ty2);
        
        scalar_t tz1 = (cz - rz - oz) / dz;
        scalar_t tz2 = (cz + rz - oz) / dz;
        scalar_t tz_min = min(tz1, tz2);
        scalar_t tz_max = max(tz1, tz2);
        
        // 计算进入和离开时间
        scalar_t t_min = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max = min(min(tx_max, ty_max), tz_max);
        
        // 如果射线与AABB相交且交点在有效范围内
        if (t_min < min_dist && t_min < t_max && t_min > 0)
            min_dist = t_min;
    }

    // 将计算得到的深度值存储到画布中
    canvas[b][u][v] = min_dist;
}

/**
 * @brief 查找最近点CUDA核心函数
 * @details 为每个无人机找到最近的障碍物点，用于碰撞避免
 * 
 * @param nearest_pt 输出最近点坐标 [batch, n_drones, 3]
 * @param balls 球体障碍物信息
 * @param cylinders 垂直圆柱体信息
 * @param cylinders_h 水平圆柱体信息
 * @param voxels 体素障碍物信息
 * @param pos 无人机位置 [batch, n_drones, 3]
 * @param drone_radius 无人机半径
 * @param n_drones_per_group 每组无人机数量
 */
template <typename scalar_t>
__global__ void nearest_pt_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> nearest_pt,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> balls,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> voxels,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> pos,
    float drone_radius,
    int n_drones_per_group) {

    // 计算当前线程对应的无人机索引
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = nearest_pt.size(1);
    const int j = idx / B;
    if (j >= nearest_pt.size(0)) return;
    const int b = idx % B;

    // 获取当前无人机位置
    const scalar_t ox = pos[j][b][0];
    const scalar_t oy = pos[j][b][1];
    const scalar_t oz = pos[j][b][2];

    // 初始化最小距离和最近点坐标
    scalar_t min_dist = max(1e-3f, oz + 1);  // 默认地面距离
    scalar_t nearest_ptx = ox;
    scalar_t nearest_pty = oy;
    scalar_t nearest_ptz = min(-1., oz - 1e-3f);

    // 检测与其他无人机的距离
    const int batch_base = (b / n_drones_per_group) * n_drones_per_group;
    for (int i = batch_base; i < batch_base + n_drones_per_group; i++) {
        if (i == b || i >= B) continue;
        scalar_t cx = pos[j][i][0];
        scalar_t cy = pos[j][i][1];
        scalar_t cz = pos[j][i][2];
        scalar_t r = 0.15;  // 无人机半径
        
        // 计算椭圆体距离（z轴权重为4）
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + 4 * (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        
        if (dist < min_dist) {
            min_dist = dist;
            // 计算最近点坐标
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // 检测与球体障碍物的距离
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[batch_base][i][0];
        scalar_t cy = balls[batch_base][i][1];
        scalar_t cz = balls[batch_base][i][2];
        scalar_t r = balls[batch_base][i][3];
        
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // 检测与垂直圆柱体的距离
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[batch_base][i][0];
        scalar_t cy = cylinders[batch_base][i][1];
        scalar_t r = cylinders[batch_base][i][2];
        
        // 2D距离计算
        scalar_t dist = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy);
        dist = max(1e-3f, sqrt(dist) - r);
        
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy + dist * (cy - oy);
            nearest_ptz = oz;  // z坐标保持不变
        }
    }
    
    // 检测与水平圆柱体的距离
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[batch_base][i][0];
        scalar_t cz = cylinders_h[batch_base][i][1];
        scalar_t r = cylinders_h[batch_base][i][2];
        
        // x-z平面距离计算
        scalar_t dist = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz);
        dist = max(1e-3f, sqrt(dist) - r);
        
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ox + dist * (cx - ox);
            nearest_pty = oy;  // y坐标保持不变
            nearest_ptz = oz + dist * (cz - oz);
        }
    }

    // 检测与体素障碍物的距离
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[batch_base][i][0];
        scalar_t cy = voxels[batch_base][i][1];
        scalar_t cz = voxels[batch_base][i][2];
        
        // 计算到AABB表面的最近点
        scalar_t max_r = max(abs(ox - cx), max(abs(oy - cy), abs(oz - cz))) - 1e-3;
        scalar_t rx = min(max_r, voxels[batch_base][i][3]);
        scalar_t ry = min(max_r, voxels[batch_base][i][4]);
        scalar_t rz = min(max_r, voxels[batch_base][i][5]);
        
        // 计算最近点坐标（投影到AABB表面）
        scalar_t ptx = cx + max(-rx, min(rx, ox - cx));
        scalar_t pty = cy + max(-ry, min(ry, oy - cy));
        scalar_t ptz = cz + max(-rz, min(rz, oz - cz));
        
        scalar_t dist = (ptx - ox) * (ptx - ox) + (pty - oy) * (pty - oy) + (ptz - oz) * (ptz - oz);
        dist = sqrt(dist);
        
        if (dist < min_dist) {
            min_dist = dist;
            nearest_ptx = ptx;
            nearest_pty = pty;
            nearest_ptz = ptz;
        }
    }
    
    // 存储最近点坐标
    nearest_pt[j][b][0] = nearest_ptx;
    nearest_pt[j][b][1] = nearest_pty;
    nearest_pt[j][b][2] = nearest_ptz;
}

/**
 * @brief 重渲染反向传播CUDA核心函数
 * @details 计算深度图的梯度，用于反向传播训练
 * 
 * @param depth 深度图 [batch, 1, H, W]
 * @param dddp 输出梯度 [batch, 3, H/2, W/2]
 * @param fov_x_half_tan 水平视场角的一半正切值
 */
template <typename scalar_t>
__global__ void rerender_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> depth,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> dddp,
    float fov_x_half_tan) {

    // 计算当前线程对应的像素索引
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = dddp.size(0);
    const int H = dddp.size(2);
    const int W = dddp.size(3);
    if (c >= B * H * W) return;
    
    const int b = c / (H * W);
    const int u = (c % (H * W)) / W;
    const int v = c % W;

    // 计算单位像素对应的角度
    const scalar_t unit = fov_x_half_tan / W;
    
    // 从2x2像素块计算平均深度和梯度
    const scalar_t d = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] + 
                       depth[b][0][u*2][v*2+1] + depth[b][0][u*2+1][v*2+1]) / 4 * unit;
    
    // 计算y和z方向的梯度
    const scalar_t dddy = (depth[b][0][u*2][v*2] + depth[b][0][u*2+1][v*2] - 
                          depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    const scalar_t dddz = (depth[b][0][u*2][v*2] - depth[b][0][u*2+1][v*2] + 
                          depth[b][0][u*2][v*2+1] - depth[b][0][u*2+1][v*2+1]) / 2 / d;
    
    // 计算归一化的梯度向量
    const scalar_t dddp_norm = max(8., sqrt(1 + dddy * dddy + dddz * dddz));
    dddp[b][0][u][v] = -1. / dddp_norm;      // x方向梯度
    dddp[b][1][u][v] = dddy / dddp_norm;     // y方向梯度
    dddp[b][2][u][v] = dddz / dddp_norm;     // z方向梯度
}

} // namespace

/**
 * @brief 渲染CUDA接口函数
 * @details 调用CUDA核心函数进行深度图渲染
 */
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
    float fov_x_half_tan) {
    
    const int threads = 1024;  // 每个块的线程数
    size_t state_size = canvas.numel();  // 总像素数
    const dim3 blocks((state_size + threads - 1) / threads);  // 计算需要的块数

    // 根据数据类型分发到对应的模板函数
    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "render_cuda", ([&] {
        render_cuda_kernel<scalar_t><<<blocks, threads>>>(
            canvas.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            flow.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R_old.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pos_old.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group,
            fov_x_half_tan);
    }));
}

/**
 * @brief 重渲染反向传播CUDA接口函数
 * @details 调用CUDA核心函数计算深度图梯度
 */
void rerender_backward_cuda(
    torch::Tensor depth,
    torch::Tensor dddp,
    float fov_x_half_tan) {
    
    const int threads = 1024;
    size_t state_size = dddp.numel();
    const dim3 blocks((state_size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(depth.type(), "rerender_backward_cuda", ([&] {
        rerender_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            depth.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            dddp.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            fov_x_half_tan);
    }));
}

/**
 * @brief 查找最近点CUDA接口函数
 * @details 调用CUDA核心函数为每个无人机找到最近的障碍物点
 */
void find_nearest_pt_cuda(
    torch::Tensor nearest_pt,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor pos,
    float drone_radius,
    int n_drones_per_group) {
    
    const int threads = 1024;
    size_t state_size = pos.size(0) * pos.size(1);  // batch * n_drones
    const dim3 blocks((state_size + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(pos.type(), "nearest_pt_cuda", ([&] {
        nearest_pt_cuda_kernel<scalar_t><<<blocks, threads>>>(
            nearest_pt.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            balls.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            cylinders_h.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            voxels.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            pos.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            drone_radius,
            n_drones_per_group);
    }));
}
