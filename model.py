import torch
from torch import nn

# =============== 梯度混合函数 ===============
def g_decay(x, alpha):
    """
    梯度混合函数：在数值和梯度之间实现加权混合
    Args:
        x: 输入张量
        alpha: 混合权重 ∈ [0,1]
    Returns:
        数值保持不变，梯度为 alpha*grad(x) + (1-alpha)*0
    用途：在某些情况下部分切断梯度流，防止梯度爆炸
    """
    return x * alpha + x.detach() * (1 - alpha)

# =============== 视觉-运动融合策略网络 ===============
class Model(nn.Module):
    """
    多模态融合策略网络：CNN视觉特征 + 状态编码 + 循环记忆 + 策略输出
    
    网络架构设计原理：
    1. 视觉分支：层次化CNN提取深度图空间特征
    2. 状态分支：MLP编码机体状态向量
    3. 融合层：视觉和状态特征的加法融合
    4. 记忆层：GRU维持时序状态记忆
    5. 策略层：输出控制动作和辅助预测
    """
    def __init__(self, dim_obs=9, dim_action=4) -> None:
        """
        初始化策略网络
        Args:
            dim_obs: 状态观测维度（7或10，取决于是否有里程计）
            dim_action: 动作输出维度（6：期望加速度3维+速度预测3维）
        """
        super().__init__()
        
        # =============== 视觉特征提取分支 ===============
        # 深度图CNN：从64x48→16x12→8x6→4x3→2x1的层次化特征提取
        self.stem = nn.Sequential(
            # 第一层：降采样+特征增强
            nn.Conv2d(1, 32, 2, 2, bias=False),  # [1,12,16] → [32,6,8] 4x下采样
            nn.LeakyReLU(0.05),                  # 小斜率避免死神经元
            
            # 第二层：空间特征提取
            nn.Conv2d(32, 64, 3, bias=False),    # [32,6,8] → [64,4,6] 感受野扩大
            nn.LeakyReLU(0.05),
            
            # 第三层：高级特征抽象
            nn.Conv2d(64, 128, 3, bias=False),   # [64,4,6] → [128,2,4] 进一步抽象
            nn.LeakyReLU(0.05),
            
            # 特征展平和降维
            nn.Flatten(),                        # [128,2,4] → [1024]
            nn.Linear(128*2*4, 192, bias=False), # 降维到192维特征
        )
        
        # =============== 状态编码分支 ===============
        # 将机体状态（速度、目标、朝向等）编码为192维特征
        self.v_proj = nn.Linear(dim_obs, 192)
        self.v_proj.weight.data.mul_(0.5)  # 初始化权重减半，平衡视觉和状态特征
        
        # =============== 循环记忆层 ===============
        # GRU维持时序状态，记忆历史信息用于序列决策
        self.gru = nn.GRUCell(192, 192)
        
        # =============== 策略输出层 ===============
        # 从记忆状态生成最终控制策略
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.fc.weight.data.mul_(0.01)  # 小初始化权重，确保训练初期策略温和
        
        # 特征融合激活函数
        self.act = nn.LeakyReLU(0.05)

    def reset(self):
        """重置网络状态：清空GRU隐状态（在新序列开始时调用）"""
        pass  # GRU隐状态通过外部传递None来重置

    def forward(self, x: torch.Tensor, v, hx=None):
        """
        前向传播：视觉-状态融合推理
        Args:
            x: 深度图输入 [B,1,H,W]，预处理后的逆深度图
            v: 状态向量 [B,dim_obs]，机体坐标系下的状态编码
            hx: GRU隐状态 [B,192]，上一时刻的记忆状态
        Returns:
            act: 动作输出 [B,6]，包含期望加速度和速度预测
            None: 兼容接口，实际未使用
            hx: 更新后的GRU隐状态
        """
        # 视觉特征提取
        img_feat = self.stem(x)  # [B,192] 深度图CNN特征
        
        # 多模态特征融合：视觉特征 + 状态编码
        x = self.act(img_feat + self.v_proj(v))  # [B,192] 加法融合+激活
        
        # 时序记忆更新
        hx = self.gru(x, hx)  # [B,192] 循环神经网络状态更新
        
        # 策略动作生成
        act = self.fc(self.act(hx))  # [B,6] 最终策略输出
        
        return act, None, hx


if __name__ == '__main__':
    Model()
