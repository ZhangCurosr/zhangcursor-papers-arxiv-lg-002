---
title: "Neural-topology-optimization-of-ship-structures-under-propul"
source: https://arxiv.org/pdf/2609.38089v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 14:09:22"
field: "结构动力学拓扑优化"
keywords: ["Topology Optimization", "Active Input Power", "Neural Reparameterization", "Ship Structures", "Forced Vibration", "KATO", "cKAN", "Structural Dynamics"]
innovations: ["首次将AIP目标应用于船舶结构强制振动拓扑优化", "证明cKAN神经生成器在动态振动问题中产生连通二值化结构", "KATO在AIP匹配下二进制静力柔顺性较GCMMA降低22-59倍"]
benchmarks: ["Static compliance benchmark vs GCMMA", "AIP reduction vs size-optimized references", "Frequency sweep 5-500 Hz with ROM"]
---

# 论文速读：Neural-topology-optimization-of-ship-structures-under-propul

## 一句话总结
本文提出了一种基于主动输入功率（AIP）的船舶结构强制振动拓扑优化方法，将神经重参数化拓扑优化（KATO/cKAN生成器）应用于发动机支撑甲板板和推进器基础框架的设计，在AIP抑制、静力柔顺性和结构连通性方面均优于传统GCMMA方法和工程参考设计。

## 研究问题与动机
- 船舶结构振动会导致疲劳损伤、噪音和仪器设备损坏，推进机械振动通过基础结构传递至船体辐射水下噪音（URN）
- 传统动态柔顺性（dynamic compliance）目标在共振和反共振频率附近存在符号变化、奇异性和非单调行为，导致基于梯度的优化器不稳定
- 现有AIP拓扑优化主要应用于学术基准问题，未探索其在大型船舶结构工程问题中的应用
- 大规模船舶结构拓扑优化的计算成本高昂，且需要特征尺寸控制和应力约束等制造约束

## 核心贡献（创新点）
- **AIP驱动的船舶结构动态拓扑优化框架**：将AIP目标从静态拓扑优化扩展到频率域强制振动设计，结合静力柔顺性正则化、Helmholtz滤波和三场投影实现特征尺寸控制
- **KATO/cKAN生成器在动态振动问题上的首次应用**：证明了卷积Kolmogorov-Arnold网络作为结构先验和隐式正则化器，能产生接近二值化、连通载荷路径的设计，且阈值处理后仍保持有效性
- **神经优化器与传统GCMMA的系统对比**：在推进器基础框架案例中，KATO在AIP上匹配GCMMA（误差<0.5 dB），但二进制静力柔顺性降低22-36倍，且在近共振300 Hz测试中二进制静力柔顺性低59倍
- **加速频率响应分析**：采用修正的200模态降阶模型（ROM），在5-500 Hz扫频中对每个频率点的评估速度比全阶求解快990-2125倍

## 方法详解
- **主动输入功率（AIP）目标**：时均输入功率 $\Pi_{\text{in}}(\omega) = \frac{1}{2}\operatorname{Re}[\mathbf{F}^H \dot{\mathbf{U}}]$，对被动结构非负，避免动态柔顺性的符号歧义；对数分贝表示 $L_\Pi = 10\log_{10}(|\Pi_{\text{in}}|/\Pi_0)$
- **混合目标函数**：$J = w_{\text{AIP}}\frac{\Pi_{\text{in}}}{\Pi_{\text{ref}}} + (1-w_{\text{AIP}})\frac{C}{C_{\text{ref}}}$，结合AIP最小化和静力柔顺性正则化，防止弱连接结构
- **应力感知正则化**：使用p-范数聚合von Mises应力 $\tilde{\sigma}_p = \left(\frac{\sum w_i \sigma_{\text{vm},i}^p}{\sum w_i + \epsilon}\right)^{1/p}$，其中$p=8$，相位无关的动态应力度量 $\sigma_{\text{vm}} = \sqrt{\sigma_{\text{vm,r}}^2 + \sigma_{\text{vm,i}}^2}$
- **KATO神经参数化**：cKAN生成器 $G_\theta$ 将潜码 $\mathbf{z}$ 映射到对数字段，经约束sigmoid（保体积）、Helmholtz PDE滤波（特征尺寸控制）、Heaviside投影（二进制化）得到物理密度场
- **伴随灵敏度分析**：通过自定义torch.autograd.Function存储LU分解，避免额外因子分解，逆系统解由存储的LU因子回代获得
- **降阶模型（ROM）**：截断200模态基加上静态残余柔量修正，低频率偏差消除，扫频效率提升3个数量级

## 实验与结果
- **数据集/案例**：100 Hz发动机支撑甲板板（双板/单板，$v_f=0.30$）和18 Hz推进器基础框架（$v_f=0.45$）；材料为钢（$E=210$ GPa，$\rho=7860$ kg/m³）
- **基线对比**：
  - 甲板案例：与尺寸优化的X型芯双层板和加强型单层板参考对比
  - 框架案例：KATO vs GCMMA（两种优化器对比）
- **主要结果**：
  - 所有8种KATO布局的AIP均低于家族匹配的参考设计；双板Unrestricted: 131.14 dB vs X-core: 136.30 dB；单板Unrestricted: 135.28 dB vs 加强型: 147.99 dB
  - 3D extrusion后，双板KATO在100 Hz处AIP低4.6-5.7 dB，单板低9.0-9.5 dB
  - KATO在18 Hz框架案例中AIP与GCMMA匹配（<0.5 dB差异），但二进制静力柔顺性低22-36倍（3.6 N m vs 80-135 N m），连通分量$N_c=1$
  - 300 Hz近共振测试：KATO二进制AIP 139.76 dB vs GCMMA 139.30 dB，但静力柔顺性低59.4倍，且1-500 Hz最大AIP低2.73 dB
  - 计算效率：缓存PARDISO比SuperLU快26.8倍；ROM比全阶快990-2125倍；KATO在应力感知变体中比GCMMA快6.4-10.4倍

## 相关工作脉络
- **SIMP密度法** [7-9]：连续密度变量+惩罚中间密度驱动二值化，本文沿用此基础参数化但用神经网络替代直接密度优化
- **动态柔顺性优化** [28-30]：早期强制振动目标，存在共振/反共振符号歧义问题；本文作者Silva等 [37] 已指出其病理特性并引入AIP
- **AIP拓扑优化** [37, 38]：Silva等人建立AIP作为非负强制振动目标的基础，本文将其扩展到船舶结构大型工程问题
- **神经重参数化TO** [49-56]：Hoyer等引入deep image prior思路，本文基于先前KATO工作 [57, 60] 扩展到动态振动领域
- **船舶结构TO应用** [39-44]：现有研究多关注静强度/重量优化，本文首次将AIP驱动动态振动优化应用于船舶结构

## 局限性与未来方向
- **材料与激励范围受限**：仅考虑各向同性钢，单频优化（100 Hz、18 Hz），可扩展至多材料和多频目标
- **结构AIP而非声辐射功率**：未直接优化辐射声功率，需耦合结构-声学模型（如BEM）才能直接最小化URN
- **制造约束不完整**：Helmholtz滤波和投影控制特征尺寸，但未考虑焊接可达性、板材轧制方向、接头细节等制造约束
- **2D优化+3D复算**：甲板拓扑在2D平面应力假设下优化，全3D拓扑优化可允许厚度方向材料变化

## 研究启发与可借鉴点
- **神经生成器作为结构先验**：cKAN生成器提供隐式空间正则化，自动产生连通载荷路径，这一思路可迁移到其他需要连通性的结构设计问题（如热传导路径、流体通道）
- **降阶模型+静态残余柔量修正**：200模态ROM配合 mode-acceleration 修正，在保持精度的同时实现3个数量级加速，适用于宽带频率响应评估场景
- **缓存稀疏因子化加速伴随分析**：复用PARDISO符号因子化和稀疏结构，比标准SciPy SuperLU快26.8倍，对需要大量灵敏度计算的迭代优化具有通用价值
- **混合AIP/静力目标策略**：仅最小化AIP可能产生弱连接结构，引入静力柔顺性正则化（$w_{\text{AIP}}=0.9$）可有效平衡动态性能与结构完整性

## 关键术语表
- **Active Input Power (AIP)**：谐波力注入结构速度场的时均实功率，非负且避免动态柔顺性的符号歧义，作为强制振动优化的目标函数
- **Neural Reparameterization**：用神经网络参数化密度场，架构本身提供隐式空间先验，无需训练数据集即可生成连通拓扑
- **Convolutional Kolmogorov-Arnold Network (cKAN)**：具有可学习B样条激活的卷积网络生成器，用于拓扑优化中的密度场参数化
- **Helmholtz PDE Filter**：基于偏微分方程的密度滤波方法，提供与网格无关的特征尺寸控制和平滑正则化
- **Heaviside Projection**：平滑近似Heaviside函数将连续密度推向0-1二值分布，阈值$\mu$通过二分法保体积调整
- **Dynamic Compliance**：外力在位移场上做功的度量，在共振/反共振附近有符号变化和奇异性，不适合直接作为振动优化目标
- **Reduced-Order Model (ROM)**：基于截断模态基的降阶模型，配合静态残余柔量修正可在保持精度的同时大幅加速频率响应分析

## 可复现要素
- **数据集**：作者声明数据与代码在论文被期刊接受后将在 https://github.com/ysyysy115/KATO_aip 开源，目前可从通讯作者合理请求获取
- **代码**：基于PyTorch实现，使用PARDISO稀疏求解器，自定义torch.autograd.Function处理伴随分析
- **关键超参**：SIMP惩罚指数$p$逐步提升；$\beta_H$投影锐度递增；学习率衰减；$w_{\text{AIP}}=0.9$（甲板）/0.8（框架）；应力权重$w_\sigma=0.05-0.25$；p-范数$p=8$；模态基数$m=200$；阻尼损失因子$\eta_s=0.02$
