---
title: "SpliTEE-Improving-LLM-Inference-on-Trusted-Hardware-with-Dif"
source: https://arxiv.org/pdf/2609.15039v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-01 17:05:09"
field: "隐私保护的机器学习系统"
keywords: ["差分隐私", "可信执行环境", "LLM推理", "Split-Inference", "隐私保护", "浮点误差分析"]
innovations: ["浮点域差分隐私掩码替代密码学加密，消除量化精度损失", "首个推导LLM各组件全局敏感度上界的理论分析（Theorem 1-3）", "Split-inference架构：线性操作外包GPU+非线性操作保留TEE，配合浮点误差上界分析指导ε选择"]
benchmarks: ["Dolly", "Exact-match/Token match/BLEU/ROUGE-L/Semantic similarity"]
---

# 论文速读：SpliTEE: Improving LLM Inference on Trusted Hardware with Differential Privacy

## 一句话总结
SpliTEE 将 LLM 推理拆分到可信执行环境（Intel TDX CPU）与不可信 GPU 之间，通过对传出的中间张量注入**差分隐私噪声**（而非密码学加密）来保护 prompt 隐私，避免了现有方案的量化精度损失，并在语义相似度上几乎无损（ε≥1 时语义相似度 ≥99.67%）。

## 研究问题与动机
1. **全 TEE 推理性能严重不足**：Intel TDX 上端到端 CPU 推理（800 tokens）需 53–83 秒，而 GPU 仅需 3–5 秒，性能差距达 10–20×。
2. **密码学方案的量化精度损失**：Slalom [34] 等方案需将输入量化到有限域做运算，导致 CNN 精度降 0.5%、LLM 精度降 1.9%。
3. **无保护的 prompt 极易被重建**：若中间输入不加任何掩码，prompt 重建攻击准确率高达 ~80%。
4. **现有方案缺乏隐私-效用可调节性**：Slalom 没有可调隐私预算参数，精度损失为固定副作用。

## 核心贡献（创新点）
1. **浮点域差分隐私掩码方案**：对传出中间表示注入 Gaussian DP 噪声，替代密码学加密，避免量化损失，保留浮点精度。
2. **首个 LLM 各组件全局敏感度上界推导**：给出 Token Embedding、LayerNorm、Attention、Activation（SiLU/FFN）的全局敏感度理论上界（Theorem 1–3），为 DP 噪声注入提供理论依据。
3. **Split-inference 架构设计（GPU 外包 + TEE 内非线性）**：线性投影（Q/K/V/O/Gate/Up/LM-head）外包至 GPU，非线性操作（LayerNorm、softmax、SiLU）留在 TEE 内，最大化 GPU 加速收益。
4. **浮点误差上界理论分析**：推导去噪后浮点误差上界（Theorem 4），验证 bound 与实际误差比值约 4.27–4.35×，可有效指导 ε 选择；证明了 Down 投影应保留在 TEE 内的原因（高维导致敏感度极大）。

## 方法详解
1. **架构划分**：TEE（Intel TDX，Xeon Silver 4514Y）执行 LayerNorm、残差连接、attention 计算、FFN 非线性、Down projection；GPU（RTX PRO 4500 Blackwell）执行 Q/K/V projection、O projection、Gate/Up projection、LM head。
2. **全局敏感度上界**（对输出为 x 的组件 f）：
   - **LN**：$\Delta_{\text{LN}} \leq 2\sqrt{d} \cdot \|\boldsymbol{\alpha}\|_4$
   - **Attention**：$\Delta_{\text{att}} \leq 2\sqrt{d} \cdot \|W_{\text{val}}\|_2 \cdot \|\boldsymbol{\alpha}\|_4$
   - **Activation（SiLU/FFN）**：$\Delta_{\text{act}} \leq 2\sqrt{d} \cdot (\sqrt{d}\|\boldsymbol{\alpha}\|_4 + \|\boldsymbol{\beta}\|_2)^2 \cdot \|W_{\text{up}}\|_2 \cdot \|W_{\text{gate}}\|_2$
3. **噪声注入**：对输出 x 的组件 f，采样 $\boldsymbol{\eta} \sim \mathcal{N}(\mathbf{0}, (\Delta_f/\epsilon)^2 I_d)$，外包 x + η 至 GPU。
4. **误差补偿**：TEE 侧预计算 mask 和 noise correction（noise bank），通过共享内存（IVSHMEM）传递 masked input，TCP socket 传递元信息（投影名、transformer layer index、shape、dtype）。
5. **浮点误差上界**（Theorem 4）：若非溢出，$|\text{fl}(\langle\mathbf{w},\mathbf{x}\rangle) - \text{fl}(\text{fl}(\langle\mathbf{w},\mathbf{x}+\boldsymbol{\eta}\rangle) - \text{fl}(\langle\mathbf{w},\boldsymbol{\eta}\rangle))| \leq \gamma_d \langle|\mathbf{w}|,|\mathbf{x}|+|\mathbf{x}+\boldsymbol{\eta}|+|\boldsymbol{\eta}|\rangle + u\langle|\mathbf{w}|,|\mathbf{x}|\rangle + O(du^2)$，适用于 binary32/binary16/bfloat16 任意格式。

## 实验与结果
- **评测模型**：Llama-3.2-3B-Instruct（hidden size d=3072，vocab=128,256，d_key=d_qry=d_val=500），兼容 Qwen/Gemma/Mistral 等。
- **设置**：100 prompts × 200 tokens，greedy decoding；169 个算子实例（6 投影 × 28 层 + LM-head），每实例 1,000 次试验。
- **输出质量 vs ε**：
  | ε | Exact-match | Token match | BLEU | ROUGE-L | Semantic sim |
  |---|---|---|---|---|---|
  | 0.5 | 67.00% | 85.43% | 91.59% | 93.26% | 99.36% |
  | 1 | 84.00% | 91.49% | 94.44% | 95.38% | 99.67% |
  | 5 | 98.00% | 99.41% | 99.57% | 99.59% | 99.96% |
  | 10/15 | 100% | 100% | 100% | 100% | 100% |
- **对比基线**：EPN（平均 ε=0.0773）语义相似度仅 95.56%；Slalom 仅 93.11%（量化副产品）。
- **浮点误差分析**：bound/error 均值比 ≈ 4.27–4.35×，bound 违反率 ~9.3–10.0%。
- **误差 vs 质量**：平均绝对误差 10⁻³ 时语义相似度 94.48%，10⁻² 时降至 82.99%。
- **性能**：全 TEE 推理比 SpliTEE 慢约 2×；比 Slalom 快 5–15 秒且精度更高。

## 相关工作脉络
1. **Slalom [34]**：流密码+一次性掩码，需量化到有限域——SpliTEE 与之本质区别是用浮点 DP 替代有限域加密，消除量化损失。
2. **SOTER [31]**：保护模型参数 IP（非 prompt 隐私）——目标不同，SpliTEE 聚焦 prompt 隐私。
3. **Twin-Shield [39]**：扩展至 attention+softmax，但使用标量掩码+置换——仍有量化精度损失，且置换方案易受代数攻击；SpliTEE 全程浮点运算无此问题。
4. **TEESlice [41]**：partition-before-training 保护模型权重——非 prompt 隐私，且需额外训练阶段。
5. **Song & Raghunathan [32]**：证明 embedding 可被反演恢复原文——为 SpliTEE 动机提供理论支撑。
6. **Dong et al. [11]**：直接在隐藏状态加 Laplace 噪声——噪声用于推理影响 utility；SpliTEE 通过全局敏感度分析和误差补偿将影响降至最低。

## 局限性与未来方向
1. **ε 需人工调参**：当前方案中 ε 为固定值，未实现自适应隐私预算分配。
2. **隐私预算不随 n 缩放**：理论上 n-token prompt 复合灵敏度为 $\sqrt{n}\epsilon$，实际采用恒定 ε 以控制噪声，安全性依赖 prompt reconstruction attack 实验观察。
3. **Down 投影无法外包**：因 FFN 维度（d_ff=8192）远大于 hidden dim（3072），导致敏感度上界极大，目前只能保留在 TEE 内。
4. **架构支持有限**：目前主要针对标准 Transformer 架构（Llama/Qwen/Gemma/Mistral），对 MoE 等新型架构未验证。
5. **浮点误差上界保守**：bound/error 均值比约 4.3×，bound 违反率 ~10%，可能在某些极端场景下不够紧。

## 研究启发与可借鉴点
1. **全局敏感度分析范式可迁移**：Theorem 1–3 的敏感度上界推导方法可用于其他 split-inference 或联邦学习中的组件外包决策。
2. **浮点误差上界指导 ε 选择**：Theorem 4 的误差分析方法为"差分隐私 + 数值计算"交叉领域提供了可复用的分析工具。
3. **Noise bank 预计算策略**：离线预计算 masks 和 corrections 的思路可应用于其他需要低延迟隐私保护的推理系统。
4. **Down 投影不外包的决策逻辑**：从高维度→高敏感度→大噪声→精度劣化的因果链分析，可作为其他组件外包可行性评估的参考框架。
5. **多维度评估体系**：同时使用 exact-match/token match/BLEU/ROUGE-L/semantic sim 进行精度评估，且将语义相似度设为关键指标，值得借鉴。

## 关键术语表
- **SpliTEE**：本文提出的 LLM 推理拆分系统，将线性操作外包至不可信 GPU，非线性操作保留在可信 TEE 内。
- **全局敏感度（Global Sensitivity）**：函数输出在不同相邻输入间的最大变化量，用于确定差分隐私所需噪声量级。
- **Gaussian DP**：高斯差分隐私，通过添加高斯噪声满足隐私预算，适用于连续值（浮点数）数据。
- **Noise Bank**：离线预计算的噪声掩码和校正值的集合，按 tensor shape 索引，支持一次使用后丢弃。
- **Prompt Reconstruction Attack**：从中间表示反推原始 prompt 的攻击方式，本文数据显示无保护时成功率高达 80%。
- **Split-Inference**：将模型推理拆分到不同信任等级的计算单元之间的推理架构。
- **IVSHMEM**：跨 VM 共享内存通信机制，用于 TEE 与 GPU 之间高效传输 masked tensor。
- **ε（Epsilon）**：差分隐私的隐私预算参数，值越小隐私保护越强但噪声越大。

## 可复现要素
- **数据集**：Dolly 数据集（100 条 prompt，100–250 tokens）；论文未明确说明是否公开，需自行下载。
- **代码/权重**：论文未明确声明开源状态。
- **关键超参**：ε 取值（0.5/1/5/10/15）、hidden size d=3072、d_key=d_qry=d_val=500、vocab size=128,256、Layer 数 28、generation cap=200 tokens。
- **硬件配置**：TEE — Intel Xeon Silver 4514Y（16核，~117GB 内存，Intel TDX）；GPU — NVIDIA RTX PRO 4500 Blackwell（~32GB 显存）。
