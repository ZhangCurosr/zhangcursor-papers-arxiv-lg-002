---
title: "FROM-EXPERIENCE-TO-EXPERTISE-ADOPTION-AWARE-MEMORY-LEARNING"
source: https://arxiv.org/pdf/2609.35568v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:03:29"
field: "AI for Compiler/KHS"
keywords: ["NPU kernel synthesis", "LLM agent", "memory learning", "credit assignment", "hardware-aware programming"]
innovations: ["Adoption-Traced Utility estimation for precise credit assignment", "Utility-Gated Consolidation with bounded resident context", "Cross-task hardware knowledge accumulation without backbone fine-tuning"]
benchmarks: ["NPUKernelBench", "xCCL transfer evaluation"]
---

# 论文速读：FROM-EXPERIENCE-TO-EXPERTISE-ADOPTION-AWARE-MEMORY-LEARNING

## 一句话总结
提出SAGE，一种面向低资源NPU kernel合成的持久化自增强agent，通过显式追踪经验采纳（ATU）和基于效用门控的热点记忆整合（UGC），使冻结参数的LLM能在跨任务中持续积累和复用硬件特定知识。

## 研究问题与动机
- **领域鸿沟**：LLM在CUDA/Triton kernel合成上表现良好，但Ascend NPU等DSA缺乏公开代码和文档支持，GPT-5.2从CUDA 92%正确率骤降至Ascend C仅14%。
- **后训练方法局限**：SFT和RL依赖稀缺的专家数据和大量训练算力，扩展性受限。
- **现有记忆学习缺陷**：Uniform credit assignment将所有检索经验一视同仁赋予相同奖励目标，未被实际采纳的经验也能获得有利估值，挤占更有潜力经验的召回优先级。
- **重复检索开销**：当仅用学习值指导检索时，跨operator泛化的高价值经验需反复检索而非保留在上下文中，增加开销且削弱跨任务指导。

## 核心贡献（创新点）
1. **提出SAGE框架**：冻结LLM背核，通过外部记忆演化实现跨任务硬件知识积累，与EvoKernel等uniform credit方法本质不同。
2. **ATU（Adoption-Traced Utility estimation）**：将显式采纳记录与终端验证器结果结合，区分被采纳与未被采纳的检索经验，解决credit分配粗糙问题。
3. **UGC（Utility-Gated Consolidation）**：利用正向效用和跨operator重复采纳证据，将可复用经验抽象为紧凑规则并入界定的resident context，避免高价值经验的重复检索开销。
4. **系统化实验验证**：在NPUKernelBench三个backbone上一致提升，GLM-5.2达95.5%执行率和86.9%超越torch_npu，sparse flash attention获43.99×加速。

## 方法详解
**整体框架**：SAGE采用冷-热分层记忆结构 $\mathcal{M} = (\mathcal{C}, \mathcal{H})$，其中$\mathcal{C}$保存源经验和item级证据，$\mathcal{H}$提供无需检索的整合规则，token预算固定为B。

**ATU机制**：
- **采纳追踪**：每个检索item $m$ 记录采纳状态 $a_{m,t} \in \{1, 0, \bot\}$ 及rationale，通过trace-completeness hook确保完整记录。
- **终端计分**：optimization episode中定义 $z = 1 - T_{best}/T_{first}$（首次正确后的相对延迟改善），失败惩罚 $-\delta$。
- **信用分配**：采纳item共享固定episode级credit budget $\xi_m = z/|\mathcal{A}|$，未采纳item获得固定负信用 $-\beta$。
- **在线效用更新**：bandit-style规则 $u_m \leftarrow (1-\eta_m)u_m + \eta_m\xi_m$，其中 $\eta_m = \max(1/(1+n_m^{ret}), \eta_{min})$ 衰减步长。

**UGC机制**：
- **准入门控**：item需满足 $u_m > 0$、$n_m^{ado} \geq K_{ado}$、$|\mathcal{O}_m| \geq K_{op}$ 才转为validated并抽象为规则。
- **Budgeted选择**：按单位token预期效用 $U(m) = u_m \hat{p}_m / L_m$ 贪心选择resident规则，$\hat{p}_m$ 为经验访问频率估计。
- **持续衰减**：resident item通过EMA更新使用估计 $\hat{p}_m \leftarrow (1-\mu)\hat{p}_m + \mu\mathbf{1}_{misused}$，持久不使用导致优先级衰减并可被驱逐。

## 实验与结果
**数据集**：NPUKernelBench（88-operator子集，含20 L1 + 50 L2 + 18 L3），另有16个更复杂的xCCL算子用于迁移评估。

**基线**：Refinement（无跨任务经验）、Static RAG（禁用记忆更新）、Value Memory（EvoKernel uniform credit）。

**主结果（GLM-5.2）**：
- Execution Rate: SAGE 95.5% vs Value Memory 84.1%（+11.4pp）
- Fast_1.0: SAGE 86.9% vs Value Memory 75.7%（+11.2pp）
- S_self: SAGE 4.47× vs Value Memory 3.36×
- Level-3 ER: SAGE 88.9% vs Value Memory 61.1%（+27.8pp）

**迁移结果（GLM-5.3 on xCCL）**：
- ER从68.8%提升至81.3%
- sparse flash attention速度提升达43.99×（vs torch_npu）

**跨backbone记忆迁移**：GLM-5.2演化的记忆对DeepSeek-V4-Flash和Qwen3-Coder-Next均有提升，ER分别+7.9pp和+9.1pp，证明知识具有部分可迁移性。

## 相关工作脉络
- **EvoKernel [10]**：值驱动跨任务记忆方法，但采用uniform credit assignment，无法区分采纳与未采纳经验，本文ATU通过采纳追踪解决此问题。
- **Memento [16] / MemRL [17] / MemQ [18]**：面向agent的记忆学习方法，但缺乏item级采纳记录，无法精确归因到具体经验。
- **AscendKernelGen [14] / NKI-Agent [23]**：依赖SFT/RL的后训练方法，需要专家数据和大量训练算力，扩展性受限。
- **AscendOptimizer [15] / AgenticCANN [4] / Hawk [25]**：frozen backbone + 检索知识辅助，但缺少显式采纳记录和跨任务知识整合机制。
- **MemGPT [31]**：fixed-context LLM agent的层次记忆管理启发，本文将其应用于NPU kernel合成场景。

## 局限性与未来方向
- **长horizon合成**：30轮交互未覆盖fused kernel和megakernel的长期合成，需要跨调度、内存布局和同步决策的多样性搜索。
- **多用户协作**：实践中可能涉及隔离sandbox或共享记忆更新，存在专业化、溯源、迁移和冲突解决的挑战。
- **跨架构泛化**：目前仅验证Ascend NPU，未来需探索其他新兴硬件架构和DSL。
- **Backbone依赖**：不同模型利用积累知识的能力差异显著（GLM-5.2 ER 95.5% vs Qwen3 31.8%），记忆质量受生成能力制约。

## 研究启发与可借鉴点
1. **采纳追踪设计**：显式记录"检索但未采纳"的经验并施加负信用，避免uniform credit导致的噪声积累，可迁移至其他代码生成agent的credit分配。
2. **冷热记忆分层**：固定预算下的hot memory整合机制，平衡检索灵活性与上下文效率，适用于任何需要跨任务知识积累的agent系统。
3. **优先级衰减策略**：EMA驱动的resident item使用频率衰减，防止过时知识永久占据有限上下文，机制简洁通用。
4. **评估协议设计**：4项anti-hacking约束（禁止高级API替代、专家实现泄露、语义捷径、无效精度权衡）确保评估真实性，值得参考。
5. **跨backbone迁移验证**：分离记忆演化与模型能力，证明知识本身的部分可迁移性，为模型无关的知识积累提供论证范式。

## 关键术语表
**SAGE**：Persistence self-improving agent for NPU kernel synthesis，通过ATU和UGC实现外部记忆演化。
**ATU (Adoption-Traced Utility estimation)**：结合采纳记录与终端验证结果的item级信用分配机制。
**UGC (Utility-Gated Consolidation)**：基于效用和跨operator采纳证据将经验整合为resident规则的机制。
**NPUKernelBench**：涵盖多类别、多级复杂度NPU kernel合成的benchmark。
**Cold-Hot Memory**：冷存储保存源经验+证据，热存储保存整合规则的分层记忆结构。
**Fast_1.0**：生成kernel超越torch_npu参考实现的已解算算子比例。
**S_self**：迭代优化带来的中位数加速比（首次正确kernel vs最佳kernel）。

## 可复现要素
- **数据集**：NPUKernelBench，论文提供了88-operator子集列表（Appendix A.1）
- **代码**：基于OpenCode [32]框架，具体SAGE实现未公开（论文提及"open source coding agent"但未提供SAGE源码链接）
- **权重**：使用Qwen3-Coder-Next、DeepSeek-V4-Flash、GLM-5.2/5.3，本地vllm-ascend部署
- **关键超参**：δ=β=0.2, η_min=0.05, γ=0.3, K_ado=K_op=3, B=10K tokens, μ=0.1
- **硬件**：Ascend 910C NPU（8 cards/server），CANN 8.5.1
- **交互预算**：每算子最多T=30轮refinement
