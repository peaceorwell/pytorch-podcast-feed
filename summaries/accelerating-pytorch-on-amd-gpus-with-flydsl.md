# FlyDSL 加速 AMD GPU 上的 PyTorch

依据：原文

来源：https://pytorch.org/blog/accelerating-pytorch-on-amd-gpus-with-flydsl/

FlyDSL 将针对 AMD GPU 的内核接入现有 PyTorch API，覆盖矩阵乘法、注意力、RMSNorm 和 TopK。当前 PyTorch 集成主要面向 MI350 系列的 gfx950 架构，收益随输入形状、精度和模型负载变化。

### 接入方式

合格的 RMSNorm 和 TopK 调用可在 eager 模式自动使用 FlyDSL。对密集、分组及 MXFP 矩阵乘法，TorchInductor 在 torch.compile 下将 FlyDSL 与 ATen、Triton 一起测速，并缓存每种负载的最快实现。F.scaled_dot_product_attention 则由 AOTriton 的调优数据库选择 FlyDSL、Triton 或 AITER ASM 内核。未覆盖的输入继续使用现有实现。

### 内核设计

FlyDSL 让开发者显式安排数据在计算块、wave 和线程间的布局，以及 LDS 缓冲、MFMA 指令和同步。密集 GEMM 用分阶段缓冲和半 tile 交错执行重叠预取与计算；MXFP GEMM 还要让操作数与对应的缩放因子同步推进。分组 GEMM 用持久化网格跨专家分配工作并跳过空组。TopK 对小 K 在寄存器内排序，对较大 K 先用基数选择缩小排序范围。注意力通过双 wave 流水线重叠 softmax 与 MFMA；head 维度为 512 时改用分片调度，以适应 160 KB 的 LDS 限制。

### 算子测试结果

在 MI355X 的稳态测试中，15 个 BF16 NT 密集 GEMM 相对逐形状较快的 ATen 或 Triton 实现，几何平均加速 1.10 倍。MXFP8 和 MXFP4 在各自 17 个 NT 形状上相对 ATen 分别加速 1.58 和 1.68 倍；24 个 BF16 分组 GEMM 相对 Triton 加速 1.20 倍。22 个 RMSNorm 测例相对 ATen 为 1.17 至 3.66 倍，最高值来自非对齐维度。10 个小 K TopK 测例在关闭确定性算法时几何平均加速 4.82 倍。AOTriton 0.14.2b 相对 0.13b 的 gfx950 注意力吞吐比，前向为 0.92 至 2.34 倍，反向为 0.93 至 2.47 倍；版本间还有其他内核和调优数据库变化，不能把差异单独归因于 FlyDSL。

### 整请求表现

作者在单张 MI355X 上测试 vLLM，使用三个模型、256 token 输入、固定 512 token 输出及 8 至 256 的并发数。BF16 相对启用 ATen 和 Triton 的基线最高加速 1.13 倍；MXFP8 和 MXFP4 相对 ATen 基线最高分别为 2.04 和 3.01 倍，两个低精度配置各自的 18 组模型与并发组合均有提升。低精度结果每组只测一次，小幅差异缺少方差验证。

### 适用边界

算子测试不计编译、首次调优或 MXFP 输入量化。MXFP 要求按 K 每 32 个值提供缩放因子、K 为 128 的倍数，并采用 A 行优先、B 列优先布局。RMSNorm 目前只加速前向；注意力支持 FP16、BF16，head 维度为 32 至 512，FlexAttention 尚未接入。TopK 对相等值或 NaN 的索引选择可能与 ATen 不同。文中的部署条件是已集成相关功能的 ROCm PyTorch nightly 与可选 FlyDSL 0.3.x 运行时；SDPA 只需链接 AOTriton 0.14.2b 或更新版本。MI450 支持、FlexAttention 和更广的训练支持仍属后续计划。
