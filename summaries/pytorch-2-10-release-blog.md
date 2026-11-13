# PyTorch 2.10 更新概览

依据：原文

来源：https://pytorch.org/blog/pytorch-2-10-release-blog/

PyTorch 2.10 于 2026 年 1 月 21 日发布，重点改进编译性能和数值调试。该版本自 2.9 以来汇集了 536 名贡献者的 4160 次提交。文中介绍了新增能力及适用条件，但没有给出性能提升的量化测试结果。

### 背景与兼容性

PyTorch 2.x 持续优化编译器栈；随着分布式强化学习用于更多模型的后训练，运行结果的确定性和数值差异排查也变得更重要。2.10 的 torch.compile() 支持 Python 3.14，并实验性支持自由线程构建版 Python 3.14t。

### 计算性能

TorchInductor 的 combo-kernels 横向融合把没有数据依赖的独立操作合并为一个 GPU kernel，以减少 kernel 启动开销。新的 torch.nn.attention.varlen_attn() 面向变长及打包序列，支持前向、反向计算和 torch.compile()；目前由 FA2 支持，计划加入 cuDNN 和 FA4 支持。它要求 NVIDIA CUDA A100 或更新的 GPU，数据类型限 BF16 和 FP16。线性代数模块还可调用 cuSOLVER 的 DnXgeev，在 NVIDIA GPU 上执行一般特征值分解。

### Intel GPU 支持

2.10 将 Windows 和 Linux 上的 Intel GPU 支持扩展到搭载 Intel Arc Graphics 的 Core Ultra Series 3。更新还包括常用基础算子及按张量、按通道缩放的 FP8 矩阵乘法，支持复数类型的 Aten MatMul，并允许通过 PyTorch C++ Extension API 的 SYCL 支持在 Windows 上实现自定义算子；Intel GPU 的单元测试覆盖范围也有所扩大。

### 确定性与数值调试

torch.compile() 现在遵循确定性算法设置。用户可调用 torch.use_deterministic_algorithms(True)，使重复编译调用执行相同操作。DebugMode 可记录派发的操作和 TorchInductor 编译的 Triton kernel，并为张量输入、输出附上确定性哈希。用同一输入运行两个模型版本时，哈希首次出现差异的位置有助于定位行为不同的操作；开发者也可注册自定义派发钩子。这些性能和调试能力在发布说明中均标为 API 尚不稳定。

### 迁移与发布安排

TorchScript 在 2.10 中已弃用，项目建议改用 torch.export。tlparse 和 TORCH_TRACE 可帮助整理编译器问题的排查信息；报告难以独立复现的问题时，项目建议附上 tlparse 日志产物。项目预计在 2026 年将发布频率从每季度一次提高到每两个月一次；1 月 28 日另有介绍版本变化的线上问答。
