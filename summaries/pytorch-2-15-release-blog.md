# PyTorch 2.15：分布式训练与多平台推理更新

依据：原文

来源：https://pytorch.org/blog/pytorch-2-15-release-blog/

PyTorch 2.15 汇集了自 2.14 以来 583 名贡献者的 3,117 次提交，重点更新分布式通信、训练内存控制、低精度推理和各硬件平台支持。

### 分布式训练

重写的 NCCL 后端现为所有 nccl 进程组的默认实现。通信器随进程组创建；在 NCCL 2.29 及以上版本，可选用更快的收缩和扩展重配置路径。集合通信卡住时，飞行记录器会保存各 rank 的现场信息。用户可通过 TORCH_DIST_USE_NCCL2=0 或 nccl-legacy 回退。新增的单边传输 API 以 NIXL 为首个后端，无需进程组即可连接对端，其 CUDA stream 传输可纳入 CUDA graph。

### 内存与通信控制

FSDP2 允许 MoE 路由门控等敏感参数保留 fp32，其余参数使用 bf16，并支持按参数设置梯度精度及将完整梯度累积步骤捕获到单个 CUDA graph。流水线并行改为按需分配接收缓冲区、发送完成后释放前向输出，还可限制未完成的发送数量，并让梯度归约与后续计算重叠。这些改动旨在降低内存占用和暴露的通信开销。

### 推理性能

在 NVIDIA Blackwell GPU 上，NVGEMM 将 ModelOpt NVFP4 线性层的标量反量化因子并入 GEMM，减少解码时每层的一次独立 kernel 启动，并改进与后续 Triton kernel 的重叠。在 Apple Silicon 上，MPS 增加 FP8、_scaled_mm、grouped_mm 的解码快路径及公开的 torch.mps.Stream API。文章报告，M3 Ultra 上的 MoE 解码吞吐量提高至原来的 1.32 至 1.34 倍；F.linear 的解码路径几何平均提速 1.23 倍。MPS 还加快了 softmax、上采样反向和 norm 归约，并优化了 QR 分解与三角方程求解。

### 平台与开发工具

默认安装包使用 CUDA 13.2，另提供 CUDA 13.4 安装包，后者实验性支持 NVIDIA Rubin。ROCm 10.0 安装包加入发行阵容；所有 ROCm 安装包现基于 AMD TheRock 构建，通过 pip 依赖安装匹配的 rocm 包，而不再内置 ROCm 库。AMD gfx950 可选用 FlyDSL 0.3.x，gfx1250（MI450）获得 Tensor Data Mover 代码生成支持。Intel XPU 增加更简洁的 SYCL 自由函数 kernel、跨进程事件共享和更细的多 GPU 性能分析。Cuspy 可把 CUDA graph 重放的 kernel 关联到捕获代码；C++ 稳定 ABI 和仅头文件接口也有所扩展。

### 兼容性与适用范围

2.15 支持 Python 3.11 至 3.15，torch.compile 支持 Python 3.15；Python 3.10 停止支持。MPS 加速要求 macOS 15 或更新版本。FSDP2 的分片与未分片梯度 dtype 行为发生变化，迁移时需检查精度设置。文中多项新 API 标为不稳定；所列性能数字来自特定算子、形状或硬件条件，不能直接推及所有工作负载。
