# PyTorch 2.11 发布概览

依据：原文

来源：https://pytorch.org/blog/pytorch-2-11-release-blog/

PyTorch 2.11 于 2026 年 3 月 23 日发布，汇集了自 2.10 以来 432 名贡献者提交的 2723 次代码变更。此次更新涉及分布式训练、注意力计算、设备算子、模型导出和执行性能。文中将主要新功能列为 API 尚不稳定的功能。

### 分布式训练与注意力

功能式集合通信现在支持自动求导，训练过程可以反向传播经过集合通信操作，使部分分布式训练方法无需编写自定义 autograd 函数。FlexAttention 在 Hopper 和 Blackwell GPU 上新增 FlashAttention-4 后端，可自动生成 CuTeDSL 的分数与掩码修改函数，并从 PyTorch 即时实例化内核。官方报告称，在计算受限负载上，它比现有 Triton 实现快 1.2 至 3.2 倍；该后端仍在开发，接口和限制可能变化。

### 设备支持与算子

Apple Silicon 的 MPS 后端扩展了算子覆盖，包括 log_normal、cauchy、geometric 分布函数、erfcx、支持全部操作模式的 grid_sampler_2d，以及支持整数和复数类型的 baddbmm、addbmm。MPS 还增加异步错误报告，可发现 GPU 索引越界，文中示例在调用 torch.mps.synchronize() 时抛出错误。ROCm 增加设备端断言，并通过共享内存缓存优化 TopK 和 radix select。CPU 则可通过 OpenBLAS 执行 FP16 GEMM；原文称这有助于加快 FP16 推理，但未提供量化结果。

### 导出与执行

LSTM、GRU 等 RNN 模块现在可在 GPU 上导出，LSTM 跟踪也支持动态形状，扩大了 torch.export 可用于生产推理的模型范围。GRU API 保持不变，新增 API 面向 LSTM。Intel GPU 的 XPUGraph 可捕获一串 XPU 操作并重复回放，以减少内核启动和 Python 运行时带来的 CPU 开销；原文未给出性能数字。

### 版本选择与迁移

从 2.11 起，x86_64 和 ARM 平台默认安装 CUDA 13 版本。需要其他构建的用户仍可从对应下载目录获取纯 CPU 或 CUDA 12.8 版本。TorchScript 已在 2.10 弃用，官方建议用 torch.export 替代 jit trace 和 script API，用 ExecuTorch 替代嵌入式运行时。2026 年的发布频率也从每季度一次改为每两个月一次。

### 发布交流

官方安排在 3 月 31 日星期二上午 10 点举行直播，介绍可微分集合通信、FlexAttention 的新后端、MPS 扩展等内容，并进行现场问答；公告未注明该时间的时区。
