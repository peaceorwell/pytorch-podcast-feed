# PyTorch 2.14 发布问答摘要

依据：视频字幕

来源：https://www.youtube.com/watch?v=pRjbQa2wHB4

这场发布直播介绍了 PyTorch 2.14 的更新，并回答社区提问。发言者称，本次发布汇集约 487 位贡献者的近 3000 次提交，重点涉及编译、分布式容错和硬件支持。直播中的性能说法并非统一的端到端测试结论。

### 编译与动态模型

Inductor 新增基于 NVIDIA CuTe DSL 和 CUTLASS 的 NVGEMM 内核，可与 Triton、ATen 等实现一同参与自动调优，支持部分后处理融合及低精度计算。它默认不启用。直播的启用说明要求在 max_autotune_gemm_backends 中加入相应后端、安装 NVIDIA CUTLASS DSL 并使用 Blackwell；发言者表示 Ampere 和 Hopper 不适用这一路径。torch.switch 扩展了 torch.cond 的多分支表达，torch.while_loop 可被 CUDA Graph 捕获，声明式 dynamic spec 供 torch.compile、torch.export 和 make_fx 共用。编译器还可选择性地将受支持的复数运算拆为实部和虚部计算。

### 分布式与容错

预览中的 NCCL 后端改写来自 torchcomms，目标是提供兼容现有后端的集合通信接口，并加入非阻塞通信器。C10D 增加进程组原地重配置、单边 RMA 窗口，以及适用于不同后端的飞行记录器。容错功能仍属实验性：失败 rank 的恢复可能需要训练流程识别可恢复错误并作调整；恢复后哪些状态仍有效，直播没有给出确定答案。

### Agentic RL 的基础设施挑战

嘉宾认为，Agentic RL 后训练的主要难点是大规模运行时的基础设施稳定性、故障恢复和扩展能力，训练任务发生故障后的重启尤其棘手。他们提到 TorchTitan、TorchForge 提供相关训练方案；新的分布式容错能力有望缓解重启问题，但直播没有给出针对 RL 工作负载的测量结果。

### 性能结果及边界

发言者称，Apple Metal Performance Primitives 的矩阵乘法在部分单算子合成测试中约快 3 倍，个别测试达到 5 倍；该段未明确比较的实现或版本，因此这些数字不能视为相对某一 PyTorch 版本或完整推理流程的收益。NVGEMM 对 Blackwell 上的低精度负载可能有益，实际收益仍取决于模型、硬件和推理框架。

### 平台与兼容性

Apple 芯片新增更多原生线性代数算子和 Metal 内核；Intel XPU 增加原生图捕获；Inductor 在 2.14 中增加面向 NVIDIA Rubin 平台的支持。ROCm 7.14 的 wheel 改由 ROCm pip SDK 构建，2.14 支持 ROCm 7.2 和 7.14，不再提供 ROCm 7.1 的 wheel。跨仓库 CI 中继让硬件合作方在自己的仓库测试 PyTorch 变更；当前 L3 阶段需给 PR 添加相应标签，测试结果才会显示在上游 PR，检查不阻止合并。torch.accelerator 旨在统一设备调用，但直播明确表示，异构硬件之间无法保证零性能损失。

### 版本迁移与使用建议

2.14 提供 CUDA 12.6、13.0 和 13.2，其中 13.0 发布到 PyPI，13.2 转为稳定版。直播时，CUDA 13.4 已在仅适用于 Linux 的 nightly 中提供；按当时计划，2.15 将停止发布 CUDA 12.6 wheel，以 13.2 为默认版本，并将支持 Rubin 架构的 CUDA 13.4 作为试验版本提供。12.6 仍拟保留源码构建测试。Maxwell、Pascal 等旧架构将失去预编译包；部分旧 GPU 不支持 CUDA 13，可能需要继续使用 PyTorch 2.14 等旧版本。2.14 的扩展默认使用 C++20。Python 3.15 的 2.14 包仅能从 download.pytorch.org 获取，普通的 pip install torch 无法安装这一组合，且该版本尚不能使用 torch.compile；2.15 计划停止支持 Python 3.10。torchvision 0.29 预计可继续配合后续 PyTorch 版本，但更换 CUDA 版本可能需要重装对应 wheel。TorchScript 已弃用，发言者建议评估 torch.export 和 AOTInductor。
