# PyTorch 2.12 发布要点

依据：原文

来源：https://pytorch.org/blog/pytorch-2-12-release-blog/

PyTorch 2.12 汇集了自 2.11 以来 457 名贡献者提交的 2,926 次修改，重点改进 CUDA 计算性能、跨设备图捕获、量化模型导出和分布式训练工具。部分功能有明确的硬件或版本条件，未来的分布式接口变更也尚未在本版默认启用。

### 计算性能

CUDA 上的批量对称矩阵和厄米矩阵特征分解改用 cuSolver 的 syevj_batched 内核。官方报告，相比上一版，torch.linalg.eigh 在适用负载中最高快 100 倍，原先耗时数分钟的批量求解可缩短到数秒。Adagrad 新增 fused=True，用单个 CUDA 内核完成优化器步骤，以减少内核启动和内存访问开销；原文未给出其加速幅度。

### 图捕获与模型导出

新的 torch.accelerator.Graph 为 CUDA、XPU 和外部后端提供统一的图捕获与重放接口，后端可注册自己的实现；流接口也新增 is_capturing()。torch.export.save 和 load 现在能处理 MXFP4、MXFP6、MXFP8 所用的 float8_e8m0fnu 块缩放指数，使这类量化模型能够完整导出和加载。借助 CUDA 12.4 的条件 IF 节点，torch.cond 分支可在 CUDA Graph 内由 GPU 判断并重放；目前适用于 eager 和 cudagraphs 后端，Inductor 支持仍待后续版本。

### 数值与分布式调试

Inductor 对 addcdiv 使用融合乘加指令，官方称这使 CUDA 和 XPU 上编译执行的结果与 eager CUDA 达到逐位一致，便于核对优化器训练过程。自定义算子和 c10d 函数式集合通信现在可直接接收 ProcessGroup 对象，也兼容字符串组名。Profiler Events API 增加事件和关联信息；同一进程组内参与同一次 NCCL 集合通信的各 rank 可用 seq_num 对齐轨迹。FlightRecorder 扩展到 ncclx、gloo 和更多 torchcomms 操作，并修复并发访问可能造成的死循环。

### 设备平台更新

CUDA Graph 可为捕获的内核加入注释，经过配套脚本处理后在性能轨迹中显示；实验性的 CUDA Green Context 工作队列上限可控制并发提交。ROCm 7.02 及以上支持可扩展内存段，AMD GPU 新增 rocSHMEM 对称内存通信。ROCm 7.12 及以上默认启用 hipSPARSELt；MI350X 支持 FP8 输入的 2:4 半结构化稀疏计算，输出为 FP32。官方报告，MI350X 上采用两阶段流水线的 FlexAttention 在所测注意力模式和形状下快 5% 至 26%。Apple Silicon 安装包提供在 macOS 26 上预编译的 Metal-4 着色器，以减少首次运行的编译开销。

### 迁移与版本边界

torchcomms 在 2.12 中已可选择启用，但官方计划在 2.13 或之后才默认采用。届时预计要求初始化 ProcessGroup 时指定设备并完成通信器初始化，同组同流的点对点操作也不再保证并发；官方还计划将 torchcomms 设为 PyTorch Distributed 的必需依赖，并弃用现有的 c10d::Backends。这些计划中的变更尚未在 2.12 生效，部分模型未来可能需要调整。TorchScript 已在 2.10 弃用，官方建议分别用 torch.export 和 ExecuTorch 替代相关导出接口与嵌入式运行时。2.12 不再发布标准 CUDA 12.8 安装包；默认版本为 CUDA 13.0，CUDA 13.2 为实验性版本。较旧的 Pascal、Volta 架构可用仍受支持的 CUDA 12.6 安装包；使用 CUDA 13.0 及以上版本的新 GPU 需要相应升级 NVIDIA 驱动。
