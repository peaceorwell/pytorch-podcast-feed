# Monarch：用 Python 管理分布式训练集群

依据：原文

来源：https://pytorch.org/blog/monarch-an-api-to-your-supercomputer/

Monarch 是 PyTorch 的分布式编程框架，旨在缩短大规模训练，尤其是分布式强化学习的开发和调试周期。文章介绍了它自 2025 年 10 月发布以来的功能进展，以及面向智能体开发的使用方式。

### 问题与设计

大型集群上的训练任务涉及资源分配、代码同步、运行状态排查和反复重启，修改一次程序往往需要较长等待。Monarch 用主机、进程和 actor 的统一模型暴露集群，让开发者在一个 Python 程序中定义训练系统，并通过少量基础接口构建容错、编排和工具集成能力。

### 开发与调试机制

Monarch 的只读 RDMA 文件系统把客户端文件分发到任务中的各台主机，用于同步代码、依赖和数据。分布式 SQL 遥测通过 DataFusion 查询各节点写入的实时状态、日志和 pyspy 跟踪；Jobs API 则允许一次申请主机后在其上运行多个任务，减少重复分配资源的等待。这些接口也让运行在开发机上的智能体能够管理任务并排查故障。

### 新增的平台能力

Kubernetes 支持包括 MonarchMesh 自定义资源、参考 operator、按需创建 Pod，以及可配合 Kueue 使用的标签传播；集群外客户端连接网关计划在 0.5 版提供。网络方面，统一 RDMA API 覆盖 InfiniBand、AWS EFA 和使用 Mellanox 网卡的 ROCm 环境；无法使用 RDMA 时可退回 actor 消息传递。项目还新增管理 API、终端界面、OpenTelemetry 集成和测试版单任务网页仪表盘。

### 结果与适用条件

作者报告，AWS EFA 测试传输了 14.5 GB 数据，耗时 7.6 秒，达到约 16 Gbps，速度为 TCP 的 10 倍；该支持目前在 PyPI 夜间构建版中。文章还称 pip wheel 体积缩小约 100 倍、冷启动快 8 倍，但未给出测试配置。自 v0.2 起，torchmonarch 不再将 torch 列为 pip 依赖；v0.4 增加 Linux ARM64 构建。

### 集成与局限

SkyPilot 集成可负责资源配置、网络和协同调度。与 VeRL 合作开发的后端在其混合引擎模式下运行 PPO 和 GRPO 训练循环，验证结果与原实现数值一致，且未观察到性能退化；文章未提供具体性能数据。作者也指出，VeRL 的 Ray API 调用分布在更广的代码库中，因此更换后端比单一控制器接口所显示的更复杂。AMD 合作方还在采用 SLURM 的 MI300、MI325 和 MI355 集群上验证了 Monarch。
