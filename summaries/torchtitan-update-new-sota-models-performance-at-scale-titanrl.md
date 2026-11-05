# TorchTitan训练框架进展

依据：原文

来源：https://pytorch.org/blog/torchtitan-update-new-sota-models-performance-at-scale-titanrl/

TorchTitan是基于原生PyTorch的生成式模型预训练和后训练平台。这次更新扩展了模型、强化学习和分布式训练能力，并报告了特定硬件与配置下的性能测试结果。

### 模型支持与配置

新增Muse Spark Open、Muse Glimmer、Qwen3-VL、Qwen3.5、Kimi K2.7、Kimi K3和DeepSeek-V4的完整训练集成，覆盖五维并行。类型化Python配置、模型配置变换和组件覆盖机制，让用户调整并行策略或替换内核，而无须维护模型分支。可选用Attention Gym及Helion内核，并保持与torch.compile兼容。

### 强化学习与正确性

TitanRL让同一模型实现贯穿预训练、后训练和vLLM生成，异步执行采样，并分别扩展训练与生成。团队称，MoE和稠密模型已在长上下文、代码执行等任务上稳定训练数百步；文中列出Qwen3.5-35B-A3B在40张GB300、64K上下文，以及20张GB300、128K上下文的测试配置。在受支持的同策略配置中，批次不变模式使训练器与生成器的对数概率逐位一致，便于排查数值差异。

### 性能机制与实测

local_compile可选择性编译关键区域，torch_remat细化激活重计算；面向NVIDIA SM100及更新GPU的Dist-MoE融合通信与计算，流水线并行组件支持CUDA图。内部GB300集群的MLPerf测试使用MXFP8训练DeepSeek V3 671B加MTP1，仅扩展流水线、数据和专家并行轴：PP2配置在256张GPU上达到每卡每秒5324.21个token，在512张GPU上达到5262.68个token，总吞吐量为每秒2694493个token。Muse Spark Open 1.2在256张GPU、96K上下文下，强制均衡路由与学习路由分别达到每卡每秒1839.52和1500.61个token；路由方式会影响结果。

### 分片与图训练

FlexShard按参数桶指定分片位置。相较等效的纯FSDP实现，分布式Muon在256张H100上训练DeepSeek V3时，优化器耗时最多降低79%；块对齐的FP8参数聚合在同规模H100上训练Llama3 405B时，单步训练时间最多缩短20%。GraphTrainer提供显式训练图，团队称其与即时执行路径的数值结果逐位一致，并可用于跨模块融合和图级调试。

### 通信、检查点与适用范围

PyTorch 2.14提供在TorchTitan上验证过的可选NCCL后端；通信器的提前初始化计划用于2.15。Dist-MoE通过对称内存在专家间传输token，TorchTitan的FSDP也可从对称内存池分配通信缓冲区。torch_checkpointing减少保存时的同步：在512张H100、DeepSeek V3 671B、每50步保存一次的测试中，平均训练阻塞从841.2毫秒降至34.1毫秒，平均写入时间从384.4秒降至13.1秒。上述加速均有指定模型、硬件和配置条件，不能直接推及其他负载。TorchTitan也已作为veRL、Miles等项目的训练后端。
