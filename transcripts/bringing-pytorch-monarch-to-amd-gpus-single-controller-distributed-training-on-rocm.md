# Bringing PyTorch Monarch to AMD GPUs: Single-Controller Distributed Training on ROCm

原文：[Bringing PyTorch Monarch to AMD GPUs: Single-Controller Distributed Training on ROCm](https://pytorch.org/blog/bringing-pytorch-monarch-to-amd-gpus-single-controller-distributed-training-on-rocm/)

## 摘要

本期介绍 PyTorch Monarch 如何把单控制器、基于 actor 的分布式训练带到 AMD Instinct GPU 和 ROCm 平台。Monarch 通过 process mesh、supervision tree 和异步执行，把故障隔离在局部，并让健康副本继续训练。结合 TorchFT 的 quorum 同步与 TorchTitan 训练引擎，系统可以通过节点间 checkpoint 传输恢复，而不必全局重启。实验显示，该方案在 SLURM 和 Kubernetes 的 MI300/MI355 集群上都能保持稳定收敛；未来还将优化 NIC 支持、重连延迟和更多训练框架。

## 对话

Ava: Imagine a language-model training job that has been running for days, then one GPU crashes—and the whole cluster has to start over.

Ava：想象一个语言模型训练任务跑了好几天，突然一块 GPU 故障，整个集群都得从头开始。

That’s the reliability problem this PyTorch Monarch work tackles on AMD GPUs.

这正是 PyTorch Monarch 在 AMD GPU 上要解决的可靠性问题。

Brian: Exactly.

Brian：没错。

The article brings Monarch to AMD Instinct GPUs with ROCm, so healthy workers can keep training while a failed worker recovers and rejoins.

文章介绍了如何通过 ROCm 将 Monarch 引入 AMD Instinct GPU，让正常的工作进程继续训练，同时故障进程恢复并重新加入。

It’s a single-controller model for large distributed jobs, with fault tolerance built into the runtime.

它用单个控制器管理大型分布式任务，并在运行时内置容错能力。

Ava: Why isn’t ordinary checkpointing enough?

Ava：常规检查点为什么不够？

Large training systems already save checkpoints regularly.

大型训练系统本来就会定期保存检查点。

Brian: Checkpointing is simple, but expensive.

Brian：保存检查点很简单，但代价很高。

Writing hundreds of gigabytes takes time and I/O bandwidth.

写入数百 GB 数据需要时间，也占用 I/O 带宽。

If a failure happens, every update since the last checkpoint is lost.

一旦发生故障，上次检查点之后的所有更新都会丢失。

The whole cluster may sit idle while a node is replaced and the job restarts.

更换节点并重启任务时，整个集群可能都得闲置。

And as the cluster grows, the chance of failing during a checkpoint interval also grows.

而且集群越大，在两次检查点之间发生故障的概率也越高。

Ava: So the goal is to recover locally instead of resetting globally.

Ava：所以目标是局部恢复，而不是让整个集群重来。

Brian: Right. Monarch lets healthy nodes continue training while failed nodes recover.

Brian：对。Monarch 让正常节点继续训练，故障节点则自行恢复。

That reduces wasted computation and keeps GPU utilization higher.

这样能减少白费的计算，并提高 GPU 利用率。

The article frames this as a shift from raw scaling to resilient scaling.

文章把这称为从单纯扩展规模转向具备韧性的扩展。

Ava: Give me the mental model for Monarch. What does a developer actually program?

Ava：怎么理解 Monarch？开发者实际要编写什么？

Brian: You write one Python program that orchestrates an entire GPU cluster.

Brian：只需编写一个 Python 程序，就能协调整个 GPU 集群。

Monarch uses an actor-based runtime, a process mesh abstraction, and asynchronous execution.

Monarch 使用基于 actor 的运行时、进程网格抽象和异步执行。

An actor has private state, so if it crashes, the failure doesn’t automatically spread to every other actor.

每个 actor 都有独立状态，因此一个 actor 崩溃不会自动波及其他所有 actor。

Ava: Actor-based runtime can sound abstract. Is each actor basically a supervised worker?

Ava：基于 actor 的运行时听起来有点抽象。每个 actor 基本上就是一个受监管的工作进程吗？

Brian: That’s a useful way to think about it. The Python API is the top layer.

Brian：可以这么理解。Python API 是最上层。

The Monarch Runtime manages actors, meshes, supervision trees, and tensor sharding.

Monarch Runtime 管理 actor、网格、监管树和张量分片。

Under that sits a Rust runtime using Tokio for performance and memory safety.

底层是使用 Tokio 的 Rust 运行时，兼顾性能和内存安全。

Infrastructure connects it to RDMA, RCCL or NCCL, SLURM, Kubernetes, and SkyPilot.

基础设施层将其连接到 RDMA、RCCL 或 NCCL，以及 SLURM、Kubernetes 和 SkyPilot。

Ava: And the supervision tree is what gives the system hierarchy?

Ava：监管树为系统提供了层级结构？

Brian: Yes. Failures are isolated and handled at the lowest practical level.

Brian：对。故障会被隔离，并尽可能在最底层处理。

A local restart can take seconds. Escalation can take minutes.

本地重启可能只需几秒；升级处理则可能需要几分钟。

Monarch also separates two concerns: the parallelism strategy inside a training replica, and the fault-tolerance mechanism across replicas.

Monarch 还把两件事分开：训练副本内部的并行策略，以及副本之间的容错机制。

That separation makes the recovery model cleaner.

这样恢复模型就更清晰了。

Ava: Moving from CUDA to ROCm sounds like the hard engineering part. What had to change?

Ava：从 CUDA 迁移到 ROCm 听起来是工程上的难点。具体要改什么？

Brian: There were three main porting paths.

Brian：主要有三条移植路径。

First, collective communications: the team used hipify_torch to convert the C++ bridge from CUDA to HIP, then linked it with RCCL, whose API mirrors NCCL’s.

首先是集合通信：团队用 hipify_torch 将 C++ 桥接代码从 CUDA 转成 HIP，再链接到 API 与 NCCL 相近的 RCCL。

Second, GPU memory management: the build detects the platform and routes CUDA driver calls through HIP equivalents.

其次是 GPU 内存管理：构建系统会检测平台，将 CUDA 驱动调用转到对应的 HIP 调用。

Ava: What about networking? GPU-direct transfers often expose platform differences.

Ava：网络方面呢？GPU 直传经常会暴露平台差异。

Brian: For RDMA, setting GPU_PLATFORM to rocm keeps the libibverbs-based path intact.

Brian：对于 RDMA，将 GPU_PLATFORM 设为 rocm，就能保留基于 libibverbs 的传输路径。

Only the GPU-side bindings switch from CUDA to HIP.

只有 GPU 侧的绑定从 CUDA 切换到 HIP。

That preserves the RDMA route for GPU-direct transfers.

这样就保留了 GPU 直传的 RDMA 路径。

Ava: Were there awkward compatibility issues?

Ava：遇到棘手的兼容性问题了吗？

Brian: Two cross-cutting issues stood out.

Brian：有两个跨模块的问题比较突出。

NVIDIA provides a static CUDA runtime library, libcudart_static. a.

NVIDIA 提供静态 CUDA 运行时库 libcudart_static.a。

ROCm has no static equivalent for libamdhip64, so the ROCm build links amdhip64 dynamically.

ROCm 没有对应的 libamdhip64 静态库，因此 ROCm 构建会动态链接 amdhip64。

Both platforms still load GPU driver API functions dynamically, including memory-creation calls, so the runtime contract stays the same.

两个平台仍会动态加载 GPU 驱动 API 函数，包括创建内存的调用，因此运行时约定保持一致。

Ava: And Rust bindings? I’d expect HIP names to leak everywhere.

Ava：Rust 绑定呢？我猜 HIP 的名称会渗透到各处。

Brian: That was the risk.

Brian：这确实是个风险。

After hipify_torch rewrites headers, bindgen produces HIP types such as hipError_t, hipDeviceptr_t, and hipStream_t.

hipify_torch 改写头文件后，bindgen 会生成 hipError_t、hipDeviceptr_t 和 hipStream_t 等 HIP 类型。

Instead of adding conditional branches at every Rust call site, the team added a rocm_compat module in nccl-sys and rdmaxcel-sys.

团队没有在每处 Rust 调用点添加条件分支，而是在 nccl-sys 和 rdmaxcel-sys 中加入 rocm_compat 模块。

It re-exports HIP symbols under CUDA names.

它以 CUDA 名称重新导出 HIP 符号。

So the rest of the Rust code remains platform-agnostic.

这样其余 Rust 代码就不依赖具体平台了。

Ava: That sounds like a small shim with a big payoff.

Ava：这个适配层虽小，作用却很大。

Brian: Exactly.

Brian：没错。

The port introduced HIP type aliases in Rust, and all one thousand one hundred seventy-one tests passed.

这次移植在 Rust 中引入了 HIP 类型别名，1171 项测试全部通过。

The article says this supports ROCm seven point zero and later.

文章称其支持 ROCm 7.0 及更高版本。

The contributions were upstreamed in pull requests number two thousand three hundred ninety-three and two thousand eight hundred ninety-one.

相关贡献已通过第 2393 和 2891 号拉取请求合入上游。

Ava: What does the completed ROCm stack support today?

Ava：完整的 ROCm 技术栈目前支持什么？

Brian: It includes the Actor runtime, RDMA, Supervision, and Tensor sharding.

Brian：包括 Actor 运行时、RDMA、监督机制和张量分片。

It runs on SLURM for high-performance computing, Kubernetes for cloud-native deployments, and SkyPilot for multi-cloud setups.

它可运行于面向高性能计算的 SLURM、面向云原生部署的 Kubernetes，以及面向多云环境的 SkyPilot。

Downstream engines include TorchTitan for training and TorchFT for fault tolerance.

下游引擎包括用于训练的 TorchTitan 和用于容错的 TorchFT。

Ava: Let’s walk through the actual failure experiment.

Ava：我们来梳理一下实际的故障实验。

How do Monarch, TorchFT, and TorchTitan divide the work?

Monarch、TorchFT 和 TorchTitan 如何分工？

Brian: Monarch is the orchestrator.

Brian：Monarch 负责统筹。

It creates ReplicaActors and a Lighthouse service, then organizes GPUs into Process Meshes.

它创建 ReplicaActor 和 Lighthouse 服务，再将 GPU 组织成进程网格。

TorchFT handles fault tolerance at the training-step level.

TorchFT 负责训练步骤级的容错。

It contacts the Lighthouse for quorum coordination, performs Quorum AllReduce, and skips failed nodes.

它联系 Lighthouse 协调仲裁组，执行 Quorum AllReduce，并跳过故障节点。

TorchTitan runs the Forward step with FSDP—Fully Sharded Data Parallel—the Backward step, and the Optimizer step.

TorchTitan 执行采用 FSDP（全分片数据并行）的前向步骤，以及反向步骤和优化器步骤。

Ava: Okay, four replica groups, right? Start with normal training.

Ava：一共四个副本组，对吧？先从正常训练说起。

Brian: The OrchestrationManager starts four ReplicaActors and one Lighthouse.

Brian：OrchestrationManager 启动四个 ReplicaActor 和一个 Lighthouse。

Each ReplicaActor starts a Replica with eight GPU processes running TorchTitan trainers.

每个 ReplicaActor 启动一个 Replica，其中有八个 GPU 进程运行 TorchTitan 训练器。

All four replicas form quorum number one.

四个副本组成第 1 个仲裁组。

DiLoCo gradient synchronization happens every twenty steps.

DiLoCo 每 20 步同步一次梯度。

Ava: Then a process in Replica zero crashes.

Ava：然后副本 0 中的一个进程崩溃了。

Brian: The Monarch supervisor captures report_training_error with the full traceback before the process dies.

Brian：Monarch 监督器在进程退出前捕获 report_training_error 及完整的错误堆栈。

Replicas one, two, and three are marked unaffected, so they continue training.

副本 1、2、3 被标记为未受影响，因此继续训练。

ReplicaActor zero performs an in-place restart: it stops the old process mesh and spawns a new one.

ReplicaActor 0 原地重启：停止旧进程网格，再启动新网格。

Ava: While that happens, the other three don’t freeze completely?

Ava：这期间，其他三个不会完全停下来吗？

Brian: They keep syncing as quorum number two.

Brian：它们作为第 2 个仲裁组继续同步。

Then the Lighthouse chooses Replica one as the donor.

随后，Lighthouse 选定副本 1 作为状态提供方。

A peer checkpoint transfer sends the model, optimizer, scheduler, and trainer state to recovering Replica zero.

副本间通过检查点传输，将模型、优化器、调度器和训练器状态发送给恢复中的副本 0。

All replicas pause briefly at the quorum boundary while the new quorum forms.

所有副本在仲裁组切换点短暂停顿，等待新仲裁组形成。

Ava: And after the transfer?

Ava：传输完成后呢？

Brian: Replica zero is synchronized, quorum number three is established with all four replicas, and DiLoCo synchronization resumes.

Brian：副本 0 完成同步，四个副本组成第 3 个仲裁组，DiLoCo 同步恢复。

There’s no manual intervention and no reload from a full global checkpoint.

无需人工干预，也不用从完整的全局检查点重新加载。

The disruption to overall throughput is small.

对整体吞吐量的影响很小。

Ava: What did the larger tests show?

Ava：更大规模的测试结果如何？

Brian: On a sixteen-node SLURM cluster with one hundred twenty-eight MI300 GPUs, they trained Llama three eight B.

Brian：他们在拥有 128 块 MI300 GPU 的 16 节点 SLURM 集群上训练了 Llama 3 8B。

RCCL failures were injected every one hundred eighty seconds, with quorum synchronization every twenty steps.

每隔 180 秒注入一次 RCCL 故障，每 20 步进行一次仲裁组同步。

Active workers fluctuated between eight and sixteen, but training continued without a full restart.

活跃工作节点数在 8 到 16 个之间波动，但训练无需完全重启。

The loss curve converged steadily and closely matched the failure-free baseline.

损失曲线稳定收敛，与无故障基线非常接近。

Ava: That’s useful, because frequent failures are the real stress test.

Ava：这很有参考价值，因为频繁故障才是真正的压力测试。

Brian: The article also notes that no single replica was down for more than thirty minutes.

Brian：文章还指出，没有任何一个副本离线超过 30 分钟。

Different replicas were killed and recovered dynamically.

不同副本被动态终止并恢复。

So the system wasn’t hiding failures; it was handling them during training.

所以系统是在训练中处理故障，而非掩盖故障。

Ava: And Kubernetes?

Ava：那 Kubernetes 呢？

Brian: They scaled to a thirty-two-node Kubernetes cluster with two hundred fifty-six MI355 GPUs.

Brian：他们将规模扩展到拥有 256 块 MI355 GPU 的 32 节点 Kubernetes 集群。

Participants stayed fairly stable, fluctuating between thirty and thirty-two during recovery.

恢复期间，参与节点数相当稳定，在 30 到 32 个之间波动。

Global average loss decreased smoothly from twelve to about four.

全局平均损失从 12 平稳降至约 4。

That suggests the model works across both SLURM and Kubernetes at this scale.

这表明该方案在这一规模下适用于 SLURM 和 Kubernetes。

Ava: Are there limitations or open questions?

Ava：还有哪些局限或待解决的问题？

Brian: The article doesn’t claim the problem is finished.

Brian：文章并未声称问题已经彻底解决。

Next steps include broader NIC support and better runtime performance, more pre-training and reinforcement-learning frameworks on ROCm, lower rejoin reload latency, and overlapping recovery with computation.

接下来的工作包括支持更多 NIC、提升运行时性能、让更多预训练和强化学习框架适配 ROCm、缩短重新加入时的加载延迟，以及让恢复与计算并行。

They also plan continued open-source collaboration with the PyTorch community.

他们还计划继续与 PyTorch 社区开展开源合作。

Ava: Let’s close with the three points I should remember.

Ava：最后说说我该记住的三点吧。

Brian: First, Monarch brings a single Python controller, actors, process meshes, and supervision trees to AMD GPUs through ROCm.

Brian：第一，Monarch 通过 ROCm 将单个 Python 控制器、Actor、进程网格和监督树带到 AMD GPU 上。

Second, the ROCm port uses hipify_torch, RCCL, HIP-aware memory and RDMA paths, plus Rust compatibility aliases.

第二，ROCm 移植使用 hipify_torch、RCCL、支持 HIP 的内存和 RDMA 路径，以及 Rust 兼容别名。

Third, Monarch, TorchFT, and TorchTitan recover failed replicas through quorum coordination and peer checkpoint transfer, so healthy replicas keep training.

第三，Monarch、TorchFT 和 TorchTitan 通过多数派协调和对等节点间的检查点传输来恢复故障副本，让正常副本继续训练。

Ava: So the big idea is stable large-scale training, even when hardware failures are expected.

Ava：所以核心就是，即使硬件故障在所难免，也能保持大规模训练稳定运行。

Brian: Exactly. More useful computation, less cluster-wide disruption.

Brian：没错。有效计算更多，整个集群受到的干扰更少。

Thanks for listening, and we’ll see you in the next episode.

感谢收听，我们下期再见。

## 术语

| Term | 释义 |
|---|---|
| PyTorch Monarch | PyTorch 的分布式运行时，用单个 Python 程序编排 GPU 集群，并提供故障处理能力。 |
| ROCm | AMD 的 GPU 软件平台和计算栈。 |
| actor-based runtime | 基于 actor 的运行时；每个 actor 拥有私有状态并可独立监督和恢复。 |
| process mesh | 把进程和 GPU 组织成可管理网格的抽象。 |
| supervision tree | 分层监督结构，用于隔离故障并在局部处理失败。 |
| RCCL | AMD 的集合通信库，API 与 NVIDIA 的 NCCL 相似。 |
| hipify_torch | 把 CUDA C++ 桥接代码转换为 HIP 代码的工具。 |
| RDMA | 远程直接内存访问，用于高效节点间数据传输。 |
| TorchFT | 处理训练步骤级故障容错、quorum 协调和 Quorum AllReduce 的组件。 |
| TorchTitan | 执行训练流程的引擎，包括 Forward、Backward 和 Optimizer 步骤。 |
| Quorum AllReduce | 只在当前健康副本组成的 quorum 中执行的 AllReduce 同步。 |
| DiLoCo | 文中使用的梯度同步机制，每二十步进行一次同步。 |
| FSDP | Fully Sharded Data Parallel，完全分片数据并行。 |
| peer checkpoint transfer | 从健康副本向恢复中副本传输模型和训练状态的过程。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| That’s the reliability problem this work tackles. | 这就是这项工作要解决的可靠性问题。 |
| Give me the mental model. | 请给我一个直观的理解框架。 |
| Let’s walk through the actual failure experiment. | 我们来具体走一遍故障实验。 |
| While that happens, the other three keep syncing. | 在此期间，另外三个仍继续同步。 |
| That sounds like a small shim with a big payoff. | 这听起来是一个很小但收益很大的兼容层。 |
| The article doesn’t claim the problem is finished. | 文章并没有声称这个问题已经彻底解决。 |
| The big idea is stable large-scale training. | 核心思想是实现稳定的大规模训练。 |
| Let’s close with the three points I should remember. | 最后总结我应该记住的三点。 |
