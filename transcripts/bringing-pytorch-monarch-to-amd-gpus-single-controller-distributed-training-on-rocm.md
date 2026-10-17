# Bringing PyTorch Monarch to AMD GPUs: Single-Controller Distributed Training on ROCm

原文：[Bringing PyTorch Monarch to AMD GPUs: Single-Controller Distributed Training on ROCm](https://pytorch.org/blog/bringing-pytorch-monarch-to-amd-gpus-single-controller-distributed-training-on-rocm/)

## 摘要

本期介绍 PyTorch Monarch 如何把单控制器、基于 actor 的分布式训练带到 AMD Instinct GPU 和 ROCm 平台。Monarch 通过 process mesh、supervision tree 和异步执行，把故障隔离在局部，并让健康副本继续训练。结合 TorchFT 的 quorum 同步与 TorchTitan 训练引擎，系统可以通过节点间 checkpoint 传输恢复，而不必全局重启。实验显示，该方案在 SLURM 和 Kubernetes 的 MI300/MI355 集群上都能保持稳定收敛；未来还将优化 NIC 支持、重连延迟和更多训练框架。

## 对话

Ava: Imagine a language-model training job that has been running for days, then one GPU crashes—and the whole cluster has to start over. That’s the reliability problem this PyTorch Monarch work tackles on AMD GPUs.

Ava：想象一个语言模型训练任务已经运行了好几天，结果一块 GPU 崩溃，整个集群都得从头开始。这就是这项面向 AMD GPU 的 PyTorch Monarch 工作要解决的可靠性问题。

Brian: Exactly. The article brings Monarch to AMD Instinct GPUs with ROCm, so healthy workers can keep training while a failed worker recovers and rejoins. It’s a single-controller model for large distributed jobs, with fault tolerance built into the runtime.

Brian：没错。文章介绍了如何通过 ROCm 将 Monarch 带到 AMD Instinct GPUs，让正常运行的工作进程在故障进程恢复并重新加入时继续训练。它为大型分布式任务提供单控制器模型，并在运行时内置容错能力。

Ava: Why isn’t ordinary checkpointing enough? Large training systems already save checkpoints regularly.

Ava：普通的检查点机制为什么不够？大型训练系统已经会定期保存检查点了。

Brian: Checkpointing is simple, but expensive. Writing hundreds of gigabytes takes time and I/O bandwidth. If a failure happens, every update since the last checkpoint is lost. The whole cluster may sit idle while a node is replaced and the job restarts. And as the cluster grows, the chance of failing during a checkpoint interval also grows.

Brian：检查点机制很简单，但代价高昂。写入数百 GB 数据需要时间，也会占用 I/O 带宽。一旦发生故障，上次保存检查点以来的所有更新都会丢失。更换节点并重启任务期间，整个集群可能都在空等。而且集群越大，在两次检查点之间发生故障的概率也越高。

Ava: So the goal is to recover locally instead of resetting globally.

Ava：所以目标是局部恢复，而不是让整个系统重置。

Brian: Right. Monarch lets healthy nodes continue training while failed nodes recover. That reduces wasted computation and keeps GPU utilization higher. The article frames this as a shift from raw scaling to resilient scaling.

Brian：对。Monarch 让正常节点在故障节点恢复期间继续训练。这样可以减少计算浪费，并保持更高的 GPU 利用率。文章将其描述为从单纯扩大规模转向有韧性的扩展。

Ava: Give me the mental model for Monarch. What does a developer actually program?

Ava：给我讲讲理解 Monarch 的基本思路。开发者实际要编写什么？

Brian: You write one Python program that orchestrates an entire GPU cluster. Monarch uses an actor-based runtime, a process mesh abstraction, and asynchronous execution. An actor has private state, so if it crashes, the failure doesn’t automatically spread to every other actor.

Brian：你只需编写一个 Python 程序，就能编排整个 GPU 集群。Monarch 使用基于 actor 的运行时、进程网格抽象和异步执行。每个 actor 都有自己的私有状态，因此一个 actor 崩溃时，故障不会自动蔓延到其他所有 actor。

Ava: Actor-based runtime can sound abstract. Is each actor basically a supervised worker?

Ava：基于 actor 的运行时听起来有些抽象。每个 actor 基本上就是一个受监管的工作进程吗？

Brian: That’s a useful way to think about it. The Python API is the top layer. The Monarch Runtime manages actors, meshes, supervision trees, and tensor sharding. Under that sits a Rust runtime using Tokio for performance and memory safety. Infrastructure connects it to RDMA, RCCL or NCCL, SLURM, Kubernetes, and SkyPilot.

Brian：这样理解很有帮助。Python API 是最上层。Monarch Runtime 管理 actors、meshes、supervision trees 和 tensor sharding。其下是使用 Tokio 的 Rust 运行时，以获得性能和内存安全性。基础设施层将它连接到 RDMA、RCCL 或 NCCL，以及 SLURM、Kubernetes 和 SkyPilot。

Ava: And the supervision tree is what gives the system hierarchy?

Ava：supervision tree 就是让系统具有层级结构的机制？

Brian: Yes. Failures are isolated and handled at the lowest practical level. A local restart can take seconds. Escalation can take minutes. Monarch also separates two concerns: the parallelism strategy inside a training replica, and the fault-tolerance mechanism across replicas. That separation makes the recovery model cleaner.

Brian：是的。故障会被隔离，并在实际可行的最低层级处理。局部重启可能只需几秒；如果需要向上升级处理，可能要几分钟。Monarch 还把两件事分开：训练副本内部的并行策略，以及副本之间的容错机制。这种分离让恢复模型更清晰。

Ava: Moving from CUDA to ROCm sounds like the hard engineering part. What had to change?

Ava：从 CUDA 迁移到 ROCm 听起来是工程上最难的部分。具体需要改动什么？

Brian: There were three main porting paths. First, collective communications: the team used hipify_torch to convert the C++ bridge from CUDA to HIP, then linked it with RCCL, whose API mirrors NCCL’s. Second, GPU memory management: the build detects the platform and routes CUDA driver calls through HIP equivalents.

Brian：移植主要有 3 条路径。第一是集合通信：团队用 hipify_torch 将 C++ 桥接代码从 CUDA 转换为 HIP，然后链接到 RCCL；RCCL 的 API 与 NCCL 的 API 对应。第二是 GPU 内存管理：构建过程会检测平台，并将 CUDA 驱动调用转到对应的 HIP 调用。

Ava: What about networking? GPU-direct transfers often expose platform differences.

Ava：网络方面呢？GPU-direct 传输常常会暴露平台差异。

Brian: For RDMA, setting GPU_PLATFORM to rocm keeps the libibverbs-based path intact. Only the GPU-side bindings switch from CUDA to HIP. That preserves the RDMA route for GPU-direct transfers.

Brian：对于 RDMA，将 GPU_PLATFORM 设为 rocm，就能保留基于 libibverbs 的路径。只有 GPU 侧的绑定从 CUDA 切换到 HIP。这样就保留了用于 GPU-direct 传输的 RDMA 路径。

Ava: Were there awkward compatibility issues?

Ava：有没有棘手的兼容性问题？

Brian: Two cross-cutting issues stood out. NVIDIA provides a static CUDA runtime library, libcudart_static.a. ROCm has no static equivalent for libamdhip64, so the ROCm build links amdhip64 dynamically. Both platforms still load GPU driver API functions dynamically, including memory-creation calls, so the runtime contract stays the same.

Brian：有 2 个贯穿整个移植过程的问题值得注意。NVIDIA 提供静态 CUDA 运行时库 libcudart_static.a。ROCm 没有与 libamdhip64 对应的静态库，因此 ROCm 构建会动态链接 amdhip64。两个平台仍会动态加载 GPU 驱动 API 函数，包括内存创建调用，因此运行时契约保持一致。

Ava: And Rust bindings? I’d expect HIP names to leak everywhere.

Ava：Rust 绑定呢？我猜 HIP 的名称会渗透到各处。

Brian: That was the risk. After hipify_torch rewrites headers, bindgen produces HIP types such as hipError_t, hipDeviceptr_t, and hipStream_t. Instead of adding conditional branches at every Rust call site, the team added a rocm_compat module in nccl-sys and rdmaxcel-sys. It re-exports HIP symbols under CUDA names. So the rest of the Rust code remains platform-agnostic.

Brian：确实有这个风险。hipify_torch 重写头文件后，bindgen 会生成 hipError_t、hipDeviceptr_t 和 hipStream_t 等 HIP 类型。团队没有在每个 Rust 调用点都添加条件分支，而是在 nccl-sys 和 rdmaxcel-sys 中加入 rocm_compat 模块。这个模块以 CUDA 名称重新导出 HIP 符号，因此其余 Rust 代码仍与平台无关。

Ava: That sounds like a small shim with a big payoff.

Ava：听起来是一个很小的适配层，却带来了很大的收益。

Brian: Exactly. The port introduced HIP type aliases in Rust, and all one thousand one hundred seventy-one tests passed. The article says this supports ROCm seven point zero and later. The contributions were upstreamed in pull requests number two thousand three hundred ninety-three and two thousand eight hundred ninety-one.

Brian：没错。这次移植在 Rust 中引入了 HIP 类型别名，1171 项测试全部通过。文章称，这支持 ROCm 7.0 及更高版本。相关贡献通过编号为 2393 和 2891 的拉取请求合并到了上游。

Ava: What does the completed ROCm stack support today?

Ava：目前完成的 ROCm 技术栈支持哪些功能？

Brian: It includes the Actor runtime, RDMA, Supervision, and Tensor sharding. It runs on SLURM for high-performance computing, Kubernetes for cloud-native deployments, and SkyPilot for multi-cloud setups. Downstream engines include TorchTitan for training and TorchFT for fault tolerance.

Brian：它包括 Actor runtime、RDMA、Supervision 和 Tensor sharding。它可以运行在用于高性能计算的 SLURM、用于云原生部署的 Kubernetes，以及用于多云部署的 SkyPilot 上。下游引擎包括负责训练的 TorchTitan 和负责容错的 TorchFT。

Ava: Let’s walk through the actual failure experiment. How do Monarch, TorchFT, and TorchTitan divide the work?

Ava：我们按步骤看一下实际的故障实验。Monarch、TorchFT 和 TorchTitan 是怎样分工的？

Brian: Monarch is the orchestrator. It creates ReplicaActors and a Lighthouse service, then organizes GPUs into Process Meshes. TorchFT handles fault tolerance at the training-step level. It contacts the Lighthouse for quorum coordination, performs Quorum AllReduce, and skips failed nodes. TorchTitan runs the Forward step with FSDP—Fully Sharded Data Parallel—the Backward step, and the Optimizer step.

Brian：Monarch 负责协调。它创建 ReplicaActors 和 Lighthouse 服务，然后把 GPUs 组织成 Process Meshes。TorchFT 负责训练步骤层面的容错：它联系 Lighthouse 协调 quorum，执行 Quorum AllReduce，并跳过故障节点。TorchTitan 执行采用 FSDP（Fully Sharded Data Parallel）的 Forward 步骤、Backward 步骤和 Optimizer 步骤。

Ava: Okay, four replica groups, right? Start with normal training.

Ava：好，是 4 个副本组，对吧？先从正常训练说起。

Brian: The OrchestrationManager starts four ReplicaActors and one Lighthouse. Each ReplicaActor starts a Replica with eight GPU processes running TorchTitan trainers. All four replicas form quorum number one. DiLoCo gradient synchronization happens every twenty steps.

Brian：OrchestrationManager 启动 4 个 ReplicaActor 和 1 个 Lighthouse。每个 ReplicaActor 启动 1 个 Replica，其中有 8 个运行 TorchTitan 训练器的 GPU 进程。全部 4 个副本组成第 1 个 quorum。每 20 步进行一次 DiLoCo 梯度同步。

Ava: Then a process in Replica zero crashes.

Ava：然后 Replica 0 中的一个进程崩溃了。

Brian: The Monarch supervisor captures report_training_error with the full traceback before the process dies. Replicas one, two, and three are marked unaffected, so they continue training. ReplicaActor zero performs an in-place restart: it stops the old process mesh and spawns a new one.

Brian：Monarch 的监督器在进程退出前捕获 report_training_error，并记录完整的调用栈。Replica 1、2、3 被标记为未受影响，因此继续训练。ReplicaActor 0 就地重启：停止旧的进程网格，再启动一个新的。

Ava: While that happens, the other three don’t freeze completely?

Ava：在这期间，另外 3 个副本不会完全停下来吗？

Brian: They keep syncing as quorum number two. Then the Lighthouse chooses Replica one as the donor. A peer checkpoint transfer sends the model, optimizer, scheduler, and trainer state to recovering Replica zero. All replicas pause briefly at the quorum boundary while the new quorum forms.

Brian：它们作为第 2 个 quorum 继续同步。随后，Lighthouse 选择 Replica 1 作为提供状态的副本。副本间的检查点传输把模型、优化器、调度器和训练器的状态发送给正在恢复的 Replica 0。新 quorum 形成时，所有副本都会在 quorum 边界短暂停顿。

Ava: And after the transfer?

Ava：传输完成之后呢？

Brian: Replica zero is synchronized, quorum number three is established with all four replicas, and DiLoCo synchronization resumes. There’s no manual intervention and no reload from a full global checkpoint. The disruption to overall throughput is small.

Brian：Replica 0 完成同步，由全部 4 个副本组成的第 3 个 quorum 建立，DiLoCo 同步随之恢复。整个过程无需人工干预，也无需从完整的全局检查点重新加载。对总体吞吐量的影响很小。

Ava: What did the larger tests show?

Ava：规模更大的测试显示了什么？

Brian: On a sixteen-node SLURM cluster with one hundred twenty-eight MI300 GPUs, they trained Llama three eight B. RCCL failures were injected every one hundred eighty seconds, with quorum synchronization every twenty steps. Active workers fluctuated between eight and sixteen, but training continued without a full restart. The loss curve converged steadily and closely matched the failure-free baseline.

Brian：他们在一个拥有 16 个节点、128 块 MI300 GPU 的 SLURM 集群上训练了 Llama 3 8B。每隔 180 秒注入一次 RCCL 故障，每 20 步进行一次 quorum 同步。活跃工作进程的数量在 8 到 16 之间波动，但训练没有经过完整重启就持续进行。损失曲线稳定收敛，与无故障基线非常接近。

Ava: That’s useful, because frequent failures are the real stress test.

Ava：这很有参考价值，因为频繁发生故障才是真正的压力测试。

Brian: The article also notes that no single replica was down for more than thirty minutes. Different replicas were killed and recovered dynamically. So the system wasn’t hiding failures; it was handling them during training.

Brian：文章还指出，没有任何一个副本停机超过 30 分钟。不同的副本会被终止，并动态恢复。所以系统并没有掩盖故障，而是在训练过程中处理故障。

Ava: And Kubernetes?

Ava：Kubernetes 上的情况呢？

Brian: They scaled to a thirty-two-node Kubernetes cluster with two hundred fifty-six MI355 GPUs. Participants stayed fairly stable, fluctuating between thirty and thirty-two during recovery. Global average loss decreased smoothly from twelve to about four. That suggests the model works across both SLURM and Kubernetes at this scale.

Brian：他们还扩展到了一个拥有 32 个节点、256 块 MI355 GPU 的 Kubernetes 集群。参与训练的副本数量相当稳定，在恢复期间于 30 到 32 之间波动。全局平均损失从 12 平稳降至约 4。这表明，在这一规模下，该模型既适用于 SLURM，也适用于 Kubernetes。

Ava: Are there limitations or open questions?

Ava：还有什么限制或待解决的问题吗？

Brian: The article doesn’t claim the problem is finished. Next steps include broader NIC support and better runtime performance, more pre-training and reinforcement-learning frameworks on ROCm, lower rejoin reload latency, and overlapping recovery with computation. They also plan continued open-source collaboration with the PyTorch community.

Brian：文章没有声称问题已经全部解决。下一步包括支持更多 NIC、提升运行时性能、让更多预训练和强化学习框架在 ROCm 上运行、缩短重新加入时的重新加载延迟，以及让恢复与计算重叠进行。他们还计划继续与 PyTorch 社区开展开源协作。

Ava: Let’s close with the three points I should remember.

Ava：最后说说我应该记住的 3 个要点。

Brian: First, Monarch brings a single Python controller, actors, process meshes, and supervision trees to AMD GPUs through ROCm. Second, the ROCm port uses hipify_torch, RCCL, HIP-aware memory and RDMA paths, plus Rust compatibility aliases. Third, Monarch, TorchFT, and TorchTitan recover failed replicas through quorum coordination and peer checkpoint transfer, so healthy replicas keep training.

Brian：第一，Monarch 通过 ROCm 把单一 Python 控制器、actors、process meshes 和 supervision trees 带到 AMD GPUs 上。第二，ROCm 移植使用了 hipify_torch、RCCL、适配 HIP 的内存和 RDMA 路径，以及 Rust 兼容性别名。第三，Monarch、TorchFT 和 TorchTitan 通过 quorum 协调与对等检查点传输来恢复故障副本，让正常副本继续训练。

Ava: So the big idea is stable large-scale training, even when hardware failures are expected.

Ava：所以核心思路是：即使硬件故障是预料之中的，大规模训练也能稳定运行。

Brian: Exactly. More useful computation, less cluster-wide disruption. Thanks for listening, and we’ll see you in the next episode.

Brian：完全正确。完成更多有用的计算，让整个集群受到更少干扰。感谢收听，我们下期再见。

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
