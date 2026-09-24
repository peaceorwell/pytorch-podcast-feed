# Bringing PyTorch Monarch to AMD GPUs: Single-Controller Distributed Training on ROCm

原文：[Bringing PyTorch Monarch to AMD GPUs: Single-Controller Distributed Training on ROCm](https://pytorch.org/blog/bringing-pytorch-monarch-to-amd-gpus-single-controller-distributed-training-on-rocm/)

## 摘要

本期介绍 PyTorch Monarch 如何把单控制器、基于 actor 的分布式训练带到 AMD Instinct GPU 和 ROCm 平台。Monarch 通过 process mesh、supervision tree 和异步执行，把故障隔离在局部，并让健康副本继续训练。结合 TorchFT 的 quorum 同步与 TorchTitan 训练引擎，系统可以通过节点间 checkpoint 传输恢复，而不必全局重启。实验显示，该方案在 SLURM 和 Kubernetes 的 MI300/MI355 集群上都能保持稳定收敛；未来还将优化 NIC 支持、重连延迟和更多训练框架。

## 对话

**Ava:** Imagine a language-model training job that has been running for days, then one GPU crashes—and the whole cluster has to start over. That’s the reliability problem this PyTorch Monarch work tackles on AMD GPUs.

**Brian:** Exactly. The article brings Monarch to AMD Instinct GPUs with ROCm, so healthy workers can keep training while a failed worker recovers and rejoins. It’s a single-controller model for large distributed jobs, with fault tolerance built into the runtime.

**Ava:** Why isn’t ordinary checkpointing enough? Large training systems already save checkpoints regularly.

**Brian:** Checkpointing is simple, but expensive. Writing hundreds of gigabytes takes time and I/O bandwidth. If a failure happens, every update since the last checkpoint is lost. The whole cluster may sit idle while a node is replaced and the job restarts. And as the cluster grows, the chance of failing during a checkpoint interval also grows.

**Ava:** So the goal is to recover locally instead of resetting globally.

**Brian:** Right. Monarch lets healthy nodes continue training while failed nodes recover. That reduces wasted computation and keeps GPU utilization higher. The article frames this as a shift from raw scaling to resilient scaling.

**Ava:** Give me the mental model for Monarch. What does a developer actually program?

**Brian:** You write one Python program that orchestrates an entire GPU cluster. Monarch uses an actor-based runtime, a process mesh abstraction, and asynchronous execution. An actor has private state, so if it crashes, the failure doesn’t automatically spread to every other actor.

**Ava:** Actor-based runtime can sound abstract. Is each actor basically a supervised worker?

**Brian:** That’s a useful way to think about it. The Python API is the top layer. The Monarch Runtime manages actors, meshes, supervision trees, and tensor sharding. Under that sits a Rust runtime using Tokio for performance and memory safety. Infrastructure connects it to RDMA, RCCL or NCCL, SLURM, Kubernetes, and SkyPilot.

**Ava:** And the supervision tree is what gives the system hierarchy?

**Brian:** Yes. Failures are isolated and handled at the lowest practical level. A local restart can take seconds. Escalation can take minutes. Monarch also separates two concerns: the parallelism strategy inside a training replica, and the fault-tolerance mechanism across replicas. That separation makes the recovery model cleaner.

**Ava:** Moving from CUDA to ROCm sounds like the hard engineering part. What had to change?

**Brian:** There were three main porting paths. First, collective communications: the team used hipify_torch to convert the C++ bridge from CUDA to HIP, then linked it with RCCL, whose API mirrors NCCL’s. Second, GPU memory management: the build detects the platform and routes CUDA driver calls through HIP equivalents.

**Ava:** What about networking? GPU-direct transfers often expose platform differences.

**Brian:** For RDMA, setting GPU_PLATFORM to rocm keeps the libibverbs-based path intact. Only the GPU-side bindings switch from CUDA to HIP. That preserves the RDMA route for GPU-direct transfers.

**Ava:** Were there awkward compatibility issues?

**Brian:** Two cross-cutting issues stood out. NVIDIA provides a static CUDA runtime library, libcudart_static.a. ROCm has no static equivalent for libamdhip64, so the ROCm build links amdhip64 dynamically. Both platforms still load GPU driver API functions dynamically, including memory-creation calls, so the runtime contract stays the same.

**Ava:** And Rust bindings? I’d expect HIP names to leak everywhere.

**Brian:** That was the risk. After hipify_torch rewrites headers, bindgen produces HIP types such as hipError_t, hipDeviceptr_t, and hipStream_t. Instead of adding conditional branches at every Rust call site, the team added a rocm_compat module in nccl-sys and rdmaxcel-sys. It re-exports HIP symbols under CUDA names. So the rest of the Rust code remains platform-agnostic.

**Ava:** That sounds like a small shim with a big payoff.

**Brian:** Exactly. The port introduced HIP type aliases in Rust, and all one thousand one hundred seventy-one tests passed. The article says this supports ROCm seven point zero and later. The contributions were upstreamed in pull requests number two thousand three hundred ninety-three and two thousand eight hundred ninety-one.

**Ava:** What does the completed ROCm stack support today?

**Brian:** It includes the Actor runtime, RDMA, Supervision, and Tensor sharding. It runs on SLURM for high-performance computing, Kubernetes for cloud-native deployments, and SkyPilot for multi-cloud setups. Downstream engines include TorchTitan for training and TorchFT for fault tolerance.

**Ava:** Let’s walk through the actual failure experiment. How do Monarch, TorchFT, and TorchTitan divide the work?

**Brian:** Monarch is the orchestrator. It creates ReplicaActors and a Lighthouse service, then organizes GPUs into Process Meshes. TorchFT handles fault tolerance at the training-step level. It contacts the Lighthouse for quorum coordination, performs Quorum AllReduce, and skips failed nodes. TorchTitan runs the Forward step with FSDP—Fully Sharded Data Parallel—the Backward step, and the Optimizer step.

**Ava:** Okay, four replica groups, right? Start with normal training.

**Brian:** The OrchestrationManager starts four ReplicaActors and one Lighthouse. Each ReplicaActor starts a Replica with eight GPU processes running TorchTitan trainers. All four replicas form quorum number one. DiLoCo gradient synchronization happens every twenty steps.

**Ava:** Then a process in Replica zero crashes.

**Brian:** The Monarch supervisor captures report_training_error with the full traceback before the process dies. Replicas one, two, and three are marked unaffected, so they continue training. ReplicaActor zero performs an in-place restart: it stops the old process mesh and spawns a new one.

**Ava:** While that happens, the other three don’t freeze completely?

**Brian:** They keep syncing as quorum number two. Then the Lighthouse chooses Replica one as the donor. A peer checkpoint transfer sends the model, optimizer, scheduler, and trainer state to recovering Replica zero. All replicas pause briefly at the quorum boundary while the new quorum forms.

**Ava:** And after the transfer?

**Brian:** Replica zero is synchronized, quorum number three is established with all four replicas, and DiLoCo synchronization resumes. There’s no manual intervention and no reload from a full global checkpoint. The disruption to overall throughput is small.

**Ava:** What did the larger tests show?

**Brian:** On a sixteen-node SLURM cluster with one hundred twenty-eight MI300 GPUs, they trained Llama three eight B. RCCL failures were injected every one hundred eighty seconds, with quorum synchronization every twenty steps. Active workers fluctuated between eight and sixteen, but training continued without a full restart. The loss curve converged steadily and closely matched the failure-free baseline.

**Ava:** That’s useful, because frequent failures are the real stress test.

**Brian:** The article also notes that no single replica was down for more than thirty minutes. Different replicas were killed and recovered dynamically. So the system wasn’t hiding failures; it was handling them during training.

**Ava:** And Kubernetes?

**Brian:** They scaled to a thirty-two-node Kubernetes cluster with two hundred fifty-six MI355 GPUs. Participants stayed fairly stable, fluctuating between thirty and thirty-two during recovery. Global average loss decreased smoothly from twelve to about four. That suggests the model works across both SLURM and Kubernetes at this scale.

**Ava:** Are there limitations or open questions?

**Brian:** The article doesn’t claim the problem is finished. Next steps include broader NIC support and better runtime performance, more pre-training and reinforcement-learning frameworks on ROCm, lower rejoin reload latency, and overlapping recovery with computation. They also plan continued open-source collaboration with the PyTorch community.

**Ava:** Let’s close with the three points I should remember.

**Brian:** First, Monarch brings a single Python controller, actors, process meshes, and supervision trees to AMD GPUs through ROCm. Second, the ROCm port uses hipify_torch, RCCL, HIP-aware memory and RDMA paths, plus Rust compatibility aliases. Third, Monarch, TorchFT, and TorchTitan recover failed replicas through quorum coordination and peer checkpoint transfer, so healthy replicas keep training.

**Ava:** So the big idea is stable large-scale training, even when hardware failures are expected.

**Brian:** Exactly. More useful computation, less cluster-wide disruption. Thanks for listening, and we’ll see you in the next episode.

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
