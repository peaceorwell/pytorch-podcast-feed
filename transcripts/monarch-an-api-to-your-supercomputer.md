# Monarch: An API to Your Supercomputer

原文：[Monarch: an API to your supercomputer](https://pytorch.org/blog/monarch-an-api-to-your-supercomputer/)

## 摘要

本期介绍 PyTorch Monarch，一个用简单 Python API 把超算集群变成可编程整体的分布式框架。它通过 RDMA 文件系统、分布式 SQL Telemetry 和 Jobs API，加速代码同步、资源管理与调试，让大规模训练更像本地开发。文章还总结了 Monarch 自 2025 年发布以来在 Kubernetes、RDMA、可观测性、安装体验和 AMD ROCm 支持上的改进。Monarch 已与 SkyPilot、VeRL 和 AMD 等项目合作，但在 Ray 依赖较深的框架中替换后端仍然比较复杂。

## 对话

Ava: What if your laptop could control a thousand GPUs as easily as it runs a Python script?

Ava：如果你的笔记本电脑能像运行 Python 脚本一样轻松控制上千块 GPU，会怎样？

That’s the promise behind Monarch, and it matters because distributed training can become painfully slow to develop and debug.

这正是 Monarch 的愿景。它很重要，因为分布式训练的开发和调试常常慢得令人头疼。

Brian: Exactly. Monarch is a distributed programming framework for PyTorch.

Brian：没错。Monarch 是一个面向 PyTorch 的分布式编程框架。

Its goal is to make a huge cluster feel like one coherent, directly controllable system, almost as if your laptop had thousands of GPUs attached.

它希望让庞大的集群像一个统一、可直接操控的系统，仿佛你的笔记本电脑连着几千块 GPU。

Ava: So this is bigger than a job launcher?

Ava：所以它不只是个任务启动器？

Brian: Much bigger. A complete training system can live in one Python program.

Brian：远不止。整套训练系统都可以写在一个 Python 程序里。

Monarch exposes simple primitives for hosts, procs, and actors.

Monarch 为主机、进程和 actor 提供了简单的基础接口。

Then higher-level features, such as fault tolerance and orchestration, can be built as reusable libraries.

容错和编排等高级功能则可以构建成可复用的库。

Ava: That sounds especially useful for distributed reinforcement learning, where many moving parts have to work together.

Ava：这对分布式强化学习似乎特别有用，毕竟很多环节都得协同工作。

Brian: Right. The article starts with that pain.

Brian：对。文章就是从这个痛点讲起的。

Huge clusters are hard to use, and distributed reinforcement learning makes the setup more complex.

大型集群本就难用，分布式强化学习又让配置更加复杂。

When you change something, the turnaround time can be very slow.

每次改动后，等待结果可能要很久。

Debugging is frustrating because the system is spread across many processes and machines.

系统分散在许多进程和机器上，调试起来也很让人头疼。

Ava: Monarch tries to make that feel local?

Ava：Monarch 想让它用起来像本地系统？

Brian: Yes. The central idea is that the distributed system should feel local.

Brian：是的。核心想法就是让分布式系统用起来像本地系统。

Monarch is also designed for agentic usage.

Monarch 也针对智能体的使用方式做了设计。

An agent can run development tasks on a dev machine, and Monarch can turn that dev machine into a supercomputer with consistent infrastructure abstractions.

智能体可以在开发机上执行开发任务，而 Monarch 借助统一的基础设施抽象，让这台开发机像超级计算机一样工作。

Ava: When you say agentic usage, what can the agent actually do?

Ava：你说智能体的使用方式，具体是指它能做什么？

Brian: It can manage and debug running code, sync dependencies and data, launch new code, and provision more hosts, procs, and actors.

Brian：它能管理和调试运行中的代码、同步依赖和数据、启动新代码，还能申请更多主机、进程和 actor。

The same model works across different deployment environments.

同一套方式适用于不同的部署环境。

Ava: Let’s start with the file system.

Ava：先说文件系统吧。

The article mentions an RDMA-Powered Remote File System.

文章提到了基于 RDMA 的远程文件系统。

Brian: The client exposes a read-only mounted filesystem.

Brian：客户端提供一个只读的挂载文件系统。

Monarch distributes those files to every host in the job through RDMA, or Remote Direct Memory Access.

Monarch 通过 RDMA，也就是远程直接内存访问，把这些文件分发到任务中的每台主机。

That makes it much faster to sync code, dependencies, and containers while you iterate.

这样在反复修改时，同步代码、依赖和容器就快得多。

Ava: So instead of repeatedly copying files through a slower path, the cluster sees a shared read-only view?

Ava：也就是说，不用反复通过较慢的方式复制文件，整个集群都能看到同一个只读文件视图？

Brian: That’s the basic idea. The RDMA filesystem is built on Monarch RDMA buffers and PyFuse.

Brian：基本就是这样。这个 RDMA 文件系统基于 Monarch RDMA 缓冲区和 PyFuse 构建。

The practical benefit is quick iteration: update your code or dependencies, and get them across the job quickly.

实际好处是迭代更快：更新代码或依赖后，很快就能同步到整个任务。

Ava: The second feature is Distributed SQL Telemetry. Why SQL?

Ava：第二项功能是分布式 SQL 遥测。为什么用 SQL？

Brian: Because SQL is a familiar way to inspect structured state, and agents already work well with SQL-based APIs.

Brian：因为 SQL 是查看结构化状态的常用方式，智能体也很擅长使用基于 SQL 的接口。

Monarch includes a lightweight distributed SQL engine.

Monarch 内置了一个轻量级分布式 SQL 引擎。

It collects live state information, pyspy traces, and logs from distributed processes and actors.

它会收集分布式进程和 actor 的实时状态、pyspy 跟踪信息和日志。

Ava: Does Monarch just collect the data, or can it query the running system directly?

Ava：Monarch 只是收集数据，还是能直接查询运行中的系统？

Brian: It can query it directly.

Brian：可以直接查询。

The team ran a DataFusion distributed SQL query engine in situ, meaning inside the running distributed environment.

团队在运行中的分布式环境里直接部署了 DataFusion 分布式 SQL 查询引擎。

Each node writes live state into tables, and an agent can query those tables efficiently while debugging.

每个节点都把实时状态写入表中，智能体调试时可以高效查询这些表。

Ava: That sounds much easier than opening logs from dozens of machines.

Ava：这可比逐台打开几十台机器的日志方便多了。

Brian: Exactly. You can explore the state from a central point.

Brian：没错。你可以从一个中心入口查看系统状态。

The article presents this as a way to reduce debugging time, because the information is accessible through one client-facing SQL endpoint.

文章认为，这能缩短调试时间，因为所有信息都能通过一个面向客户端的 SQL 接口访问。

Ava: Then there’s the Jobs API. What problem does that solve?

Ava：接下来是 Jobs API。它解决什么问题？

Brian: It lets you provision resources, such as hosts, once, and run many jobs on them.

Brian：它让你一次申请主机等资源，然后在上面运行多个任务。

You avoid paying the repeated allocation penalty every time a job restarts.

这样每次重启任务就不用重新等待资源分配。

Monarch supports Kubernetes and SLURM, and other schedulers can be connected by implementing a Monarch Job.

Monarch 支持 Kubernetes 和 SLURM；其他调度器也可以通过实现 Monarch Job 接入。

Ava: So the three pieces work together: sync quickly, inspect quickly, and restart quickly.

Ava：所以这三项功能配合起来，就是快速同步、快速查看、快速重启。

Brian: That’s a good recap.

Brian：总结得很到位。

Monarch’s toolbox helps agents restart jobs fast, sync new code and data fast, and debug fast.

Monarch 的工具让智能体能够快速重启任务、同步新代码和数据，并迅速调试。

The system feels local even though the work is distributed.

工作虽然分布在多台机器上，用起来却像在本地。

Ava: The article says Monarch launched at the PyTorch conference in October twenty twenty-five.

Ava：文章说，Monarch 在 2025 年 10 月的 PyTorch 大会上发布。

What changed since then?

此后有什么变化？

Brian: Kubernetes support became first-class.

Brian：Kubernetes 成了一等支持平台。

There’s now a dedicated open-source repository called monarch-kubernetes.

现在有一个专门的开源仓库，叫 monarch-kubernetes。

It includes a MonarchMesh Custom Resource Definition, a reference KubeBuilder operator, and a hello-world demo.

其中包含 MonarchMesh 自定义资源定义、参考 KubeBuilder Operator 和 hello-world 演示。

Ava: What does just-in-time pod provisioning mean here?

Ava：这里的按需创建 Pod 是什么意思？

Brian: Pods are allocated when they’re needed instead of being reserved upfront.

Brian：需要时才分配 Pod，而不是预先预留。

That can improve cluster utilization.

这样可以提高集群利用率。

Monarch also added an external gateway, so out-of-cluster clients can connect to meshes running inside Kubernetes.

Monarch 还增加了外部网关，让集群外的客户端能连接 Kubernetes 内运行的 Mesh。

That feature is landing in zero point five.

这个功能将随 0.5 版推出。

Ava: And there are versioned and nightly Docker containers?

Ava：还有正式版和每日构建版的 Docker 容器？

Brian: Yes.

Brian：对。

They’re published to GHCR, the GitHub Container Registry, to support reproducible deployments.

它们发布到 GitHub 容器注册表 GHCR，便于复现部署。

MonarchMesh labels can also enable scheduling through Kueue.

MonarchMesh 标签也可以启用 Kueue 调度。

Ava: Let’s move to networking. RDMA appears everywhere in this project.

Ava：接着聊网络。这个项目里到处都能看到 RDMA。

Brian: Monarch added several RDMA backends and a higher-level API.

Brian：Monarch 增加了多个 RDMA 后端和一个更高层的 API。

AWS EFA, or Elastic Fabric Adapter, is now supported by Monarch’s RDMABuffer.

Monarch 的 RDMABuffer 现已支持 AWS EFA，即弹性网络适配器。

The article says it was validated at sixteen gigabits per second, ten times faster than TCP, with fourteen point five gigabytes transferred in seven point six seconds.

文章称，实测速度为每秒 16 吉比特，比 TCP 快 10 倍，7.6 秒传输了 14.5 吉字节。

Ava: That’s a concrete result. Is the API tied to AWS hardware?

Ava：这个结果很具体。API 依赖 AWS 硬件吗？

Brian: No. The Unified RDMA API is designed to be hardware-portable.

Brian：不依赖。统一 RDMA API 旨在跨硬件运行。

It works across InfiniBand with mlx5, AWS EFA, and ROCm.

它支持使用 mlx5 的 InfiniBand、AWS EFA 和 ROCm。

You can write once and run on different fabrics, or fall back to Monarch actor messaging when RDMA isn’t available.

代码写一次，就能在不同网络架构上运行；没有 RDMA 时，也能回退到 Monarch Actor 消息传递。

Ava: ROCm brings AMD GPUs into the picture.

Ava：ROCm 也让 AMD GPU 加入进来。

Brian: Right.

Brian：没错。

GPU-direct RDMA and RCCL collective communication now work on AMD GPUs through ROCm with Mellanox interfaces.

借助 ROCm 和 Mellanox 网卡，AMD GPU 现在支持 GPU 直连 RDMA 和 RCCL 集合通信。

AMD also validated Monarch on ROCm for MI300, MI325, and MI355 clusters, with SLURM-based orchestration.

AMD 还在采用 SLURM 编排的 MI300、MI325 和 MI355 集群上验证了 Monarch 的 ROCm 支持。

Ava: That’s useful for teams that don’t run NVIDIA hardware.

Ava：这对不用 NVIDIA 硬件的团队很有帮助。

Brian: Yes, and the article points out that AMD clusters with Mellanox network interfaces can use RDMA for fast GPU-to-GPU communication.

Brian：是的。文章还指出，配备 Mellanox 网卡的 AMD 集群可以用 RDMA 实现高速 GPU 间通信。

That combination is available from major cloud providers such as Azure and Oracle.

Azure 和 Oracle 等大型云服务商都提供这种配置。

Ava: What about observability?

Ava：可观测性方面呢？

The article seems to treat it as a major feature, not an afterthought.

文章似乎把它当作主要功能，而不是事后补上的。

Brian: That’s fair.

Brian：这么说没错。

Monarch has a client-accessible Distributed SQL Telemetry endpoint, an Admin API, and a Terminal UI for inspecting and managing live jobs.

Monarch 提供客户端可访问的分布式 SQL 遥测端点、管理 API 和终端界面，可检查和管理正在运行的作业。

It also supports OpenTelemetry for metrics, logs, and visualization on Kubernetes.

它还支持通过 OpenTelemetry 在 Kubernetes 上采集指标、日志并进行可视化。

Ava: Can it fit into existing DevOps tools?

Ava：它能接入现有的 DevOps 工具吗？

Brian: Yes.

Brian：能。

The OpenTelemetry integration can connect with Prometheus, Loki, Grafana, and other common open-source tooling.

OpenTelemetry 集成可连接 Prometheus、Loki、Grafana 等常见开源工具。

There’s also a per-job open-source dashboard in beta for visualizing and debugging distributed jobs without external tools.

另有一个测试版开源仪表盘，可按作业可视化和调试分布式任务，无须外部工具。

Ava: The installation story changed too, right?

Ava：安装方式也改进了，对吧？

Brian: A lot.

Brian：改进很大。

The pip wheel became one hundred times smaller, and startup became eight times faster.

pip wheel 的体积缩小了 100 倍，启动速度提高了 8 倍。

The project removed libpython linking requirements.

项目移除了链接 libpython 的要求。

As of version zero point two, torchmonarch no longer pulls in torch as a pip dependency, which avoids version conflicts.

从 0.2 版起，torchmonarch 不再将 torch 作为 pip 依赖引入，从而避免版本冲突。

Ava: And uv support is built in?

Ava：还内置了 uv 支持？

Brian: Yes. Monarch works out of the box with uv, the fast Python package manager.

Brian：对。Monarch 开箱即用地支持高速 Python 包管理器 uv。

The example flow is simple: clone the repository, change into the directory, and run the example with uv.

示例流程很简单：克隆仓库，进入目录，再用 uv 运行示例。

Packaging was consolidated under one torchmonarch name, with PEP four forty pre-release versions for nightlies.

软件包统一使用 torchmonarch 这一名称，每日构建版采用符合 PEP 440 的预发布版本号。

ARM sixty-four Linux builds were added in version zero point four.

0.4 版增加了 ARM64 Linux 构建。

Ava: That sounds much friendlier for interactive work too.

Ava：听起来交互式开发也方便多了。

Brian: The article mentions improved interactive SPMD support.

Brian：文章提到了交互式 SPMD 支持的改进。

SPMD means Single Program, Multiple Data.

SPMD 指单程序多数据。

The goal is to make notebook-style development more practical for these jobs.

目标是让这类作业更适合在笔记本式环境中开发。

The RDMA File System also helps with convenient file syncing across hosts.

RDMA 文件系统也让跨主机同步文件更方便。

Ava: The collaborations section is interesting. What did SkyPilot add?

Ava：合作部分挺有意思。SkyPilot 带来了什么？

Brian: The SkyPilot integration lets users run Monarch workloads on any Kubernetes cluster or cloud with one command, without changing their Monarch code.

Brian：集成 SkyPilot 后，用户只需一条命令，就能在任意 Kubernetes 集群或云平台上运行 Monarch 工作负载，无需修改 Monarch 代码。

SkyPilot handles node provisioning, networking, and gang scheduling, so teams can focus on training logic.

SkyPilot 负责节点配置、网络和成组调度，让团队专注于训练逻辑。

Ava: And it connects through Monarch’s JobTraits API?

Ava：它是通过 Monarch 的 JobTraits API 接入的吗？

Brian: Yes.

Brian：对。

That API lets SkyPilot plug in as the job backend, without requiring separate operators on Kubernetes clusters.

这个 API 让 SkyPilot 能作为作业后端接入，无需在 Kubernetes 集群上另装 operator。

Ava: What about VeRL? That seems closer to distributed RL.

Ava：那 VeRL 呢？它似乎更接近分布式强化学习。

Brian: VeRL is an open-source framework for distributed RLHF post-training.

Brian：VeRL 是用于分布式 RLHF 后训练的开源框架。

The teams built a Monarch backend for VeRL’s single-controller architecture.

双方团队为 VeRL 的单控制器架构构建了 Monarch 后端。

It includes resource pool abstractions based on the Monarch Jobs API, colocated multi-role workers, an RDMA transport for VeRL’s DataProto exchange pattern, and a vLLM server integration.

它包括基于 Monarch Jobs API 的资源池抽象、同机部署的多角色工作进程、用于 VeRL DataProto 数据交换的 RDMA 传输，以及 vLLM 服务器集成。

Ava: Did the training results match?

Ava：训练结果一致吗？

Brian: The article says VeRL’s PPO and GRPO training loops ran on Monarch through this backend, using hybrid-engine training mode.

Brian：文章称，VeRL 的 PPO 和 GRPO 训练循环通过这个后端在 Monarch 上以混合引擎模式运行。

The results were numerically identical, with no performance regression.

结果在数值上完全一致，性能也没有下降。

Ava: That sounds almost effortless. Was it?

Ava：听起来几乎不费力。真是这样吗？

Brian: Not quite.

Brian：也不尽然。

One important finding was that VeRL’s single-controller interface is cleanly abstracted, but Ray API usage appears throughout the broader codebase.

一个重要发现是：VeRL 的单控制器接口抽象得很清晰，但整个代码库中都用到了 Ray API。

So replacing the backend was more involved than the interface alone suggested.

因此，替换后端比单看接口所预想的更复杂。

The article says this is common in frameworks built on Ray, and the communities can keep collaborating on it.

文章说，这在基于 Ray 构建的框架中很常见，双方社区可以继续合作解决。

Ava: So Monarch improves the infrastructure layer, but existing framework assumptions can still make migration complicated.

Ava：也就是说，Monarch 改进了基础设施层，但现有框架的设计前提仍可能让迁移变复杂。

Brian: Exactly. That’s one of the limitations the article reveals.

Brian：没错。这是文章揭示的局限之一。

Monarch provides a coherent system and strong tools, but integrating it into a mature stack may require work beyond swapping one interface.

Monarch 提供了一套连贯的系统和强大的工具，但要将它融入成熟的技术栈，可能不只是替换一个接口那么简单。

Ava: Let’s close with the main takeaways. Give me the three-point version.

Ava：最后总结一下。用三点概括吧。

Brian: First, Monarch turns a supercomputer into a programmable Python system, so distributed development feels more like local development.

Brian：第一，Monarch 把超级计算机变成可用 Python 编程的系统，让分布式开发更像本地开发。

Second, RDMA file syncing, Distributed SQL Telemetry, the Jobs API, and the admin tools reduce iteration time.

第二，RDMA 文件同步、分布式 SQL 遥测、Jobs API 和管理工具缩短了迭代时间。

Third, since its launch, Monarch has expanded Kubernetes, networking, observability, installation, AMD support, and integrations with SkyPilot and VeRL.

第三，自发布以来，Monarch 扩展了 Kubernetes、网络、可观测性、安装方式和 AMD 支持，并集成了 SkyPilot 与 VeRL。

Ava: And the bigger direction is speed and simplicity for both people and agents.

Ava：更大的方向，是让人和智能体都能更快、更简单地开展工作。

Brian: Right. Monarch is presented as an API for your supercomputer.

Brian：对。Monarch 被定位为超级计算机的 API。

If you work on large-scale PyTorch training, it’s worth watching how this ecosystem develops.

如果你从事大规模 PyTorch 训练，值得关注这个生态如何发展。

Ava: That’s our episode. Thanks for listening, and we’ll talk to you next time.

Ava：本期节目就到这里。感谢收听，我们下次再聊。

## 术语

| Term | 释义 |
|---|---|
| Monarch | 面向 PyTorch 的分布式编程框架，用 Python API 控制大型集群 |
| RDMA | Remote Direct Memory Access，远程直接内存访问 |
| RDMA-Powered Remote File System | 通过 RDMA 向作业中所有主机分发文件的只读文件系统 |
| Distributed SQL Telemetry | 用分布式 SQL 引擎收集并查询运行时状态、日志和追踪信息 |
| Jobs API | 用于申请主机等资源并在其上运行多个作业的 API |
| Kubernetes | 用于部署和管理容器化工作负载的平台 |
| SLURM | 常用于 HPC 集群资源调度和作业管理的系统 |
| AWS EFA | Amazon Elastic Fabric Adapter，高性能网络适配器 |
| Unified RDMA API | 跨 InfiniBand、AWS EFA 和 ROCm 网络的硬件可移植 RDMA 接口 |
| ROCm | AMD GPU 的软件平台 |
| OpenTelemetry | 用于指标、日志和可视化的开放式可观测性标准 |
| SPMD | Single Program, Multiple Data，单程序多数据 |
| RLHF | Reinforcement Learning from Human Feedback，基于人类反馈的强化学习 |
| PPO | Proximal Policy Optimization，一种强化学习算法 |
| GRPO | VeRL 使用的另一种强化学习训练方法，文章未展开其全称 |

## 口语表达

| Phrase | 释义 |
|---|---|
| What if your laptop could control a thousand GPUs? | 如果你的笔记本能控制一千块 GPU 呢？ |
| That’s the basic idea. | 这就是基本思路。 |
| Let’s move to networking. | 我们继续看网络部分。 |
| That’s a good recap. | 这个总结得很好。 |
| What problem does that solve? | 它解决的是什么问题？ |
| It can query it directly. | 它可以直接查询它。 |
| That sounds much easier than... | 这听起来比……容易多了。 |
| Not quite. | 也不完全是。 |
| One important finding was that... | 一个重要发现是…… |
| Let’s close with the main takeaways. | 我们用主要要点来收尾。 |
