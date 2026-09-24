# Monarch: An API to Your Supercomputer

原文：[Monarch: an API to your supercomputer](https://pytorch.org/blog/monarch-an-api-to-your-supercomputer/)

## 摘要

本期介绍 PyTorch Monarch，一个用简单 Python API 把超算集群变成可编程整体的分布式框架。它通过 RDMA 文件系统、分布式 SQL Telemetry 和 Jobs API，加速代码同步、资源管理与调试，让大规模训练更像本地开发。文章还总结了 Monarch 自 2025 年发布以来在 Kubernetes、RDMA、可观测性、安装体验和 AMD ROCm 支持上的改进。Monarch 已与 SkyPilot、VeRL 和 AMD 等项目合作，但在 Ray 依赖较深的框架中替换后端仍然比较复杂。

## 对话

**Ava:** What if your laptop could control a thousand GPUs as easily as it runs a Python script? That’s the promise behind Monarch, and it matters because distributed training can become painfully slow to develop and debug.

**Brian:** Exactly. Monarch is a distributed programming framework for PyTorch. Its goal is to make a huge cluster feel like one coherent, directly controllable system, almost as if your laptop had thousands of GPUs attached.

**Ava:** So this is bigger than a job launcher?

**Brian:** Much bigger. A complete training system can live in one Python program. Monarch exposes simple primitives for hosts, procs, and actors. Then higher-level features, such as fault tolerance and orchestration, can be built as reusable libraries.

**Ava:** That sounds especially useful for distributed reinforcement learning, where many moving parts have to work together.

**Brian:** Right. The article starts with that pain. Huge clusters are hard to use, and distributed reinforcement learning makes the setup more complex. When you change something, the turnaround time can be very slow. Debugging is frustrating because the system is spread across many processes and machines.

**Ava:** Monarch tries to make that feel local?

**Brian:** Yes. The central idea is that the distributed system should feel local. Monarch is also designed for agentic usage. An agent can run development tasks on a dev machine, and Monarch can turn that dev machine into a supercomputer with consistent infrastructure abstractions.

**Ava:** When you say agentic usage, what can the agent actually do?

**Brian:** It can manage and debug running code, sync dependencies and data, launch new code, and provision more hosts, procs, and actors. The same model works across different deployment environments.

**Ava:** Let’s start with the file system. The article mentions an RDMA-Powered Remote File System.

**Brian:** The client exposes a read-only mounted filesystem. Monarch distributes those files to every host in the job through RDMA, or Remote Direct Memory Access. That makes it much faster to sync code, dependencies, and containers while you iterate.

**Ava:** So instead of repeatedly copying files through a slower path, the cluster sees a shared read-only view?

**Brian:** That’s the basic idea. The RDMA filesystem is built on Monarch RDMA buffers and PyFuse. The practical benefit is quick iteration: update your code or dependencies, and get them across the job quickly.

**Ava:** The second feature is Distributed SQL Telemetry. Why SQL?

**Brian:** Because SQL is a familiar way to inspect structured state, and agents already work well with SQL-based APIs. Monarch includes a lightweight distributed SQL engine. It collects live state information, pyspy traces, and logs from distributed processes and actors.

**Ava:** Does Monarch just collect the data, or can it query the running system directly?

**Brian:** It can query it directly. The team ran a DataFusion distributed SQL query engine in situ, meaning inside the running distributed environment. Each node writes live state into tables, and an agent can query those tables efficiently while debugging.

**Ava:** That sounds much easier than opening logs from dozens of machines.

**Brian:** Exactly. You can explore the state from a central point. The article presents this as a way to reduce debugging time, because the information is accessible through one client-facing SQL endpoint.

**Ava:** Then there’s the Jobs API. What problem does that solve?

**Brian:** It lets you provision resources, such as hosts, once, and run many jobs on them. You avoid paying the repeated allocation penalty every time a job restarts. Monarch supports Kubernetes and SLURM, and other schedulers can be connected by implementing a Monarch Job.

**Ava:** So the three pieces work together: sync quickly, inspect quickly, and restart quickly.

**Brian:** That’s a good recap. Monarch’s toolbox helps agents restart jobs fast, sync new code and data fast, and debug fast. The system feels local even though the work is distributed.

**Ava:** The article says Monarch launched at the PyTorch conference in October twenty twenty-five. What changed since then?

**Brian:** Kubernetes support became first-class. There’s now a dedicated open-source repository called monarch-kubernetes. It includes a MonarchMesh Custom Resource Definition, a reference KubeBuilder operator, and a hello-world demo.

**Ava:** What does just-in-time pod provisioning mean here?

**Brian:** Pods are allocated when they’re needed instead of being reserved upfront. That can improve cluster utilization. Monarch also added an external gateway, so out-of-cluster clients can connect to meshes running inside Kubernetes. That feature is landing in zero point five.

**Ava:** And there are versioned and nightly Docker containers?

**Brian:** Yes. They’re published to GHCR, the GitHub Container Registry, to support reproducible deployments. MonarchMesh labels can also enable scheduling through Kueue.

**Ava:** Let’s move to networking. RDMA appears everywhere in this project.

**Brian:** Monarch added several RDMA backends and a higher-level API. AWS EFA, or Elastic Fabric Adapter, is now supported by Monarch’s RDMABuffer. The article says it was validated at sixteen gigabits per second, ten times faster than TCP, with fourteen point five gigabytes transferred in seven point six seconds.

**Ava:** That’s a concrete result. Is the API tied to AWS hardware?

**Brian:** No. The Unified RDMA API is designed to be hardware-portable. It works across InfiniBand with mlx5, AWS EFA, and ROCm. You can write once and run on different fabrics, or fall back to Monarch actor messaging when RDMA isn’t available.

**Ava:** ROCm brings AMD GPUs into the picture.

**Brian:** Right. GPU-direct RDMA and RCCL collective communication now work on AMD GPUs through ROCm with Mellanox interfaces. AMD also validated Monarch on ROCm for MI300, MI325, and MI355 clusters, with SLURM-based orchestration.

**Ava:** That’s useful for teams that don’t run NVIDIA hardware.

**Brian:** Yes, and the article points out that AMD clusters with Mellanox network interfaces can use RDMA for fast GPU-to-GPU communication. That combination is available from major cloud providers such as Azure and Oracle.

**Ava:** What about observability? The article seems to treat it as a major feature, not an afterthought.

**Brian:** That’s fair. Monarch has a client-accessible Distributed SQL Telemetry endpoint, an Admin API, and a Terminal UI for inspecting and managing live jobs. It also supports OpenTelemetry for metrics, logs, and visualization on Kubernetes.

**Ava:** Can it fit into existing DevOps tools?

**Brian:** Yes. The OpenTelemetry integration can connect with Prometheus, Loki, Grafana, and other common open-source tooling. There’s also a per-job open-source dashboard in beta for visualizing and debugging distributed jobs without external tools.

**Ava:** The installation story changed too, right?

**Brian:** A lot. The pip wheel became one hundred times smaller, and startup became eight times faster. The project removed libpython linking requirements. As of version zero point two, torchmonarch no longer pulls in torch as a pip dependency, which avoids version conflicts.

**Ava:** And uv support is built in?

**Brian:** Yes. Monarch works out of the box with uv, the fast Python package manager. The example flow is simple: clone the repository, change into the directory, and run the example with uv. Packaging was consolidated under one torchmonarch name, with PEP four forty pre-release versions for nightlies. ARM sixty-four Linux builds were added in version zero point four.

**Ava:** That sounds much friendlier for interactive work too.

**Brian:** The article mentions improved interactive SPMD support. SPMD means Single Program, Multiple Data. The goal is to make notebook-style development more practical for these jobs. The RDMA File System also helps with convenient file syncing across hosts.

**Ava:** The collaborations section is interesting. What did SkyPilot add?

**Brian:** The SkyPilot integration lets users run Monarch workloads on any Kubernetes cluster or cloud with one command, without changing their Monarch code. SkyPilot handles node provisioning, networking, and gang scheduling, so teams can focus on training logic.

**Ava:** And it connects through Monarch’s JobTraits API?

**Brian:** Yes. That API lets SkyPilot plug in as the job backend, without requiring separate operators on Kubernetes clusters.

**Ava:** What about VeRL? That seems closer to distributed RL.

**Brian:** VeRL is an open-source framework for distributed RLHF post-training. The teams built a Monarch backend for VeRL’s single-controller architecture. It includes resource pool abstractions based on the Monarch Jobs API, colocated multi-role workers, an RDMA transport for VeRL’s DataProto exchange pattern, and a vLLM server integration.

**Ava:** Did the training results match?

**Brian:** The article says VeRL’s PPO and GRPO training loops ran on Monarch through this backend, using hybrid-engine training mode. The results were numerically identical, with no performance regression.

**Ava:** That sounds almost effortless. Was it?

**Brian:** Not quite. One important finding was that VeRL’s single-controller interface is cleanly abstracted, but Ray API usage appears throughout the broader codebase. So replacing the backend was more involved than the interface alone suggested. The article says this is common in frameworks built on Ray, and the communities can keep collaborating on it.

**Ava:** So Monarch improves the infrastructure layer, but existing framework assumptions can still make migration complicated.

**Brian:** Exactly. That’s one of the limitations the article reveals. Monarch provides a coherent system and strong tools, but integrating it into a mature stack may require work beyond swapping one interface.

**Ava:** Let’s close with the main takeaways. Give me the three-point version.

**Brian:** First, Monarch turns a supercomputer into a programmable Python system, so distributed development feels more like local development. Second, RDMA file syncing, Distributed SQL Telemetry, the Jobs API, and the admin tools reduce iteration time. Third, since its launch, Monarch has expanded Kubernetes, networking, observability, installation, AMD support, and integrations with SkyPilot and VeRL.

**Ava:** And the bigger direction is speed and simplicity for both people and agents.

**Brian:** Right. Monarch is presented as an API for your supercomputer. If you work on large-scale PyTorch training, it’s worth watching how this ecosystem develops.

**Ava:** That’s our episode. Thanks for listening, and we’ll talk to you next time.

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
