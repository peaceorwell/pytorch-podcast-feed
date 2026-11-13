# KVCache Meetup Korea 2026: Where Cache Reuse Meets Data Movement

原文：[KVCache Meetup Korea 2026: Event Recap - PyTorch](https://pytorch.org/blog/kvcache-meetup-korea-2026-event-recap/)

## 摘要

2026 年 9 月 30 日的 KVCache Meetup Korea 涵盖 CXL 内存扩展、GPU 与 HBM 利用率、RNGD 请求调度和 LMCache MP Mode。SK hynix 的实验使用两块 H100 GPU、Llama 3.1 8B 和 vLLM：预填充阶段主要投影操作的 GPU 计算利用率较高，解码阶段的 HBM 带宽利用率较高。FuriosaAI 展示的测试使用 12,370 token 的提示词，每次处理一个请求，每种条件测量三次；读取主机缓存并生成首个 token 约需 92 ms，重算缓存约需 900 ms。基于 72 份问卷的反馈还提到，有人希望增设同领域开发者交流的专门环节。

## 对话

Ava: KV Cache reuse can cut prefill work, but moving that cache also costs time. KVCache Meetup Korea 2026 put that trade-off beside GPU bottlenecks, scheduling, and cache management.

Ava：复用 KV Cache 可以减少 prefill 阶段的计算量，但搬运缓存也需要时间。KVCache Meetup Korea 2026 将这一权衡与 GPU 瓶颈、调度和缓存管理放在一起讨论。

Brian: PyTorchKR and XCENA co-hosted the event on Sep 30, 2026, with sponsorship from Tensormesh. Speakers came from XCENA, SK hynix, FuriosaAI, and LMCache.

Brian：PyTorchKR 和 XCENA 于 2026 年 9 月 30 日联合举办了这场活动，Tensormesh 提供赞助。演讲者来自 XCENA、SK hynix、FuriosaAI 和 LMCache。

Ava: Let's start with CXL. So capacity is only part of it. The inference engine still needs to get useful cache to the GPU.

Ava：我们先从 CXL 说起。容量只是其中一方面，推理引擎仍需把有用的缓存送到 GPU。

Brian: Jooho Lee explained that CXL expands the memory space CPUs can access with load and store instructions. He described a pool shared by multiple hosts. Frequently used cache stays in DRAM, while less frequently used cache goes on SSDs. XCENA's open-source storage engine, Maru, connects the inference engine to those layers. The path and cost of moving cache to the GPU matter too.

Brian：Jooho Lee 解释说，CXL 扩展了 CPU 可以通过加载和存储指令访问的内存空间。他描述了一个由多个主机共享的内存池。经常使用的缓存留在 DRAM，使用频率较低的缓存则存放在 SSD 上。XCENA 的开源存储引擎 Maru 将推理引擎与这些存储层连接起来。将缓存送到 GPU 所经过的路径及其代价也很重要。

Ava: The talk also covered splitting mixture-of-experts work between a GPU and MX1. Some expert weights sit in memory near MX1 and are processed there. The design transfers activations and results instead of full weights.

Ava：演讲还讨论了如何让 GPU 和 MX1 分担混合专家模型的工作。部分专家权重存放在靠近 MX1 的内存中，并由 MX1 在那里处理。这种设计传输的是激活值和结果，而不是完整的权重。

Brian: Sejin Oh looked at GPU compute and HBM bandwidth. The experiments used two H100 GPUs, Llama 3.1 8B, and vLLM. GPU compute use was high during the main projection operations in prefill. During decode, HBM bandwidth use was high.

Brian：Sejin Oh 研究了 GPU 算力和 HBM 带宽。实验使用了两块 H100 GPU、Llama 3.1 8B 和 vLLM。在 prefill 阶段的主要投影运算中，GPU 算力利用率很高；在 decode 阶段，HBM 带宽利用率很高。

Ava: That helps explain why reusing KV Cache can change the bottleneck. If prefill needs less computation, memory bandwidth can become the limit.

Ava：这有助于解释为什么复用 KV Cache 会改变瓶颈。如果 prefill 所需的计算减少，内存带宽就可能成为限制因素。

Brian: Whether a service needs more memory capacity or bandwidth depends on input and output lengths, batch size, and other operating conditions. The talk also separated TTFT, time to the first output token, from TPOT, time per output token.

Brian：服务更需要内存容量还是带宽，取决于输入和输出长度、批大小及其他运行条件。演讲还区分了 TTFT，即首个输出 token 的生成时间，和 TPOT，即每个输出 token 的生成时间。

Ava: Donghun Lee's FuriosaAI talk moved from resource limits to request scheduling on RNGD. He followed requests from arrival to execution.

Ava：Donghun Lee 在 FuriosaAI 的演讲中，从资源限制转向了 RNGD 上的请求调度。他追踪了请求从到达到执行的过程。

Brian: He covered PagedAttention for KV Cache blocks, Continuous batching for active requests, and Chunked prefill for smaller prompt segments. He also introduced buckets, which are tensor shapes compiled before execution.

Brian：他介绍了用于管理 KV Cache 块的 PagedAttention、用于处理活跃请求的 Continuous batching，以及将提示词分成较小片段的 Chunked prefill。他还介绍了 bucket，也就是在执行前编译好的张量形状。

Ava: Combining requests can raise throughput, but a larger batch may need a larger bucket and add latency. Padding in a fixed shape has a cost too.

Ava：合并请求可以提高吞吐量，但更大的批次可能需要更大的 bucket，并增加延迟。固定形状中的填充也有代价。

Brian: Furiosa-LLM measures execution cost on the actual hardware. It merges two batches when their combined cost is lower than processing them separately, accounting for each request's context length and the cost of padding.

Brian：Furiosa-LLM 会在实际硬件上测量执行代价。如果合并两个批次的代价低于分别处理的代价，它就会将两者合并；计算时会考虑每个请求的上下文长度和填充代价。

Ava: What about the cost of retrieving cache instead of recomputing it? The talk compared those two paths.

Ava：那么，读取缓存与重新计算缓存相比，代价如何？演讲比较了这两条路径。

Brian: The slides used a 12,370-token prompt, one request at a time, and three measurements for each condition. Retrieving the host cache and generating the first token took about 92 ms. Recomputing the cache took about 900 ms. Under those test conditions, retrieving the host cache was faster.

Brian：幻灯片中的测试使用了长度为 12,370 个 token 的提示词，每次处理一个请求，每种条件测量三次。读取主机端缓存并生成首个 token 大约需要 92 ms，重新计算缓存大约需要 900 ms。在这些测试条件下，读取主机端缓存更快。

Ava: The benefit depends on how much data moves and the available bandwidth. The talk described overlapping cache transfer with computation at each layer. Experimental features included a network-shared cache and running prefill and decode on separate resources.

Ava：收益取决于需要传输的数据量和可用带宽。演讲描述了如何在每一层让缓存传输与计算重叠进行。实验性功能包括网络共享缓存，以及让 prefill 和 decode 在不同资源上运行。

Brian: Baoloong Mao's LMCache talk focused on MP Mode. It moves cache management out of the inference engine and into a separate process. That separates inference computation from cache I/O to reduce competition for resources.

Brian：Baoloong Mao 关于 LMCache 的演讲重点介绍了 MP Mode。它将缓存管理从推理引擎移到独立进程中，使推理计算与缓存 I/O 分离，以减少资源争用。

Ava: The processes still need to coordinate. They exchange control information such as block IDs, while CUDA IPC lets them share GPU memory.

Ava：这些进程仍然需要协调。它们交换块 ID 等控制信息，同时通过 CUDA IPC 共享 GPU 内存。

Brian: Within a node, they use a shared CPU memory pool to reduce duplicate storage. MP Mode also reads and loads data from several storage layers in parallel.

Brian：在单个节点内，它们使用共享的 CPU 内存池来减少重复存储。MP Mode 还会并行地从多个存储层读取和加载数据。

Ava: The operations side matters as well. The talk covered Kubernetes deployment, cache hit rates, cache lifetimes, and isolating failures between inference and cache processes.

Ava：运维方面同样重要。演讲涉及 Kubernetes 部署、缓存命中率、缓存生命周期，以及如何隔离推理进程与缓存进程的故障。

Brian: In the Q&A, attendees asked about separating prefill and decode compared with mixed batching, which processes them together. They also asked about LMCache alongside vLLM's built-in cache features and plans to extend MP Mode across nodes.

Brian：在问答环节，参会者询问了将 prefill 和 decode 分开运行，与通过混合批处理一起处理两者相比有何不同。他们还询问了 LMCache 如何与 vLLM 的内置缓存功能配合，以及将 MP Mode 扩展到跨节点运行的计划。

Ava: The event included time to connect. At registration, attendees could choose up to two interest stickers from Model, Serving, Infra, Hardware, and Memory. They later gathered in small groups and examined XCENA's CXL server at the venue.

Ava：活动也安排了交流时间。签到时，参会者可以从 Model、Serving、Infra、Hardware 和 Memory 中选择最多两张兴趣贴纸。之后，他们分成小组交流，并在现场查看了 XCENA 的 CXL 服务器。

Brian: The feedback summary was based on 72 survey responses. In a question that allowed multiple selections, 48 respondents chose the program as a positive point, or 66.7%. The venue and facilities received 30 selections, or 41.7%.

Brian：反馈汇总基于 72 份问卷答复。在一道可多选的问题中，48 位受访者将活动内容选为优点，占 66.7%；场地和设施获得了 30 次选择，占 41.7%。

Ava: Respondents raised microphone quality and volume, screen visibility from the back rows, and limited space. Some wanted interpretation or translation for the English talks. Others asked for a dedicated session where developers in the same field could talk.

Ava：受访者提到了麦克风音质和音量、后排的屏幕可见度，以及场地空间有限的问题。一些人希望英语演讲能提供口译或翻译，另一些人则希望设置专门的交流环节，让同一领域的开发者能够交谈。

Brian: For future events, respondents asked for real production examples, vLLM and SGLang internals, NPU compilers, recent inference optimization techniques, and demos. Across these talks, cache capacity, transfer cost, compute, bandwidth, and scheduling all shaped the discussion of LLM serving. Thanks for listening.

Brian：对于未来的活动，受访者希望看到真实的生产环境案例、vLLM 和 SGLang 的内部机制、NPU 编译器、近期的推理优化技术，以及演示。纵观这些演讲，缓存容量、传输代价、算力、带宽和调度共同构成了 LLM 服务的讨论。感谢收听。

## 术语

| Term | 释义 |
|---|---|
| KV Cache | 推理时保存可复用注意力数据的缓存。 |
| CXL | Compute Express Link；文中用于扩展 CPU 可通过加载和存储指令访问的内存空间。 |
| Maru | XCENA 开源的 KV Cache 存储引擎，连接推理引擎与存储层。 |
| HBM | 高带宽内存；文中观察到解码阶段的 HBM 带宽利用率较高。 |
| prefill | 处理输入提示词的阶段。 |
| decode | 生成输出 token 的阶段。 |
| TTFT | 生成首个输出 token 所需的时间。 |
| TPOT | 生成每个输出 token 所需的时间。 |
| PagedAttention | 文中用于管理 KV Cache 块的机制。 |
| Continuous batching | 文中用于处理活跃请求的批处理方式。 |
| Chunked prefill | 把提示词处理拆成较小片段的方式。 |
| buckets | 执行前已编译的张量形状。 |
| MP Mode | LMCache 将缓存管理移至独立进程的模式。 |
| CUDA IPC | 让进程共享 GPU 内存的进程间通信机制。 |
| mixed batching | 在同一批处理中一起处理 prefill 和 decode 工作。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let's start with | 我们先从……谈起。 |
| So capacity is only part of it. | 所以容量只是其中一部分。 |
| That helps explain why | 这有助于解释为什么…… |
| on the actual hardware | 在实际硬件上。 |
| one request at a time | 每次处理一个请求。 |
| Under those test conditions | 在那些测试条件下。 |
| The operations side matters as well. | 运维方面也很重要。 |

## 中文内容总结

2026 年 9 月 30 日，PyTorchKR 与 XCENA 在韩国合办 KVCache Meetup，Tensormesh 提供赞助。XCENA、SK hynix、FuriosaAI 和 LMCache 的讲者讨论了大语言模型推理中的 KV Cache 容量、数据传输、GPU 资源利用和请求调度。活动还安排了问答、硬件展示及交流。

### 缓存容量与数据移动

XCENA 介绍了利用 CXL 扩展 CPU 可访问内存、让多台主机共享内存池的方案。其开源存储引擎 Maru 连接推理引擎与由 DRAM 和 SSD 组成的缓存层，将常用数据留在 DRAM。讲者强调，缓存能存多少只是问题之一，将数据送回 GPU 的路径和代价也影响推理效率。针对混合专家模型，报告还介绍了让 MX1 就近处理部分专家权重、只传输激活值和计算结果的设计。

### GPU 计算与 HBM 带宽

SK hynix 结合 Roofline 分析和运行测量，比较了预填充与解码阶段的资源使用。实验采用两块 H100 GPU、Llama 3.1 8B 和 vLLM：预填充的主要投影运算具有较高 GPU 计算利用率，解码则大量使用 HBM 带宽。复用 KV Cache 可减少预填充计算，但之后内存带宽可能成为瓶颈。容量和带宽孰更紧张，取决于输入、输出长度及批大小等运行条件。

### RNGD 上的请求调度

FuriosaAI 介绍了面向 RNGD 的 Furiosa-LLM 推理引擎，包括 PagedAttention、连续批处理、分块预填充和预编译张量形状。合并请求可提高吞吐量，却可能因更大的张量形状和填充增加延迟；引擎依据实际硬件执行成本，判断合并是否比单独处理划算。一次使用 12,370 token 提示词、每次处理一个请求且每种条件测量三次的测试中，从主机取回缓存并生成首 token 约需 92 毫秒，重新计算约需 900 毫秒。这个结果受传输量和可用带宽约束。

### 独立缓存管理进程

LMCache 介绍了将缓存管理移出推理引擎的 MP Mode。它通过 CUDA IPC 共享 GPU 内存，进程间交换块 ID 等控制信息，并使用节点内共享 CPU 内存池减少重复存储；多个存储层可并行读取和加载。报告还涉及 Kubernetes 部署、缓存命中率与生命周期监控、进程间故障隔离及外部存储插件。跨节点扩展 MP Mode 仍是讨论中的计划。

### 活动反馈

会后问答涉及预填充与解码分离、混合批处理，以及 LMCache 与 vLLM 内置缓存功能的分工。72 份问卷中，48 人选择活动内容为优点，占 66.7%；30 人选择场地与设施，占 41.7%，该题允许多选。反馈也提到话筒音量、后排屏幕可见度、空间和英文报告的翻译需求，并希望未来增加真实生产案例、推理框架内部机制与演示。
