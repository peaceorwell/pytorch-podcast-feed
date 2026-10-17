# Why Agent Sessions Change the Rules of Inference

原文：[Session-Aware Agentic Inference with NVIDIA Dynamo - PyTorch](https://pytorch.org/blog/session-aware-agentic-inference-with-nvidia-dynamo/)

## 摘要

智能体在调用工具时可能暂停生成，但它的上下文仍占用 KV 缓存；并发会让这种占用迅速累积。NVIDIA Dynamo 用稳定的会话 ID 串联请求，使追踪回放、会话级调度和缓存感知路由成为可能。文章报告，在指定的 SWE-bench 配置下，会话感知调度比单独使用 KV 感知路由提高了约百分之十二到十六的吞吐量。共享缓存索引仍处于实验阶段，而 KvHint 等主动移动缓存的机制仍在开发中。

## 对话

Ava: An agent can use GPU memory while it's waiting for a tool. How can a paused conversation do that?

Ava：代理在等待工具返回时仍可能占用 GPU 显存。一个暂停的对话怎么会占用显存？

Brian: Its key-value, or KV, cache can stay in memory between model calls. With many agents running, those waiting contexts add up. That's why this matters for serving.

Brian：它的键值（KV）缓存可以在两次模型调用之间留在内存中。许多代理同时运行时，这些等待中的上下文就会累积起来。这就是它对推理服务很重要的原因。

Ava: So the server sees more than a stream of unrelated questions?

Ava：所以服务器看到的不只是一连串互不相关的问题？

Brian: Exactly. An agentic session is a chain of model calls and tool calls. A coding session may start with a huge prompt, then keep adding to that context on every turn.

Brian：正是如此。一个智能体会话由一连串模型调用和工具调用组成。一次编程会话可能以一个巨大的提示词开始，随后每一轮都继续向这份上下文添加内容。

Ava: And a single task can start subagents too.

Ava：而且一项任务也可以启动子代理。

Brian: Right. Those subagents may run in parallel. Meanwhile, the main agent spends much of its time waiting for tools. Its cached context may still occupy space.

Brian：对。这些子代理可能并行运行。与此同时，主代理大部分时间都在等待工具返回，而它缓存的上下文可能仍占着空间。

Ava: What happens at the start of a long session?

Ava：长会话开始时会发生什么？

Brian: The model does a prefill: it processes the input prompt and builds the KV cache it can reuse during generation. If that cache is evicted, a later call may need to process the context again.

Brian：模型会进行预填充（prefill）：处理输入提示词，并构建可在生成过程中复用的 KV 缓存。如果这份缓存被驱逐，后续调用可能需要重新处理这段上下文。

Ava: That sounds expensive when the context keeps growing.

Ava：如果上下文不断增长，听起来成本会很高。

Brian: It is the problem the article focuses on. The serving system has to keep useful context available across turns while making room for many sessions at once.

Brian：这正是文章关注的问题。服务系统必须在同时容纳许多会话的情况下，让有用的上下文在各轮次之间保持可用。

Ava: What does NVIDIA Dynamo add to see the whole session?

Ava：NVIDIA Dynamo 加入了什么，才能看到整个会话？

Brian: A stable session ID. Every large language model request in one agent's chain carries the same identifier. A child session can also carry its parent's ID.

Brian：一个稳定的会话 ID。同一个代理调用链中的每个大语言模型请求都携带相同的标识符。子会话也可以携带父会话的 ID。

Ava: So the ID is like a ticket that stays with the agent, even when it leaves to use a tool?

Ava：所以这个 ID 就像一张始终跟着代理的票，即使它去调用工具也是如此？

Brian: That's a useful analogy. Dynamo can connect the returning request to earlier turns. The session information is available to several optimizations, and each one is opt-in.

Brian：这个比喻很贴切。Dynamo 可以把返回的请求与先前的轮次关联起来。多种优化都可以使用会话信息，而且每种优化都需要选择启用。

Ava: Does an agent developer have to change their code?

Ava：代理开发者需要修改代码吗？

Brian: For Claude Code, Codex, and OpenCode, Dynamo recognizes identity headers they already send. Other harnesses can use plugins. A custom harness can send one canonical session header.

Brian：对于 Claude Code、Codex 和 OpenCode，Dynamo 能识别它们已经发送的身份标头。其他运行框架（harness）可以使用插件。自定义 harness 可以发送一个规范的会话标头。

Ava: By harness, you mean the software coordinating the model and its tools?

Ava：你说的 harness，是指协调模型及其工具的软件？

Brian: Yes. Dynamo maps those different headers into the same internal session information. Downstream components don't need to know which harness sent the request.

Brian：是的。Dynamo 会把这些不同的标头映射成同一种内部会话信息。下游组件无需知道请求是由哪个 harness 发出的。

Ava: All right. Once requests are connected, what can engineers measure?

Ava：明白了。请求关联起来后，工程师能测量什么？

Brian: They can collect session-linked traces. Each request gets a record after its response stream ends, with timing, output token count, finish reason, KV cache metrics, and tool-call names.

Brian：他们可以收集按会话关联的追踪数据。每个请求的响应流结束后都会生成一条记录，其中包括时间信息、输出 token 数量、结束原因、KV 缓存指标以及工具调用名称。

Ava: Does that record include the user's prompt?

Ava：这条记录包含用户的提示词吗？

Brian: By default, it doesn't store prompt, response, or tool-call content. Those can be enabled separately. The default replay information includes input length and hashes of sequence blocks.

Brian：默认情况下，它不存储提示词、响应或工具调用的内容。这些内容可以分别启用存储。默认的回放信息包含输入长度和序列块的哈希值。

Ava: Hashes of blocks? What are they for?

Ava：块的哈希值？它们有什么用？

Brian: They let replay preserve which prompt prefixes are shared across turns and sessions. Think of matching fingerprints. The replay can recognize a repeated piece of context without using its original tokens.

Brian：它们让回放能够保留不同轮次和会话之间共享了哪些提示词前缀。可以把它想成比对指纹。回放不需要使用原始 token，也能识别重复出现的上下文片段。

Ava: Wait, so one captured session can become a repeatable workload?

Ava：等等，也就是说，一个捕获下来的会话可以变成可重复运行的工作负载？

Brian: Yes. The capture records the requests the agent sent. Engineers can replay that schedule without asking the original agent to make its decisions again or calling its tools again.

Brian：对。捕获过程会记录代理发出的请求。工程师可以按原来的时序回放这些请求，无须让原代理再次做决策，也无须再次调用它的工具。

Ava: And there are two ways to replay it?

Ava：回放有两种方式？

Brian: Offline, AISimulate runs the request graph through a simulated scheduler, router, and KV cache. That lets engineers compare policies or cache sizes without spending GPU time.

Brian：在离线模式下，AISimulate 会让请求图经过模拟的调度器、路由器和 KV 缓存。这样工程师就能在不耗费 GPU 时间的情况下比较不同策略或缓存大小。

Ava: What about the second way?

Ava：第二种方式呢？

Brian: AIPerf replays the same graph against a real Dynamo endpoint on real GPUs. That's where they measure actual latency, throughput, and cache behavior.

Brian：AIPerf 会在真正的 GPU 上，把同一个请求图回放到真实的 Dynamo 端点。工程师由此测量实际延迟、吞吐量和缓存行为。

Ava: So the trace connects agent behavior to what the inference stack did. What's the routing problem?

Ava：所以追踪数据把代理行为和推理栈的运行情况联系了起来。路由方面有什么问题？

Brian: A request-level router places one request at a time. It asks which worker has the best cache overlap with this prompt. That's useful, but it can't see the whole agent's growing working set.

Brian：按请求路由的路由器一次只安排一个请求。它会判断哪个工作节点的缓存与这个提示词重叠最多。这很有用，但它看不到整个代理不断增长的工作集。

Ava: Working set means the context the session may need again?

Ava：工作集指的是会话之后可能还要用到的上下文？

Brian: Here, yes: the cached tokens assigned to active programs. If many agents reach the same step, their combined working sets can exceed GPU memory, then CPU memory.

Brian：在这里，是的：它指分配给活跃程序的已缓存 token。如果许多代理走到同一步，它们合起来的工作集可能先超过 GPU 内存，再超过 CPU 内存。

Ava: And then the server throws away context it'll soon need.

Ava：然后服务器就会丢弃很快还要用到的上下文。

Brian: Exactly. It evicts cache from live sessions and later does repeated prefill. The article calls that cache thrashing. The extra work can cut throughput.

Brian：正是如此。它会驱逐仍在运行的会话的缓存，之后又得重复进行预填充。文章把这叫作缓存抖动。额外的工作可能降低吞吐量。

Ava: Could the router slow agents down before that happens?

Ava：路由器能在这发生前让代理慢下来吗？

Brian: That's the idea behind session-aware admission control. Admission control decides when a program can proceed. Dynamo's strategy can apply backpressure when an agent reaches a tool boundary.

Brian：这就是会话感知准入控制背后的思路。准入控制决定程序何时可以继续运行。Dynamo 的策略可以在代理到达工具边界时施加背压。

Ava: Why choose the tool boundary?

Ava：为什么选择工具边界？

Brian: The current model turn can finish. Then the program can be paused logically before its next turn. There's no need to interrupt generation halfway through.

Brian：当前模型轮次可以先完成。然后，程序可以在下一轮开始前从逻辑上暂停。不需要在生成进行到一半时打断它。

Ava: Let's make that concrete. What does the scheduler track?

Ava：举个具体例子。调度器会跟踪什么？

Brian: It groups requests by session ID. Each program has two states: REASONING or ACTING, and ACTIVE or PAUSED. It enters ACTING at a tool boundary.

Brian：它按会话 ID 对请求分组。每个程序都有两组状态：REASONING 或 ACTING，以及 ACTIVE 或 PAUSED。程序会在工具边界进入 ACTING 状态。

Ava: What triggers a pause?

Ava：什么会触发暂停？

Brian: The scheduler estimates each worker's KV utilization from its assigned programs' token weights and the worker's cache capacity. At ninety-five percent, it pauses the smallest ACTING programs until utilization falls to eighty percent.

Brian：调度器根据分配给每个工作节点的程序的 token 权重，以及该节点的缓存容量，估算其 KV 利用率。当利用率达到 95% 时，它会暂停规模最小的 ACTING 程序，直到利用率降至 80%。

Ava: Does it resume them as soon as utilization drops a little?

Ava：利用率稍微降一点，就会立刻恢复它们吗？

Brian: It waits until utilization reaches eighty-five percent or less. That gap keeps pause and resume from flipping back and forth near the limit.

Brian：它会等到利用率降至 85% 或更低。这个区间可以防止暂停和恢复操作在阈值附近来回切换。

Ava: And if there's no ACTING program to pause?

Ava：如果没有处于 ACTING 状态的程序可暂停呢？

Brian: It marks the smallest REASONING program for a pause at its next tool boundary. When there's room again, paused programs are considered from smallest token count upward.

Brian：它会标记最小的 REASONING 程序，让它在下一个工具边界暂停。再次有空间时，会按 token 数量从少到多考虑恢复已暂停的程序。

Ava: Could one small session stay paused forever?

Ava：某个小会话会不会一直暂停下去？

Brian: The design includes a thirty-minute forced-resume cap to prevent indefinite starvation. It also briefly boosts the priority of resumed requests.

Brian：该设计设有 30 分钟的强制恢复时限，以防程序无限期得不到运行机会。它还会短暂提高恢复执行的请求的优先级。

Ava: What did the tests show?

Ava：测试结果如何？

Brian: In the article's SWE-bench setup, two TP4 MiniMax-M2 replicas ran on one node with eight H100 GPUs. Program-aware scheduling improved throughput by roughly twelve to sixteen percent over KV-aware routing alone.

Brian：在文章所用的 SWE-bench 测试配置中，一个配有八块 H100 GPU 的节点运行了两个 TP4 MiniMax-M2 副本。与单独使用 KV 感知路由相比，程序感知调度让吞吐量提高了约百分之十二到百分之十六。

Ava: So that number belongs to that setup, not every deployment.

Ava：所以这个数字只适用于那套配置，不能套用到所有部署。

Brian: Right. The article ties the gain to avoiding repeated prefill in that workload. It doesn't claim the same percentage for every model or concurrency level.

Brian：对。文章把这项提升归因于在该工作负载中避免了重复的 prefill，并没有声称所有模型或并发水平都能获得同样比例的提升。

Ava: Did they test the idea on agent training workloads too?

Ava：他们也在智能体训练工作负载上测试了这个想法吗？

Brian: Yes, on agentic reinforcement learning rollouts. At low concurrency, their strategy was roughly even with VERL's default Global LB, or global load balancing.

Brian：测试了，用的是智能体强化学习 rollout。在低并发下，他们的策略与 VERL 默认的 Global LB，也就是全局负载均衡，表现大致相当。

Ava: What changed when more agents ran together?

Ava：同时运行更多智能体后，情况有什么变化？

Brian: At medium concurrency in that test, model-token throughput was eleven to fourteen point six percent higher. At high concurrency, Global LB throughput dropped sharply while the session-aware strategy kept scaling.

Brian：在那次测试的中等并发水平下，模型 token 吞吐量高出百分之十一到百分之十四点六。到了高并发水平，Global LB 的吞吐量急剧下降，而会话感知策略仍能随并发增加而扩展。

Ava: That's the scheduling part. What happens when useful KV cache moves out of GPU memory?

Ava：这是调度方面的情况。如果有用的 KV 缓存移出了 GPU 内存，会发生什么？

Brian: The router needs to know where it went. Dynamo already considered cache on a worker's GPU and in its native CPU offload. The article adds an experimental index for an external shared store.

Brian：路由器需要知道它被移到了哪里。Dynamo 原本就会考虑工作节点 GPU 上的缓存和其原生的 CPU 卸载缓存。文章又增加了一个用于外部共享存储的实验性索引。

Ava: The shared store here is Mooncake?

Ava：这里的共享存储是 Mooncake 吗？

Brian: Yes. Mooncake Store publishes events when objects are stored or removed. Dynamo uses those events to maintain a shared-pool index, so it can account for reusable cache when placing requests.

Brian：是。Mooncake Store 会在对象存入或移除时发布事件。Dynamo 利用这些事件维护共享池索引，从而在分配请求时把可复用的缓存考虑进去。

Ava: Can it count a page as reusable if only part of that page is in the store?

Ava：如果一个页只有一部分在存储中，它能把这个页算作可复用吗？

Brian: No. Every physical object needed for the worker's layout must be present. Think of a kit with several parts: one missing part means the kit isn't ready.

Brian：不能。工作节点的布局所需的每个物理对象都必须在场。可以把它想成一套由多个零件组成的工具包：少了一个零件，整套工具包就还没备齐。

Ava: Here's the catch: couldn't the router count the same prefix twice, once on the GPU and once in Mooncake?

Ava：但有个问题：路由器会不会把同一个前缀算两次，一次算在 GPU 上，一次算在 Mooncake 里？

Brian: It avoids that. Shared-pool credit applies only beyond the device-resident prefix. The placement score then combines native cache overlap, shared-pool credit, and live worker load.

Brian：它会避免重复计算。共享池的计分只适用于设备驻留前缀之后的部分。然后，分配评分会综合原生缓存的重叠程度、共享池的计分以及工作节点的实时负载。

Ava: So far, the router finds cache and controls concurrency. What's next?

Ava：到目前为止，路由器负责寻找缓存和控制并发。接下来是什么？

Brian: The article looks toward programmatic KV cache management. Dynamo knows session lifecycles across workers. An inference engine such as vLLM or SGLang knows the exact cache state it holds.

Brian：文章接着探讨程序化 KV 缓存管理。Dynamo 了解跨工作节点的会话生命周期。vLLM 或 SGLang 这样的推理引擎则知道自己持有的准确缓存状态。

Ava: How do those two views work together?

Ava：这两种视角要怎么配合？

Brian: Dynamo would send a soft KvHint expressing what it wants. The engine would decide how to carry that out against its actual cache state. It can clip, delay, or ignore a hint.

Brian：Dynamo 会发送一个软性的 KvHint，表达它希望发生什么。引擎则会根据实际缓存状态决定如何执行；它可以限制、推迟或忽略这条提示。

Ava: What might a hint ask for?

Ava：提示可能会要求做什么？

Brian: Share could move a cached prefix between workers. Prefetch could bring KV toward faster memory before it's needed. Demote could move it toward a colder tier during a long tool call.

Brian：Share 可以在工作节点之间移动已缓存的前缀。Prefetch 可以在需要 KV 之前把它移向更快的内存。Demote 可以在漫长的工具调用期间把它移向更冷的存储层。

Ava: Are all those movements working today?

Ava：这些移动方式现在都能用了吗？

Brian: The article says Share has a working implementation in HiCache. Prefetch and Demote are designed to build on that machinery. The broader hint interface is still a proposal.

Brian：文章说，Share 在 HiCache 中已有可运行的实现。Prefetch 和 Demote 的设计会以这套机制为基础。更广泛的提示接口仍处于提案阶段。

Ava: What tells Dynamo which cached blocks belong to which session?

Ava：Dynamo 怎么知道哪些缓存块属于哪个会话？

Brian: Its Session-Prefix Indexer. It's a radix tree, a tree that groups shared prefixes. Its branches represent sessions or subagent sessions, and it tracks their KV block locations and lineage.

Brian：靠它的 Session-Prefix Indexer。这是一棵基数树，也就是一棵按共享前缀分组的树。树枝代表会话或子智能体会话，它还会跟踪这些会话的 KV 块位置和沿袭关系。

Ava: So it can see a shared beginning and the different paths agents take afterward.

Ava：所以它能看出共同的开头，以及智能体随后走出的不同路径。

Brian: Exactly. Engine cache events and router prefix hits keep that view updated. Dynamo is experimenting with policies that use it to decide which hints to send.

Brian：没错。引擎的缓存事件和路由器的前缀命中会持续更新这一视图。Dynamo 正在试验利用它来决定发送哪些提示的策略。

Ava: Give me the three-point recap before my own context gets evicted.

Ava：趁我自己的上下文还没被逐出，给我来个三点回顾。

Brian: First, a stable session ID connects agent turns. Second, traces and session-aware scheduling help measure work and reduce cache thrashing. Third, shared-cache indexing and proposed hints aim to keep useful context available across memory tiers.

Brian：第一，稳定的会话 ID 将智能体的各轮交互关联起来。第二，追踪数据和会话感知调度有助于衡量工作量并减少缓存反复失效。第三，共享缓存索引和拟议中的提示机制旨在让有用的上下文在不同内存层级之间保持可用。

Ava: Thanks for listening. We'll catch you next time.

Ava：感谢收听。下次再见。

## 术语

| Term | 释义 |
|---|---|
| KV cache | 键值缓存；保存模型已处理上下文的信息，供后续生成或请求复用。 |
| prefill | 预填充；模型处理输入提示并建立缓存的阶段。 |
| session ID | 会话标识；把同一智能体轨迹中的多次请求关联起来的稳定 ID。 |
| harness | 智能体运行框架；协调模型调用与工具调用的软件。 |
| session-linked traces | 会话关联追踪记录；将多次请求的性能信息连成完整轨迹。 |
| request-level router | 请求级路由器；逐个请求决定由哪个工作节点处理。 |
| working set | 工作集；运行中的会话可能再次使用的缓存上下文。 |
| cache thrashing | 缓存抖动；有用缓存反复被驱逐，之后又不得不重新计算。 |
| session-aware admission control | 会话感知准入控制；按会话状态和缓存容量决定何时让程序继续运行。 |
| shared-pool index | 共享缓存池索引；记录外部存储中哪些 KV 数据可供复用。 |
| KvHint | KV 缓存提示；由 Dynamo 向推理引擎表达缓存移动意图的软提示。 |
| Session-Prefix Indexer | 会话前缀索引器；关联共享前缀、会话分支及 KV 块位置的结构。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Wait, so | 等等，也就是说；用于确认刚听到的要点。 |
| That's a useful analogy. | 这个类比很贴切。 |
| Let's make that concrete. | 我们说得具体一点。 |
| What did the tests show? | 测试结果如何？ |
| Here's the catch: | 问题在这里；用于提出一个关键难点。 |
| What changed when more agents ran together? | 更多智能体同时运行时，情况有什么变化？ |
| How do those two views work together? | 这两种视角如何配合？ |
| Give me the three-point recap | 用三点帮我回顾一下。 |
