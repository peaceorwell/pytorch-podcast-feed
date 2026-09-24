# Flight Recorder: A New Lens for Understanding NCCL Watchdog Timeouts

原文：[Flight Recorder: A New Lens for Understanding NCCL Watchdog Timeouts](https://pytorch.org/blog/flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/)

## 摘要

本期解释 PyTorch 中 NCCL watchdog timeout 的成因，以及为什么这个错误特别难排查。节目梳理了 CPU 侧卡顿或跨 rank 分歧、GPU kernel hang、collective 参数错误、网络或硬件问题四大类根因。重点介绍 Flight Recorder：它在每个 rank 上记录 collective 的类型、状态、数据类型、大小和调用栈，并在超时后汇总分析。最后讨论 Meta 的案例、可视化方法，以及 Flight Recorder 未来对 TorchComm、CUDA graphs 和其他后端的支持计划。

## 对话

**Ava:** Your large-scale training job runs for hours, then one line appears: NCCL watchdog timeout. The message tells you almost nothing, and the whole job is stuck. Why should engineers care about Flight Recorder?

**Brian:** Because Flight Recorder can turn that vague timeout into a cross-rank timeline. It helps show which collective diverged, which rank went missing, and where the original call came from.

**Ava:** Let’s start with the basics. What exactly is a collective in PyTorch?

**Brian:** In distributed training, ranks need to synchronize or exchange results. A collective is an operation that involves all ranks in a process group. For example, all_reduce sums a tensor across the group and writes the result back to that tensor.

**Ava:** And different training systems use different collectives?

**Brian:** Right. DDP commonly uses all-reduce. FSDP, which means Fully Sharded Data Parallel, uses all-gather and reduce-scatter. TorchRec often uses all-to-all.

**Ava:** Where does the call go when I write dist.all_reduce?

**Brian:** It passes through the C++ PyTorch dispatcher and a PyBind layer, then reaches c10d. The c10d layer calls a communication backend, such as NCCL for GPU communication or Gloo for CPU communication.

**Ava:** Why have c10d in the middle? Why not call NCCL directly?

**Brian:** PyTorch needs control information before handing work to the communication library. One important feature is the c10d watchdog. This post focuses on NCCL, but the watchdog mechanism can monitor other distributed backends too.

**Ava:** So what does the NCCL watchdog actually watch?

**Brian:** NCCL collectives are scheduled by the CPU and run asynchronously on the GPU. NCCL itself doesn’t provide built-in error checking for a misused collective. If ranks call different collectives, pass invalid arguments, or otherwise misuse the API, the GPU operation can hang forever.

**Ava:** And PyTorch catches that with a Work object?

**Brian:** Yes. The CPU-side Work object wraps the NCCL call with two CUDA events, one before and one after. The watchdog thread periodically checks whether the collective completed on the GPU within a user-defined timeout. The default is ten minutes.

**Ava:** When the limit is exceeded, it throws the famous error.

**Brian:** Exactly. But the name is a little misleading. The error is raised by PyTorch’s NCCL watchdog, not by the NCCL library itself. NCCL appears in the name because the timed-out collective was launched through the NCCL backend.

**Ava:** Why is this error so hard to debug? The log has a sequence number and an operation type, but that doesn’t seem enough.

**Brian:** There are two major reasons. First, it’s a catch-all error. Almost anything that makes a rank wait indefinitely can end in the same timeout: a CPU-side hang, a GPU hang, a CUDA deadlock, invalid collective arguments, or a network problem.

**Ava:** So the timeout doesn’t identify the layer that failed.

**Brian:** Right. Second, the error is raised from the watchdog thread. To avoid that thread hanging too, the log contains limited metadata. The stack trace is from the watchdog thread, not from the main PyTorch thread that scheduled the collective.

**Ava:** That’s a pretty important distinction.

**Brian:** It is. Also, the rank that reports the timeout is rarely the rank that caused it. And the collective shown in the error may simply be the one where the problem became visible, not the original cause.

**Ava:** So rerunning with CUDA_LAUNCH_BLOCKING can take hours and still be frustrating.

**Brian:** Exactly. The article says debugging without Flight Recorder can take hours or longer, often requiring another run with extra debugging flags.

**Ava:** Let’s talk about how collectives normally execute.

**Brian:** The CPU schedules a sequence of compute kernels and NCCL collectives onto the GPU. Later, at a CPU-GPU synchronization point, the CPU waits for the GPU to finish. There are two important synchronization types.

**Ava:** The first is between GPUs?

**Brian:** Yes. Inter-rank GPU-GPU synchronization means every GPU in the process group synchronizes before continuing. Barrier collectives, such as all_reduce, are common sources of watchdog timeouts.

**Ava:** And the second is one rank’s CPU waiting for its own GPU.

**Brian:** Exactly. That can be explicit, such as torch.cuda.synchronize, or implicit, such as moving a tensor between CPU and GPU.

**Ava:** The article says a collective can time out in two broad ways.

**Brian:** Either the collective kernel itself runs too long or hangs, or the ranks become desynchronized. Their collective metadata or state differs when the timeout occurs.

**Ava:** Which one is more common in practice?

**Brian:** Across Meta’s fleet, almost all observed timeouts came from collective desynchronization, not simple slowness or a collective kernel hang. That means increasing the timeout usually won’t fix the issue. You have to resolve the desync.

**Ava:** What causes that desync?

**Brian:** The article groups causes into four categories: CPU-side issues, GPU compute kernel hangs, misconfigured collective arguments, and network or hardware issues.

**Ava:** Let’s take CPU-side issues first.

**Brian:** For a process group, every rank must execute the same, or complementary, collectives in exactly the same order. If one rank schedules nothing, or schedules a different collective, another rank can wait forever and eventually time out.

**Ava:** What if the CPU is simply slow?

**Brian:** Suppose data loading, checkpointing, or PT2 compilation makes some ranks take longer than the watchdog threshold. Those ranks don’t schedule the next collective, while another rank does. The rank that did schedule it may be the one that reports the timeout.

**Ava:** PT2 compilation sounds especially tricky.

**Brian:** It is data-dependent. Compiler cache behavior and dynamic-shape recompilation can make compilation times differ across ranks. If that difference exceeds the watchdog threshold, a timeout can result. This problem motivated PT2 compiler collectives.

**Ava:** And CPU execution divergence means the ranks follow different code paths.

**Brian:** Yes. Data-dependent conditionals can make ranks schedule different collectives or skip one entirely. Data imbalance is one example. If one rank runs out of data earlier, it may leave the training loop one iteration sooner.

**Ava:** Error handling can create the same problem?

**Brian:** Absolutely. An exception handler is rank-specific logic. It might get stuck on the CPU, call another GPU synchronization during teardown, or swallow the exception and continue training. Any of those can leave ranks waiting in incompatible states.

**Ava:** There’s also an edge case involving multiple process groups.

**Brian:** Right. In N-D parallelism, such as FSDP, different process groups may schedule collectives back-to-back on the same GPU. Without proper synchronization, GPU communication execution order can differ across ranks. NCCL 2.26 introduces NCCL_LAUNCH_ORDER_IMPLICIT to enforce the same order as scheduling.

**Ava:** What about a real GPU compute hang?

**Brian:** CUDA execution is sequential within a stream. If a compute kernel hangs, the GPU can’t reach a scheduled collective. The CPU then blocks at a later synchronization point. Depending on the timing, the symptoms can look like a missing collective or a collective state mismatch.

**Ava:** Next is invalid collective arguments.

**Brian:** Usually, input and output tensor types and shapes must agree across ranks. Some operations have global requirements too. For all_to_all_single, the input and output size splits define the communication topology.

**Ava:** So if rank X expects more data from rank Y than Y sends, a receive can wait forever.

**Brian:** Exactly. PyTorch can’t fully verify those NCCL assumptions. An invalid split can leave ncclRecv blocked forever, which later appears as a watchdog timeout.

**Ava:** And the fourth category is the infrastructure layer.

**Brian:** Network or hardware issues account for roughly twenty to thirty percent of the timeouts observed in the article. Transient link or port flaps are common examples. Faulty GPU hardware can also cause hangs across multiple unrelated jobs, often with repeated failures or hardware signals such as XIDs.

**Ava:** Now let’s get to Flight Recorder. What is it recording?

**Brian:** It’s a per-rank, CPU-side ring buffer shared across all process groups. For each collective launch, it records the type, state, input and output data types, input and output sizes, and the CPU call stack. Python and C++ stacks can both be recorded.

**Ava:** What states does a collective move through?

**Brian:** Four states: not scheduled, also called missing; scheduled from the CPU; started on the GPU; and completed on the GPU. Each collective also gets a monotonically increasing sequence ID within each process group.

**Ava:** How does the data get dumped when the job is already falling apart?

**Brian:** If TORCH_NCCL_DUMP_ON_TIMEOUT is set, the watchdog triggers an immediate dump to storage. Users can also retrieve records through a Python API, write to a pipe file, or use an HTTP request.

**Ava:** But what if the rank that’s hanging can’t dump its own records?

**Brian:** That was a key design problem. Flight Recorder uses a side TCP/IP channel through TCPStore. One rank broadcasts a timeout signal, and a dedicated monitor thread on each rank receives it and triggers the dump.

**Ava:** How do they avoid several process groups dumping at once?

**Brian:** Only the monitor thread for the default process group checks signals and starts the dump. Other monitor threads briefly sleep, giving the main dump time to finish. The design is best effort because the system is fragile during a timeout, but inside Meta it achieved a near one-hundred-percent full dump rate.

**Ava:** Why analyze after the timeout instead of coordinating live?

**Brian:** The system is already fragmented, and a single dead rank can break coordination. Waiting live would also waste training resources. So the design first writes raw records, then aggregates and analyzes them offline.

**Ava:** What does the post-timeout analysis look for?

**Brian:** It aligns records from all ranks by sequence ID, collective type, and scheduling order, then groups them by process group. The goal is to find missing ranks and mismatches in type, state, call stack, data type, or size.

**Ava:** Can those mismatches point to the root-cause category?

**Brian:** Usually. CPU slowness often looks like a missing rank or state mismatch. CPU divergence can show a different collective type or call stack. Misconfigured arguments often show dtype or size mismatches. Network issues may show every rank in the started state, with no mismatch.

**Ava:** Is there a tool for doing that alignment?

**Brian:** PyTorch provides fr_trace. It enumerates collective mismatches, participating ranks, metadata, and one representative call stack.

**Ava:** Meta also built a visualization, right?

**Brian:** Yes. They place global collective scheduling order on the horizontal axis. Each row represents a rank and process-group combination. Cells represent collectives, and colors identify combinations of collective type and call stack. Selecting cells opens an icicle chart of the call stacks.

**Ava:** That sounds useful for scanning a huge distributed trace.

**Brian:** That’s the idea. The visualization makes mismatched colors easy to spot and helps compare call stacks quickly. Grouping by process group is especially useful for N-D parallelism and cross-process-group scheduling races.

**Ava:** But Flight Recorder alone doesn’t show what the CPU main thread was doing.

**Brian:** Correct. Effective debugging also needs distributed CPU call-stack telemetry. You want to know whether the CPU was loading data, compiling, waiting at a barrier, synchronizing with the GPU, or handling an exception. The article mentions open-source tools such as py-spy for collecting that information.

**Ava:** What did the case studies reveal?

**Brian:** In one recommendation-system case, metric computation used collectives, but faulty conditional logic caused some ranks to skip them. Flight Recorder’s divergent collective and call stacks isolated the problem.

**Ava:** And the all_to_all case was more subtle.

**Brian:** Yes. Some GPUs were stuck in two different, consecutive all_to_all calls. At first, every rank seemed to run all_to_all, so the cause was unclear. The call stacks showed that different variants came from different operations. Invalid input sizes let some ranks finish the first collective and move to the next while others were still stuck in the first.

**Ava:** That explains why looking only at the operation name was not enough.

**Brian:** Exactly. Sequence, call stack, and argument metadata matter together.

**Ava:** What’s next for Flight Recorder?

**Brian:** The team plans to integrate it with TorchComm, validate it with host-side optimizations such as CUDA graphs, and extend it beyond NCCL to backends including MTIA and Gloo.

**Ava:** Let’s close with the three points I should remember. First?

**Brian:** A watchdog timeout is a catch-all symptom. The reporting rank and the reported collective may not be the original cause.

**Ava:** Second?

**Brian:** Most timeouts observed at Meta came from cross-rank desynchronization, especially CPU-side slowness or divergent execution. Raising the timeout usually doesn’t solve that.

**Ava:** And third?

**Brian:** Flight Recorder preserves per-rank collective history, then aligns it offline so you can locate missing ranks, metadata mismatches, and the relevant call stacks.

**Ava:** That’s a much better lens than staring at one generic stack trace.

**Brian:** Exactly. When the watchdog fires, the useful question is not just, “Which rank timed out?” It’s, “Where did the ranks first stop agreeing?”

**Ava:** Thanks for listening. We’ll see you next time.

## 术语

| Term | 释义 |
|---|---|
| NCCL watchdog timeout | PyTorch 的 NCCL watchdog 检测到 collective 超过超时时间后抛出的错误 |
| collective | 需要多个 rank 协同执行的分布式通信操作 |
| process group | 需要彼此同步的一组 rank |
| c10d | PyTorch 分布式通信层，负责连接上层调用与通信后端 |
| Work object | 在 CPU 侧跟踪 GPU collective 生命周期的对象 |
| CPU-GPU synchronization | CPU 阻塞等待 GPU 完成已调度操作的同步点 |
| collective desynchronization | 不同 rank 在 collective 类型、顺序或状态上出现不一致 |
| PT2 compilation | PyTorch 2 编译流程，编译时间可能受数据和动态 shape 影响 |
| N-D parallelism | 同时使用多个并行维度和多个 process group 的并行方式 |
| all_to_all_single | 按 rank 之间的发送和接收切分进行数据交换的 collective |
| Flight Recorder | 记录各 rank collective 元数据并在超时后导出的诊断机制 |
| ring buffer | 循环覆盖写入的固定大小缓冲区 |
| TCPStore | 基于 TCP/IP 的键值存储，用于在 rank 之间广播超时信号 |
| fr_trace | 用于对齐和聚合 Flight Recorder dump 的诊断工具 |
| icicle chart | 用于展示调用栈层级结构的可视化图表 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let’s start with the basics. | 我们先从基础开始。 |
| That’s a pretty important distinction. | 这是一个相当重要的区别。 |
| So the timeout doesn’t identify the layer that failed. | 所以超时并不能指出是哪一层失败了。 |
| You have to resolve the desync. | 你必须解决这种不同步。 |
| That was a key design problem. | 这是一个关键的设计问题。 |
| The cause was unclear. | 原因并不清楚。 |
| That explains why... | 这就解释了为什么…… |
| Let’s close with the three points I should remember. | 最后总结我应该记住的三点。 |
