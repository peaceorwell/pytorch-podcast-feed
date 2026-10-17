# PyTorch 2.14: A Faster GEMM Backend, a Rewritten NCCL, and Declarative Dynamic Shapes

原文：[PyTorch 2.14 Release Blog](https://pytorch.org/blog/pytorch-2-14-release-blog/)

## 摘要

PyTorch 2.14 发布，本期主要聊三块内容。第一是 NVGEMM，一个基于 CuTeDSL 生成 CUTLASS 内核的新矩阵乘法后端，这次补上了 epilogue 融合，还支持低精度格式，能跟 Triton、ATen 一起参与自动调优。第二是分布式那块的大动作：重写版 NCCL 后端以预览形式进屋，容错重配置和单边 RMA 窗口成了 c10d 的一等概念，Flight Recorder 也不再只服务 NCCL 一家。第三是 Apple Silicon，SVD、QR、Cholesky 这些线性代数运算终于有了原生 Metal 内核，解码路径上一个 8.5 倍的性能坑也被填上了。另外还顺带提了 torch.switch 多路分支、@dynamic_spec 声明式动态形状，以及 ROCm、XPU、Rubin 架构的平台支持进展。这些功能大多还标着 API Unstable，说明接口本身可能还会变。

## 对话

Ava: Okay so PyTorch just dropped 2.14. Almost three thousand commits since 2.13. That's a lot to unpack.

Ava：好，PyTorch 刚发布了 2.14。自 2.13 以来将近三千次提交，信息量真大。

Brian: Yeah, 2,995 commits, 487 contributors. It's a big one. But there's a clear thread running through it if you squint.

Brian：是啊，2,995 次提交，487 位贡献者。这次更新规模很大。不过仔细看，还是能发现一条贯穿其中的主线。

Ava: Which is?

Ava：是什么？

Brian: PyTorch trying to be one platform that works the same whether you're on NVIDIA, AMD, Intel, or a Mac. Faster matmuls, a rewritten networking backend, native math on Apple GPUs. Same story, three fronts.

Brian：PyTorch 想成为一个统一的平台，让你用 NVIDIA、AMD、Intel，还是 Mac，都能以同样的方式工作。更快的矩阵乘法、重写的网络通信后端、Apple GPU 上的原生数学运算。一个方向，三个方面。

Ava: Let's start with the matmul thing. NVGEMM. What even is that?

Ava：先说矩阵乘法吧。NVGEMM 到底是什么？

Brian: So, GEMM is just General Matrix Multiply — the operation that eats most of your GPU time in a transformer. NVGEMM is a new backend for it inside Inductor, PyTorch's compiler.

Brian：GEMM 就是 General Matrix Multiply，也就是通用矩阵乘法。在 transformer 里，它会占掉 GPU 的大部分运行时间。NVGEMM 是 PyTorch 编译器 Inductor 内部为这类运算新增的后端。

Ava: And it's built on CuTeDSL?

Ava：它是基于 CuTeDSL 构建的？

Brian: Right, that landed in 2.13 as a way to generate CUTLASS kernels — CUTLASS is NVIDIA's template library for writing fast GPU math. In 2.13 it could only spit out a standalone kernel. Whatever came after, like a bias add, stayed in a separate kernel that had to re-read the result from memory.

Brian：对。CuTeDSL 在 2.13 中加入，用来生成 CUTLASS 内核；CUTLASS 是 NVIDIA 用于编写高性能 GPU 数学运算的模板库。在 2.13 中，它只能生成一个独立的内核。后续操作，比如加偏置，仍得放在另一个内核里，再从内存中读取前一个内核的结果。

Ava: That sounds wasteful.

Ava：听起来很浪费。

Brian: It is. So the big addition in 2.14 is epilogue fusion. Now the bias add, the activation, the rescale — they get folded right into the same kernel. No extra memory round trip.

Brian：确实。所以 2.14 的一项重要新增功能是 epilogue fusion。现在，加偏置、激活、重新缩放都能融合进同一个内核，不用再往返内存。

Ava: And this competes against what PyTorch already had?

Ava：那它会和 PyTorch 已有的方案竞争？

Brian: Exactly, it's not a replacement, it's a third candidate. Inductor autotunes NVGEMM alongside Triton and ATen and picks whichever is fastest for your shape. It also reaches the low-precision paths now, so things like NVFP4 — a four-bit floating format — get their scaling folded into the kernel too, instead of paying for a separate multiply.

Brian：没错。它不是替代品，而是第三个候选方案。Inductor 会把 NVGEMM、Triton 和 ATen 放在一起自动调优，针对你的数据形状选出最快的那个。它现在也覆盖低精度路径，所以像 NVFP4 这种四位浮点格式，其缩放操作也能融合进内核，不必再为一次单独的乘法付出开销。

Ava: Any catch?

Ava：有什么限制吗？

Brian: You have to opt in, add NVGEMM to the autotune backend list. And NVFP4 support needs Blackwell hardware. If an epilogue is too weird for it to express, it just falls back to Triton, so you're not worse off.

Brian：你得主动启用，把 NVGEMM 加进自动调优后端列表。而且 NVFP4 支持需要 Blackwell 硬件。如果某个 epilogue 太特殊，它无法表达，就会回退到 Triton，所以性能不会因此变差。

Ava: Okay, let's move to distributed training. This is the part I actually care about.

Ava：好，接着聊分布式训练。这才是我真正关心的部分。

Brian: Big one here too. There's a preview of a rewritten NCCL backend — NCCL being NVIDIA's Collective Communications Library, the thing that handles all-reduce and friends across GPUs.

Brian：这里也有个大更新：重写后的 NCCL 后端推出了预览版。NCCL 是 NVIDIA 的 Collective Communications Library，负责处理 GPU 之间的 all-reduce 等操作。

Ava: Rewritten how?

Ava：怎么重写的？

Brian: It's ported from a project called torchcomms, which showed up in 2.13. Now it lands in-tree as basically a drop-in replacement, nicknamed nccl2, and it's slated to become the default backend in 2.15.

Brian：它是从一个叫 torchcomms 的项目移植过来的，这个项目在 2.13 中出现。现在它以基本可以直接替换旧后端的形式进入 PyTorch 代码库，昵称是 nccl2，并计划在 2.15 中成为默认后端。

Ava: What does it actually buy you over the old one?

Ava：相比旧后端，它到底带来什么好处？

Brian: Nonblocking communicators, one-sided memory access, and — this is the headline for me — fault tolerance built in as a first-class concept, not a hack bolted onto NCCL specifically.

Brian：非阻塞通信器、单边内存访问，还有最让我关注的一点：把容错作为一项基础能力内置进去，而不是专门给 NCCL 拼凑一个补丁。

Ava: Wait, back up. What does fault tolerance mean in this context? Isn't that usually a checkpoint-and-restart thing?

Ava：等等，先解释一下。这里的容错是什么意思？这通常不就是保存检查点再重启吗？

Brian: That's exactly the pain point. Normally when one rank dies in a big job, you tear down the whole process group and restart it, which throws away warm state across the entire cluster.

Brian：这正是痛点。通常，大型任务中的一个 rank 一旦故障，你就得拆掉整个进程组再重启，这会让整个集群已经预热的状态付诸东流。

Ava: Ouch. That's expensive at scale.

Ava：哎哟，规模大了这成本可不低。

Brian: Very. So now the Backend and ProcessGroup objects expose reconfiguration interfaces — a group can be rebuilt in place instead of destroyed. And this isn't NCCL-only, Gloo gets it too.

Brian：非常高。所以现在 Backend 和 ProcessGroup 对象提供了重新配置接口，可以原地重建进程组，而不必销毁它。而且这不只适用于 NCCL，Gloo 也支持。

Ava: You mentioned one-sided memory access. What's that, exactly?

Ava：你提到了单边内存访问。那具体是什么？

Brian: Think RMA, remote memory access. Normally a collective needs both sides to show up and synchronize. One-sided means one rank can just read or write another rank's memory directly, no matching call needed on the other end.

Brian：可以把它理解成 RMA，也就是远程内存访问。通常，集合通信需要双方都参与并同步。单边的意思是，一个 rank 可以直接读取或写入另一个 rank 的内存，不需要另一端发起配对调用。

Ava: When would you want that?

Ava：什么情况下会需要它？

Brian: Irregular access patterns. Embedding lookups, weight transfer, expert routing in mixture-of-experts models. Anywhere the access pattern isn't a clean all-to-all.

Brian：访问模式不规则的时候，比如查找 embedding、传输权重、在 mixture-of-experts 模型中进行专家路由。只要访问模式不是规整的 all-to-all，就可能用得上。

Ava: And there was something about a Flight Recorder?

Ava：还有一个叫 Flight Recorder 的东西？

Brian: Right, that's the trace buffer people use to debug hung or mismatched collectives. It used to be wired specifically into NCCL. Now it hooks through ProcessGroup generically, so it works for Gloo or any custom backend too. Debugging a hang doesn't mean giving up your trace just because you're not on NCCL.

Brian：对，那是人们用来调试卡住或不匹配的集合通信操作的跟踪缓冲区。它过去专门绑定在 NCCL 上，现在通过 ProcessGroup 的通用接口接入，所以 Gloo 和任何自定义后端也能用。调试卡顿时，不会因为没用 NCCL 就失去跟踪记录。

Ava: Makes sense. Let's switch gears — literally, since there's a feature called torch.switch.

Ava：有道理。我们换个话题——这次还真是字面上的“切换”，因为有个功能叫 torch.switch。

Brian: Ha, yeah. So torch.cond already existed for a two-way branch inside a compiled graph — if this, else that. But if you needed, say, eight-way branching, like picking one of eight experts in an MoE model, you had to nest torch.cond calls.

Brian：哈，是啊。torch.cond 之前就能在编译图中实现双分支，也就是满足这个条件走一边，否则走另一边。但如果你需要，比如说，八路分支，像在 MoE 模型里从八个专家中选一个，就得把 torch.cond 调用层层嵌套。

Ava: Which gets ugly fast.

Ava：那很快就会变得很难看。

Brian: Right, it bloats the traced graph and hides what you're actually doing. torch.switch does multi-way branching natively, dispatching on an index. It's aimed squarely at mixture-of-experts routing.

Brian：对，它会让跟踪得到的图膨胀，也掩盖了你实际要做的事。torch.switch 原生支持多路分支，根据索引分派，目标就是 mixture-of-experts 路由。

Ava: There's also something about while loops and CUDA graphs?

Ava：还有关于 while 循环和 CUDA 图的更新？

Brian: torch.while_loop can now be captured inside a CUDA graph. Normally a data-dependent loop count forces a device-to-host copy to decide how many iterations to run, and that breaks graph capture right where you'd want it intact.

Brian：现在可以把 torch.while_loop 捕获进 CUDA 图。通常，循环次数如果依赖数据，就得把数据从设备复制到主机来决定运行多少次，而这恰好会在你最想保持图完整的地方打断图捕获。

Ava: So now the loop count can vary at replay time without breaking capture?

Ava：所以现在重放时循环次数可以变化，而且不会破坏图捕获？

Brian: Exactly. The condition gets re-evaluated on GPU at the end of each iteration using CUDA's native while-node support. It's not a speed win by itself, it just means a variable-length loop no longer kicks you out of the graph entirely.

Brian：完全正确。每次迭代结束时，都会利用 CUDA 原生的 while 节点支持，在 GPU 上重新判断条件。这本身不会带来速度提升，只是意味着可变长度的循环不再让整个执行流程脱离图。

Ava: Okay, now the one I've been waiting for — Apple Silicon. My Mac finally gets real linear algebra?

Ava：好，终于说到我一直在等的 Apple Silicon 了。我的 Mac 总算能做真正的线性代数运算了？

Brian: Yes! This was a genuine gap before. SVD, eigh — that's the eigendecomposition for Hermitian matrices — QR, and Cholesky. All were leaning on Apple's MPSGraph framework, or just falling back to CPU.

Brian：没错！这之前确实是个缺口。SVD、eigh——也就是 Hermitian 矩阵的特征分解——QR 和 Cholesky，此前都依赖 Apple 的 MPSGraph 框架，或者干脆回退到 CPU。

Ava: Which kills performance if you're doing anything numerical.

Ava：只要做点数值计算，这就会拖垮性能。

Brian: Right, mixed CPU and MPS round trips are exactly the kind of thing that quietly tanks your throughput. Now SVD, eigh, and lstsq run via native Jacobi-style Metal kernels for float32. Cholesky got a faster panel-factorization algorithm too, something like 1.2 to 2.8x faster depending on matrix size.

Brian：对，CPU 和 MPS 之间来回传数据，正是那种会悄悄拖垮吞吐量的事。现在 SVD、eigh 和 lstsq 在 float32 下通过原生的 Jacobi 风格 Metal 内核运行。Cholesky 也换用了更快的 panel-factorization 算法，根据矩阵大小，速度大约提高了 1.2 to 2.8x。

Ava: Any limits?

Ava：有什么限制？

Brian: float64 still falls back to CPU, because Metal simply has no double-precision type. And there's a whole separate five-part rewrite of reduction kernels — full reductions, strided ones, argmax and argmin — moving off MPSGraph onto hand-written Metal.

Brian：float64 仍会回退到 CPU，因为 Metal 根本没有双精度类型。另外还有一项分为五个部分的归约内核重写工作，涵盖全量归约、带步幅的归约、argmax 和 argmin，把它们从 MPSGraph 迁移到手写的 Metal 内核。

Ava: I saw something about an 8.5x slowdown fix too.

Ava：我还看到一个修复 8.5x 性能下降的问题。

Brian: Yeah, that one's specific but important if you run models locally. Single-token decode — the shape you get when you're generating one token at a time — was falling off the fast path in F.linear on MPS. Eight point five times slower than it should've been, on bf16 and fp16.

Brian：是啊。这个问题很具体，但如果你在本地运行模型，就很重要。单 token 解码，也就是一次生成一个 token 时的数据形状，在 MPS 上的 F.linear 中没走到快速路径。用 bf16 和 fp16 时，速度比应有水平慢了八点五倍。

Ava: That's huge for anyone doing local inference on a Mac.

Ava：对任何在 Mac 上做本地推理的人来说，这影响都很大。

Brian: Exactly, it's one of the bigger remaining gaps between MPS and CUDA for autoregressive generation, and this closes a good chunk of it.

Brian：没错。这是自回归生成方面，MPS 与 CUDA 之间尚存的较大差距之一；这次修复缩小了其中相当一部分差距。

Ava: Quick one — what's @dynamic_spec?

Ava：快速问一个——@dynamic_spec 是什么？

Brian: It's a cleanup, not a new capability exactly. Before this, declaring which dimensions of your input can vary at runtime meant a different mechanism depending on whether you were using torch.compile, torch.export, or make_fx.

Brian：它主要是整理现有机制，严格说不算一项新能力。在此之前，要声明输入的哪些维度可以在运行时变化，得看你用的是 torch.compile、torch.export 还是 make_fx，各有不同的机制。

Ava: So three different ways to say the same thing.

Ava：所以这是表达同一件事的三种不同方式。

Brian: Right, and the declaration usually lived far away from the model code. Now you define a shape spec once, attach it to your function with a decorator, and all three entry points pick it up automatically.

Brian：对，而且声明通常离模型代码很远。现在，你只需定义一次形状规范，用装饰器把它附加到函数上，三个入口就会自动获取它。

Ava: And platform support — I saw ROCm and Rubin mentioned.

Ava：还有平台支持方面——我看到提到了 ROCm 和 Rubin。

Brian: ROCm 7.14 wheels now build from AMD's TheRock SDK. Intel XPU gets native graph capture. And Inductor now targets Rubin, NVIDIA's next architecture after Blackwell.

Brian：ROCm 7.14 的 wheels 现在基于 AMD 的 TheRock SDK 构建。Intel XPU 获得了原生图捕获支持。Inductor 现在也以 Rubin 为目标，它是 NVIDIA 在 Blackwell 之后的下一代架构。

Ava: Anything we should flag as not fully baked yet?

Ava：有什么尚未完全成熟的地方需要提醒大家吗？

Brian: Almost everything in this release is tagged API Unstable, so expect some churn. Complex-tensor support in torch.compile is explicitly experimental — not every complex operation is covered yet. And Python 3.15 support landed, but it's eager-mode only for now — calling torch.compile under 3.15 just raises an error rather than quietly running slower.

Brian：这个版本里几乎所有内容都标记为 API Unstable，所以要做好接口可能变动的准备。torch.compile 对复数张量的支持明确处于实验阶段——目前还没有覆盖所有复数运算。Python 3.15 支持也已经加入，但眼下只支持 eager 模式——在 3.15 下调用 torch.compile 会直接报错，而不会悄悄以较慢的速度运行。

Ava: Good to know before someone upgrades blind. So, wrapping up — three things to remember?

Ava：这点最好在有人盲目升级前知道。那么，最后总结一下——有哪三点需要记住？

Brian: One: NVGEMM brings epilogue fusion and low-precision support to PyTorch's fastest matmul path. Two: fault tolerance and one-sided memory access are now built into distributed training, not bolted on. Three: Apple Silicon finally gets native linear algebra and a major decode-speed fix.

Brian：第一，NVGEMM 为 PyTorch 最快的矩阵乘法路径带来了 epilogue 融合和低精度支持。第二，容错和单边内存访问现在已内置于分布式训练，而不是事后加装。第三，Apple Silicon 终于有了原生线性代数支持，解码速度方面的一个重大问题也得到了修复。

Ava: That's PyTorch 2.14. Thanks for listening, and we'll catch you next release.

Ava：以上就是 PyTorch 2.14。感谢收听，我们下个版本再见。

## 术语

| Term | 释义 |
|---|---|
| NVGEMM | 基于 CuTeDSL 生成 CUTLASS 内核的新矩阵乘法后端，作为 Inductor 的候选算子与 Triton、ATen 竞争 |
| GEMM | 通用矩阵乘法（General Matrix Multiply），GPU 上最耗时的核心运算之一 |
| CuTeDSL | 用于生成 CUTLASS 内核的领域特定语言（DSL） |
| epilogue fusion | 将矩阵乘法之后的操作（如加偏置、激活函数、缩放）融合进同一个内核，避免额外的内存读写 |
| NCCL | NVIDIA Collective Communications Library，NVIDIA 的集合通信库，用于多 GPU 间的 all-reduce 等操作 |
| torchcomms | 2.13 引入的通信后端项目，2.14 中其能力以 nccl2 形式合入主干 |
| c10d | PyTorch 分布式通信的核心库，管理进程组和集合通信 |
| RMA / 单边操作 | Remote Memory Access，无需对端配合即可直接读写远程内存的通信方式 |
| Flight Recorder | 用于诊断集合通信挂起或不匹配问题的跟踪缓冲区工具 |
| torch.switch | 新的高阶算子，支持基于索引的多路分支，替代嵌套的 torch.cond |
| torch.while_loop | 支持数据依赖循环次数的控制流算子，2.14 起可被捕获进 CUDA graph |
| CUDA graph | 将一系列 GPU 操作预先捕获并可重放的机制，减少每次调用的开销 |
| @dynamic_spec | 声明式动态形状 API，让 torch.compile、torch.export、make_fx 共用同一套形状声明 |
| MPS / Metal | Apple Silicon 上的 GPU 计算框架（Metal Performance Shaders）及其底层 Metal 内核 |
| MoE | Mixture of Experts，混合专家模型架构 |

## 口语表达

| Phrase | 释义 |
|---|---|
| That's a lot to unpack. | 信息量很大，得慢慢梳理 |
| if you squint | 仔细一看／粗略看的话（引出一个不太明显的规律） |
| Same story, three fronts. | 同样的思路，在三个方向上重复 |
| That sounds wasteful. | 这样听起来很浪费 |
| Any catch? | 有什么坑／附加条件吗？ |
| Wait, back up. | 等等，倒回去解释一下 |
| Which gets ugly fast. | 这样很快就会变得很糟糕 |
| That's huge for anyone doing X. | 这对做某件事的人来说影响很大 |
| Good to know before someone upgrades blind. | 在人们盲目升级之前，这点很值得知道 |
| quietly tanks your throughput | 悄无声息地拖垮你的吞吐量 |
