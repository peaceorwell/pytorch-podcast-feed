# Why Is PyTorch Compile So Fast: Kernel Fusion

原文：[Why Is PyTorch Compile So Fast: Kernel Fusion](https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/)

## 摘要

本期讨论 PyTorch 编译器为什么能让模型运行得更快，核心原因是 kernel fusion，也就是把多个 GPU 操作合并到更少的 kernel 中。Ava 和 Brian 通过逐元素计算示例，解释 vertical fusion 如何减少 kernel 启动次数、中间缓冲区和全局内存访问。节目还介绍 reduction fusion、GEMM plus epilogue fusion、prologue fusion 和 horizontal fusion。最后，两位主持人说明如何用 TORCH_LOGS 查看 Inductor 生成的 Triton kernel，并总结 fusion 的价值与适用边界。

## 对话

**Ava:** If you use PyTorch’s compiler, your model can run up to ten times faster. That sounds huge. But what is actually making it faster?

**Brian:** The short answer is kernel fusion. PyTorch’s compiler takes several operations that would normally run separately and combines them into one efficient GPU kernel.

**Ava:** Okay, so let’s start before fusion. What happens when we run ordinary PyTorch operations on a GPU?

**Brian:** Without compilation, the GPU runs a kernel, which is a function on the GPU, for each torch operation in your code. Every kernel launch has overhead. And every intermediate result usually has to move through memory.

**Ava:** So there are two costs: starting kernels and moving data?

**Brian:** Exactly. The GPU launches one kernel, writes an intermediate tensor to memory, then launches another kernel that reads it back. Repeat that several times, and the memory traffic becomes expensive.

**Ava:** Where does Inductor fit into this?

**Brian:** PyTorch’s Inductor compiler automatically groups dependent operations into single Triton kernels. Triton is the language used for these generated GPU kernels. The goal is to keep data in fast memory, close to the registers, and reduce kernel-launch overhead.

**Ava:** You said dependent operations. Is that what vertical fusion means?

**Brian:** Yes. Think of vertical fusion as linking steps. The output of one operation goes straight into the next. On a computation graph, the operations stack vertically because each one depends on the result above it.

**Ava:** Why is that pattern so common in deep learning?

**Brian:** Because neural networks are often chains. You might normalize data, pass it through a linear layer, and then apply an activation function. Vertical fusion joins those dependent steps.

**Ava:** And the main benefit is avoiding intermediate tensors?

**Brian:** Right. Those temporary tensors don't need to be written to global memory and read back later. They can stay in registers, which are the fastest memory available to the GPU.

**Ava:** Let's make that concrete. What example does the article use?

**Brian:** A pointwise fusion example. Pointwise operations apply simple math independently to each element. Addition, multiplication, and activation functions are typical examples.

**Ava:** What's the unfused version?

**Brian:** Imagine three operations in sequence. First, multiply an input by another tensor. Second, add a bias. Third, apply sigmoid. Without fusion, Inductor creates three separate Triton kernels.

**Ava:** So kernel one multiplies, kernel two adds, and kernel three applies sigmoid?

**Brian:** Exactly. Each kernel loads its inputs, performs one operation, and writes its result. The syntax may look intimidating, but the pattern is simple.

**Ava:** How much memory traffic does that create?

**Brian:** Across the three kernels, there are eight memory operations. You read the inputs for multiplication, read the multiplication result and the bias for addition, read the addition result for sigmoid, and write all three results.

**Ava:** Wait, all three results get written, even though only the final result matters?

**Brian:** In the unfused sequence, yes. The intermediate multiply result and add result have to be stored so the next kernel can use them.

**Ava:** That sounds wasteful.

**Brian:** It is a lot of memory traffic. And there are also three kernel launches, each with its own overhead.

**Ava:** Now show me the fused version.

**Brian:** With fusion, torch.compile creates one kernel. It loads all the inputs once, performs multiply, add, and sigmoid in a row, and stores only the final result.

**Ava:** What happens to the two intermediate values?

**Brian:** They stay in registers. The temporary values, called tmp2 and tmp4 in the example, never touch slower global memory.

**Ava:** So the GPU keeps the values close while it finishes the chain.

**Brian:** Exactly. It is like doing three calculations on a whiteboard before putting anything into a filing cabinet. You avoid repeatedly storing and retrieving temporary notes.

**Ava:** What are the exact improvements in this example?

**Brian:** Kernel launches drop from three to one. Two intermediate buffers disappear: the multiply result and the add result. And memory traffic drops from eight operations to four.

**Ava:** Four? Walk me through that number.

**Brian:** The fused kernel reads three full tensors and writes one full tensor. That's four memory operations in total, compared with reading five full tensors and writing three full tensors before fusion. The article describes that as a fifty percent reduction in memory traffic.

**Ava:** That explains the speedup much better than simply saying the compiler optimizes things.

**Brian:** Yes. The compiler is reducing work around the math. The arithmetic is still there, but the data movement and launch overhead are smaller.

**Ava:** Is pointwise fusion the only kind of vertical fusion?

**Brian:** No. It is one example. Inductor also has reduction fusion. That combines reducing operations such as max, mean, or sum with operations before and after them.

**Ava:** Why is reduction fusion important?

**Brian:** It matters for patterns such as batch normalization. The reduction and nearby operations can be kept together instead of repeatedly writing intermediate data.

**Ava:** The article also mentions GEMM plus epilogue fusion. What does that mean?

**Brian:** GEMM is a heavy matrix calculation. In GEMM plus epilogue fusion, simple math is attached to the end of that calculation. Instead of doing a matrix multiply, writing the result, reading it again to add bias, and then applying ReLU, those final steps happen right after the multiply in the same kernel.

**Ava:** So epilogue means the work at the end.

**Brian:** Right. Prologue fusion is the opposite idea. Preprocessing happens as data loads. For example, input normalization can happen on the fly as the data comes in before matrix multiplication.

**Ava:** And horizontal fusion is different because the operations are independent?

**Brian:** Exactly. Horizontal fusion runs multiple independent operations on the same input at once. The article's example computes sin of x and cos of x in one kernel, so x is loaded once instead of twice.

**Ava:** Vertical fusion follows a chain. Horizontal fusion shares an input.

**Brian:** That's a good way to remember it. Vertical means dependent steps are linked. Horizontal means independent work is placed side by side.

**Ava:** Can engineers see fusion in their own code?

**Brian:** Yes. The article suggests starting with a complete reduction example. You create a small Python file, run it with the TORCH_LOGS environment variable, and inspect the generated code.

**Ava:** What should we look for in that output?

**Brian:** You may see a generated Triton kernel with a name like triton_per_fused_add_mul_sum_0. The per prefix means per-reduction, and the name tells you that add, mul, and sum were fused together.

**Ava:** That sounds useful for checking what the compiler really did.

**Brian:** Exactly. You don't have to guess. The generated kernels show which operations Inductor grouped.

**Ava:** Are there limitations discussed in the article?

**Brian:** The article keeps the discussion focused. It explains that fusion is one of the most important optimizations in torch.compile, especially because memory traffic and kernel overhead are often major slowdowns in GPU work. It doesn't claim every pattern will fuse in the same way.

**Ava:** So we should view fusion as a compiler optimization to inspect, rather than a promise that every operation becomes one kernel.

**Brian:** Yes. The practical advice is to try torch.compile on your own code and inspect what it generates when you need more detail. You don't have to rewrite your implementation; you add the compiler decorator and let the compiler do the work.

**Ava:** Before we finish, give us the three-point recap.

**Brian:** First, without fusion, every torch operation can mean another kernel launch and more global-memory traffic. Second, vertical fusion keeps dependent intermediate values in registers, while horizontal fusion combines independent operations that share inputs. Third, in the pointwise example, fusion changes three kernels into one, removes two intermediate buffers, and cuts memory traffic from eight operations to four.

**Ava:** And the takeaway for a PyTorch engineer?

**Brian:** When your model runs faster under torch.compile, kernel fusion is a major reason. Try it, use TORCH_LOGS to inspect the generated Triton kernels, and learn which operations were fused.

**Ava:** That's all for today. Thanks for listening.

**Brian:** See you next time, and keep your tensors moving less.

## 术语

| Term | 释义 |
|---|---|
| PyTorch compiler | PyTorch 编译器，用于把模型操作编译成更高效的 GPU 代码 |
| kernel | 在 GPU 上执行的函数 |
| kernel fusion | 将多个操作合并到一个 kernel 中执行的优化 |
| Inductor | PyTorch 的编译器后端，会生成优化后的 Triton kernel |
| Triton kernel | 使用 Triton 生成的 GPU kernel |
| vertical fusion | 把计算图中相互依赖、上下连接的操作合并起来 |
| pointwise operation | 对每个元素独立执行的简单数学操作 |
| global memory | GPU 上较慢、用于读写大量数据的内存 |
| register | GPU 中速度最快、靠近计算单元的内存 |
| reduction fusion | 将 max、mean、sum 等归约操作与前后操作融合 |
| GEMM | 通用矩阵乘法，heavy matrix calculation |
| epilogue fusion | 将矩阵计算后的简单操作附加到同一个 kernel 末尾 |
| prologue fusion | 在数据加载时执行预处理操作的融合方式 |
| horizontal fusion | 将共享同一输入的多个独立操作放进一个 kernel |
| TORCH_LOGS | 用于查看 Inductor 生成代码的环境变量 |

## 口语表达

| Phrase | 释义 |
|---|---|
| The short answer is... | 简短回答是…… |
| Let's make that concrete. | 我们把这个讲得具体一点。 |
| Walk me through that number. | 请带我解释一下这个数字。 |
| What happens to... | ……会发生什么？ |
| That's a good way to remember it. | 这是个很好的记忆方法。 |
| You don't have to guess. | 你不需要靠猜。 |
| Before we finish... | 在结束之前…… |
| The takeaway for... | 对……来说，核心启示是…… |
