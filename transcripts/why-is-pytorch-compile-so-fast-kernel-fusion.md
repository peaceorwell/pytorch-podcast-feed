# Why Is PyTorch Compile So Fast: Kernel Fusion

原文：[Why Is PyTorch Compile So Fast: Kernel Fusion](https://pytorch.org/blog/why-is-pytorch-compile-so-fast-kernel-fusion/)

## 摘要

本期讨论 PyTorch 编译器为什么能让模型运行得更快，核心原因是 kernel fusion，也就是把多个 GPU 操作合并到更少的 kernel 中。Ava 和 Brian 通过逐元素计算示例，解释 vertical fusion 如何减少 kernel 启动次数、中间缓冲区和全局内存访问。节目还介绍 reduction fusion、GEMM plus epilogue fusion、prologue fusion 和 horizontal fusion。最后，两位主持人说明如何用 TORCH_LOGS 查看 Inductor 生成的 Triton kernel，并总结 fusion 的价值与适用边界。

## 对话

Ava: If you use PyTorch’s compiler, your model can run up to ten times faster.

Ava：使用 PyTorch 编译器，模型运行速度最高可提升到 10 倍。

That sounds huge. But what is actually making it faster?

听起来提升很大。到底是什么让它变快的？

Brian: The short answer is kernel fusion.

Brian：简单说，就是内核融合。

PyTorch’s compiler takes several operations that would normally run separately and combines them into one efficient GPU kernel.

PyTorch 编译器把原本分别运行的多个操作合并成一个高效的 GPU 内核。

Ava: Okay, so let’s start before fusion.

Ava：好，那先看看融合之前是什么样。

What happens when we run ordinary PyTorch operations on a GPU?

普通的 PyTorch 操作在 GPU 上运行时会发生什么？

Brian: Without compilation, the GPU runs a kernel, which is a function on the GPU, for each torch operation in your code.

Brian：不经过编译，代码里的每个 torch 操作都会让 GPU 运行一个内核，也就是在 GPU 上执行的函数。

Every kernel launch has overhead.

每次启动内核都有开销。

And every intermediate result usually has to move through memory.

而且每个中间结果通常都要经过内存。

Ava: So there are two costs: starting kernels and moving data?

Ava：所以有两项开销：启动内核和搬运数据？

Brian: Exactly.

Brian：没错。

The GPU launches one kernel, writes an intermediate tensor to memory, then launches another kernel that reads it back.

GPU 启动一个内核，把中间张量写入内存，再启动另一个内核把它读回来。

Repeat that several times, and the memory traffic becomes expensive.

重复几次，内存数据传输的开销就很大了。

Ava: Where does Inductor fit into this?

Ava：Inductor 在其中起什么作用？

Brian: PyTorch’s Inductor compiler automatically groups dependent operations into single Triton kernels.

Brian：PyTorch 的 Inductor 编译器会自动把相互依赖的操作合并到同一个 Triton 内核中。

Triton is the language used for these generated GPU kernels.

Triton 是编写这些生成的 GPU 内核所用的语言。

The goal is to keep data in fast memory, close to the registers, and reduce kernel-launch overhead.

目标是让数据留在靠近寄存器的高速存储中，并减少内核启动开销。

Ava: You said dependent operations. Is that what vertical fusion means?

Ava：你说的相互依赖的操作，就是纵向融合吗？

Brian: Yes. Think of vertical fusion as linking steps.

Brian：对。可以把纵向融合理解为把连续步骤串起来。

The output of one operation goes straight into the next.

一个操作的输出直接交给下一个操作。

On a computation graph, the operations stack vertically because each one depends on the result above it.

在计算图上，这些操作纵向排列，因为每一步都依赖上一步的结果。

Ava: Why is that pattern so common in deep learning?

Ava：为什么这种模式在深度学习中这么常见？

Brian: Because neural networks are often chains.

Brian：因为神经网络通常由一连串操作组成。

You might normalize data, pass it through a linear layer, and then apply an activation function.

比如先对数据做归一化，再经过线性层，最后应用激活函数。

Vertical fusion joins those dependent steps.

纵向融合就是把这些相互依赖的步骤合在一起。

Ava: And the main benefit is avoiding intermediate tensors?

Ava：主要好处是省去中间张量？

Brian: Right.

Brian：对。

Those temporary tensors don't need to be written to global memory and read back later.

这些临时张量不用写入全局内存，之后再读回来。

They can stay in registers, which are the fastest memory available to the GPU.

它们可以留在寄存器里，那是 GPU 可用的最快存储。

Ava: Let's make that concrete. What example does the article use?

Ava：举个具体例子吧。文章用了什么例子？

Brian: A pointwise fusion example.

Brian：一个逐元素融合的例子。

Pointwise operations apply simple math independently to each element.

逐元素操作就是对每个元素独立进行简单计算。

Addition, multiplication, and activation functions are typical examples.

加法、乘法和激活函数都是典型例子。

Ava: What's the unfused version?

Ava：未融合时是什么样？

Brian: Imagine three operations in sequence. First, multiply an input by another tensor.

Brian：设想依次执行三个操作。第一步，把输入和另一个张量相乘。

Second, add a bias. Third, apply sigmoid.

第二步，加上偏置。第三步，应用 sigmoid 函数。

Without fusion, Inductor creates three separate Triton kernels.

未融合时，Inductor 会创建三个独立的 Triton 内核。

Ava: So kernel one multiplies, kernel two adds, and kernel three applies sigmoid?

Ava：所以第一个内核做乘法，第二个做加法，第三个应用 sigmoid？

Brian: Exactly. Each kernel loads its inputs, performs one operation, and writes its result.

Brian：没错。每个内核都读取输入、执行一个操作，再写出结果。

The syntax may look intimidating, but the pattern is simple.

代码语法可能看着复杂，但模式很简单。

Ava: How much memory traffic does that create?

Ava：这会产生多少次内存读写？

Brian: Across the three kernels, there are eight memory operations.

Brian：三个内核总共进行八次内存读写。

You read the inputs for multiplication, read the multiplication result and the bias for addition, read the addition result for sigmoid, and write all three results.

乘法要读取两个输入；加法要读取乘法结果和偏置；sigmoid 要读取加法结果；三个结果还都要写入内存。

Ava: Wait, all three results get written, even though only the final result matters?

Ava：等等，虽然只需要最终结果，三个结果还是都要写入内存？

Brian: In the unfused sequence, yes.

Brian：在未融合的执行过程中，是的。

The intermediate multiply result and add result have to be stored so the next kernel can use them.

乘法和加法的中间结果必须存起来，供下一个内核使用。

Ava: That sounds wasteful.

Ava：这听起来很浪费。

Brian: It is a lot of memory traffic.

Brian：确实会产生大量内存读写。

And there are also three kernel launches, each with its own overhead.

而且还要启动三个内核，每次都有开销。

Ava: Now show me the fused version.

Ava：现在看看融合后的版本。

Brian: With fusion, torch. compile creates one kernel.

Brian：通过融合，torch.compile 只生成一个内核。

It loads all the inputs once, performs multiply, add, and sigmoid in a row, and stores only the final result.

它只加载一次所有输入，依次完成乘法、加法和 sigmoid，最后只写入最终结果。

Ava: What happens to the two intermediate values?

Ava：那两个中间值呢？

Brian: They stay in registers.

Brian：它们留在寄存器里。

The temporary values, called tmp2 and tmp4 in the example, never touch slower global memory.

示例中叫 tmp2 和 tmp4 的临时值，完全不用访问较慢的全局内存。

Ava: So the GPU keeps the values close while it finishes the chain.

Ava：所以 GPU 在完成整串计算时，一直把这些值留在近处。

Brian: Exactly.

Brian：没错。

It is like doing three calculations on a whiteboard before putting anything into a filing cabinet.

就像在白板上算完三步，才把结果放进文件柜。

You avoid repeatedly storing and retrieving temporary notes.

这样就不用反复存取临时记录。

Ava: What are the exact improvements in this example?

Ava：这个例子具体改善了什么？

Brian: Kernel launches drop from three to one.

Brian：内核启动次数从三次降到一次。

Two intermediate buffers disappear: the multiply result and the add result.

乘法和加法结果对应的两个中间缓冲区也省掉了。

And memory traffic drops from eight operations to four.

内存读写次数则从八次降到四次。

Ava: Four? Walk me through that number.

Ava：四次？这个数字怎么算的？

Brian: The fused kernel reads three full tensors and writes one full tensor.

Brian：融合后的内核读取三个完整张量，写入一个完整张量。

That's four memory operations in total, compared with reading five full tensors and writing three full tensors before fusion.

总共四次内存读写；融合前则要读取五个完整张量、写入三个完整张量。

The article describes that as a fifty percent reduction in memory traffic.

文章称，内存流量因此减少了 50%。

Ava: That explains the speedup much better than simply saying the compiler optimizes things.

Ava：这比只说编译器做了优化，更能解释为什么会提速。

Brian: Yes. The compiler is reducing work around the math.

Brian：对。编译器减少的是计算之外的开销。

The arithmetic is still there, but the data movement and launch overhead are smaller.

算术运算还在，但数据搬运和内核启动的开销更小了。

Ava: Is pointwise fusion the only kind of vertical fusion?

Ava：逐元素融合是纵向融合的唯一形式吗？

Brian: No. It is one example. Inductor also has reduction fusion.

Brian：不是，它只是一个例子。Inductor 还支持归约融合。

That combines reducing operations such as max, mean, or sum with operations before and after them.

它把求最大值、均值或总和等归约操作，与前后的操作融合起来。

Ava: Why is reduction fusion important?

Ava：归约融合为什么重要？

Brian: It matters for patterns such as batch normalization.

Brian：它对批量归一化之类的计算模式很有用。

The reduction and nearby operations can be kept together instead of repeatedly writing intermediate data.

归约及其相邻操作可以一起执行，避免反复写入中间数据。

Ava: The article also mentions GEMM plus epilogue fusion. What does that mean?

Ava：文章还提到 GEMM 加尾声融合。那是什么意思？

Brian: GEMM is a heavy matrix calculation.

Brian：GEMM 是计算量很大的矩阵运算。

In GEMM plus epilogue fusion, simple math is attached to the end of that calculation.

GEMM 加尾声融合，就是把简单运算接在矩阵计算的末尾。

Instead of doing a matrix multiply, writing the result, reading it again to add bias, and then applying ReLU, those final steps happen right after the multiply in the same kernel.

矩阵相乘后，无须先写入结果、再读出来加偏置、最后应用 ReLU；这些步骤会紧接着乘法，在同一个内核中完成。

Ava: So epilogue means the work at the end.

Ava：所以尾声就是末尾的工作。

Brian: Right. Prologue fusion is the opposite idea. Preprocessing happens as data loads.

Brian：对。序言融合则是相反的思路：加载数据时就做预处理。

For example, input normalization can happen on the fly as the data comes in before matrix multiplication.

比如，输入数据进入时，就可以即时完成归一化，再进行矩阵乘法。

Ava: And horizontal fusion is different because the operations are independent?

Ava：横向融合的区别在于，各项操作互不依赖？

Brian: Exactly.

Brian：没错。

Horizontal fusion runs multiple independent operations on the same input at once.

横向融合让多个独立操作同时处理同一个输入。

The article's example computes sin of x and cos of x in one kernel, so x is loaded once instead of twice.

文章中的例子用一个内核计算 x 的 sin 和 cos，因此 x 只需加载一次，而不是两次。

Ava: Vertical fusion follows a chain. Horizontal fusion shares an input.

Ava：纵向融合沿着计算链进行，横向融合则共享输入。

Brian: That's a good way to remember it. Vertical means dependent steps are linked.

Brian：这样记很好。纵向就是把相互依赖的步骤连起来。

Horizontal means independent work is placed side by side.

横向就是把独立的操作放在一起。

Ava: Can engineers see fusion in their own code?

Ava：工程师能在自己的代码中看到融合效果吗？

Brian: Yes. The article suggests starting with a complete reduction example.

Brian：能。文章建议从一个完整的归约示例入手。

You create a small Python file, run it with the TORCH_LOGS environment variable, and inspect the generated code.

创建一个小型 Python 文件，设置 TORCH_LOGS 环境变量后运行，再查看生成的代码。

Ava: What should we look for in that output?

Ava：输出中该看什么？

Brian: You may see a generated Triton kernel with a name like triton_per_fused_add_mul_sum_0.

Brian：你可能会看到名为 triton_per_fused_add_mul_sum_0 之类的 Triton 内核。

The per prefix means per-reduction, and the name tells you that add, mul, and sum were fused together.

前缀 per 表示按归约处理，名称也表明 add、mul 和 sum 被融合在了一起。

Ava: That sounds useful for checking what the compiler really did.

Ava：这很适合用来检查编译器到底做了什么。

Brian: Exactly. You don't have to guess.

Brian：没错，不用猜。

The generated kernels show which operations Inductor grouped.

生成的内核会显示 Inductor 把哪些操作组合在了一起。

Ava: Are there limitations discussed in the article?

Ava：文章讨论了哪些局限性吗？

Brian: The article keeps the discussion focused.

Brian：文章始终围绕这个主题展开讨论。

It explains that fusion is one of the most important optimizations in torch.

文章解释了，融合是 torch.compile 中最重要的优化之一。

compile, especially because memory traffic and kernel overhead are often major slowdowns in GPU work.

尤其因为在 GPU 计算中，内存传输和内核启动开销往往是主要瓶颈。

It doesn't claim every pattern will fuse in the same way.

文章并没有说所有计算模式都会以同样的方式融合。

Ava: So we should view fusion as a compiler optimization to inspect, rather than a promise that every operation becomes one kernel.

Ava：所以，融合是值得检查的编译器优化，不能指望所有操作都变成一个内核。

Brian: Yes. The practical advice is to try torch.

Brian：对。实用的建议是试试 torch.compile。

compile on your own code and inspect what it generates when you need more detail.

在自己的代码上使用它，需要了解详情时再查看编译结果。

You don't have to rewrite your implementation; you add the compiler decorator and let the compiler do the work.

不用重写代码；加上编译器装饰器，让编译器完成优化即可。

Ava: Before we finish, give us the three-point recap.

Ava：结束前，请用三点总结一下。

Brian: First, without fusion, every torch operation can mean another kernel launch and more global-memory traffic.

Brian：第一，没有融合时，每个 torch 操作都可能带来一次内核启动和更多全局内存访问。

Second, vertical fusion keeps dependent intermediate values in registers, while horizontal fusion combines independent operations that share inputs.

第二，垂直融合把有依赖关系的中间值留在寄存器中；水平融合则合并共享输入的独立操作。

Third, in the pointwise example, fusion changes three kernels into one, removes two intermediate buffers, and cuts memory traffic from eight operations to four.

第三，在逐元素计算的例子中，融合把三个内核变成一个，省去两个中间缓冲区，并将内存访问次数从 8 次减到 4 次。

Ava: And the takeaway for a PyTorch engineer?

Ava：对 PyTorch 工程师来说，关键是什么？

Brian: When your model runs faster under torch. compile, kernel fusion is a major reason.

Brian：如果模型用了 torch.compile 后跑得更快，内核融合是一个重要原因。

Try it, use TORCH_LOGS to inspect the generated Triton kernels, and learn which operations were fused.

试一试，用 TORCH_LOGS 查看生成的 Triton 内核，弄清哪些操作被融合了。

Ava: That's all for today. Thanks for listening.

Ava：今天就到这里。感谢收听。

Brian: See you next time, and keep your tensors moving less.

Brian：下次见，让张量少跑点路。

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
