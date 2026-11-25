# Integrating SGLang into Spyre in Under a Week with AI

原文：[Integrating SGLang into Spyre in Under a Week with AI - PyTorch](https://pytorch.org/blog/integrating-sglang-into-spyre-in-under-a-week-with-ai/)

## 摘要

两人团队借助 AI 编程代理，在不到一周内把 SGLang 集成到 IBM Spyre，插件约一千五百行。集成复用了 PyTorch 和 torch.compile 的硬件支持，以及 SGLang 的平台注册、注意力后端、KV pool 和算子分发扩展点。SGLang 在 Spyre 上的吞吐量约为基于 torch-spyre 的参考 PyTorch 推理实现的三点六倍；在 Granite 3.3 8B 和 Gemma 4 26B A4B 的实测工作负载上，性能与 vLLM 持平。RadixCache 的前缀共享无需 Spyre 专属修改，未来还计划支持动态形状和 Triton。

## 对话

Ava: What if adding a new accelerator to a serving framework took less than a week? That’s what happened when a two-person team brought SGLang to IBM Spyre with AI coding agents, using a plugin of roughly 1,500 lines.

Ava：如果给一个服务框架新增加速器只需不到一周，会怎样？这正是一个两人团队借助 AI 编程代理，将 SGLang 接入 IBM Spyre 时发生的情况，他们使用了一个约 1,500 行代码的插件。

Brian: And the result was measurable: on Spyre, SGLang reached approximately 3.6× the throughput of the reference PyTorch inference implementation using torch-spyre. On the measured workloads for Granite 3.3 8B and Gemma 4 26B A4B, SGLang was on par with vLLM.

Brian：结果是可量化的：在 Spyre 上，使用 torch-spyre 的参考 PyTorch 推理实现时，SGLang 达到了约 3.6× 的吞吐量。在针对 Granite 3.3 8B 和 Gemma 4 26B A4B 测得的工作负载上，SGLang 与 vLLM 持平。

Ava: Let’s unpack why that moved so quickly. Spyre is IBM’s accelerator for enterprise AI inference, and torch-spyre registers it as a PyTorch device. It also connects Spyre to torch.compile, so the hardware appears through standard PyTorch APIs.

Ava：让我们拆解一下它为何进展如此迅速。Spyre 是 IBM 面向企业 AI 推理的加速器，而 torch-spyre 将其注册为 PyTorch 设备。它还把 Spyre 接入 torch.compile，使硬件能够通过标准 PyTorch API 呈现。

Brian: That shared layer was the starting point. The team could use existing PyTorch implementations of Spyre operations as reference code, then adapt the integration code to SGLang’s backend interfaces. The AI agents mainly connected hardware support that already existed to a new serving framework.

Brian：这一共享层是起点。团队可以把 Spyre 各项操作的现有 PyTorch 实现作为参考代码，然后调整集成代码以适配 SGLang 的后端接口。AI 代理主要做的是把已经存在的硬件支持连接到一个新的服务框架。

Ava: What actually changed in SGLang?

Ava：SGLang 中究竟发生了哪些变化？

Brian: Four small patches extended backend support. Two made existing execution paths available without CUDA-specific dependencies. The other two registered Spyre as a supported device and as an attention backend that does not use Triton. The compatibility changes can also help other accelerators, while the registration changes are specific to Spyre.

Brian：四个小补丁扩展了后端支持。其中两个让现有执行路径无需依赖 CUDA 特有组件即可使用。另两个将 Spyre 注册为受支持的设备，以及注册为不使用 Triton 的注意力后端。兼容性改动也能帮助其他加速器，而注册改动则是 Spyre 特有的。

Ava: So the plugin stayed small because SGLang already separated the integration points?

Ava：所以，插件之所以保持精简，是因为 SGLang 已经将集成接入点分离开来了吗？

Brian: Exactly. Platform registration, the attention backend, the KV pool, and operator dispatch created bounded tasks. The existing scheduler, request handling, and serving loop stayed in place. The initial plugin was roughly 1,500 lines, compared with about 3,500 lines for the existing vLLM Spyre backend.

Brian：没错。平台注册、注意力后端、KV 池以及算子分发把工作拆成了范围明确的任务。现有的调度器、请求处理和服务循环都保留不变。初始插件约 1,500 行，而现有的 vLLM Spyre 后端约 3,500 行。

Ava: That’s a large difference. Was it mostly less hardware code?

Ava：差距很大。主要是因为硬件代码更少吗？

Brian: A major part came from reuse. SGLang’s attention base class supplied the surrounding logic, so the Spyre backend only had to override the inner scaled dot-product attention operation. Reusing that base class accounted for roughly two-thirds of the line-count difference between the two plugins.

Brian：很大一部分来自复用。SGLang 的注意力基类提供了外围逻辑，因此 Spyre 后端只需重写内部的缩放点积注意力操作。复用这个基类，约占两个插件行数差异的三分之二。

Ava: What else came from the vLLM implementation?

Ava：还有哪些部分来自 vLLM 实现？

Brian: The team reused how the vLLM Spyre integration grouped inputs into buckets and handled tensor shapes. It also reused the PyTorch-based approach to allocating the KV cache. SGLang’s device-agnostic KV pool needed only a single method override for buffer creation, although allocation still needed PyTorch-based handling where the existing path relied on Triton.

Brian：团队复用了 vLLM Spyre 集成将输入分组到桶中以及处理张量形状的方式。它还复用了基于 PyTorch 分配 KV 缓存的方法。SGLang 与设备无关的 KV 池在创建缓冲区时只需重写一个方法，不过在现有路径依赖 Triton 的部分，分配仍需要采用基于 PyTorch 的处理方式。

Ava: And the ordinary model operators?

Ava：那普通的模型算子呢？

Brian: For RMSNorm, SiLU, GeLU, rotary embeddings, and linear layers, the integration connected existing Spyre implementations to SGLang’s operator dispatch. That’s another example of wiring together implementations instead of rewriting them.

Brian：对于 RMSNorm、SiLU、GeLU、旋转嵌入和线性层，这项集成把现有的 Spyre 实现接入 SGLang 的算子分发。这又是一个通过连接现有实现来完成工作、而不是重写它们的例子。

Ava: Did every serving feature need a Spyre-specific adaptation?

Ava：每项服务功能都需要针对 Spyre 做适配吗？

Brian: No. RadixCache is the clearest example. Prefix sharing is resolved entirely within SGLang’s scheduler before execution reaches the accelerator, so the existing scheduling logic worked unchanged. The plugin is available as open source alongside upstream SGLang.

Brian：不需要。RadixCache 是最清楚的例子。前缀共享完全由 SGLang 的调度器在执行到达加速器之前解决，因此现有调度逻辑无需修改即可工作。该插件已与上游 SGLang 一同以开源形式提供。

Ava: For someone trying it, what does the setup look like?

Ava：如果有人想尝试，设置流程是什么样的？

Brian: The example assumes the Spyre runtime, torch-spyre, the SGLang plugin, and its compatibility patches are installed and configured. An engine is created with device set to spyre, the spyre_paged attention backend, float16 dtype, page size 64, max running requests set to 1, and CUDA graphs disabled. It then generates up to 64 new tokens for a prompt and shuts the engine down.

Brian：这个示例假定 Spyre 运行时、torch-spyre、SGLang 插件及其兼容性补丁都已安装并配置好。创建引擎时，将设备设为 spyre，注意力后端设为 spyre_paged，数据类型设为 float16，页面大小设为 64，最大运行请求数设为 1，并禁用 CUDA graphs。随后，它会根据一个提示词生成最多 64 个新 token，然后关闭引擎。

Ava: Let’s be precise about the performance claims. The 3.6× figure compares throughput with the reference PyTorch inference implementation using torch-spyre, right?

Ava：让我们准确说明性能声明。3.6× 这一数字比较的是使用 torch-spyre 的参考 PyTorch 推理实现的吞吐量，对吗？

Brian: Right. And the vLLM comparison is separate. Across the workloads measured for Granite 3.3 8B and Gemma 4 26B A4B, SGLang’s performance on Spyre was on par with vLLM. The relative-performance figure is normalized to vLLM equal to 1.0, with lower values better.

Brian：对。而且与 vLLM 的比较是另一回事。在针对 Granite 3.3 8B 和 Gemma 4 26B A4B 测量的所有工作负载上，SGLang 在 Spyre 上的性能都与 vLLM 持平。相对性能数值以 vLLM 等于 1.0 进行归一化，数值越低越好。

Ava: What’s next for the integration?

Ava：这项集成接下来有什么计划？

Brian: The Spyre software stack is still evolving. Over the coming months, the team expects dynamic-shape support, which should reduce much of today’s shape management. Later this year, they also expect Triton support for Spyre and RadixAttention on top of it, opening the door to additional kernel optimizations.

Brian：Spyre 软件栈仍在不断演进。在接下来的几个月里，团队预计会支持动态形状，这应能减少目前大量的形状管理工作。今年晚些时候，他们还预计会支持 Spyre 上的 Triton，并在其基础上实现 RadixAttention，从而为更多内核优化打开空间。

Ava: So the reusable foundation matters beyond this one plugin.

Ava：所以，可复用的基础不只对这一个插件有意义。

Brian: Yes. Keeping hardware support in PyTorch and integration code at SGLang’s extension points gives future compiler and runtime improvements a path into the serving stack. The next integration can start from concrete implementations and interfaces that both developers and AI coding tools can work with.

Brian：是的。将硬件支持保留在 PyTorch 中，并把集成代码放在 SGLang 的扩展点上，能让未来的编译器和运行时改进进入服务栈。下一次集成可以从开发者和 AI 编程工具都能使用的具体实现与接口起步。

Ava: Here’s the recap: a two-person team integrated SGLang with Spyre in less than a week using a plugin of roughly 1,500 lines. Reuse of PyTorch hardware support, the vLLM Spyre code, and SGLang’s extension points kept the work focused. SGLang reached approximately 3.6× the throughput of the reference PyTorch inference implementation using torch-spyre, was on par with vLLM on the measured Granite 3.3 8B and Gemma 4 26B A4B workloads, and used RadixCache without Spyre-specific changes.

Ava：总结一下：一个两人团队在不到一周内，借助约 1,500 行的插件将 SGLang 与 Spyre 集成起来。复用 PyTorch 的硬件支持、vLLM 的 Spyre 代码以及 SGLang 的扩展点，让工作始终聚焦。SGLang 达到了使用 torch-spyre 的参考 PyTorch 推理实现约 3.6× 的吞吐量；在测得的 Granite 3.3 8B 和 Gemma 4 26B A4B 工作负载上与 vLLM 持平；并且无需针对 Spyre 做改动即可使用 RadixCache。

Brian: That’s the useful pattern: keep accelerator behavior in shared PyTorch code, and make the serving integration satisfy clear framework interfaces. Thanks for listening.

Brian：这就是值得借鉴的模式：将加速器行为保留在共享的 PyTorch 代码中，并让服务集成满足清晰明确的框架接口。感谢收听。

## 术语

| Term | 释义 |
|---|---|
| IBM Spyre | IBM 面向企业 AI 推理的加速器 |
| torch-spyre | 将 Spyre 注册为 PyTorch 设备并提供硬件支持的软件组件 |
| torch.compile | PyTorch 的编译接口，可连接加速器硬件支持 |
| SGLang | 用于大模型服务的框架 |
| vLLM | 另一种大模型推理与服务框架 |
| attention backend | 负责注意力计算实现的后端 |
| KV pool | 管理键值缓存缓冲区的组件 |
| operator dispatch | 把算子调用分派到具体实现的机制 |
| scaled dot-product attention | 缩放点积注意力 |
| KV cache | 保存注意力键和值以支持生成的缓存 |
| RadixCache | SGLang 中用于前缀共享的缓存机制 |
| RadixAttention | 建立在 Triton 支持之上的注意力优化方向 |
| dynamic shapes | 运行时形状可变化的张量支持 |
| Triton | 用于编写和优化加速器内核的工具 |
| throughput | 单位时间内处理的请求或输出量 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let’s unpack why that moved so quickly. | 我们来拆解一下为什么进展这么快。 |
| That shared layer was the starting point. | 那个共享层就是起点。 |
| What actually changed in SGLang? | SGLang 实际改了什么？ |
| So the plugin stayed small because SGLang already separated the integration points? | 所以插件能保持精简，是因为 SGLang 已经分离好了集成点？ |
| A major part came from reuse. | 很大一部分来自复用。 |
| Did every serving feature need a Spyre-specific adaptation? | 每个服务特性都需要针对 Spyre 的适配吗？ |
| Let’s be precise about the performance claims. | 我们准确说明一下性能结论。 |
| What’s next for the integration? | 接下来这个集成要做什么？ |
| Here’s the recap: | 总结一下： |
| That’s the useful pattern: | 这个模式的实用之处是： |

## 中文内容总结

IBM 团队用两名开发者和 AI 编码代理，在不到一周内把 SGLang 推理服务栈接入 Spyre 加速器，形成约 1500 行的插件。集成主要复用了 PyTorch、torch-spyre 和现有 vLLM Spyre 后端中的硬件实现，并利用 SGLang 明确的扩展接口完成设备、注意力后端、KV 缓存和算子调度对接。测试显示，SGLang 在 Spyre 上的吞吐量约为基于 torch-spyre 的参考 PyTorch 推理实现的 3.6 倍；在 Granite 3.3 8B 和 Gemma 4 26B A4B 的已测工作负载上，性能与 vLLM 相当。

### 问题与起点

团队希望让 IBM Spyre 用户使用 SGLang 的服务能力，包括内置的前缀共享。Spyre 面向企业 AI 推理，通过 torch-spyre 注册为 PyTorch 设备，并接入 torch.compile，以标准 PyTorch API 暴露硬件能力。已有可用的编译器、PyTorch 硬件支持和推理服务框架，为后续集成提供了共同实现层。

### 集成方案与代码复用

AI 编码代理以现有 PyTorch Spyre 算子实现和 vLLM 的 Spyre 后端为参考，将硬件代码改接到 SGLang 的后端接口。四个小补丁中，两个让已有执行路径摆脱 CUDA 专属依赖，两个注册 Spyre 设备及不依赖 Triton 的注意力后端。SGLang 的平台注册、注意力后端、KV 池和算子调度把工作拆成边界清晰的任务，同时保留原有调度器、请求处理和服务循环。现有 vLLM Spyre 后端约 3500 行，SGLang 插件约 1500 行，其中约三分之二的差异来自复用注意力基类提供的外围逻辑，只需在 Spyre 后端覆盖内部的缩放点积注意力操作。团队还复用了输入分桶、张量形状处理和基于 PyTorch 的 KV 缓存分配方式，并把 RMSNorm、SiLU、GeLU、旋转位置编码和线性层接入 SGLang 的算子调度。

### 性能结果与条件

文章报告的实测结果显示，SGLang 在 Spyre 上达到参考 PyTorch 推理实现约 3.6 倍的吞吐量。团队还在 Spyre 上比较了 SGLang 与 vLLM，测试模型为 Granite 3.3 8B 和 Gemma 4 26B A4B；在这两个模型的已测工作负载中，SGLang 性能与 vLLM 相当。文章没有给出更广泛模型、绝对吞吐量或所有测试配置，因此这些结果只支持所列工作负载范围内的比较。

### 无需专用改动的能力

RadixCache 的前缀共享由 SGLang 调度器在执行到加速器之前完成，因此无需增加 Spyre 专用代码即可工作。SGLang 的设备无关 KV 池也只需覆盖一次缓冲区创建方法；当原有路径依赖 Triton 时，实际分配仍需使用 PyTorch 处理。

### 限制、后续计划与实际使用

示例要求预先安装并配置 Spyre 运行时、torch-spyre、SGLang 插件和兼容性补丁，并使用 device 为 spyre、attention_backend 为 spyre_paged、float16、page_size 64、单个运行请求和禁用 CUDA Graph 等参数。插件已随上游 SGLang 开源。作者预计未来数月加入动态形状支持，以减少当前的形状管理；并预计在当年晚些时候加入 Triton 支持，再在其上启用 RadixAttention。这些是后续计划，不是当前测得的结果。整体经验表明，把硬件支持放在 PyTorch，把框架适配集中在 SGLang 扩展点，可以降低新加速器接入服务栈的工作量，并让后续编译器和运行时改进继续复用同一基础。
