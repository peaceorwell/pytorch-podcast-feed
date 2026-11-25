# ExecuTorch WebGPU: PyTorch Models in the Browser

原文：[ExecuTorch WebGPU: PyTorch Models on the GPU, in the Browser - PyTorch](https://pytorch.org/blog/executorch-webgpu-pytorch-models-on-the-gpu-in-the-browser/)

## 摘要

ExecuTorch 的 WebGPU 后端让 PyTorch 模型通过 torch.export 导出为 .pte 文件，在支持 WebGPU 的浏览器中使用 GPU 运行，无需转换为其他框架的格式；推理输入和输出留在设备上。后端通过 WGSL 计算着色器执行计算，并用按最大尺寸分配的缓冲区、逐次派发重算工作组等方式处理动态形状。在 Apple M1 Pro 的 Google Chrome 中，量化的 Llama 3.2 1B 和 Qwen3-0.6B 相比同机的多线程 WebAssembly + XNNPACK CPU 路径，prefill 快 4–7 倍，decode 快 2–10 倍；两条路径使用的激活格式不同。文章还列出视觉、语音模型及设备端微调案例，并说明该后端仍处于早期开发阶段。

## 对话

Ava: A PyTorch model can now run on the GPU inside a browser tab, while prompts and outputs stay on the device. That's a useful path for private or offline inference.

Ava：现在，PyTorch 模型可以在浏览器标签页内的 GPU 上运行，而提示词和输出都留在设备上。这为私密或离线推理提供了一条实用路径。

Brian: ExecuTorch's new WebGPU backend makes that possible without converting the model to another framework's format. You export it with torch.export to one .pte file, then run that file in a WebGPU browser.

Brian：ExecuTorch 新增的 WebGPU 后端让这成为可能，而且无需把模型转换成其他框架的格式。你可以用 torch.export 将模型导出为一个 .pte 文件，然后在支持 WebGPU 的浏览器中运行这个文件。

Ava: The export path sounds simple. Where does the GPU work actually happen?

Ava：导出流程听起来很简单。GPU 上的计算实际发生在哪里？

Brian: The export uses WebGPUPartitioner. At runtime, ExecuTorch allocates GPU buffers and dispatches WGSL compute shaders through WebGPU command buffers.

Brian：导出时会用到 WebGPUPartitioner。运行时，ExecuTorch 会分配 GPU 缓冲区，并通过 WebGPU 命令缓冲区调度 WGSL 计算着色器。

Ava: The same .pte file is meant to run across browsers and devices. What handles the differences?

Ava：同一个 .pte 文件预期能在不同浏览器和设备上运行。这些差异由什么来处理？

Brian: WGSL is translated to Metal Shading Language, SPIR-V, or HLSL, depending on the platform. For the browser, the C++ runtime is compiled to WebAssembly with Emscripten, and emdawnwebgpu connects its WebGPU calls to the browser's JavaScript API.

Brian：根据平台不同，WGSL 会被转换为 Metal Shading Language、SPIR-V 或 HLSL。在浏览器中，C++ 运行时通过 Emscripten 编译成 WebAssembly，emdawnwebgpu 则将它的 WebGPU 调用连接到浏览器的 JavaScript API。

Ava: A browser workload can change shape as a prompt grows. Does the backend reallocate GPU memory for every length?

Ava：随着提示词变长，浏览器中的计算任务形状也可能变化。后端会针对每种长度重新分配 GPU 内存吗？

Brian: For dynamic shapes, it allocates buffers at the maximum size, recalculates workgroups for each dispatch, and routes shapes to specialized WGSL kernels. That's how it supports batched prefill.

Brian：对于动态形状，它会按最大尺寸分配缓冲区，在每次调度时重新计算工作组数量，并将不同形状分派给专门的 WGSL 内核。它就是这样支持批量预填充的。

Ava: That connects to the speed claim. What exactly was compared?

Ava：这就涉及速度方面的说法了。具体比较了什么？

Brian: The authors tested quantized Llama 3.2 1B and Qwen3-0.6B in Google Chrome on an Apple M1 Pro. They compared WebGPU with ExecuTorch's own multithreaded WebAssembly CPU backend using XNNPACK on the same machine.

Brian：作者在搭载 Apple M1 Pro 的设备上，用 Google Chrome 测试了量化后的 Llama 3.2 1B 和 Qwen3-0.6B。他们将 WebGPU 与同一台设备上 ExecuTorch 自己的多线程 WebAssembly CPU 后端进行比较，后者使用 XNNPACK。

Ava: And the reported gain is for prefill and decode, not one combined inference number, right?

Ava：报告的速度提升分别针对预填充和解码，而不是一个合并后的推理指标，对吗？

Brian: Right. WebGPU was 4–7× faster on prefill and 2–10× faster on decode in that setup. Each point is the median of 5 runs after 3 warmups, across contexts from 128 to 8K tokens, with identical token IDs.

Brian：对。在这套测试环境中，WebGPU 的预填充速度快了 4–7×，解码速度快了 2–10×。上下文长度从 128 到 8K 个 token，每个数据点都是预热 3 次后运行 5 次所得的中位数，并且使用了相同的 token ID。

Ava: I'm looking at the precision settings before treating that as a pure GPU-versus-CPU result. Were the activation formats the same?

Ava：在把它看作单纯的 GPU 与 CPU 对比结果之前，我想先看看精度设置。激活值的格式相同吗？

Brian: They weren't. WebGPU used 4-bit groupwise weights with group size 64 and fp32 activations. The CPU path used XNNPACK's 4-bit weights and 8-bit dynamic activations. Both paths were timed with wall-clock time because GPU timestamps aren't available for CPU execution.

Brian：不同。WebGPU 使用分组大小为 64 的 4 位逐组量化权重和 fp32 激活值。CPU 路径使用 XNNPACK 的 4 位权重和 8 位动态量化激活值。两条路径都用实际经过时间计时，因为 GPU 时间戳无法用于 CPU 执行。

Ava: What does the model coverage look like beyond text generation?

Ava：除了文本生成，模型覆盖范围还有哪些？

Brian: SmolVLM handles image questions and captions. The listed vision models cover depth estimation, image segmentation, object detection, super-resolution, and background removal. Whisper base covers speech-to-text, and another listed model covers text-to-speech.

Brian：SmolVLM 可处理图像问答和图像描述。列出的视觉模型涵盖深度估计、图像分割、目标检测、超分辨率和背景移除。Whisper base 用于语音转文字，另有一个列出的模型用于文字转语音。

Ava: And the article includes on-device fine-tuning, not just inference.

Ava：文章还介绍了设备端微调，不只是推理。

Brian: Yes. It lists LoRA and DiReFT fine-tuning for Llama 3.2 1B, Qwen2.5-0.5B, and Qwen3-0.6B, aimed at personalized assistants and domain adaptation.

Brian：是的。文章列出了针对 Llama 3.2 1B、Qwen2.5-0.5B 和 Qwen3-0.6B 的 LoRA 与 DiReFT 微调，面向个性化助手和领域适配。

Ava: If I want to try it, what would I actually ship?

Ava：如果我想试用，实际需要部署什么？

Brian: The article describes serving a .pte file over a CDN on a web page. The ExecuTorch WebGPU Playground runs many of these models on the same WebAssembly + WebGPU runtime. Each model's Hugging Face repository includes .pte files, a model card, and the exact export recipe.

Brian：文章介绍了在网页上通过 CDN 提供 .pte 文件的做法。ExecuTorch WebGPU Playground 使用同一套 WebAssembly + WebGPU 运行时运行其中许多模型。每个模型的 Hugging Face 仓库都包含 .pte 文件、模型卡和确切的导出步骤。

Ava: So the useful takeaway is one PyTorch export for local browser GPU inference, with a measured speedup over ExecuTorch's CPU path under the stated test conditions.

Ava：所以，实用的结论是：只需一次 PyTorch 导出，就能在本地通过浏览器 GPU 进行推理；在所述测试条件下，相比 ExecuTorch 的 CPU 路径还有经过测量的速度提升。

Brian: And there's a broad set of validated models, plus on-device fine-tuning examples. The backend's still in early development. The team plans to improve speed and stability and expand operator, data type, and model coverage. Thanks for listening.

Brian：此外，还有一系列经过验证的模型，以及设备端微调示例。这个后端仍处于早期开发阶段。团队计划提升速度和稳定性，并扩大算子、数据类型和模型的覆盖范围。感谢收听。

## 术语

| Term | 释义 |
|---|---|
| ExecuTorch | PyTorch 的设备端运行方案；本文使用其新 WebGPU 后端在浏览器中运行模型。 |
| WebGPU | 浏览器中的 GPU 接口，支持计算着色器等能力。 |
| torch.export | 本文用于将 PyTorch 模型导出到 ExecuTorch 部署流程的接口。 |
| .pte file | ExecuTorch 的模型文件；本文介绍的同一文件可用于支持 WebGPU 的浏览器。 |
| WebGPUPartitioner | 本文所述模型导出流程中使用的 WebGPU 后端分区组件。 |
| WGSL compute shaders | WebGPU 后端用于在 GPU 上执行计算的着色器。 |
| dynamic shapes | 运行时会变化的张量形状；该后端使用按最大尺寸分配的缓冲区等机制处理。 |
| batched prefill | 对批量输入执行 prefill；本文的动态形状处理机制支持这一场景。 |
| prefill | 语言模型处理已有输入上下文的阶段；本文单独报告该阶段的速度。 |
| decode | 语言模型逐步生成后续 token 的阶段；本文单独报告该阶段的速度。 |
| XNNPACK | 本文对比所用的 ExecuTorch 多线程 WebAssembly CPU 路径中的计算后端。 |
| LoRA | 本文列出的设备端微调方法之一。 |
| DiReFT | 本文列出的另一种设备端微调方法。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Where does the GPU work actually happen? | GPU 计算实际在哪一步发生？ |
| What handles the differences? | 这些差异由什么机制处理？ |
| What exactly was compared? | 具体比较了什么？ |
| in that setup | 在那组测试条件下 |
| on the same machine | 在同一台机器上 |
| Were the activation formats the same? | 激活值的格式相同吗？ |
| What does the model coverage look like beyond text generation? | 除了文本生成，模型覆盖范围还有哪些？ |
| under the stated test conditions | 在所述测试条件下 |

## 中文内容总结

ExecuTorch 新增 WebGPU 后端。开发者可用 torch.export 将 PyTorch 模型导出为 .pte 文件，在支持 WebGPU 的浏览器中运行，无需先转换为 GGUF、ONNX 或 TFLite。推理在浏览器标签页内完成，提示词、图像和音频等输入留在设备上。

### 要解决的问题

浏览器覆盖桌面和移动设备，但以往部署 PyTorch 模型通常要转换格式。ExecuTorch 希望让同一份 .pte 文件跨 WebGPU 浏览器使用，以减少部署步骤，并为离线推理、较低网络延迟和减少服务端推理开销提供条件。

### 导出与运行机制

模型经 torch.export 和 WebGPUPartitioner 导出后，由编译成 WebAssembly 的 ExecuTorch C++ 运行时加载。Emscripten 和 emdawnwebgpu 将运行时的 WebGPU 调用接到浏览器 API；后端分配 GPU 缓冲区，并调度 WGSL 计算着色器。针对动态形状，运行时按最大尺寸分配缓冲区，在每次调度时重算工作组，并按形状选择专用内核。WebGPU 在不同平台上可将 WGSL 转为 Metal Shading Language、SPIR-V 或 HLSL。

### 性能与测量条件

作者在 Apple M1 Pro 的 Google Chrome 中，以量化的 Llama 3.2 1B 和 Qwen3-0.6B 比较 WebGPU 与同机多线程 ExecuTorch WebAssembly CPU 路径（XNNPACK）。在 128 至 8K token 上，每项结果经 3 次预热后取 5 次运行的中位数，两个后端使用相同 token ID，并以实际经过时间计时。报告的 WebGPU 加速比为预填充 4 至 7 倍、逐 token 解码 2 至 10 倍。两条路径的量化配置不同：GPU 使用组大小为 64 的 4 位权重和 fp32 激活，CPU 使用 XNNPACK 的 4 位权重和 8 位动态激活；这些数字应按该测试配置理解。

### 模型覆盖与使用

作者称已验证的模型包括 Llama 3.2、Qwen2.5、Qwen3 和 Gemma 3 等语言模型，SmolVLM 图文理解，以及深度估计、图像分割、目标检测、超分辨率、语音识别和语音合成模型。文章还列出针对部分小型语言模型的 LoRA 与 DiReFT 设备端微调。Playground 用同一套 WebAssembly 与 WebGPU 运行时演示其中多种模型；各模型的 Hugging Face 仓库提供 .pte 文件、模型卡和导出步骤。

### 现状与意义

该后端仍处于早期开发阶段，作者计划继续改进速度和稳定性，扩展算子、数据类型及模型覆盖。对开发者而言，网页可通过 CDN 提供 .pte 文件，由后端处理管线编译、内存管理和调度；实际可运行的模型仍取决于当前支持范围以及用户浏览器和设备的 WebGPU 能力。
