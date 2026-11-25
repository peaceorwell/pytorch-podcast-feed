# ExecuTorch WebGPU：在浏览器 GPU 上运行 PyTorch 模型

依据：原文

来源：https://pytorch.org/blog/executorch-webgpu-pytorch-models-on-the-gpu-in-the-browser/

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
