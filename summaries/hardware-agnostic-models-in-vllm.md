# vLLM 的硬件无关模型层

依据：原文

来源：https://pytorch.org/blog/hardware-agnostic-models-in-vllm/

vLLM 正为新模型开发面向特定硬件的“扁平”实现，同时建设独立的硬件无关层，以继续支持树外加速器、旧款 GPU 和更多模型。作者希望两条路径各自演进：前者追求新硬件上的极致性能，后者重视可移植性与可接受的吞吐量。

### 现有架构面临的问题

vLLM 的扁平模型、旧版模型和 transformers 后端，目前大多共用同一套注意力、线性投影等基础层。新模型的结构和专用内核日益多样，开发者若要保持整图 torch.compile 兼容，还需让 Dynamo 能追踪模型，并为新内核注册带有模拟实现及正确修改标注的算子。面向新 GPU 的扁平模型正转向定制融合，底层实现也可能失去 torch.compile 和 CustomOp 扩展支持。这会增加树外插件维护模型与层的负担，并可能使依赖 transformers 后端的旧模型或特殊模型在 GPU 上明显变慢。

### 设计方案

新层与硬件专用层隔离，放在 model_executor/hw_agnostic 等位置。设计要求模型可整图编译，保留 CustomOp 和 PluggableLayer 供树外插件覆盖实现，并尽量使用 PyTorch、Triton 或 Helion 编写可移植算子；不支持这些框架的加速器仍可通过扩展机制适配。旧版模型逐步移除后，作者计划让 transformers 后端及各扁平模型都有使用硬件无关层的路径。

### 实现进度与验证

transformers 后端改接新层的部分代码已进入 vLLM 主分支，目前只覆盖有限的层，可通过 USE_HW_AGNOSTIC=1 启用。团队已借助 IBM Spyre 树外插件验证 Gemma 4、Qwen3 和 Granite 4.2 等模型。为每个扁平模型提供相应 model.py 的方案尚未落地；可复用层将集中存放，模型专用层则随模型放置，并遵守同一套设计要求。

### 性能与适用范围

在 NVIDIA H100 上，作者比较了 transformers 后端开启和关闭 USE_HW_AGNOSTIC 时三个近期模型的总 token 吞吐量。按三个模型的几何平均值，硬件无关层与使用 FlashAttention、CUTLASS 等 CUDA 优化库的原生实现相差在 3.4% 以内，个别模型略高。这是特定硬件和模型上的测量结果，不能推及其他设备。作者明确表示，这条路径的目标是跨平台运行和保持性能，而非在 Blackwell、CDNA 4 等新硬件上达到最高性能；整体工作仍在推进中。
