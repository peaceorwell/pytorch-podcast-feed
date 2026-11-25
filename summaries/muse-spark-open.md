# Muse Spark Open 开放权重与推理优化

依据：原文

来源：https://pytorch.org/blog/muse-spark-open/

Meta 于 2026 年 10 月 15 日发布 Muse Spark Open 权重，开发者可在 Hugging Face 按 Muse Model License 获取。模型总参数量为 816B、每次激活 42B，上下文窗口为 100 万 token，与 Muse Spark 1.3 共用基础模型。文章重点介绍其编程代理能力、Helion 推理算子和 TorchTitan 微调支持。

### 模型能力

Meta 称，Muse Spark Open 针对多文件重构、调试和长期编程任务训练，旨在提高首次尝试的准确率、工具调用可靠性和代码库理解能力。模型可处理文本、图像、文档、音频及音视频输入；其代理训练涵盖网页浏览、多模态浏览、跨站点任务，以及在代理框架中协调子代理、操作桌面、移动端和浏览器。模型还能在执行动作前单独说明意图。开放权重版本增加了安全后训练，以适应脱离系统防护措施的独立使用。上述能力和安全性主要是发布方的描述，文章没有提供相应的基准测试数据。

### Helion 推理优化

Muse Spark Open 支持在 SGLang 和 vLLM 上优化推理，面向 NVIDIA、AMD 和 Intel 硬件。发布方用 PyTorch 原生内核语言 Helion 实现 RoPE、RMSNorm、量化等操作。在 NVIDIA GB300 上，这些算子相对以 CUDA、CuTe DSL 等硬件专用语言编写的定制内核，测得 1.03 至 3.95 倍加速；移植到 AMD MI350 后，发布方称算子获得相近改进，端到端吞吐量整体提高 3%。文章还介绍了 SGLang 中用 Helion 编写的 megakernel：它将多个操作融合进一个常驻内核，以改善低延迟推理，并可按形状和硬件自动调优。算子级加速范围不能直接视为整个模型的推理加速范围。

### 微调与实际使用

TorchTitan 新增 Muse Spark Open 的模型定义，支持多轮监督微调、LoRA，以及通过 TitanRL 进行强化学习微调。开发者可获取权重，并按任务调整模型；Helion 内核也可针对加速器架构、量化格式、张量形状和并行策略继续调优。文章没有给出微调效果、资源需求或各类硬件配置下的完整性能数据，因此实际部署效果仍取决于具体工作负载和配置。
