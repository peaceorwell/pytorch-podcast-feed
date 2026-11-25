# 2026 年北美 PyTorch 大会回顾

依据：原文

来源：https://pytorch.org/blog/pytorch-conference-north-america-2026-accelerating-open-intelligence-together/

2026 年北美 PyTorch 大会于 10 月 20 日至 21 日在圣何塞举行，吸引超过 3,500 人参加。大会以“共同加速开放智能”为主题，面向研究人员、开发者和企业，展示 PyTorch 基金会项目如何支持模型训练、推理和部署，并推动工作负载在不同芯片与云环境中运行。

### 跨硬件支持

Google 与 Meta 合作开发的开源 TorchTPU 为 Google TPU 提供原生 PyTorch 后端，支持即时执行和 torch.compile，并已验证与 TorchTitan、vLLM、SGLang 和 Ray 的集成。迁移部分工作负载可能只需更换设备配置，但达到峰值性能仍可能需要调整张量形状、分片和操作。面向 AWS Trainium 的 TorchNeuron 于 10 月 20 日以 Apache 2.0 许可推出公开测试版，支持即时执行、torch.compile、DTensor 和 torch.distributed。大会还介绍了 IBM Spyre 的编译流程，以及面向 AMD MI350 系列 GPU 的可选 FlyDSL 后端。

### 训练、推理与端侧更新

TorchTitan 在现有 5D 并行体系中扩展了模型支持，并通过 TitanRL 加入强化学习流程，使同一模型实现可用于预训练、后训练及 vLLM 内的执行。Helion 与 NVIDIA CompileIQ 的集成让开发者在内核自动调优时一并评估编译器控制选项。ExecuTorch 的 Cortex-M 后端支持通过 Arm CMSIS-NN 加速，在 Arm Cortex-M CPU 上运行 PyTorch 模型，覆盖 Scalar、DSP 和 MVE/Helium 能力层级。

### 案例与验证范围

Zephon 与 TorchTitan 的集成旨在让训练改变 GPU 数量后仍重现相同的全局批次。在一个 10 亿参数模型测试中，训练从 16 块 GPU 的检查点分别恢复到 8、32 和 64 块 GPU，每个全局批次的 SHA-256 哈希均与原运行一致。Shopify 则介绍了使用 PyTorch 和 vLLM 的持续学习流程，并称该流程每天将生产故障反馈纳入模型权重、质量超过其比较的前沿模型，推理服务成本降低 96%；原文未给出这一质量比较的评测细节。

### 社区活动与后续安排

10 月 19 日的会前活动包括 PyTorch 培训、Jupyter Day 和研究项目落地工作坊。正式议程包含超过 100 场技术会议和 144 场海报展示，另设演示、维护者问答、专题讨论及贡献者奖。PyTorch 基金会宣布新增九家成员机构，其中 Nebius、Nscale 和 Shopify 为白金会员。大会表示会后将发布会议资料与录像；下一届大会定于 2027 年 10 月 6 日至 7 日在圣何塞举行。
