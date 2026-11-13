# PyTorch 基金会新增九家成员

依据：原文

来源：https://pytorch.org/blog/pytorch-foundation-welcomes-nine-new-members-as-momentum-builds-around-open-source-ai-center-of-gravity/

PyTorch 基金会于 2026 年 10 月 20 日在圣何塞举行的 PyTorch Conference North America 2026 上宣布新增九家成员。基金会隶属 Linux Foundation，希望通过开放治理，把研究者、开发者和企业连接到涵盖训练、推理及部署的开源项目生态中。

### 新增成员

白金级成员为 Nebius、Nscale 和 Shopify Inc.；银级成员为 LightSeek Foundation 和 Lablup Inc.；关联成员为 SKOPOS INDIA FOUNDATION、Siksha ‘O’ Anusandhan、Messiah University 和 Valencia College。基金会认为，成员增长有助于研究、模型、基础设施和硬件领域的成果相互流动，并支持 PyTorch、vLLM、DeepSpeed、Ray、Helion 和 Safetensors 等项目。这是基金会对生态作用的表述，并非本次公告测得的成效。

### 生态项目进展

公告汇集了多项近期更新。ExecuTorch 的 Cortex-M 后端提供从 PyTorch 训练到 Arm Cortex-M 设备端推理的路径，使用 Arm CMSIS-NN 加速，并支持 Scalar、DSP 和 MVE/Helium 三种能力层级。开源后端 TorchNeuron 面向 AWS Trainium 及 EC2 Trn2、Trn3，宣称只需更改一项配置即可运行 PyTorch 工作负载，同时支持 eager execution、torch.compile、DTensor、torch.distributed 和 TorchTitan。Google 宣布面向 TPU 开发者公开提供原生 PyTorch 项目 TorchTPU，目标是以少量代码改动部署模型，并接入 vLLM、SGLang、Ray 和 LLM-d。

### 案例结果与范围

IBM 的 Spyre 软件栈扩展了 PyTorch Inductor 编译器和原生运行时。公告称，Spyre 专用编译器代码减少了 7 倍，且在已测试的工作负载上保持性能相当；借助 AI 辅助开发，以 44 个适配器和约 1 万行代码支持数千个模型。Shopify 的案例称，其使用 PyTorch 和 vLLM 建立持续学习循环，每天将生产环境中的失败案例用于更新模型权重，模型质量超过所比较的前沿模型，服务成本降低 96%。公告没有给出这些比较的基准、测量方法或适用范围，不宜将案例结果视为所有部署都能达到的效果。

### 会议与实际意义

同期会议面向训练、推理、内核工程和负责任 AI 等领域的从业者，讨论从底层基础设施到应用和智能体的技术。对开发者而言，本次公告展示了基金会项目向不同芯片和部署场景延伸的方向；各项性能与成本数字则分别来自特定案例。
