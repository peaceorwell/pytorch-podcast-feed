# ExecuTorch 将 PyTorch 模型部署到 Cortex-M CPU

依据：原文

来源：https://pytorch.org/blog/bringing-executorch-to-arm-cortex-m-cpus/

ExecuTorch 新增 Cortex-M CPU 后端，使开发者能在 PyTorch 体系内完成模型导出、量化和设备部署，无须转换到其他框架格式。该后端以 CMSIS-NN 加速推理，也可与已有的 Ethos-U NPU 委托执行配合。

### 问题与部署流程

微控制器通常只有不足 1 MB 的 SRAM、数 MB 闪存和毫瓦级功耗预算。过去，将 PyTorch 模型部署到 Cortex-M 往往需要格式转换、另一套工具链，并排查转换引入的数值差异。新流程使用 torch.export 导出模型，通过 PT2E 和 CortexMQuantizer 校准、量化为 INT8，再按目标 CPU 编译成 .pte 文件，与运行时和 CMSIS-NN 一起集成进固件。

### 核心机制

编译阶段的转换会把适用的量化算子替换为 CMSIS-NN 或其他 Cortex-M 优化实现，并按目标指令集选择内核：M0 等使用标量指令，M4 等使用 DSP，M55 和 M85 使用 MVE（Helium）。编译还会消除多余的数据类型转换和量化操作，预计算量化参数，把卷积权重从 OIHW 转为 OHWI，并预先安排张量及临时缓冲区在 TCM、SRAM 等内存中的位置。作者称这可减少运行时开销，并避免张量动态分配。对带 Ethos-U 的芯片，受支持算子可交给 NPU；留在 CPU 上的算子满足量化、布局和形状条件时使用 CMSIS-NN，否则回退到未加速的通用内核。

### 性能测量及条件

作者在 Alif Ensemble E8 的单个 400 MHz Cortex-M55 高性能核心上测试了五个 MLPerf Tiny 模型中的四个，未使用板载 NPU。四者均采用 INT8 量化和算子选择性编译。DS-CNN、MobileNet v1 0.25、ResNet-8 和异常检测自编码器的延迟分别为 5.825、21.450、15.290 和 1.208 毫秒，峰值 RAM 分别为 26.00、85.15、55.14 和 5.20 KiB。包含运行时与内核的代码闪存占用为 14.97 至 36.03 KiB，另需存放模型。编译条件包括 GCC 15.2.1、关闭日志、核心运行时使用 -Oz、内核使用 -O3。这些是作者自己的测量，不是官方 MLPerf Tiny 提交，不能直接与排行榜成绩比较。

### 演示、适用范围与后续工作

在 Cortex-M85 的 Renesas EK-RA8D1 开发板上，µYOLO 人形检测演示使用 128×128 RGB 输入，INT8 .pte 模型为 452 KiB，总闪存占用 640 KiB，峰值 RAM 为 254 KiB。将耗时最高的算子权重放入 DTCM 并启用指令缓存后，480 MHz 下单帧推理延迟从 191 毫秒降至 24 毫秒；端到端吞吐量为 12.7 FPS。仓库内的演示应用面向 Corstone FVP，开发板移植代码在仓库外。作者计划扩展混合精度、浮点内核、注意力等算子和硬件验证；尚未测试的第五个 MLPerf Tiny 流式唤醒词任务还需要多帧推理测试设施。
