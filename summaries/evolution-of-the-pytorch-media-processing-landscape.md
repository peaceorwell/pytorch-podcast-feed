# PyTorch 媒体处理库的分工调整

依据：原文

来源：https://pytorch.org/blog/evolution-of-the-pytorch-media-processing-landscape/

PyTorch 将图像、视频和音频的解码与编码集中到 TorchCodec；TorchVision 负责图像和视频变换，TorchAudio 负责音频变换。这项调整旨在让媒体文件与张量之间的处理流程更清晰，并减轻各库的维护负担。

### 原有问题

过去，媒体读写能力分散在 TorchVision 和 TorchAudio，部分功能重复。TorchVision 的视频读取有 PyAV、C++ FFmpeg 和 CUDA/NVCUVID 三种后端，只有 PyAV 默认可用；其余后端需要从源码构建，并受特定 FFmpeg 版本约束。TorchAudio 也有 StreamReader、StreamWriter 和使用不同后端的音频解码工具。用户需要在多个入口中选择，维护者则要分别处理底层依赖、构建和发布。

### 新的处理流程

TorchCodec 统一承担图像、视频和音频的解码与编码，将媒体文件或已编码字节转换为张量，再把张量编码为媒体文件；处理张量时，图像和视频使用 TorchVision 的变换，音频使用 TorchAudio 的变换。文中的视频示例用 VideoDecoder 取出帧，再交给 torchvision.transforms.v2 裁剪并转换数据类型；音频示例用 AudioDecoder 读取采样及采样率，再用 MelSpectrogram 生成梅尔频谱。编码接口包括 VideoEncoder、AudioEncoder、JpegEncoder 和 PngEncoder。

### 范围、收益与条件

作者称，集中开发使 TorchCodec 通常比先前分散的实现更快，尤其是 CUDA 视频解码，但本文没有给出基准测试或提速数值。媒体读写涉及 FFmpeg 4 至 9 共六个主要版本、NVIDIA 编解码 SDK，以及各图像格式的原生库；将这些依赖集中到 TorchCodec，简化了 TorchVision 和 TorchAudio 的维护。三个库现已实现 ABI 稳定：某个版本不再只绑定一个 PyTorch 版本，也可与其后的 PyTorch 版本配合使用，因此发布节奏不必再与 PyTorch 一致。

### 迁移与边界

TorchVision 和 TorchAudio 现在聚焦变换；原有模型、数据集和流水线等功能不再积极开发，作者指出 HuggingFace 等生态库已有替代选择。原先位于两库的媒体读写 API 已弃用或移除，现有用户应参考迁移指南转向 TorchCodec。TorchAudio 的调整影响较大，不过团队根据社区反馈保留了部分原计划移除的常用 API。文章未提供完整解码与变换流水线的性能调优方法，作者表示将另文介绍。
