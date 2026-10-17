# PyTorch Media Processing: Three Libraries, Clearer Roles

原文：[Evolution of the PyTorch Media Processing Landscape](https://pytorch.org/blog/evolution-of-the-pytorch-media-processing-landscape/)

## 摘要

PyTorch 将图像、视频和音频的编解码能力集中到 TorchCodec，TorchVision 和 TorchAudio 则聚焦张量变换。此次调整旨在统一用户入口、集中性能优化投入，并减轻原生依赖带来的维护负担；模型、数据集和流水线等其他功能已不再积极开发，文章建议考虑更广泛的生态替代方案。节目通过视频和音频示例解释三个库如何配合，同时说明文章没有提供具体性能测试数据，TorchAudio 的迁移也曾带来较大影响。三个库现已实现 ABI 稳定，不再需要随每次 PyTorch 发布重新构建；后续文章将介绍各库新功能及完整处理流水线的性能优化。

## 对话

Ava: You want to train a model on video, but first you've got to choose a decoder. Why does that feel like a research project before the research project?

Ava：你想用视频训练模型，但首先得选一个解码器。为什么感觉研究还没开始，就得先做一个研究项目？

Brian: That's exactly the problem this article addresses. PyTorch has brought media decoding and encoding together in TorchCodec. There's now a clearer answer to where your images, video, and audio should enter the system.

Brian：这正是这篇文章要解决的问题。PyTorch 已经在 TorchCodec 中整合了媒体解码和编码。现在，图像、视频和音频该从哪里进入系统，答案更清楚了。

Ava: And I'm guessing this matters even more when the model generates media. You need a way to get the result back out, too.

Ava：我猜模型生成媒体时，这一点就更重要了。你还得有办法把生成的结果导出来。

Brian: Right. Decoding turns media files or encoded bytes into tensors, the arrays a model works with. Encoding takes tensors back into media files. Generative models need that return journey.

Brian：对。解码把媒体文件或编码后的字节转换成张量，也就是模型处理的数组。编码则把张量转换回媒体文件。生成式模型需要这个反向过程。

Ava: Let's get the names straight. I'm Ava, a software engineer outside the PyTorch core team. Brian, you're here to help me understand who does what.

Ava：先把名称和分工说清楚。我是 Ava，一名不在 PyTorch 核心团队的软件工程师。Brian，你来帮我弄明白各个工具负责什么。

Brian: Gladly. TorchCodec handles decoding and encoding. TorchVision transforms images and video. TorchAudio transforms audio. Think of TorchCodec as the entrance and exit, with transforms doing work on tensors in between.

Brian：乐意效劳。TorchCodec 负责解码和编码。TorchVision 对图像和视频做变换。TorchAudio 对音频做变换。可以把 TorchCodec 看作入口和出口，而各种变换在两者之间处理张量。

Ava: When you say transforms, you mean operations that change the data, like cropping an image?

Ava：你说的变换，是指裁剪图像这类改变数据的操作吗？

Brian: Exactly. We'll get to a crop example shortly. The basic division is straightforward: move between media and tensors with TorchCodec, then use the other two libraries to transform those tensors.

Brian：没错。我们很快会讲到裁剪的例子。基本分工很简单：用 TorchCodec 在媒体和张量之间转换，再用另外两个库对张量做变换。

Ava: Okay, take me back. What made the old setup confusing? Having a vision library and an audio library sounds reasonable.

Ava：好，先回顾一下。以前的安排为什么让人困惑？有一个视觉库和一个音频库，听起来挺合理的。

Brian: The difficulty was that decoding and encoding were scattered and partly duplicated. Each library built its own stack. Even within TorchVision, video reading had multiple entry points and three backends, meaning different underlying implementations.

Brian：难点在于，解码和编码功能分散在各处，而且部分功能重复。每个库都构建了自己的技术栈。即使在 TorchVision 内部，视频读取也有多个入口和3种后端，也就是说，底层实现各不相同。

Ava: Three backends for video? That already sounds like a choice I'd rather make after coffee. What were the options?

Ava：视频有3种后端？听起来这种选择我还是喝完咖啡再做比较好。都有哪些选项？

Brian: There was PyAV in Python, a C plus plus backend using FFmpeg, a media processing library, and a CUDA backend for NVIDIA graphics processors. Only the PyAV option was available out of the box.

Brian：有 Python 中的 PyAV、使用媒体处理库 FFmpeg 的 C plus plus 后端，以及面向 NVIDIA 图形处理器的 CUDA 后端。只有 PyAV 选项是开箱即用的。

Ava: Wait, so choosing another backend also meant taking on build work?

Ava：等等，也就是说，选择其他后端还得自己做构建工作？

Brian: Yes. The other two required building from source, and they were tied to a specific FFmpeg version. TorchAudio also had readers and writers for video and audio, plus additional audio decoding utilities with different backends.

Brian：是的。另外2种后端需要从源码构建，而且都绑定了特定的 FFmpeg 版本。TorchAudio 也有视频和音频的读取器、写入器，还有使用不同后端的其他音频解码工具。

Ava: And image decoding lived in TorchVision. I can see how the answer to 'where do I decode this?' could get complicated.

Ava：图像解码则在 TorchVision 里。我能理解为什么“这个东西该在哪里解码？”会变成一个复杂的问题。

Brian: That's the background. Over the last two years, the team consolidated the media stack. TorchCodec now brings together decoding and encoding for images, video, and audio, on CPUs, or central processing units, and CUDA.

Brian：背景就是这样。过去2年里，团队整合了媒体处理技术栈。现在，TorchCodec 将图像、视频和音频的解码与编码功能整合在一起，支持 CPU（中央处理器）和 CUDA。

Ava: The article gives three reasons for that move, right? Let's start with the one users feel directly.

Ava：文章给出了这样做的3个理由，对吧？先从用户能直接感受到的那个说起。

Brian: One place to go. Media I/O, meaning input and output, now belongs in TorchCodec. Users get a library designed as a whole, instead of having to choose among separate interfaces spread across libraries.

Brian：有了一个统一的去处。媒体 I/O，也就是输入和输出，现在由 TorchCodec 负责。用户得到的是一个整体设计的库，不必再从分散在不同库中的接口之间做选择。

Ava: So there's less detective work before I can start. What's the second reason?

Ava：这样我在开始之前就不用费力查找了。第二个原因是什么？

Brian: Performance work has a clear home. Previously, effort was spread across implementations, and it wasn't obvious which one should receive an optimization. Now improvements go into TorchCodec, where users can share the benefit.

Brian：性能优化有了明确的归属。以前，精力分散在不同实现上，也不清楚该优化哪一个。现在，改进会进入 TorchCodec，让用户都能受益。

Ava: Does the article say it's actually faster, or just easier to optimize?

Ava：文章说它实际运行得更快，还是只说它更容易优化？

Brian: It says TorchCodec is generally more performant than the previous implementations, particularly for CUDA video decoding. But it doesn't give benchmark numbers or test conditions. We can't turn that into a specific speedup.

Brian：文章说 TorchCodec 的性能总体上优于以前的实现，尤其是在 CUDA 视频解码方面。但它没有给出基准测试数字或测试条件，所以我们不能据此说出具体快了多少。

Ava: Fair enough. What's the third reason? I suspect someone had a very long list of dependencies.

Ava：说得对。第三个原因是什么？我猜有人列出了一份很长的依赖项清单。

Brian: They did. At the time of writing, the stack involved six major FFmpeg versions, four through nine. It also involved NVIDIA's codec development kit for decoding on a GPU, or graphics processing unit.

Brian：没错。写这篇文章时，这套技术栈涉及从第4版到第9版的6个主要 FFmpeg 版本，还涉及 NVIDIA 的编解码器开发工具包，用于在 GPU（图形处理器）上解码。

Ava: And images bring their own libraries into the picture?

Ava：图像也会带来各自需要的库？

Brian: Yes, libraries such as libjpeg and libpng for different image formats. These dependencies also have licensing rules affecting what can ship. And they're C or C plus plus libraries, rather than pure Python dependencies.

Brian：是的，比如处理不同图像格式的 libjpeg 和 libpng。这些依赖项还有许可规则，会影响哪些内容能够随产品发布。而且它们是 C 或 C plus plus 库，不是纯 Python 依赖项。

Ava: Which makes builds and releases harder. The complexity hasn't disappeared, then. It's been gathered into one place.

Ava：这会让构建和发布更困难。所以复杂性并没有消失，而是被集中到了一个地方。

Brian: Exactly. That place is TorchCodec. It greatly reduces the maintenance burden for TorchVision and TorchAudio. To recap this part: users get one entry point, optimization gets one home, and dependency maintenance gets contained.

Brian：正是。这个地方就是 TorchCodec。它大幅减轻了 TorchVision 和 TorchAudio 的维护负担。回顾一下这一部分：用户有了统一入口，优化有了统一归属，依赖项的维护也集中起来了。

Ava: Let's move on to those other libraries. They used to offer much more than transforms. What's happening to models and datasets?

Ava：再说说另外两个库。它们过去提供的功能远不止变换。模型和数据集现在怎么办？

Brian: TorchVision and TorchAudio are now focused on transforms. The rest of their broad feature set isn't under active development. The article points to the wider ecosystem, including HuggingFace libraries, for models, datasets, and pipelines.

Brian：TorchVision 和 TorchAudio 现在专注于变换。它们原先广泛的其他功能不再处于积极开发中。对于模型、数据集和流水线，文章指向了更广泛的生态系统，包括 HuggingFace 的库。

Ava: Why draw the boundary there? Is that based on how people use the libraries?

Ava：为什么把界限划在这里？这是根据人们使用这些库的方式决定的吗？

Brian: Yes. The team observed that transforms were the most active usage areas. Models, datasets, and pipelines had lagged as alternatives grew. Narrowing the scope lets the team concentrate on strengths that other ecosystem tools don't cover.

Brian：是的。团队观察到，变换是使用最活跃的领域。随着替代方案的发展，模型、数据集和流水线方面已经落后。缩小范围能让团队集中精力做好其他生态工具没有覆盖的强项。

Ava: That sounds sensible, but changing the boundaries of a library can hurt existing users. How rough was this transition?

Ava：听起来合理，但改变库的功能范围可能会影响现有用户。这次过渡有多难？

Brian: The article says it was particularly disruptive for TorchAudio. Many APIs, meaning application programming interfaces, were deprecated and eventually removed. Community feedback also changed the plan: several popular APIs originally marked for removal were kept.

Brian：文章说，这对 TorchAudio 的影响尤其大。许多 API，也就是应用程序编程接口，先被弃用，最终又被移除。社区反馈也改变了计划：几个原本标记为将被移除的热门 API 保留了下来。

Ava: That's a useful detail. We shouldn't hear 'smaller library' and assume every older feature vanished.

Ava：这个细节很有用。听到“库变小了”，我们不能就认定所有旧功能都消失了。

Brian: Right. The article doesn't list those retained APIs here. Its argument is that a smaller TorchAudio can actually be maintained, giving users something they can continue to rely on.

Brian：对。文章没有在这里列出那些保留下来的 API。它的观点是，规模更小的 TorchAudio 才能得到切实维护，让用户今后也能继续依赖它。

Ava: Let's make this concrete. Suppose I've got a video file. Walk me through the example without making me listen to every bracket.

Ava：说得具体一点。假设我有一个视频文件。带我过一遍示例，但别让我听你逐个念方括号。

Brian: Deal. TorchCodec's VideoDecoder opens the file. The example selects ten frames, starting at frame index ten and stopping before twenty. That returns a FrameBatch, a batch object containing the frame data.

Brian：好。TorchCodec 的 VideoDecoder 打开文件。示例选取了10帧，从帧索引10开始，到20之前结束。它返回一个 FrameBatch，也就是包含帧数据的批次对象。

Ava: What does that frame data look like before the transforms touch it?

Ava：变换处理之前，帧数据是什么样的？

Brian: It's a tensor of unsigned eight-bit values. Its dimensions are frames, channels, height, and width. The example passes that data into TorchVision's v2 transforms, meaning its version two transform interface.

Brian：它是一个由无符号8位值组成的张量，维度依次是帧、通道、高和宽。示例把这些数据传入 TorchVision 的 v2 transforms，也就是它的第2版变换接口。

Ava: And that's where the crop happens?

Ava：裁剪就是在这一步发生的？

Brian: Yes. It combines RandomResizedCrop, a random crop and resize operation with an output size of two hundred twenty-four, and ToDtype. The latter converts the data to thirty-two-bit floating point, with scaling enabled.

Brian：对。示例组合使用了 RandomResizedCrop 和 ToDtype。前者会随机裁剪并调整大小，输出尺寸为224；后者会在启用缩放的情况下，把数据转换为32位浮点数。

Ava: So the handoff is the tensor. The decoder gets the frames out, and the transforms prepare those frames for what comes next.

Ava：所以交接的是张量。解码器取出帧，变换操作再把这些帧准备好，供后续步骤使用。

Brian: Exactly. Audio follows the same pattern. TorchCodec's AudioDecoder reads all the samples in the example. Those samples go into TorchAudio's MelSpectrogram transform, which represents audio in terms of time and frequency using the mel scale.

Brian：没错。音频遵循同样的模式。示例中，TorchCodec 的 AudioDecoder 读取所有采样数据，再把它们交给 TorchAudio 的 MelSpectrogram 变换；它使用梅尔刻度，以时间和频率来表示音频。

Ava: There's also a sample rate in that example. Where does it come from?

Ava：那个示例里还有一个采样率。它是从哪里来的？

Brian: From the decoded samples. The example uses that sample rate to configure MelSpectrogram. Images follow the same overall pattern: decoding returns plain tensors that can go straight into TorchVision's version two transforms.

Brian：从解码后的样本中获取。示例用这个采样率来配置 MelSpectrogram。图像也遵循相同的整体流程：解码返回普通张量，可以直接传入 TorchVision 的第2版变换接口。

Ava: And when the model produces something we want to save, we head back through TorchCodec?

Ava：模型生成了我们想保存的内容时，就再通过 TorchCodec 输出？

Brian: Yes. The article names VideoEncoder, AudioEncoder, JpegEncoder, and PngEncoder for that direction. The key point is the same across media: decode, transform the tensors, and encode when you need a media file.

Brian：是的。文章提到了用于这个方向的 VideoEncoder、AudioEncoder、JpegEncoder 和 PngEncoder。对各种媒体来说，关键流程都一样：解码、对张量做变换，并在需要媒体文件时编码。

Ava: There's one more change for framework engineers: releases. What does ABI stable mean here?

Ava：对于框架工程师来说，还有一项变化是版本发布。这里的 ABI stable 是什么意思？

Brian: ABI means application binary interface. The article says all three libraries are now ABI stable. A given library version isn't tied to just one PyTorch version; it keeps working with subsequent PyTorch versions.

Brian：ABI 指应用程序二进制接口。文章说，这三个库现在都实现了 ABI 稳定性。某个库版本不再只绑定一个 PyTorch 版本；它也能继续与后续的 PyTorch 版本配合使用。

Ava: So they don't need a rebuild for every PyTorch release?

Ava：所以他们不需要为 PyTorch 的每次发布重新构建？

Brian: Correct. Their release schedules no longer match PyTorch's, and that's expected. For users still decoding or encoding through TorchVision or TorchAudio, the article recommends moving to TorchCodec, starting with the migration guide.

Brian：对。它们的发布周期不再与 PyTorch 同步，这是预期之中的。对于仍在通过 TorchVision 或 TorchAudio 进行解码或编码的用户，文章建议迁移到 TorchCodec，并从迁移指南入手。

Ava: And if something's missing?

Ava：如果缺少某些功能呢？

Brian: Report it on GitHub. There's also a follow-up post planned about new features and getting the most performance from a full decoding and transform pipeline. This article establishes the direction, but leaves those details for later.

Brian：到 GitHub 上反馈。后续还计划发布一篇文章，介绍新功能，以及如何让完整的解码和变换流水线发挥最佳性能。本文明确了方向，但把这些细节留待以后讨论。

Ava: Let's finish with three things to remember. First, use TorchCodec for decoding and encoding images, video, and audio.

Ava：最后来总结三点。第一，使用 TorchCodec 解码和编码图像、视频及音频。

Brian: Second, use TorchVision for image and video transforms, and TorchAudio for audio transforms. Look to the wider ecosystem for the other needs discussed here.

Brian：第二，使用 TorchVision 进行图像和视频变换，使用 TorchAudio 进行音频变换。至于这里讨论的其他需求，可以看看更广泛的生态系统。

Ava: Third, the smaller scopes and ABI stability change maintenance and releases. Expect independent release schedules, and plan your move from the older media interfaces.

Ava：第三，范围缩小和 ABI 稳定性会改变维护与发布方式。要做好各自按独立周期发布的准备，并规划如何从旧的媒体接口迁移。

Brian: Thanks for listening, and we'll catch you next time.

Brian：感谢收听，我们下次再见。

## 术语

| Term | 释义 |
|---|---|
| TorchCodec | 集中提供图像、视频和音频编解码能力的 PyTorch 媒体库。 |
| TorchVision | 本文中定位为聚焦图像和视频张量变换的库。 |
| TorchAudio | 本文中定位为聚焦音频张量变换的库。 |
| decoding | 解码：将媒体文件或编码后的字节转换为张量。 |
| encoding | 编码：将张量转换回媒体文件。 |
| transforms | 对张量执行的变换操作，例如裁剪、调整大小或生成音频特征。 |
| backend | 后端：接口背后实际执行工作的底层实现。 |
| FFmpeg | 媒体处理软件及库；文中多个旧编解码实现依赖它。 |
| CUDA | NVIDIA GPU 计算平台；文中用于描述 GPU 上的媒体处理能力。 |
| FrameBatch | 视频帧批次对象，示例通过其 data 属性取得帧张量。 |
| RandomResizedCrop | 随机裁剪并调整到指定输出尺寸的变换。 |
| MelSpectrogram | 梅尔频谱图变换，用梅尔频率尺度表示音频的时频信息。 |
| sample rate | 采样率；示例用解码所得音频的采样率配置音频变换。 |
| ABI | 应用程序二进制接口；本文的 ABI 稳定意味着库不再需要随每次 PyTorch 发布重新构建。 |
| deprecated | 已弃用：接口被标记为不再推荐使用，与已经移除并不相同。 |

## 口语表达

| Phrase | 释义 |
|---|---|
| Let's get the names straight. | 我们先把这些名称和各自的角色理清楚。 |
| Take me back. | 带我回顾一下之前的情况。 |
| Wait, so choosing another backend also meant taking on build work? | 等等，所以选另一个后端还意味着得自己承担构建工作？ |
| Fair enough. | 有道理；可以理解。 |
| Let's make this concrete. | 我们用具体例子来说明。 |
| Walk me through the example. | 带我一步步过一下这个例子。 |
| The key point is the same across media. | 不同媒体类型的核心要点都是一样的。 |
| Let's finish with three things to remember. | 最后用三个需要记住的要点收尾。 |
