# PyTorch Media Processing: Three Libraries, Clearer Roles

原文：[Evolution of the PyTorch Media Processing Landscape](https://pytorch.org/blog/evolution-of-the-pytorch-media-processing-landscape/)

## 摘要

PyTorch 将图像、视频和音频的编解码能力集中到 TorchCodec，TorchVision 和 TorchAudio 则聚焦张量变换。此次调整旨在统一用户入口、集中性能优化投入，并减轻原生依赖带来的维护负担；模型、数据集和流水线等其他功能已不再积极开发，文章建议考虑更广泛的生态替代方案。节目通过视频和音频示例解释三个库如何配合，同时说明文章没有提供具体性能测试数据，TorchAudio 的迁移也曾带来较大影响。三个库现已实现 ABI 稳定，不再需要随每次 PyTorch 发布重新构建；后续文章将介绍各库新功能及完整处理流水线的性能优化。

## 对话

Ava: You want to train a model on video, but first you've got to choose a decoder.

Ava：想用视频训练模型，首先得选个解码器。

Why does that feel like a research project before the research project?

怎么感觉研究还没开始，就得先做一番研究？

Brian: That's exactly the problem this article addresses.

Brian：这正是这篇文章要解决的问题。

PyTorch has brought media decoding and encoding together in TorchCodec.

PyTorch 通过 TorchCodec 统一了媒体解码和编码。

There's now a clearer answer to where your images, video, and audio should enter the system.

现在，图像、视频和音频该从哪里进入系统，有了更明确的答案。

Ava: And I'm guessing this matters even more when the model generates media.

Ava：我猜模型生成媒体内容时，这一点更重要。

You need a way to get the result back out, too.

还得有办法把结果输出出来。

Brian: Right.

Brian：没错。

Decoding turns media files or encoded bytes into tensors, the arrays a model works with.

解码把媒体文件或编码后的字节转换成张量，也就是模型处理的数组。

Encoding takes tensors back into media files.

编码则把张量转换回媒体文件。

Generative models need that return journey.

生成式模型也需要这一步。

Ava: Let's get the names straight.

Ava：先把几个名字理清楚。

I'm Ava, a software engineer outside the PyTorch core team.

我是 Ava，一名不在 PyTorch 核心团队的软件工程师。

Brian, you're here to help me understand who does what.

Brian，你来帮我弄清楚各个库负责什么。

Brian: Gladly. TorchCodec handles decoding and encoding.

Brian：好啊。TorchCodec 负责解码和编码。

TorchVision transforms images and video. TorchAudio transforms audio.

TorchVision 负责图像和视频变换，TorchAudio 负责音频变换。

Think of TorchCodec as the entrance and exit, with transforms doing work on tensors in between.

可以把 TorchCodec 看作入口和出口，变换操作则在中间处理张量。

Ava: When you say transforms, you mean operations that change the data, like cropping an image?

Ava：你说的变换，是指裁剪图像这类改变数据的操作？

Brian: Exactly. We'll get to a crop example shortly.

Brian：对。稍后我们会讲一个裁剪的例子。

The basic division is straightforward: move between media and tensors with TorchCodec, then use the other two libraries to transform those tensors.

分工很简单：用 TorchCodec 在媒体和张量之间转换，再用另外两个库变换张量。

Ava: Okay, take me back. What made the old setup confusing?

Ava：好，先说说以前的情况。旧方案为什么让人困惑？

Having a vision library and an audio library sounds reasonable.

一个视觉库、一个音频库，听起来挺合理。

Brian: The difficulty was that decoding and encoding were scattered and partly duplicated.

Brian：问题是解码和编码功能分散在各处，还有部分重复。

Each library built its own stack.

每个库都建立了自己的一套处理方式。

Even within TorchVision, video reading had multiple entry points and three backends, meaning different underlying implementations.

即使在 TorchVision 内部，读取视频也有多个入口和三种后端，也就是三种底层实现。

Ava: Three backends for video?

Ava：视频有三种后端？

That already sounds like a choice I'd rather make after coffee. What were the options?

光听着就觉得得先喝杯咖啡再选。都有哪些？

Brian: There was PyAV in Python, a C plus plus backend using FFmpeg, a media processing library, and a CUDA backend for NVIDIA graphics processors.

Brian：有 Python 版的 PyAV、使用媒体处理库 FFmpeg 的 C++ 后端，以及用于 NVIDIA 图形处理器的 CUDA 后端。

Only the PyAV option was available out of the box.

只有 PyAV 开箱即用。

Ava: Wait, so choosing another backend also meant taking on build work?

Ava：等等，选其他后端还得自己编译？

Brian: Yes.

Brian：对。

The other two required building from source, and they were tied to a specific FFmpeg version.

另外两种都需要从源码构建，而且依赖特定版本的 FFmpeg。

TorchAudio also had readers and writers for video and audio, plus additional audio decoding utilities with different backends.

TorchAudio 也提供视频和音频的读取与写入功能，另有采用不同后端的音频解码工具。

Ava: And image decoding lived in TorchVision.

Ava：图像解码则在 TorchVision 里。

I can see how the answer to 'where do I decode this? ' could get complicated.

难怪“我该在哪里解码？”会变成复杂的问题。

Brian: That's the background. Over the last two years, the team consolidated the media stack.

Brian：背景就是这样。过去两年，团队整合了媒体处理功能。

TorchCodec now brings together decoding and encoding for images, video, and audio, on CPUs, or central processing units, and CUDA.

现在 TorchCodec 统一提供图像、视频和音频的解码与编码，支持 CPU（中央处理器）和 CUDA。

Ava: The article gives three reasons for that move, right?

Ava：文章给出了这样调整的三个理由，对吧？

Let's start with the one users feel directly.

先说用户最有感受的那个。

Brian: One place to go. Media I/O, meaning input and output, now belongs in TorchCodec.

Brian：入口统一了。媒体 I/O，也就是输入和输出，现在都由 TorchCodec 负责。

Users get a library designed as a whole, instead of having to choose among separate interfaces spread across libraries.

用户可以使用一套统一设计的库，不必在分散于各个库的接口之间做选择。

Ava: So there's less detective work before I can start. What's the second reason?

Ava：这样开始之前就不用费劲摸索了。第二个理由呢？

Brian: Performance work has a clear home.

Brian：性能优化有了明确的落脚点。

Previously, effort was spread across implementations, and it wasn't obvious which one should receive an optimization.

以前，精力分散在不同实现上，也不清楚该优化哪一个。

Now improvements go into TorchCodec, where users can share the benefit.

现在改进都集中在 TorchCodec，用户也能共同受益。

Ava: Does the article say it's actually faster, or just easier to optimize?

Ava：文章说它确实更快，还是只说更容易优化？

Brian: It says TorchCodec is generally more performant than the previous implementations, particularly for CUDA video decoding.

Brian：文章说 TorchCodec 通常比之前的实现性能更好，尤其是在 CUDA 视频解码方面。

But it doesn't give benchmark numbers or test conditions.

但文章没有提供基准测试数据或测试条件。

We can't turn that into a specific speedup.

所以不能据此给出具体的加速倍数。

Ava: Fair enough. What's the third reason?

Ava：有道理。第三个理由呢？

I suspect someone had a very long list of dependencies.

我猜有人列了一长串依赖项。

Brian: They did.

Brian：确实如此。

At the time of writing, the stack involved six major FFmpeg versions, four through nine.

写作本文时，这套技术栈涉及 FFmpeg 4 到 9，共六个主要版本。

It also involved NVIDIA's codec development kit for decoding on a GPU, or graphics processing unit.

还需要 NVIDIA 的编解码器开发套件，用 GPU（图形处理器）解码。

Ava: And images bring their own libraries into the picture?

Ava：处理图像也得引入专门的库吧？

Brian: Yes, libraries such as libjpeg and libpng for different image formats.

Brian：对，比如处理不同图像格式的 libjpeg 和 libpng。

These dependencies also have licensing rules affecting what can ship.

这些依赖项还有许可要求，会影响哪些内容可以随软件发布。

And they're C or C plus plus libraries, rather than pure Python dependencies.

而且它们是 C 或 C++ 库，并非纯 Python 依赖。

Ava: Which makes builds and releases harder. The complexity hasn't disappeared, then.

Ava：这就让构建和发布更难了。所以复杂性并没有消失。

It's been gathered into one place.

只是被集中到了一处。

Brian: Exactly. That place is TorchCodec.

Brian：没错，这一处就是 TorchCodec。

It greatly reduces the maintenance burden for TorchVision and TorchAudio.

它大大减轻了 TorchVision 和 TorchAudio 的维护负担。

To recap this part: users get one entry point, optimization gets one home, and dependency maintenance gets contained.

总结一下：用户有了统一入口，优化有了统一归处，依赖维护也集中起来了。

Ava: Let's move on to those other libraries. They used to offer much more than transforms.

Ava：再说说另外两个库。它们以前提供的远不止变换功能。

What's happening to models and datasets?

模型和数据集现在怎么样了？

Brian: TorchVision and TorchAudio are now focused on transforms.

Brian：TorchVision 和 TorchAudio 现在专注于变换功能。

The rest of their broad feature set isn't under active development.

它们原先提供的其他大量功能已不再积极开发。

The article points to the wider ecosystem, including HuggingFace libraries, for models, datasets, and pipelines.

文章建议通过更广泛的生态系统，包括 HuggingFace 的库，来获取模型、数据集和流水线。

Ava: Why draw the boundary there? Is that based on how people use the libraries?

Ava：为什么把边界划在这里？是根据大家的使用情况决定的吗？

Brian: Yes. The team observed that transforms were the most active usage areas.

Brian：对。团队观察到，变换功能是使用最活跃的部分。

Models, datasets, and pipelines had lagged as alternatives grew.

随着替代方案发展，模型、数据集和流水线方面的功能逐渐落后。

Narrowing the scope lets the team concentrate on strengths that other ecosystem tools don't cover.

缩小范围后，团队就能专注于其他生态工具没有覆盖的优势领域。

Ava: That sounds sensible, but changing the boundaries of a library can hurt existing users.

Ava：听起来合理，但调整库的功能范围可能影响现有用户。

How rough was this transition?

这次过渡有多不顺利？

Brian: The article says it was particularly disruptive for TorchAudio.

Brian：文章说，对 TorchAudio 用户的影响尤其大。

Many APIs, meaning application programming interfaces, were deprecated and eventually removed.

许多 API（应用程序编程接口）先被标记为弃用，最终被移除。

Community feedback also changed the plan: several popular APIs originally marked for removal were kept.

社区反馈也改变了计划：一些原定移除的热门 API 被保留了下来。

Ava: That's a useful detail.

Ava：这点很值得一提。

We shouldn't hear 'smaller library' and assume every older feature vanished.

不能一听说“库变小了”，就以为所有旧功能都没了。

Brian: Right. The article doesn't list those retained APIs here.

Brian：对。文章这里没有列出具体保留了哪些 API。

Its argument is that a smaller TorchAudio can actually be maintained, giving users something they can continue to rely on.

文章的观点是，规模更小的 TorchAudio 才能得到持续维护，让用户继续放心使用。

Ava: Let's make this concrete. Suppose I've got a video file.

Ava：说具体点吧。假设我有一个视频文件。

Walk me through the example without making me listen to every bracket.

带我过一遍这个例子，别逐个念那些括号。

Brian: Deal. TorchCodec's VideoDecoder opens the file.

Brian：好。TorchCodec 的 VideoDecoder 会打开文件。

The example selects ten frames, starting at frame index ten and stopping before twenty.

示例从索引为 10 的帧开始，取到索引 20 之前，共十帧。

That returns a FrameBatch, a batch object containing the frame data.

它返回一个 FrameBatch，也就是包含帧数据的批量对象。

Ava: What does that frame data look like before the transforms touch it?

Ava：变换处理之前，帧数据是什么样的？

Brian: It's a tensor of unsigned eight-bit values.

Brian：它是一个由无符号 8 位数值组成的张量。

Its dimensions are frames, channels, height, and width.

它的维度依次是帧、通道、高度和宽度。

The example passes that data into TorchVision's v2 transforms, meaning its version two transform interface.

示例把这些数据传给 TorchVision 的 v2 变换，也就是第二版变换接口。

Ava: And that's where the crop happens?

Ava：裁剪就是在这一步做的？

Brian: Yes.

Brian：对。

It combines RandomResizedCrop, a random crop and resize operation with an output size of two hundred twenty-four, and ToDtype.

它组合了 RandomResizedCrop 和 ToDtype；前者随机裁剪并缩放，输出尺寸为 224。

The latter converts the data to thirty-two-bit floating point, with scaling enabled.

后者将数据转换为 32 位浮点数，并启用数值缩放。

Ava: So the handoff is the tensor.

Ava：所以传递给变换的是张量。

The decoder gets the frames out, and the transforms prepare those frames for what comes next.

解码器提取帧，变换则为后续处理准备这些帧。

Brian: Exactly. Audio follows the same pattern.

Brian：没错。音频也是同样的流程。

TorchCodec's AudioDecoder reads all the samples in the example.

示例中，TorchCodec 的 AudioDecoder 会读取全部采样数据。

Those samples go into TorchAudio's MelSpectrogram transform, which represents audio in terms of time and frequency using the mel scale.

这些采样数据传给 TorchAudio 的 MelSpectrogram 变换，用梅尔刻度表示音频随时间变化的频率信息。

Ava: There's also a sample rate in that example. Where does it come from?

Ava：那个示例里还有采样率。它是从哪里来的？

Brian: From the decoded samples. The example uses that sample rate to configure MelSpectrogram.

Brian：来自解码后的样本。示例用这些样本的采样率来配置 MelSpectrogram。

Images follow the same overall pattern: decoding returns plain tensors that can go straight into TorchVision's version two transforms.

图像也遵循同样的模式：解码得到普通张量，可以直接传给 TorchVision 的 v2 变换。

Ava: And when the model produces something we want to save, we head back through TorchCodec?

Ava：模型生成了想保存的内容时，就再交给 TorchCodec？

Brian: Yes.

Brian：对。

The article names VideoEncoder, AudioEncoder, JpegEncoder, and PngEncoder for that direction.

文章提到，这一步可用 VideoEncoder、AudioEncoder、JpegEncoder 和 PngEncoder。

The key point is the same across media: decode, transform the tensors, and encode when you need a media file.

各种媒体的关键步骤都一样：解码、变换张量，需要媒体文件时再编码。

Ava: There's one more change for framework engineers: releases.

Ava：对框架工程师来说，还有一个变化：发布方式。

What does ABI stable mean here?

这里说的 ABI 稳定是什么意思？

Brian: ABI means application binary interface.

Brian：ABI 指应用程序二进制接口。

The article says all three libraries are now ABI stable.

文章说，这三个库现在都实现了 ABI 稳定。

A given library version isn't tied to just one PyTorch version; it keeps working with subsequent PyTorch versions.

某个库版本不再只对应一个 PyTorch 版本，后续的 PyTorch 版本也能继续使用它。

Ava: So they don't need a rebuild for every PyTorch release?

Ava：所以每次 PyTorch 发布新版本，都不用重新构建这些库？

Brian: Correct. Their release schedules no longer match PyTorch's, and that's expected.

Brian：没错。它们的发布周期不再与 PyTorch 同步，这是预期之内的。

For users still decoding or encoding through TorchVision or TorchAudio, the article recommends moving to TorchCodec, starting with the migration guide.

仍通过 TorchVision 或 TorchAudio 解码、编码的用户，文章建议从迁移指南入手，转用 TorchCodec。

Ava: And if something's missing?

Ava：如果缺少某项功能呢？

Brian: Report it on GitHub.

Brian：到 GitHub 上反馈。

There's also a follow-up post planned about new features and getting the most performance from a full decoding and transform pipeline.

后续还计划发一篇文章，介绍新功能，以及如何让完整的解码与变换流程发挥最佳性能。

This article establishes the direction, but leaves those details for later.

本文明确了方向，具体细节留待后续介绍。

Ava: Let's finish with three things to remember.

Ava：最后总结三点。

First, use TorchCodec for decoding and encoding images, video, and audio.

第一，用 TorchCodec 解码和编码图像、视频及音频。

Brian: Second, use TorchVision for image and video transforms, and TorchAudio for audio transforms.

Brian：第二，用 TorchVision 处理图像和视频变换，用 TorchAudio 处理音频变换。

Look to the wider ecosystem for the other needs discussed here.

本文提到的其他需求，可以在更广泛的生态中寻找解决方案。

Ava: Third, the smaller scopes and ABI stability change maintenance and releases.

Ava：第三，职责范围缩小和 ABI 稳定会改变维护与发布方式。

Expect independent release schedules, and plan your move from the older media interfaces.

各库会独立发布，也要规划好从旧媒体接口迁移。

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
