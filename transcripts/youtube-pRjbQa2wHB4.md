# PyTorch 2.14 Release Live Q&A

来源：[https://www.youtube.com/watch?v=pRjbQa2wHB4](https://www.youtube.com/watch?v=pRjbQa2wHB4)

## 简介

PyTorch 2.14 introduces updates across compilation, distributed training, performance, dynamic shapes, Apple Silicon, and accelerator platforms. Highlights include NVGEMM, a CuTeDSL-generated GEMM backend for Inductor; the new nccl2 backend for PyTorch Distributed; fault-tolerant process-group reconfiguration in c10d; native linear algebra on Apple Silicon; and declarative dynamic shapes with @dynamic_spec.

Bring your questions about the release to our live Q&A. Andrey Talman (Meta), Nikkita Shulga (Thinking Machines Lab), Joe Spisak (Reflection AI), and Chris Gottbrath (Gottbrath Tech, moderator) will share an overview of PyTorch 2.14 and answer community questions about PyTorch and the new capabilities in the release.

Topics will include:

- NVGEMM and CuTeDSL-generated CUTLASS kernels in Inductor
- The new nccl2 backend for PyTorch Distributed
- Fault-tolerant collectives and process-group reconfiguration in c10d
- Native linear algebra and additional Metal kernel improvements on Apple Silicon
- torch.switch and CUDA graph capture for torch.while_loop
- Declarative dynamic shapes with @dynamic_spec
- Experimental torch.compile support for complex-valued tensors
- Expanded ROCm, Intel XPU, and NVIDIA platform support

PyTorch 2.14 includes 2,995 commits from 487 contributors since PyTorch 2.13. The release includes work across compilation, distributed communication, device support, and accelerator platforms. PyTorch Conference North America 2026 takes place October 20–21 in San Jose, with sessions spanning compiler and runtime work, distributed communication, device portability, release engineering, CI, observability, accelerator integration, contributor infrastructure, and more. Explore PyTorch Conference North America 2026.

## 字幕

### 00:00:06 - 00:00:33

Okay. Uh hello everyone. Um and welcome to our PyTorch uh 2.14 uh release live uh Q&A. Um uh let me scroll back up to my uh my comments here. Um this session will start with uh with introducing our experts um and then a short overview of the key updates in PyTorch uh 21.14 followed by um the fun part which is an open Q&A covering your questions about this release.

好的，大家好，欢迎参加我们的 PyTorch 2.14 版本发布直播问答。我先往上翻一下，找到我的备注。本场活动会先介绍几位专家，再简要概述 PyTorch 21.14 的主要更新，然后进入有趣的部分：开放问答，回答大家关于这个版本的问题。

### 00:00:30 - 00:00:55

Um first I'll introduce myself. I'm I'm Chris Scott. I'm your moderator for today's webinar. I've been involved in the PyTorch community for about eight years focusing on focusing on product and marketing uh type angles. Um these days I run a consulting practice uh focusing on emerging AI technologies and I'm delighted to welcome a panel of three absolute PyTorch experts. We have Andre Talman uh Joe Spizac and Nikita Schulga.

先介绍一下我自己。我是 Chris Scott，今天这场网络研讨会的主持人。我参与 PyTorch 社区大约八年了，主要关注产品和营销方面。现在我经营一家专注于新兴 AI 技术的咨询公司。很高兴请来三位顶尖的 PyTorch 专家：Andre Talman、Joe Spizac 和 Nikita Schulga。

### 00:00:56 - 00:01:19

Uh and they're here to answer your questions about PyTorch and generally to you know spread the word. Um so Andre is a release engineer in the PyTorch Dev Info team here at Meta or at Meta. I'm sorry. I used to be at Meta so that just rolls off my tongue. It's no longer accurate. at Meta. Uh he drives uh the release process and keeps uh the continuous integration and continuous development infrastructure humming.

他们会在这里回答大家关于 PyTorch 的问题，也会向更多人介绍它。Andre 是 Meta 的 PyTorch Dev Info 团队的一名发布工程师。抱歉，我刚才说“这里的 Meta”，是因为我以前在 Meta 工作，说顺口了，现在已经不准确了。他负责推动发布流程，并保障持续集成和持续开发基础设施顺畅运行。

### 00:01:16 - 00:01:48

Uh Joe um is the uh VP of product uh and head of open source at Reflection AI. Uh he's one of the PyTorch core maintainers and among the strongest advocates that I know of for open source, open models and open infrastructure across the AI community. Uh and always a fun uh fun person to to hear from. Nikita is a PyTorch core maintainer uh and a key contributor and reviewer really active across just a bunch of areas uh in in the core area of PyTorch.

Joe 是 Reflection AI 的产品副总裁兼开源负责人。他是 PyTorch 的核心维护者之一，也是我认识的 AI 社区中最坚定倡导开源、开放模型和开放基础设施的人之一。听他讲话总是很有意思。Nikita 是 PyTorch 的核心维护者，也是重要的贡献者和评审者，在 PyTorch 核心的许多领域都很活跃。

### 00:01:46 - 00:02:09

So you probably see his name on on lots of things. Um I'll give a brief overview now in a in a moment here about the release uh for a few minutes and I'm would encourage you if you're in the in the audience uh to to use that time to think about what questions you might have um for our experts. uh you can put those in LinkedIn and and the and the YouTube tools.

所以你们可能在很多地方都见过他的名字。接下来我会花几分钟简要介绍这个版本。如果你正在观看，也请趁这段时间想想有什么问题要问我们的专家。你可以通过 LinkedIn 和 YouTube 的工具提交问题。

### 00:02:06 - 00:02:35

Uh we'll uh we'll cue them up for our experts to answer. We will we'll also have a few seed questions up uh on our you know prepared uh based on ongoing community discussions which you can join by uh looking them up in uh discuss.pytor.org and dev discuss.pytor.org or hopping on PyTorch Slack. Um those are just here to to kind of keep things flowing and make sure we touch on some important topics but it's more fun really to answer the questions that come in from the community.

我们会把问题整理出来，交给专家回答。我们也准备了几个引导性问题，依据的是正在进行的社区讨论；大家可以到 discuss.pytor.org、dev discuss.pytor.org 查找并参与讨论，或者加入 PyTorch Slack。这些问题只是用来让讨论顺畅进行，并确保谈到一些重要话题；回答社区发来的问题其实更有意思。

### 00:02:32 - 00:03:07

So, please don't be shy. Okay, so an overview of the release. Uh, this actually seems to me to be a pretty significant release despite, you know, the the numbering 21.14 makes it seem small, but it's not really, I don't think, at all. Um, uh, it really moves the needle on like compilation, uh, with some new concepts in compilation, fault tolerance, uh, and, uh, platform support. As always, it's a huge community left with almost 3,000 commits from almost 500 uh different uh contributors, 487 this year or this time.

所以，请大家尽管提问。下面概述一下这个版本。在我看来，这其实是一次相当重要的发布。虽然编号 21.14 看起来像是一次小更新，但我完全不觉得它小。它在编译方面带来了实质进展，包括一些新的编译概念，也推进了容错和平台支持。和往常一样，这背后是庞大的社区工作：近 3,000 次提交，来自近 500 位不同的贡献者，准确说这次是 487 位。

### 00:03:05 - 00:03:42

Um they made improvements all across PyTorch and I encourage folks to look at the release release blog and the release notes for uh for more detail to supplement this uh this Q&A. A few of the most spec uh most significant changes this time around are uh NV gym from Nvidia brings cute DSL uh generated cutless kernels to inductor. So this is like a performance uh improvement uh for uh modern uh GPUs uh with epilog fusion scaled uh NV FP4 gems and grouped uh reduction epilogues autotuned alongside Triton and ATIN.

他们改进了 PyTorch 的方方面面。建议大家阅读发布博客和发布说明，了解更多细节，作为这场问答的补充。这次最重要的变化之一是，Nvidia 的 NV gym 把由 cute DSL 生成的 cutless kernels 带进了 inductor。这给现代 GPUs 带来了性能改进，其中有 epilog fusion、scaled NV FP4 gems 和 grouped reduction epilogues，并与 Triton 和 ATIN 一起进行自动调优。

### 00:03:42 - 00:04:13

We have a preview of our uh some really deep changes in our in our um uh distributed backend. So rewritten nickel backend for PyTorch. This is ported from sort of a an area where we've been doing some experimentation called torchcoms implementing uh the full collective contract uh with non-blocking communication communicators and advanced features such as fault tolerance and windows uh designed as drag and drop replacements for existing nickel C10 backend.

我们还预览了分布式后端的一些深层改动：为 PyTorch 重写的 nickel backend。它移植自我们一直用于实验的一个领域，叫 torchcoms，实现了完整的 collective contract，提供 non-blocking communication communicators，以及容错和 windows 等高级功能，设计上可以直接替换现有的 nickel C10 backend。

### 00:04:10 - 00:04:37

Um, so this I think is really uh the fault tolerance I think is going to be is going to be a big uh thing as we uh go forward and I think this is really laying really strong uh uh foundations for that. Uh fault tolerance also becomes a first class citizen in the C10 uh D uh with in place process group reconfiguration, one-sided RMA windows and a flight recorder that works for any backend rather than only for nickel.

我认为，随着我们继续推进，容错会成为一件大事，而这些改动正在为它打下非常扎实的基础。容错在 C10 D 中也成为一等公民，支持原地重新配置 process group、单侧 RMA windows，以及适用于任何 backend、而不只适用于 nickel 的 flight recorder。

### 00:04:35 - 00:05:11

Uh Apple silicon also got a lot more u functionality with more native linear uh algebra operators including ig uh QR and Cholski alongside uh fast production operations and further uh uh MPS graph to metal uh kernel migrations. So performance improvements uh in many places on Apple um torch switch generalize uh the torch cond to multi-way branching. So this is like a language improvement in the way you can think about um uh uh expressing uh things that used to cause graph breaks in PyTorch.

Apple silicon 也获得了更多功能，包括更多原生线性代数算子，例如 ig、QR 和 Cholski，还有快速的 production operations，以及进一步将 MPS graph 迁移到 metal kernels。因此，Apple 平台上很多地方的性能都有提升。torch switch 把 torch cond 扩展为多路分支。这可以看作表达方式上的改进，让你能够表达过去会在 PyTorch 中导致 graph breaks 的内容。

### 00:05:08 - 00:05:50

Uh with torch while loop uh also can now be captured in a CUDA graph. Um declarative dynamic shapes uh are also uh added. This is another sort of language feature. There's now a dynamic spec uh uh decorator at dynamic spec uh which is shared across torch compile torch export and make.fx FX experimental torch compiler uh support is also available for complex value tensors. Uh this is an opt-in thing uh which decomposes uh supported complex operations into real and imaginary computations enabling compiler backends to optimize complex number workloads which are super important for science and I think uh have been hard to do in PyTorch for a long time.

torch while loop 现在也可以被 CUDA graph 捕获。此外，还增加了声明式动态形状，这是另一项偏语言层面的功能。现在有一个名为 at dynamic spec 的 dynamic spec 装饰器，可在 torch compile、torch export 和 make.fx 之间共用。FX experimental torch compiler 也开始支持复数值张量。这项功能需要主动启用，它会把受支持的复数运算分解为实部和虚部计算，让编译器后端能够优化复数工作负载；这对科学计算极其重要，而我认为长期以来在 PyTorch 中一直很难做好。

### 00:05:47 - 00:06:21

Um broader platform support uh is you know we also made improvements across a bunch of different platforms including Rockcom. There's now 7.14 wheels uh produced um from the rock pip uh SDK and Intel adding uh native graph capture for the XPU and inductor uh target inductor targeting uh inductor uh functionality targeting the new Ruben platform from Nvidia. So that was a mouthful but that's the sort of like highlights from this particular release.

平台支持也更广了。我们改进了许多不同平台，包括 Rockcom。现在有从 rock pip SDK 构建的 7.14 wheels；Intel 为 XPU 增加了原生 graph capture；inductor 方面也增加了面向 Nvidia 新 Ruben 平台的功能。刚才信息量很大，但这些就是本次发布的大致亮点。

### 00:06:19 - 00:06:46

Having set that stage let's move to questions. Um so at the first question I think that we might have will be what CUDA version does PyTorch 2.14 ship uh and which one do I get by default and what packaging changes for CUDA do we expect for the upcoming release? Andre

背景介绍完了，我们来回答问题。我想第一个问题是：PyTorch 2.14 附带哪些 CUDA 版本？默认会安装哪个？接下来的版本在 CUDA 打包方面预计会有哪些变化？Andre？

### 00:06:46 - 00:06:46

[snorts]

[轻哼声]

### 00:06:46 - 00:07:26

um yes so basically for the current 21.14 release uh we support exactly same cuda matrix as previous release 21.13 so it's cuda 12.6 kuda 13.0 and cuda 13.2 um cuda 13.0 is being published to pipy um cuda 13.2 two becomes stable. Um it for previous release it was prototype considered as prototype right now it's stable. Um so um for the next uh also for this release we included rocker modifications.

嗯，是的。基本上，对于当前的 21.14 版本，我们支持的 CUDA 版本组合与上一版 21.13 完全相同，也就是 CUDA 12.6、CUDA 13.0 和 CUDA 13.2。CUDA 13.0 正在发布到 PyPI，而 CUDA 13.2 已成为稳定版本。上一版中，它还被视为原型版本；现在它是稳定版本了。另外，这个版本还包含了对 Rocker 的修改。

### 00:07:23 - 00:07:58

So rocom 714 wheels are published um right now build with uh the rock uh pip SDK. Um so for the next release 2.15 it's I think where the lot of changes are coming um first of all 2 um 14 release was the last release that we released CUDA 12.x wheels so 12.6 six for the 2.15 release. We will uh remove those wheels. There will no longer be shipped with PyTorch.

ROCm 714 的 wheel 现在已经发布，使用 ROCm pip SDK 构建。至于下一版 2.15，我认为会有很多变化。首先，2.14 是我们发布 CUDA 12.x wheel 的最后一个版本，因此到了 2.15，我们会移除 CUDA 12.6 wheel。它们将不再随 PyTorch 一起发布。

### 00:07:55 - 00:08:47

However, you are able you will still be able to compile with those manually um because and we are keeping one CUDA 12.6 CI built on our system. So we are ensuring that it's still the case. So there is no regression in this case. Um with CUDA 12.6 six support um uh that we're removing we're no longer be supporting pre-built wheels for Maxwell, Pascal and VA architectures. Um so the next part uh it's CUDA 13.2 will be published as default CUDA version on PIP and we will introduce CUDA 13.4 four as a prototype uh CUDA version with support for ver Rubin architecture.

不过，你仍然可以手动编译这些版本，因为我们会在系统中保留一项 CUDA 12.6 的 CI 构建，以确保这仍然可行，在这方面不会出现回归。随着我们移除 CUDA 12.6 的支持，我们也将不再为 Maxwell、Pascal 和 VA 架构提供预构建的 wheel。接下来，CUDA 13.2 将作为 PyPI 上默认的 CUDA 版本发布；我们还会引入 CUDA 13.4，作为支持 Vera Rubin 架构的原型 CUDA 版本。

### 00:08:45 - 00:08:52

Um I think this covers pretty much it in terms of cuda and a little bit of rocom. Please

我想关于 cuda，以及一点 rocom 的情况，差不多就是这些。请……

### 00:08:52 - 00:09:27

if I can make a small comment there's a like a lot of numbers lot of versions please track the pytor issues and dev discuss where we publish all those versions because it's they hard to remember and sometimes we have to pivot depending on some new hardware capabilities or whatever but that's the current plans. Thank you. Yeah, and just to add, there's a release MD file in the PyTorch PyTorch where the current matrix is published and yeah, you can go ahead and take a look there and ask any questions.

如果我能补充一句：这里有很多数字和版本，请关注 pytor issues 和 dev discuss，我们会在那里公布所有这些版本。它们确实很难记，而且有时我们必须根据新的硬件能力或其他情况调整方向；不过，这就是目前的计划。谢谢。对，再补充一点，PyTorch PyTorch 里有一个 release MD 文件，公布了当前的版本组合。大家可以去看看，有问题也可以提。

### 00:09:27 - 00:09:53

Okay. Um, do we have a next question queued up? [snorts] Okay. Um, thank you, uh, Michael for asking this question. With Agentic RL becoming a bigger part of training, um are you all seeing these workloads create different infrastructure challenges than traditional model training? Anything PyTorch needs still needs to get better at there? I'm actually going to um direct this one to Joe first and then Andre and Nikita, you guys can jump in after.

好的。下一个问题准备好了吗？[轻哼声] 好的。谢谢 Michael 提出这个问题：随着 Agentic RL 在训练中占据越来越重要的位置，你们有没有发现这类工作负载带来的基础设施挑战与传统模型训练不同？PyTorch 在这方面还有哪些地方需要改进？我先请 Joe 回答，然后 Andre 和 Nikita 可以接着补充。

### 00:09:51 - 00:09:56

Um Joe, I know you care a lot about agentic RL. So

Joe，我知道你很关注 agentic RL，所以……

### 00:09:56 - 00:09:58

yeah,

对。

### 00:09:58 - 00:10:35

I mean I think um you know if you look at it's it's pretty interesting the the models coming out these days are coming out fast and furious and I think there there was at least a for a while a perception that people are are pre-training models at an increased pace right but I think a lot of the gains that that we've seen actually you know some of the Chinese models for example that are coming out and and versioned really fast and the Quen models a lot of it is actually due to RL uh and you know I think one of the things that we've learned over the is you know if you build a really strong pre-trained base um to build on you can actually RL over it and you can actually throw more compute

我觉得这很有意思。现在模型推出得又快又密集。至少有一段时间，大家觉得人们正在加快预训练模型的速度，对吧？但我认为，我们看到的很多进步，比如一些中国模型和 Quen 模型快速推出新版本，其实很大程度上归功于 RL。过去这段时间我们学到的一点是：如果先建立一个非常强的预训练基础，就可以在其上开展 RL，还可以投入更多算力……

### 00:10:33 - 00:11:18

on it and I think the deepseek paper actually taught us a lot of lessons that kind of confirms for the the bitter lesson pill people among us including myself that you know if you continue to throw more compute at stable stable RL using that word stable very very uh carefully purposefully uh then you can actually extend capabilities of models over time. And so I think like you know uh in the earlier days I guess of RL um and and we started this project called OpenM last year when I was actually almost one year ago exactly uh at the PyTorch conference we uh launched that project and it continues to grow and I think one the whole goal of it was to democratize the use of reinforcement learning um because it's it's a lot easier to

……进去。我认为 deepseek 的论文给了我们很多启发，也印证了我们这些信奉“苦涩教训”的人的看法，包括我自己：如果持续给稳定的 RL 投入更多算力——我用“稳定”这个词非常谨慎，也很有意——就能逐步扩展模型的能力。回想 RL 早期，去年我们启动了一个叫 OpenM 的项目。差不多整整一年前，在 PyTorch 大会上，我们发布了这个项目；它还在不断发展。我认为它的整个目标之一就是让更多人能够使用强化学习，因为现在要创建环境、设置任务，尤其是有 Nemo RL、TRL、unsloth 这些工具可用，已经容易多了……

### 00:11:16 - 00:11:58

kind of create an environment create your tasks um and then be able to to set up you know especially today with all the tools available Nemo RL TRL unsloth, you can actually uh use reinforcement learning um you know way easier than you could probably like five or 10 years ago in the earlier days um and actually add capabilities to your models bas basically be able to postrain using reinforcement learning for openw weight models and I think that's something we want to continue to obviously de democratize and and enable with developers um and so like Daniel from Unsloth continues to bear that that flag for many uh and obviously we contribute uh significantly to open project and that fits nicely into the PyTorch ecosystem and extends it as people use PyTorch and other tools to uh

……你可以使用强化学习，比五年或 10 年前的早期容易得多，还能为模型增加能力，也就是基本上能够用强化学习对 openw weight models 做后训练。这显然是我们想继续向开发者推广并帮助他们做到的事。Unsloth 的 Daniel 继续为很多人扛着这面旗帜；我们显然也为开放项目做出了大量贡献。这与 PyTorch 生态系统很契合，也扩展了它，因为人们会使用 PyTorch 和其他工具来……

### 00:11:58 - 00:12:03

to to to customize their models for their tasks.

……针对自己的任务定制模型。

### 00:12:03 - 00:12:16

But fundamentally, PyTorch has a lot of the things that that folks need. Um we need these sort of layered packages like OpenM but but PyTorch itself. Are there any changes that you see uh you see needed now or in the future in PyTorch?

但从根本上说，PyTorch 已经具备了大家需要的很多东西。我们需要 OpenM 这样的分层软件包，不过就 PyTorch 本身而言，你觉得现在或将来还需要做哪些改变？

### 00:12:16 - 00:12:20

Well, I think Oh, sorry. Go for it.

嗯，我觉得……哦，抱歉，你先说。

### 00:12:20 - 00:12:44

No, no, go ahead. I want to hear your point of view. on the infrastructure side like uh the nickel 2 and like the safer nickel is one of the things that is constantly painful for post training workflows when you have to restart. So that would be an infrastructure part but also we have as Joel Joe said a lot of great partnership. I encourage people to check out well u maybe not encourage but there are two frameworks that exist in the pytor side.

不，不，你先说。我想听听你的看法。就基础设施而言，nickel 2 和更安全的 nickel 正是后训练工作流中不断令人头疼的事情之一，尤其是在必须重新启动的时候。这是基础设施方面的问题。而且，正如 Joel Joe 所说，我们也有很多很好的合作。我想请大家看看——也不一定是鼓励吧——pytor 这边有两个现成的框架。

### 00:12:43 - 00:13:09

There is a torch titan and torch forge that kind of give you some recipes on the post training and I think you should kind of direct your attention more towards those repositories when you're asking a specific questions about the frameworks rather about the like harness rather than like a training framework itself but support for MFP4 is also very important for post training pipelines so like all the infrastructures that is needed is there but Joe to continue

一个是 torch titan，另一个是 torch forge，它们提供了一些后训练的实践方案。如果你问的是框架方面的具体问题，我认为应该更多关注这些代码仓库，关注运行这些流程的框架，而不是训练框架本身。另外，支持 MFP4 对后训练流水线也非常重要。所以，所需的基础设施都在那里了。Joe，你接着说。

### 00:13:09 - 00:13:34

no I I think you're spot on I think there you know infra I mean we're we're using a pretty large large amount of compute to do RL here in reflection. We obviously did a lot of RL meta. Um I think infrastructure is the challenge. I think there there's a lot of work I think to um you know to to scale it up to make it stable. I think I'm just looking at even like I'm I can't really use examples of what we're doing internally yet because we're still pretty salty.

对，我觉得你说得很准。我是说，在 reflection，我们正用相当大规模的算力做 RL；我们显然也在 meta 做过很多 RL。我认为基础设施是挑战所在。要把规模扩大、让系统保持稳定，还有很多工作要做。我甚至正在想一些例子，但目前还不能拿我们内部的工作举例，因为我们现在仍然相当谨慎。

### 00:13:31 - 00:13:57

But like even when we were uh dealing with like fault tolerance in like the llama days, you know, most of our uh the faults that we found even in pre-training or even in post- training came from obviously GPUs failing over. um you know we had uh yeah we I think what something like 30% of of those failures or we had close to like 400 and something GPUs out of the 16,000 GPUs we were training on for the large model were failing.

不过，即使回想 llama 时期我们处理容错问题的经历，无论是预训练还是后训练，发现的大多数故障显然都来自 GPUs 故障切换。当时我们训练大型模型用了 16,000 个 GPUs，其中有大约 400 多个出现故障；我记得这大概占那些故障的 30% 左右。

### 00:13:56 - 00:14:23

We wrote that into the paper. You can kind of look at the graph there. I think like infrastructure stability and um and fall tolerance and I think we'll get to that in some of the other questions but that's probably one of the biggest biggest things and I think that's one of the reasons why like Tinker is actually like pretty interesting from from thinking machines. It's like a, you know, not not to to plug the his product here, but like it it actually does um it's actually a really uh cool API that actually abstracts that away.

我们把这些写进了论文，你们可以看看里面的图。我认为基础设施的稳定性和容错非常关键，后面几个问题可能还会谈到，但这大概是最重要的事情之一。我觉得这也是 thinking machines 的 Tinker 很有意思的原因。我不是要在这里给他的产品打广告，不过它确实提供了一个很棒的 API，把那些复杂性抽象掉了。

### 00:14:22 - 00:14:38

And I think that's for those especially the academics I've talked to and researchers like that is by far the hardest part. And obviously it's powered by PyTorch which is really great. Um but that that is like the the thing that that needs to be solved for this to be something that's widespread used. So

对我聊过的学者和研究人员来说，那显然是最难的部分。而它由 PyTorch 提供支持，这也非常好。不过，要让这项技术得到广泛使用，就必须解决这个问题。

### 00:14:38 - 00:14:59

awesome. I think that's a pretty good pretty solid answer to that question. Um, next up, um, why didn't you add LLM? Why don't didn't you add LLM support into the UI so people can talk to an agent instead of reading through the documentation? Uh, who wants to answer this? Uh, [laughter]

很好，我觉得这个问题回答得相当充分。接下来一个问题：为什么没有在 UI 中加入 LLM 支持，让大家可以直接和 agent 对话，而不必阅读文档？谁想回答？[笑声]

### 00:14:59 - 00:15:34

I'm not sure what UI we're talking about, but yes, like you can ask questions to an agent uh on the pytor.org, work but there's also kind of a pragmatic approach that um like running LM for public consumption is not cheap and via foundation right so there are lots of open LLMs that are very very good about answering questions about PyTorch documentation and we partnered with one of the providers early on so you can ask PyTorch questions that I don't remember job probably have more upto-ate details about the partnership like who is the provider but yeah

我不确定这里说的是哪个 UI，不过，是的，你可以在 pytor.org 上向 agent 提问。但还有一个很实际的问题：让 LM 面向公众运行并不便宜，而且还要通过基金会来做。现在有很多开放的 LLM，非常擅长回答有关 PyTorch 文档的问题。我们早期就和其中一家提供商合作，让大家可以提 PyTorch 相关的问题。我记不清了，Joe 可能更了解这项合作的最新情况，比如提供商是谁，不过，是的。

### 00:15:34 - 00:15:39

yeah go to collab and talk an agent, right? And it answers your questions about PyTorch.

对，去 collab 和 agent 聊聊，对吧？它会回答你关于 PyTorch 的问题。

### 00:15:39 - 00:15:40

Actually,

其实……

### 00:15:40 - 00:15:41

I would like to add Go ahead.

我还想补充一句。你先说。

### 00:15:41 - 00:15:43

Go for it, Andre.

Andre，你说吧。

### 00:15:43 - 00:16:09

So, basically, yes. Um, go ahead and create an issue for us describing exactly where you want to add the LLM and maybe we can look into it. Yes, because we have extensive documentation and extensive um uh basically release notes and there there are multiple sources of documentation. So I think if you're more direct about it what you want to do maybe we can come come up with the solutions for this.

基本上，是的。请创建一个 issue，明确说明你希望在哪里加入 LLM，我们可以研究一下。因为我们有大量文档和很详尽的发布说明，文档来源也有好几个。所以，如果你能更具体地说明想做什么，我们也许能想出解决办法。

### 00:16:09 - 00:16:12

Yeah and an issue on GitHub is the right way to do that. Go ahead Joe.

对，在 GitHub 上提 issue 是正确的方式。Joe，你说。

### 00:16:12 - 00:16:42

Yeah. No, I was just gonna say I mean in the if the the the search obviously we use kind of an LM based search now I think there and I think you know you can pragmatically um you can pragmatically use like Gemini as well and I think all of our documentation is in distribution right and uh it's it's well well indexed um and so you know I go into Google and and and actually Gemini does a really great job of like of scouring our documentation and and distilling things down to what I need.

对，我只是想说，现在我们的搜索显然已经在使用某种基于 LM 的搜索。我认为大家也可以务实地使用 Gemini。我们的文档都已分发出去，而且索引做得很好。所以，我会用 Google 搜索；Gemini 在检索我们的文档并把内容提炼成我需要的信息方面，确实做得很出色。

### 00:16:40 - 00:17:02

So, um, we don't actually even need to do a lot embedded in our site, but obviously make sure our docs are updated. Um, and and I mean, we like there's obviously the last year we've been trending towards how do you build things in a way obviously are good for humans, but also like consumable by LMS as well. And so, I think the team has done a great job kind of getting things in a place where machines can read things um along with the humans.

所以，我们其实不一定要在网站里嵌入很多东西，但显然需要确保文档持续更新。过去一年，我们一直在思考怎样构建既方便人阅读、也方便 LMS 使用的内容。我觉得团队做得很好，把内容整理到了机器和人都能阅读的状态。

### 00:17:02 - 00:17:11

So, um, but we continue to improve.

不过，我们还会继续改进。

### 00:17:11 - 00:17:42

Sounds great. Uh, next question. Um, Demila B uh asks, "What is the what is still the biggest unresolved limitation in PyTorch today for training large dynamic GNN's especially when the graph topology say changes at every step?" That's a great question. Um, who here feels most confident talking about graph neural networks? Joe, is that an area you've you've looked into or not so much? I'm probably not the GNN expert here.

听起来很好。下一个问题。Demila B 问：“对于训练大型动态 GNN's，PyTorch 如今仍未解决的最大局限是什么，尤其是在图拓扑结构每一步都会变化的情况下？”这是个好问题。在座谁对图神经网络最有把握？Joe，你研究过这个领域吗，还是接触得不多？我可能不是这里的 GNN 专家。

### 00:17:39 - 00:18:09

We have used GNN's in the past for interesting kind of scientific use cases. Um especially in the chemistry work we did uh with with uh with open chemistry. Um and there's they're certainly useful um for those types because you have very complex data that you're you're trying to embed that information into the model. So, a lot of different modalities, but I'm I'm probably far from a GNN expert.

我们过去在一些有意思的科学应用中使用过 GNN's，尤其是在 open chemistry 的化学研究里。它们在这类场景中确实有用，因为你要把非常复杂的数据中的信息嵌入模型，还涉及很多不同的模态。不过，我离 GNN 专家还差得远。

### 00:18:09 - 00:18:10

Yeah.

对。

### 00:18:10 - 00:18:55

Yeah. I think the good answer is the one that Andrew made to the previous question like, you know, if you have a particular like I think you should probably file an issue against PyTorch Geometric and if they say that you know that there's a fundamental issue in PyTorch, you should file an issue against us because we don't know until we like hear about it. But like I would say GNN's is not the most hot topic in uh artificial intelligence in the last 12 months and so we don't like have a particular focus on those but they should work and there shouldn't be any limitations other than inherent limitations in the architecture that like post training is somewhat slow because your graph propagation like backward propagation across the graph like introduce lots and lots of gradients

对。我觉得比较好的回答，就是 Andrew 对上一个问题给出的回答：如果你有具体问题，可能应该先向 PyTorch Geometric 提交 issue。如果他们认为 PyTorch 存在根本性问题，再向我们提交 issue，因为只有听到反馈，我们才会知道问题所在。不过，我得说，在过去 12 个月里，GNN's 并不是人工智能领域最热门的话题，所以我们没有特别聚焦它们。但它们应该能够工作，除了架构固有的限制之外，不应该再有其他限制；比如后训练会有点慢，因为沿着图进行传播，尤其是反向传播，会引入大量梯度。

### 00:18:54 - 00:19:06

which are somewhat unstable and you need to like be very careful because the trading electron rate should be very low but that's not new. [snorts]

这些东西有些不稳定，你得非常小心，因为交易电子率应该非常低，但这也不是新情况。[哼了一声]

### 00:19:06 - 00:19:14

Okay, thank you folks for taking a swing at a challenging question.

好的，谢谢大家尝试回答这个有挑战性的问题。

### 00:19:14 - 00:19:28

Um Adam's uh question is next. Um how does an inductor choose between kublos cutless and cute DSL kernels and does the user have any flexibility here? Is this cho is this choice exposed um uh for example in FX uh layer?

接下来是 Adam 的问题：inductor 如何在 kublos、cutless 和 cute DSL 内核之间做选择？用户在这方面有调整空间吗？比如，这种选择会在 FX 层暴露出来吗？

### 00:19:28 - 00:20:00

So I can probably answer this one. Well like inductor tries to do the best perf right and it like tries to select the ones that it believes will be better. Uh you can also do max autotune when it should just dutifully iterate over all the options and try to pick the one that works for you. Also there is a hardware limitations and there should be knobs in the inductor to say use or doesn't use certain backends but I don't think there is a very easy way to express it through the FX graph but you can suggest this one as an improvement.

这个问题我大概可以回答。inductor 会尽力取得最佳性能，并选择它认为表现更好的选项。你也可以启用 max autotune，让它逐一尝试所有选项，选出适合你的那个。另外，硬件本身有限制，inductor 里应该也有配置项，可以指定使用或不使用某些后端。但我觉得没有很方便的办法通过 FX 图表达这种选择；你可以把它作为改进建议提出来。

### 00:19:58 - 00:20:36

So once again file an issue saying that you want this one but I kind of want you to think about it like why do you want to be explicit like don't we want to converge to one model and I suspect that cute DSL is probably something we kind of leaning towards on the most modern architectures like blackfields and probably have common verbs but I think like kublast will probably give you a much better perf on and cutless on an architectures because there's kind of a cut off there.

所以，再次建议你提交一个 issue，说明你想要这个功能。不过我也想请你想一想：为什么要明确指定呢？我们不是希望最终收敛到一个模型吗？我猜在 blackfields 这类最新架构上，我们可能更倾向于 cute DSL，而且可能有一些通用的 verbs。但我认为，在另一些架构上，kublast 和 cutless 很可能会带来好得多的性能，因为这里存在一个分界点。

### 00:20:36 - 00:21:03

Okay, any other uh responses on this one? Okay, let's move to the next one. We have I love that there are so many audience questions. So, thank you folks for putting those in. Uh I think that's great. Okay. Um we have somebody from LinkedIn who asked uh which PyTorch 20 uh 21.14 uh feature gives the biggest performance boost for production LLM inference and how easy is it uh to adopt?

好的，关于这个问题还有其他回答吗？那我们看下一个。观众提了这么多问题，我很高兴，谢谢大家提交问题，我觉得这太好了。好，有位来自 LinkedIn 的朋友问：PyTorch 20 21.14 的哪个功能能为生产环境中的 LLM 推理带来最大的性能提升？采用起来有多容易？

### 00:20:59 - 00:21:33

Um I don't know that we did an ablation study of the various different features but does anybody have a um have a thought here? I suspect cudgraphs uh ability to trace more functions, avoid cudograph breaks and ability to like trace coms uh would be the one maybe some of the torch compiler features but that depends on the frameworks that you're using for your LLM inference, right? And they should be pretty easy to adopt.

我不知道我们是否对各项功能做过消融研究。有人有什么想法吗？我猜可能是 cudgraphs：它能够追踪更多函数、避免 cudograph 中断，还能追踪 coms。也可能是某些 torch compiler 功能，但这取决于你用什么框架做 LLM 推理，对吧？这些功能应该都比较容易采用。

### 00:21:33 - 00:21:54

So like lots of those features were driven by like asks from users such as like VLM or SJAN. So I hope that they will be incorporating those features into their frameworks and NVJMS is also a great one that probably important for performance.

其中很多功能都是应 VLM 或 SJAN 等用户的需求开发的。所以我希望他们会把这些功能整合进各自的框架。NVJMS 也是个很棒的功能，很可能对性能很重要。

### 00:21:54 - 00:21:54

Okay.

好的。

### 00:21:54 - 00:22:09

Yeah, I think NVJ specifically on black with low precision I think it's important. Yeah.

对，我觉得 NVJ 在 black 上使用低精度时尤其重要。对。

### 00:22:09 - 00:22:41

Okay. Um the next one is from Asam. Uh, and I might be mispronouncing there. I apologize if I did. Um, PyTorch, uh, 21.14 makes fault tolerance a first class CTN concept. Does the new model allow a training job to recover from a failed rank without reconstructing the entire distributed process group? And what state is expected to remain valid after recovery? Do we have any anyone here who's expert in the new um torchcom's model?

好的，下一个问题来自 Asam。如果我读错了名字，先向你道歉。PyTorch 21.14 将容错变成了一等 CTN 概念。新模型是否允许训练任务在某个 rank 失败后恢复，而无须重建整个分布式进程组？恢复后，哪些状态预计仍然有效？这里有人熟悉新的 torchcom's 模型吗？

### 00:22:41 - 00:22:41

[clears throat]

[清了清嗓子]

### 00:22:41 - 00:22:42

None of us are.

我们都不是这方面的专家。

### 00:22:42 - 00:22:44

Okay. Not me.

好吧，我也不是。

### 00:22:44 - 00:23:09

I think we're I think we're going to have to refer this question to um uh to the discussion. So this would be a great question for example for discuss uh.pytorch or dev discuss. Um so I think uh the the specific uh details about like what part of the state uh would be expected to remain valid uh is the kind of thing one of the distributed uh uh engineers would be really strong rice question.

我想我们得把这个问题转到讨论区。这很适合在 discuss uh.pytorch 或 dev discuss 上问。至于状态中的哪些部分在恢复后预计仍然有效，这类具体细节很适合请分布式领域的工程师回答；这是个很好的 rice 问题。

### 00:23:09 - 00:23:10

Yeah.

对。

### 00:23:10 - 00:23:34

Yep. Exactly. But I want to just flag that this is experimental support and we indeed like seeking your feedback. So please try to obtain into this like nickel ch and other CTN default tolerance features and let us know if they work for you or not. But yes, like a new model allows for a more graceful recovery, but it doesn't guarantee that it will work out of the box.

没错。不过我想提醒一下：这仍是实验性支持，我们确实希望收到大家的反馈。所以请试试 nickel ch 以及其他 CTN 容错功能，告诉我们它们对你是否有效。新模型确实能让恢复过程更平稳，但不能保证开箱即用。

### 00:23:32 - 00:23:50

You probably need to like adjust your training pipeline to uh like take benefit of that and like recognize this error as recoverable. But yes, sounds like a great discuss question.

你可能还需要调整训练流水线，才能利用这些功能，并让它把这种错误识别为可恢复错误。不过没错，这很适合拿到 discuss 上讨论。

### 00:23:50 - 00:24:17

Next question. Yes, it's the um uh Shiva Shush um uh again pronunciation caveat on Apple silicon uh 21.14 adds uh native linear algebra and more metal kernels uh for LLM decode on unified memory. Which path actually moves the needle uh versus 2.13 the new metal kernels or inductor picking up better epilogs? Any measured tokens per second deltas um on the M series or fixed to code loop?

下一个问题来自 Shiva Shush；还是那句话，如果名字读错了请见谅。关于 Apple silicon，21.14 为统一内存上的 LLM decode 增加了原生线性代数支持和更多 metal 内核。与 2.13 相比，真正带来明显提升的是新的 metal 内核，还是 inductor 选用了更好的 epilogs？在 M 系列上，或者在 fixed to code loop 中，有没有测得的每秒 token 数变化？

### 00:24:15 - 00:24:24

And I I believe Nikita, you've been really active in the past with some of the Apple changes. So I'm gonna toss this one to you.

Nikita，我记得你过去一直很积极地参与一些 Apple 相关的改动，所以这个问题交给你。

### 00:24:24 - 00:25:12

Sure. Well, first of all, we don't specifically measure and also it's very subjective because like different M series have a different behavior. But there is one feature that was introduced in 214. we use uh like a new apple silicon or like metal framework feature called metal performance primitives which is similar to uh I don't know probably to cut in a sense so it's very different it allows one to express matrix multiplication it's done in eager mode and it can in some cases produce just of this particular operation like 3x or even sometimes measured 5x but this is more like a synthetic benchmark not an end to end one performance boost so you should give it tries there is no inductor peing better epilogues necessarily for Apple silicon and yes we we don't do measurements because

当然。首先，我们没有专门测量过，而且这个问题也很因机型而异，因为不同的 M 系列表现不同。不过 214 引入了一个功能：我们使用了一个新的 apple silicon，或者说 metal 框架功能，叫作 metal performance primitives。我也说不准，它从某种意义上可能和 cut 类似，但又很不一样。它允许你表达矩阵乘法，在 eager mode 下运行；在某些情况下，单看这一项操作，性能可以达到 3x，有时甚至测到 5x。但这更像是合成基准测试的结果，不是端到端的性能提升，所以你应该试试看。至于 Apple silicon，未必是因为 inductor peing 到了更好的 epilogues。我们确实也没有做测量，因为

### 00:25:11 - 00:25:37

again it's kind of you cannot measure on the pietorch alone you need to measure on a particular framework but you can probably check those numbers at hugging face transformers on one of their um frameworks if this is what you're using for your inference pipelines

还是那句话，你不能只在 pietorch 上测量；你需要针对某个具体框架来测。不过，如果你的推理流水线用的是 hugging face transformers 的某个框架，你大概可以去那里查看这些数字。

### 00:25:37 - 00:26:01

okay Um yes, let's do so. Uh Shibash had three several questions in a row, so I don't know if we'll get to all of them, but let's take this next one here. [snorts] Uh the declarative uh dynamic spec looks useful for Asian workloads where context length um changes midsession. Uh if the live shape set expands beyond the declared spec, does inductor recompile quietly or can you make it a hard fail uh so that you can have more predictable latency?

好的，我们继续。Shibash 连着提了三好几个问题，我不知道能不能全部答到，但先看下一个。[哼了一声] 对于上下文长度会在会话中途变化的 Asian 工作负载，声明式 dynamic spec 看起来很有用。如果运行时的形状集合超出了声明的 spec，inductor 会悄悄重新编译，还是可以让它直接失败，以便获得更可预测的延迟？

### 00:25:57 - 00:26:04

Um does anybody know the answer to this one? Um

有人知道这个问题的答案吗？

### 00:26:04 - 00:26:14

I don't know on top of my head but knowing the development patterns of the inductor I suspect there should be an op to make it fail

我一时想不起来，但根据 inductor 的开发方式，我猜应该有一个 op 能让它失败。

### 00:26:14 - 00:26:34

but there also should be a knob like that was a default orch compile behavior for quite some time that like if it fails to do something instead of ering out it just falls back to either or like to the previous behavior. So there are both nodes exist but you should ask this question and de discuss. Um it's kind of yeah hard to know all the specifics.

不过也应该有相应的配置项。orch compile 在很长一段时间里的默认行为是：如果某件事做不成，它不会 ering out，而是回退到 either，或者回退到之前的行为。所以这两种 nodes 应该都存在，但你应该去 de discuss 上问这个问题。具体细节确实很难全都记得。

### 00:26:34 - 00:27:09

Yeah I feel I I feel I feel like the if we don't already have a knob to do this this would be a knob that that the team would likely receive. Well um because that that latency uh predictability is is pretty important to a lot of use cases. Okay. Um let's go ahead and move on to the to the next uh question in the list here. Um, how will dropping a CUDA 12 wheels in 2.15 affect downstream uh C++ libr uh libraries uh or binaries for for projects such as to sharp, but I know there's there's others.

对，我也觉得，如果现在还没有控制这一行为的配置项，团队应该会愿意加入一个，因为延迟的可预测性对很多使用场景都很重要。好，我们继续看列表里的下一个问题：在 2.15 中停止提供 CUDA 12 wheels，会怎样影响 to sharp 等项目下游的 C++ 库或二进制文件？我知道还有其他项目。

### 00:27:06 - 00:27:10

Um, Andre, do you want to take this one?

Andre，你想回答这个问题吗？

### 00:27:10 - 00:27:39

Yeah. Yes. Uh, so basically I think right now it would be good time for you to start planning to migrating for coder 13 and probably the best version would be 13.2 two since this is going to be our um stable version for the next 2.15 release. Um and yeah, I think it's and if you still want to use CUDA2.x, you should be able to compile yourself from the source.

对，是的。基本上，我认为现在是你开始规划迁移到 CUDA 13 的好时机，最好选择 13.2，因为它会是下一个 2.15 版本的稳定版本。另外，如果你仍想使用 CUDA2.x，应该可以自行从源码编译。

### 00:27:36 - 00:27:59

So this is should still be supported. So you can take a look at how we compile liptor binaries and basically duplicate the workflow and compile it yourself on your workers. um that that could be an option for short term like if you cannot migrate uh faster um I don't know maybe Nikita wants to chime in here any additional

所以这应该仍然受支持。你可以看看我们如何编译 liptor 二进制文件，照着这个流程在自己的工作机器上编译。如果你没法更快地完成迁移，这可以是一个短期办法。我不知道 Nikita 是否还想补充什么。

### 00:27:59 - 00:28:29

um sure I guess like uh uh two question like two two points one is that uh like if you're on an older GPU unfortunately CUDA 13 doesn't support some of them so you have to stay on the older PyTorch versions but also one should not expect any performance uh gains or uh like new features for this older hardware with new releases. So like there is no reason for you to upgrade.

当然，我想说两个问题，或者说两点。第一，如果你用的是较老的 GPU，很遗憾，CUDA 13 不支持其中一些型号，所以你必须继续使用较旧的 PyTorch 版本。而且，不应指望新版本能为这些旧硬件带来性能提升或新功能。因此，你也没有升级的理由。

### 00:28:26 - 00:29:01

You can continue using 214 uh for your older like pipelines and TPU binaries will still be there, right? I don't know how much to sharp is indeed about like I guess it's most frequently used on Windows and I don't know how much of that is like relevant to people on Windows like using blackville features like like VFP4 because like not available. So use 214 but also yeah like CUDA 12 was released five years ago if I'm not mistaken.

对于较旧的流水线，你可以继续使用 214，TPU binaries 也还会提供，对吧？我不太清楚 to sharp 到底涉及多少这方面的东西；我猜它最常在 Windows 上使用，也不知道 blackville 的 VFP4 之类的功能与 Windows 用户有多大关系，因为它们不可用。所以用 214 就好。另外，CUDA 12 如果我没记错的话，是五年前发布的。

### 00:28:59 - 00:29:11

So it's it's a good time to upgrade like ask yourself why do you need to be on CUDA 12 in 2026 or like heading towards 2027 [snorts]

所以现在是升级的好时候。问问自己，为什么到了 2026，或者眼看就要到 2027 了，还需要留在 CUDA 12？[哼了一声]

### 00:29:12 - 00:29:37

though I' though I've known people who have stayed on old CUDA versions for a long time but but yes okay next question um okay so this one is a simple one and this uh came from the the community uh questions ahead of time uh if you're on AMD what change for uh rockcomm in 2.14. Andre, do you want to talk to this one?

不过，我确实认识一些人，他们在旧版 CUDA 上待了很久。好，下一个问题比较简单，是社区提前提交的：如果你使用 AMD，2.14 中 rockcomm 有什么变化？Andre，你想谈谈吗？

### 00:29:37 - 00:30:19

Yes. Yes. So, basically for romcom specifically, uh we removed the rocom 7.1 support. Uh no wheels are no longer shipped with 214 PyTorch. So, um uh if you're on roam 7.1, I think you need to migrate to either Rocomm 7.2 or 714. Uh those are two versions that are supported currently by 214 PyTorch. Um 714 is the latest one and it is using uh the rock pdk 7.2 it's the same version that we also shipped with previous release of torch.

好的。具体到 romcom，我们移除了对 rocom 7.1 的支持；214 PyTorch 已经不再提供相应的 wheels。所以如果你在使用 roam 7.1，我认为需要迁移到 Rocomm 7.2 或 714。这是目前 214 PyTorch 支持的两个版本。714 是最新的，使用的是 rock pdk 7.2；这与上一个 torch 版本附带的版本相同。

### 00:30:16 - 00:30:37

So it's uh more similar to what rocom 7.1 was. So I think you so you have a choice basically. I think the best way is to migrate straight to 714. Um but yeah, I think that's about it. Yes. [snorts]

所以它与 rocom 7.1 更相似。基本上你有选择；我认为最好的办法是直接迁移到 714。大概就是这些。[哼了一声]

### 00:30:37 - 00:30:37

Okay.

好的。

### 00:30:37 - 00:30:54

And there are some advanced features which are available in 714 like he file which is similar to coup file like direct storage access from GPUs and you can use more hip solver functions for example torch and alf uses that one.

714 还提供一些高级功能，比如 he file；它类似于 coup file，让 GPU 直接访问存储。你也可以使用更多 hip solver 函数，比如 torch and alf 就用了它。

### 00:30:54 - 00:31:05

Yes. And the packaging is actually a lot better and a lot more modern for 714.

对，而且 714 的打包方式确实好了很多，也现代化了许多。

### 00:31:05 - 00:31:28

Okay. Um I think our next question is going to be kind of a little bit of a broader question. Um uh so it seems like the diversity in the Okay, here we are. Uh it seems like the diversity in hardware in the AI world is exploding lately. At least that's what I see on on on uh uh in my news passing by. uh my uh screen every day.

好的，下一个问题会更宽泛一些。最近 AI 领域的硬件似乎越来越多样化了，至少我每天在屏幕上掠过的新闻给我的感觉是这样。

### 00:31:26 - 00:31:38

Uh how do we think about platform support um within the PyTorch community? I'd love to hear from from Joe on this and then I'm sure Andrea and Nikita may also want to chime in with details.

PyTorch 社区是如何考虑平台支持的？我很想先听听 Joe 的看法，相信 Andrea 和 Nikita 也可能想补充一些细节。

### 00:31:38 - 00:31:39

Yeah,

对。

### 00:31:39 - 00:32:06

I mean we know we've been I'm coming at Harbor 180 for a long time with PyTorch. I think we've been trying uh to support a lot of different backends. We there's working groups in the in the foundation. I think you know in some ways like you know there's a couple different angles. One obviously we wanted to have a diverse set of of hardware vendors and and a hardware ecosystem to give choice to developers because not just at the cloud.

我们知道，PyTorch 很长时间以来都在做 Harbor 180。我想我们一直在努力支持许多不同的后端；基金会里也有相关工作组。这个问题可以从几个角度来看。其中一个显然是，我们希望有多样化的硬件厂商和硬件生态，让开发者有选择，因为使用场景不只在云端。

### 00:32:05 - 00:32:30

I think everyone sort of thinks of like the cloud side is like the default but like running on device running you know um within wearables like there's there's like a lot going on in different hardware architectures and that's why this like compiler and kernel ecosystems is so diverse. I think that's like one piece. I think pragmatically though like like compute is scarce these days and you know being able to run on Nvidia, being able to run on AMD, being able to run on TPUs.

我想大家往往把云端视为默认场景，但设备端运行、在可穿戴设备中运行等场景也很多，涉及各种硬件架构。这也是编译器和内核生态如此多样化的原因之一。从实际情况看，如今算力紧缺；能够在 Nvidia、AMD 和 TPU 上运行也很重要。

### 00:32:29 - 00:32:59

I think it's actually um you know things have like you know big labs or folks that that need a lot of compute have have had to learn how to run on on you know a lot of different architectures. I think like if I look at anthropic they run on Nvidia, they run on Tranium, they run on TPUs. Um even in like different workflows like I know we're talking a lot about training typically but you know training RL like we're talking about running on GPUs.

我认为，大型实验室或其他需要大量算力的人，已经不得不学会在许多不同的架构上运行。比如 anthropic，他们使用 Nvidia、Tranium 和 TPU。即使在不同的工作流程中也是如此：我们通常谈得更多的是训练，但训练 RL 时，我们也会说到在 GPU 上运行。

### 00:32:56 - 00:33:34

We run environments in CPUs. So all of a sudden you see the Intel stock up these days because they're selling a lot more CPUs for RL which is good. Um you know when you you run reinforcement learning you're running a lot of inference. um at inference time you're running a ton of inference obviously like you're doing a lot of compute time inference time compute um scaling that up in different axes um and so I think like you know you might run that on on different architecture you might run AMD you might run on on on different things so I think the hardware diversity is like is great for developers and I think that even going back like a year or longer we've been talking about this and um for PyTorch even back in we've been

我们在 CPU 上运行环境。所以你最近会看到 Intel 股价上涨，因为他们为 RL 卖出了更多 CPU，这挺好。做强化学习时，你会运行大量推理；到了推理阶段，显然也会运行海量推理，投入大量推理时计算，并沿不同方向扩大规模。所以你可能会在不同架构上运行，可能用 AMD，也可能用其他东西。我认为硬件多样化对开发者是好事。事实上，回头看一年甚至更久以前，我们就一直在谈这个问题。对于 PyTorch，我们早在那时就一直在

### 00:33:34 - 00:34:01

working with AMD since what 2018 um I'm so you remember those days right So, I I think it's it's I I wouldn't even say it's exploding lately. I actually think it's been exploding for a long time and I think um we just continue to grow and obviously Nvidia gets a lot of the headlines as they as they well deserve. Um but there's definitely a lot of architectures out there that are are needed and um and it's just going to continue to kind of grow in different directions.

与 AMD 合作了，差不多从 2018 开始吧，你还记得那些日子，对吧？所以我甚至不会说这种多样化是最近才突然增加的。我觉得它已经发展了很长时间，而且还在持续增长。Nvidia 得到了很多头条关注，这也是他们应得的。但确实还有很多必需的架构，今后也会继续朝不同方向发展。

### 00:33:59 - 00:34:22

Um, and I think we're going to get inference is obviously much more important these days for many. Um, and it has to be much more efficient to be able to scale and that's just going to necessitate different types of architectures. Disagregated inference for example where you separate pre-fill and decode um is a major trend right now and has been for the last several months. Um, so I think that's it's going to continue to innovate obviously where the workloads are going to be.

我认为推理如今对很多人显然重要得多，而且必须变得更高效才能扩大规模；这也必然需要不同类型的架构。比如 Disagregated inference，把 pre-fill 和 decode 分开，就是目前的一大趋势，过去几个月一直如此。因此，随着工作负载的发展，这方面显然还会继续创新。

### 00:34:20 - 00:34:25

So would love others to to chime in though.

我也很想听听其他人的看法。

### 00:34:25 - 00:34:34

Yeah. Um, Nikita and Andre, do you want to say anything about distributed or sorry about uh heterogeneous hardware?

对。Nikita 和 Andre，你们想谈谈分布式，抱歉，我是说异构硬件吗？

### 00:34:34 - 00:35:16

Uh, sure. Um, yes. Uh, basically on PyTorch, PyTorch specifically, we've been working on the project called CRCR, it's a cross repository CI relay. So, it lets partner back ends to run their own CI against PyTorch changes. So on the PyTorch PRS and report result back to us um on the PyTorch HUD page. Uh so we developed this with the partners um in the past few months and basically it's pretty much released right now for this um 2015 um 2014 release and this is a new feature in the Hut.

好的。具体说到 PyTorch，我们一直在做一个叫 CRCR 的项目，它是一个跨仓库 CI 中继。它让合作伙伴的后端能够针对 PyTorch 的变更运行自己的 CI，也就是针对 PyTorch PRS 运行测试，并把结果报告回 PyTorch HUD 页面。过去几个月，我们和合作伙伴一起开发了它；针对这次 2015、2014 版本发布，它现在基本已经推出了。这是 Hut 中的一项新功能。

### 00:35:14 - 00:35:50

So if you go to our hot page, there's the CRCR button. You can go and access this to have a like a preview. Uh so basically it's the way it's um configured. It's um the four levels of access. So level one is on boarding. So events are forwarded to downstream uh repositories and um upstream receive no feedback. So PyTorch we don't know what our tests are running on the downstream repos but tests are happening.

如果你打开我们的 hot 页面，会看到 CRCR 按钮。你可以点进去预览一下。它的配置分为四个访问级别。第一级是接入：事件会转发到下游仓库，但上游收不到反馈。所以在 PyTorch 这边，我们不知道下游仓库正在运行哪些测试，不过测试确实在运行。

### 00:35:46 - 00:36:46

So L2 is observation. It's when we actually uh PyTorch receives the results uh of the downstream um tests and able to visualize and see what are the failures and this helps like um uh observation uh and uh um L3 it's stable um it's where we are currently at um it's non-blocking run check on PR so it's basically you apply um flow out of tree lab out CR flow CRCR label to the PR and the results from the uh downstream repositories appear on your PyTorch PR and um currently we're working with Nindia Windowsci Red Hat IBM Spire Ascent NPU Google TPU and VLM repos for that.

L2 是观察阶段。这时 PyTorch 能收到下游测试的结果，并查看有哪些失败，这有助于观察测试情况。L3 是稳定阶段，也是我们目前所处的阶段：它会在 PR 上运行非阻断式检查。具体来说，你给 PR 加上 flow out of tree lab out CR flow CRCR label，下游仓库的结果就会出现在你的 PyTorch PR 上。目前，我们正与 Nindia Windowsci Red Hat IBM Spire Ascent NPU Google TPU 和 VLM 仓库合作推进这件事。

### 00:36:42 - 00:37:22

So [clears throat] this will help with um onboarding different um vendors, different architectures and different repositories uh and uh maintainers of those repositories should be able to onboard very quickly and get results testing for PyTorch pretty quickly and supported on PR testing and on nightly testing. So there are kind of two modes for this and yeah I think this is a very uh interesting uh part of the PyTorch project that we're currently developing and going to develop for um I don't know maybe Nikita

这将有助于接入不同的供应商、架构和仓库。那些仓库的维护者应该能很快完成接入，也能很快拿到针对 PyTorch 的测试结果；PR 测试和 nightly 测试都支持，所以大致有两种模式。我觉得这是 PyTorch 项目中很有意思的一部分，我们正在开发，也会继续开发。也许 Nikita……

### 00:37:22 - 00:38:15

yeah and maybe uh like one more yeah one more thing that like on the API side there's a torch accelerate API which tries to abstract away different devices different names different capabilities so you can write once for accelerated compute and it should on not I don't know I don't want to repeat the Java everywhere but it will run on multiple compute and we try to like align all our operator implementation this is why CRCR is there is that uh like vendors can make sure that like their hardware report the same results as we expect it would and developers will have fewer surprises when they use this like abstraction to dispatch their job to accelerator available but yes it is very important because you need to be on more platforms.

对，也许还要补充一点：在 API 这边，有一个 torch accelerate API，试图抽象不同设备、不同名称和不同能力，让你只需为加速计算写一次代码，它就应该能在……我不知道，我不想再到处重复 Java 那句话了，但它会在多种计算设备上运行。我们也在努力让所有算子实现保持一致。CRCR 之所以存在，是为了让供应商能确认他们的硬件报告的结果与我们的预期相同，这样开发者使用这一抽象把任务分派给可用的加速器时，遇到的意外就会更少。这确实非常重要，因为你需要覆盖更多平台。

### 00:38:15 - 00:38:40

Yeah, I think I think I mean I I think this whole area is really fascinating within PyTorch because like uh like how do you how do you have like a consistent experience across the such different hardware and then how do we do QA? How do we deal with like new hardware emerging which may or may not be in the cloud for us to test? and how do we do like the the the the um the the the continuous integration testing in a distributed fashion.

是的，我觉得 PyTorch 的整个领域都很有意思：面对差别这么大的硬件，怎样提供一致的体验？我们又怎样做 QA？新硬件不断出现，而我们未必能在云端使用它来测试，这该怎么应对？还有，怎样以分布式方式开展持续集成测试？

### 00:38:38 - 00:38:58

Um so so this is an area where like just a lot of effort has gone and I think we've just gotten like this part of PyTorch has gotten a lot more mature. So I imagine for hardware vendors it's way easier to sort of contemplate getting their their platform properly supported in PyTorch now than it was you know two years ago three years ago. So okay I think

这个领域已经投入了大量精力，我觉得 PyTorch 的这一部分成熟了许多。所以我想，对硬件供应商来说，现在考虑让自己的平台在 PyTorch 中得到妥善支持，比两三年前容易多了。好的，我觉得……

### 00:38:58 - 00:39:26

want to add sorry last last point to this and it's the way it's developed right now it's a lot more secure than it used to be. So there's no um sharing of the keys nothing like that. So on boarding and from infrastructure point of view it's very fast and secure and uh uh very straightforward. We set up specific repository CRC test when [clears throat] you can go and look at the on boarding guide and can start on boarding.

抱歉，我想再补充最后一点：按现在的开发方式，它比以前安全得多。不需要共享密钥，也没有类似的事。从基础设施的角度看，接入既快又安全，流程也很直接。我们设置了专门的仓库 CRC test；你可以去看接入指南，然后开始接入。

### 00:39:26 - 00:40:04

Yeah. Awesome. Okay. I think we have a next question lined up. Okay. And this again thank you Shivashish uh for asking uh three interesting and challenging questions uh uh for us to answer here. Uh so for draft then verify loops uh which would be sort of speculative decoding style style is could a graph capture um in the uh torch while loop u uh mode usable on the verify step in 214 or does that control uh flow still force a graph break um do we do we know the answer to that among the

对，太好了。接下来应该有一个问题。再次感谢 Shivashish 提出三个有意思、也很有挑战性的问题，让我们在这里回答。关于 draft then verify 循环，也就是类似 speculative decoding 的方式：在 214 中，用 torch while loop u mode 捕获的图，能用于 verify 步骤吗？还是说那里的控制流仍会迫使图中断？我们当中有人知道这个问题的答案吗？

### 00:40:04 - 00:40:44

I think generically you cannot answer this question because it depends on the workflow it should be usable and that's uh what the new um like abstractions that we talked about in the highlights are for but there could be some cases when it's not but we you already have uh toggles to test that you can just wrap your code with like aborton graph break and then you know that you produced a graph break or not but it's kind of I I don't think there's one one answer that fits them all like we try to make those not cause graph breaks but depending on the CUDA version you're on the hardware capabilities is a Python version and so on and so forth.

我觉得这个问题不能一概而论，因为它取决于具体工作流。它应该是可用的，我们在亮点介绍中谈到的新抽象就是为此设计的；但也可能有些情况无法使用。不过你已经有开关可以测试：只要用 aborton graph break 包住代码，就能知道有没有发生图中断。我认为不存在适用于所有情况的单一答案。我们会尽量避免这些操作造成图中断，但这也取决于你使用的 CUDA 版本、硬件能力、Python 版本等因素。

### 00:40:43 - 00:40:59

It can cause graph break. So please check if it doesn't create great if it does file an issue but we try to eliminate as graph breaks as much as possible because it slows down your pipelines.

它可能造成图中断。所以请检查它有没有造成图中断；如果有，请提交一个 issue。我们会尽可能消除图中断，因为它会拖慢你的流水线。

### 00:40:59 - 00:41:13

Okay, awesome. Um, can we jump next to the question about Envy Gym uh Q5? Uh, Basil,

好的，太好了。我们能跳到关于 Envy Gym 的问题 Q5 吗？Basil，

### 00:41:13 - 00:41:40

sorry to do that verbally rather than in the chat. So, this is one of the key things that we highlighted at the be at the beginning and I wanted to make sure we we took a moment to talk about it among the questions. Uh so what is NVGEM? Um kind of a generic question and how do I turn it on? Um uh so uh Nikita do you want to jump on this or Andre?

抱歉，我直接说出来了，没有在聊天里发。这是我们一开始重点提到的事情之一，我想确保我们在问答环节花点时间谈谈它。那么，NVGEM 是什么？这是个比较宽泛的问题；另外，我该怎么启用它？Nikita 或 Andre，你们谁来回答？

### 00:41:40 - 00:41:57

Sure. But we already somewhat covered it when the other user asked a question about different uh good gen backends and I said that there is a like you can't do it over FX graph but there is a toggles and specifically for NVJ there's max to tune gem backends

可以。不过，之前另一位用户问到不同的 good gen 后端时，我们已经部分讲过了。我当时说过，它不能通过 FX graph 来做，但有相应的开关。具体到 NVJ，有一个 max to tune gem backends

### 00:41:57 - 00:42:50

list and if you add NVJM here so it's not enabled by default but then it will be enabled and it will allow some more advanced epilog fusion and yeah NVJM is um QGSL gem back end that based on cutless and it allows you to like yeah fuse mules and fusions and get generate even faster uh yeah uh kernels specifically if you do low precision data types like in VFP4 but it requires you to have an Nvidia cutless DSL installed on your system and blackwell as well so like you unfortunately cannot use 44 [laughter] on your ampers or hoppers Okay, I see a question from Jose Silva on the on the chat.

列表。如果你把 NVJM 加到这里，它就会启用；默认情况下它没有启用。启用后可以做一些更高级的 epilog fusion。NVJM 是一个基于 cutless 的 QGSL gem 后端，能让你融合 mules 和 fusions，生成速度更快的 kernels；使用像 VFP4 这样的低精度数据类型时尤其如此。但你的系统需要安装 Nvidia cutless DSL，还需要 blackwell。所以很遗憾，你不能在 ampers 或 hoppers 上使用 44。（笑）我看到 Jose Silva 在聊天里提了一个问题。

### 00:42:45 - 00:43:13

Um, okay, awesome. Thank you, Basil. Um, if I train a vision model on CUDA and export it using AOT inductor to C++ binaries for a machine running an Intel Iris iGPU. Um, is dynamic shape and anti-quantization supported out of the box on those targets? On those XPU targets? [laughter]

好的，太好了。谢谢你，Basil。如果我在 CUDA 上训练一个视觉模型，然后用 AOT inductor 将它导出为 C++ binaries，供一台搭载 Intel Iris iGPU 的机器使用，那么这些目标设备开箱即用地支持 dynamic shape 和 anti-quantization 吗？这些 XPU 目标设备呢？（笑）

### 00:43:13 - 00:43:14

That's quite a specific question.

这个问题相当具体。

### 00:43:14 - 00:43:20

I was just going to say I don't know that case is covered in rci. Uh, [laughter]

我正想说，我不知道 rci 是否覆盖了这种情况。（笑）

### 00:43:20 - 00:43:28

it might be covered in rci, but maybe not intel Iris iGPU in particular. Like I think they're trying to cover a more

rci 可能覆盖了它，但未必特别覆盖 intel Iris iGPU。我觉得他们想覆盖的是更

### 00:43:28 - 00:43:38

modern GPUs. Uh but I guess you never know until you try and then you can probably file an issue if it doesn't work.

现代的 GPU。不过，在试过之前也说不准；如果不能用，你大概可以提交一个 issue。

### 00:43:38 - 00:43:39

Yeah, file is

对，提交一个……

### 00:43:39 - 00:43:42

exactly [laughter]

没错。（笑）

### 00:43:42 - 00:43:42

help us out.

帮我们排查一下。

### 00:43:42 - 00:44:08

And we do have Yeah, we do have support for XPU devices. So on both PyTorch and torch vision. So uh if you have a PR in mind, you can go and open a PR against PyTorch Vision CI. But like I guess if I read the question correctly, if you train model on CUDA, can you export it for XPU? You need to make some changes to your model first, right?

对，我们确实支持 XPU 设备，PyTorch 和 torch vision 都支持。如果你有具体的 PR 想法，可以针对 PyTorch Vision CI 提一个 PR。不过，如果我没理解错这个问题，你是问在 CUDA 上训练模型后，能不能将它导出供 XPU 使用？你需要先对模型做一些修改，对吧？

### 00:44:06 - 00:44:57

You need to like transfer your model to XPU, make sure that it works and then AOT inductor should work on XPU same as it works on any other back ends. But I'm not even sure if int 8 contisation is supported through inductor even for CUDA. like I think that there are lots of like layers to unload here. So please file an issue but I also kind of think why do you want an int 8 rather than like well int 8 and and check the spec if iGPU even support an int8ate uh fast smart n I know that CPUs do but I'm not sure about GPUs in particular.

你需要把模型转移到 XPU，确认它能运行，然后 AOT inductor 应该能像支持其他后端一样支持 XPU。但我甚至不确定 inductor 是否支持 CUDA 上的 int 8 contisation。这里面有很多层问题要弄清楚，所以请提交一个 issue。不过我也在想，你为什么想用 int 8？另外也要查一下规格，看看 iGPU 是否支持 int8ate 的 fast smart n。我知道 CPU 支持，但不确定 GPU 是否支持。

### 00:44:57 - 00:45:26

Um I think Basil's pulling the one that we mentioned to him here. Maybe in the meantime we can do a small uh note about PyTorch release matrix. So one of the changes that is in 214 is that we mandate that all of your extensions will be compiled with oh here you go C++ 20 by default. So if you're using an older compiler, you really need to update.

我想 Basil 正在找我们刚才向他提到的那个问题。与此同时，也许可以简单说一下 PyTorch release matrix。214 中的一项变化是，我们要求你的所有扩展默认都用 C++ 20 编译。哦，找到了。所以如果你还在使用旧编译器，确实需要更新。

### 00:45:23 - 00:45:37

It is 2026 and majority of the compiler should be able to support C++ 20. Uh but that's backward compatibility breaking change. And with that, I don't know, Andrew, do you want to Oh, Chris, do you want to read the question first?

现在是 2026，大多数编译器应该都能支持 C++ 20。但这是一项破坏向后兼容性的变更。说到这里，Andrew，你要不要……哦，Chris，你要不要先读一下问题？

### 00:45:37 - 00:45:54

Yeah. Yeah. Yeah. So, so this this question which I think was going in the same direction that that Nikita was was thinking is do I still have to upgrade um torch vision every time I upgrade grade pietorch which I know was a headache that people uh had to deal with for a very long time that um Andre do you want to respond to this?

好。这个问题我想和 Nikita 刚才的思路是同一个方向：每次升级 pietorch，我还必须升级 torch vision 吗？我知道这件事长期以来一直让大家头疼。Andre，你想回答一下吗？

### 00:45:54 - 00:46:39

Yes. Yes. Um so torch vision uh 0.29 is a stable uh with respect to 21.14. So it will keep working with later version 2.15 2.16 and later. Uh the only catch here if you need to um introduce new CUDA version you may need to download um version of torch vision compiled with this CUDA version. Um but yes um like as um as a basic you no longer need matching torch vision to install every torch upgrade and uh in the future we may not do torch vision at the same time as torch.

好的。torch vision 0.29 对于 21.14 是稳定的，所以它会继续适用于之后的 2.15、2.16 以及更晚的版本。唯一需要注意的是，如果你要引入新的 CUDA 版本，可能需要下载使用该 CUDA 版本编译的 torch vision。但一般来说，你不再需要每次升级 torch 时都安装一个匹配的 torch vision。今后，我们也可能不再同时发布 torch vision 和 torch。

### 00:46:35 - 00:46:50

So the releases could be um a little bit decoupled because of this. So I think it's something good for us. It's a good news.

因此，这两个版本的发布时间可以稍微错开。我觉得这对我们是件好事，也是个好消息。

### 00:46:50 - 00:46:50

Just next next question, please.

请看下一个问题。

### 00:46:50 - 00:46:53

Next question.

下一个问题。

### 00:46:53 - 00:47:06

Okay. So, what is new for writing dynamic models that still compile? Well, um Andre, do you want to take this one as as as well?

好的。要编写仍然能够编译的动态模型，有什么新方法？Andre，这个问题你也来回答吗？

### 00:47:06 - 00:47:45

I think we answered this question kind of twice already. There was a question separate. Like if we want to highlight the features there is a torch switch uh torch while loop that is captured across codraph and dynamic spec and people already asked about dynamic spec and whether it will cause inductor to fail when it goes above like an upper boundary. People talked about uh asked about specialization but yes those are the features that you should try and like uh they will improve your VSC to compile model with dynamically shaped inputs like you will have fun um compiled artifacts that works for multiple shapes.

我觉得我们已经回答过这个问题大概两次了，之前有一个单独的问题。如果要重点介绍这些功能，有 torch switch、torch while loop，它们可以跨 codraph 被捕获，还有 dynamic spec。大家也已经问过 dynamic spec，以及超过某个上界时它会不会导致 inductor 失败；也有人问过 specialization。总之，这些都是你应该试试的功能。它们会改善你编译具有动态形状输入的模型时的 VSC，让你得到能够处理多种形状的编译产物。

### 00:47:43 - 00:47:51

So once again it's in the torch switch [laughter] torch loop and dynamics back.

再说一次，就是 torch switch（笑）、torch loop 和 dynamics back。

### 00:47:51 - 00:48:45

Uh can we jump to question 14 in our prepared ones? Um this is the everpresent Torch script uh topic [snorts] trying to touch on things that we haven't already discussed. Okay. So I still use Torch script. Should I be worried? Um Nikita, do you want to answer this one? Yes, we've been saying that to script has been deprecated since I think 211 or 212 and it has not been really working even like you cannot use script on you cannot trace the store script on Python 314 and I'm not sure what is the status of 312 or 313 like it's just being deprecated there are bugs there are security problems it supports less and less operators I don't think you can capture anything that support flotate or whatever.

我们能跳到准备好的问题 14 吗？这是一直绕不开的 Torch script 话题。（哼笑）我想谈谈还没讨论过的内容。好，问题是：我还在使用 Torch script，需要担心吗？Nikita，你来回答好吗？是的，我们一直在说 to script 已弃用，我记得是从 211 或 212 开始。它实际上早就不太能用了：在 Python 314 上，你不能使用 script，也不能 trace store script。我不确定它在 312 或 313 上是什么状态。总之，它正在被弃用，存在 bug，也有安全问题，支持的算子越来越少。我觉得你甚至无法捕获任何支持 flotate 之类东西的内容。

### 00:48:43 - 00:49:12

So don't use to script anymore. So AOTI expert is a pretty uh like mature framework at this point. Use this one. If you need to use script, you can use an older PyTorch but like be aware of the limitation. Be like we will be happy if there is a community maintainer who want to like look into the torch script but we are not actively working on new features or have anyone who actively like triages in common issues.

所以不要再用 to script 了。AOTI expert 到目前为止已经是一个相当成熟的框架，用它吧。如果你必须使用 script，可以用旧版 PyTorch，但要了解它的限制。如果有社区维护者愿意接手 torch script，我们会很高兴；但我们没有积极开发新功能，也没有人持续分流和处理常见的 issue。

### 00:49:10 - 00:49:25

So unless they are security related they will probably stay as is and craft and script is not a security issue unfortunately.

因此，除非 issue 与安全有关，否则它们很可能会保持现状。遗憾的是，craft and script 不属于安全问题。

### 00:49:25 - 00:49:30

Okay we have I think we have one left one more remaining um prepared question. Um

好的，我想我们还剩一个准备好的问题。

### 00:49:30 - 00:49:33

we switch to community ones or whatever.

或者我们转去回答社区的问题之类的。

### 00:49:33 - 00:49:49

No I okay I think we have one more prepared one. If there are community ones let's go back and make sure we catch them before we end. Um, so I hear that uh 21.14 supports Python 315. Can I just pip install Torch on it? Um, Andre, do you want to take this one?

不，我想我们还有一个准备好的问题。如果有社区问题，我们结束前再回头确认都答到了。我听说 21.14 支持 Python 315。我能直接在上面运行 pip install Torch 吗？Andre，你来回答好吗？

### 00:49:49 - 00:50:26

Um, yes. Uh, so basically the short answer, we do support 315. Uh, but it's only available via download pytorch.org. So pip install torch will not work with 315 for this release 2014. Um having said that in the future uh we are planning on um uh publishing for the next release 2.15 we're publishing uh 315 Python to the pipy as a default one. We also be duplicating Python 3.10 for the next release.

可以。简短地说，我们确实支持 315，但目前只能通过 download pytorch.org 获取。所以，对于这次 2014 版本发布，pip install torch 在 315 上不能用。不过，我们计划在下一个版本 2.15 中，将 315 Python 默认发布到 pipy。我们还会在下一个版本中 duplicating Python 3.10。

### 00:50:25 - 00:51:00

I think it's very important if you're still running 3.10 for next release it will not be supported. no wheels published and um compiling from source will not be possible. So we'll be migrating or supporting minimum 311. Um but yes for the current version 310 3.15 including uh free threaded 3.15T is are supported. However, one more call out is that torch compile is still not enabled for 315 and 315T.

我觉得这一点非常重要：如果你仍在使用 3.10，下一个版本就不再支持它了。届时不会发布 wheels，也无法从源码编译。因此，我们会转向最低支持 311。不过，对当前版本来说，310 和 3.15，包括 free threaded 3.15T，都是受支持的。另外还要提醒一点：torch compile 在 315 和 315T 上仍未启用。

### 00:51:00 - 00:51:45

Um yes I think that's that's about uh covers it. I don't know if anybody else have any additional thoughts. Oh, maybe like kind of a reason why we don't support 310 anymore because uh it reached its end of life per um like uh Python end of life policy for the releases and we also will be like so we constantly reviewing the supported platforms and unfortunately retires the ones that like go out of support so we probably also be stopping supporting like Mac OS I forget what was released from three years ago but the typical and life cycle for those releases are also three So we will be returning that one as well.

我想这基本涵盖了要说的内容。不知道其他人有没有补充？也许还可以解释一下为什么我们不再支持 310：按照 Python 的版本生命周期政策，它已经到了生命周期终点。我们也会持续审查受支持的平台，遗憾的是，停止支持那些已经失去支持的平台。所以我们可能也会停止支持某个 Mac OS 版本，我忘了那是三年前发布的哪个版本，但这些版本通常也只有三年的生命周期；我们也会让那个版本退出支持。

### 00:51:42 - 00:52:04

And again CUDA 12 that was mentioned before. Let's repeat it again that CUDA 12 support is going away. So you should still be able to build from source. We try not to break that thing. And I don't think we will actively kind of break 310 but we will not be done like testing it. So it will get broken eventually. Cool.

还有前面提过的 CUDA 12，再重复一遍：CUDA 12 的支持即将结束。你应该仍然能从源码构建；我们会尽量不破坏这一点。我想我们也不会主动破坏 310，但不会继续测试它，所以它最终会坏掉。好。

### 00:52:04 - 00:52:17

Okay. I think we have a couple more questions lined up. Um, so, uh, Brian asks, "How can I use PyTorch to create my own LLMs?" Anyone [laughter] want to?

好的，我想我们还有几个问题。Brian 问：“我怎样用 PyTorch 创建自己的 LLMs？”有人想回答吗？（笑）

### 00:52:17 - 00:52:23

There are lots of great tutorials. I want to say that like that tell how you can write

有很多很棒的教程，我想说，它们会教你怎样编写

### 00:52:23 - 00:52:46

like LLM training and inference like Andrew Corp's uh like tiny story will be like a good one if you want to get familiar with LLMs. But if you want to train your own [laughter] like I don't know uh leading model using just PyTorch you need a pretty substantial hardware investments uh which might be a bigger blocker

LLM 训练和推理代码。比如 Andrew Corp 的 tiny story，如果你想熟悉 LLMs，那会是个不错的例子。但如果你想只用 PyTorch 训练自己的，嗯，我不知道，领先的模型，就需要投入相当可观的硬件，这可能是更大的障碍。（笑）

### 00:52:46 - 00:52:56

and also you need a lot of data right so like I think the bottleneck is not your framework but your infrastructure and your data

而且你还需要大量数据，对吧？所以我认为瓶颈不在框架，而在你的基础设施和数据。

### 00:52:56 - 00:52:59

yeah I was going to say

对，我正想这么说。

### 00:52:59 - 00:53:07

there there are some folks out there creating opensource models models that you could then maybe [laughter] refine. I don't know, Joe.

外面有些人在开发开源模型，之后你也许可以（笑声）进一步优化它们。我也说不准，Joe。

### 00:53:07 - 00:53:32

Hopefully soon. Yeah, hopefully. Uh when those models models are hard to build, I have to say. Um no, I I I think like building models from scratch, like really good ones obviously is expensive. I think, you know, to Nikita's point, there's a definitely hardware is is a big challenge. I think there there's some really cool like I I'll I'll plug Unslo because like Daniel and Michael are really great and they have a great ecosystem.

希望很快就能有。对，希望如此。不过我得说，那些模型很难构建。我觉得从零开始构建模型，尤其是真正优秀的模型，显然很昂贵。我想，正如 Nikita 所说，硬件肯定是个很大的挑战。还有一些很棒的项目，我想推荐一下 Unslo，因为 Daniel 和 Michael 都非常出色，他们也建立了很好的生态。

### 00:53:30 - 00:53:53

They have great tools. Uh you could basically pick out, you know, open source models and you can use unsloth and and and kind of like do RL and and SFT and and be able to do it really quick like quickly and cheaply. Uh even in notebooks um like there's actually a ton of examples there. So uh but like creating it from scratch is like quite expensive u quite involved.

他们有很棒的工具。你基本上可以选一个开源模型，用 unsloth 做 RL 和 SFT，而且做起来又快又便宜。甚至可以在 notebooks 里做，那里有很多示例。不过，从零开始创建模型确实相当昂贵，也相当费力。

### 00:53:50 - 00:54:14

Uh we've kind of um we're we're at the point now as a community where it's just it's like prohibitively expensive to to do that as an individual. Um, but that said, there's a ton of great open models out there. Um, and you can play with them and you can customize them and you can run them locally now with Olana and other platforms. Um, so I think that's like um, yeah, obviously Karpath's uh, project is is super cool as well.

我们这个社区现在已经到了这样一个阶段：个人从零开始做这件事，成本高得几乎无法承担。不过，现成的优秀开放模型有很多。你可以拿它们做实验、按自己的需要定制，现在还可以通过 Olana 和其他平台在本地运行它们。所以我觉得，Karpath 的项目显然也非常棒。

### 00:54:12 - 00:54:21

U, there's a lot going on there. Um, but I wouldn't worry about training from scratch because there's just a ton to build on right now. It's just readily available.

这方面正在发生很多事。但我觉得不用担心从零开始训练，因为眼下有大量现成的东西可以作为基础，而且都很容易获取。

### 00:54:21 - 00:54:45

Yeah. Awesome. Uh, I think we had one or two more questions. Um so uh this one is from Manukumar. Uh are there any uh Google Summer of Code uh projects uh focused around PyTorch? I don't know if there are. Does anybody know among this group?

对，太好了。我想我们还有一两个问题。这一个来自 Manukumar：有没有围绕 PyTorch 的 Google Summer of Code 项目？我不知道有没有。在座各位有人知道吗？

### 00:54:45 - 00:55:02

I don't know. But that's a good idea. Like I think maybe pitch a few few ideas to Google. But like as name suggest, it's a Google summer of code. We don't run it. So if somebody wants to organize it, that sounds great. And like

我不知道。但这是个好主意，也许可以向 Google 提几个想法。不过，顾名思义，这是 Google Summer of Code，不是我们主办的。如果有人想组织，那听起来很棒。而且……

### 00:55:02 - 00:55:02

yeah,

对。

### 00:55:02 - 00:55:05

if somebody

如果有人……

### 00:55:05 - 00:55:07

Yeah, we'll be happy to support.

对，我们很乐意提供支持。

### 00:55:07 - 00:55:08

Yep.

没错。

### 00:55:08 - 00:55:14

We should we should reach out to our uh our our Google um PyTorch Foundation. Uh uh

我们应该联系一下我们在 Google 的 PyTorch Foundation 那边的人。

### 00:55:14 - 00:55:15

there you go.

对，就是这样。

### 00:55:15 - 00:55:18

And see what we can help.

看看我们能帮上什么忙。

### 00:55:18 - 00:55:49

Exactly. Okay. Awesome. Um, Basil, were there any more questions in the in the comments? Oh, okay. Uh, no, I think Okay. Well, yeah, this is a slightly different question. With the silicon market exploding, how is PyTorch ensuring that the XPU, you know, torch.xpu can compile and run um models across Nvidia, AMD, and Intel chips with um zero performance loss. Um

正是。好，太好了。Basil，评论区还有其他问题吗？哦，好。没有，我想……好吧，有一个稍微不同的问题：随着芯片市场迅速扩张，PyTorch 如何确保 XPU，也就是 torch.xpu，能够在 Nvidia、AMD 和 Intel 的芯片上编译并运行模型，而且实现零性能损失？

### 00:55:49 - 00:55:52

I think it's a loaded question but also like uh it's very hard to define what zero component

我觉得这是个信息量很大的问题，而且很难界定什么是零组件。

### 00:55:52 - 00:55:53

right

对。

### 00:55:53 - 00:56:04

don't think there is any like hardware spec that you can say well this this piece of silicon is exactly the same as another piece of silicon and also like a pure

我觉得不存在这样的硬件规格，能让你说一块芯片和另一块芯片完全一样；而且还有所谓纯粹的……

### 00:56:04 - 00:56:44

uh like a feature parity but as Andrew say there is CRC and there is in general and you should direct like project experience development partnership with Intel which is part of the PyTorch Foundation and they doing their best and community doing their best and what you can do to make sure that it is the case if you notice that something was working on your black fill and doesn't work on XPU please file an issue and I guess there will be people who happy to answer if this can be addressed in some way but yeah like we we want heterogeneity we want uh like models to work.

比如功能对等的问题。不过，正如 Andrew 所说，有 CRC，也有整体上的工作；还有项目经验以及与 Intel 的开发合作。Intel 是 PyTorch Foundation 的一员，他们和社区都在尽最大努力。你能做的是，如果发现某个功能在你的 black fill 上能用，在 XPU 上却不能用，请提交一个 issue。我想会有人愿意回答，看看是否能以某种方式解决。但我们确实希望支持不同的硬件，也希望模型能够运行。

### 00:56:42 - 00:56:56

We talked about torch accelerate. We talked about CRCR. We talked a lot about like a newer romcom. So there are lots of initiative but I guess zero performance loss is just unattainable and it's also I'm not sure what it means by

我们谈到了 torch accelerate，谈到了 CRCR，也谈了很多关于较新的 romcom 的事情。所以有很多相关举措，但我觉得零性能损失根本无法实现，而且我也不确定这里的意思是……

### 00:56:56 - 00:57:07

not well defined. It's not well defined because different hardware performs you know just just behaves differently. Um and we try and give you lots of knobs and lots of capabilities but uh

没有明确的定义。确实没有明确的定义，因为不同的硬件性能不同，你知道，它们的表现就是不一样。我们努力提供大量可调选项和功能，但是……

### 00:57:07 - 00:57:19

we try to reduce overhead with every release like the majority of the releases the features the kind of unspoken features that we don't talk about is we always try to bring a little bit of performance here and there.

我们每次发布新版本都努力降低开销。大多数版本里，那些我们没有特别提及的功能改进，其实就是在各个地方一点一点地提升性能。

### 00:57:19 - 00:57:44

Yeah, just to add to it I think one of the interesting project maybe for you to look into is VLM. This is like they run multiple different models on different hardware and like as an example so you can explore it a little bit and I think it's open source and the CI is open source so um just as an example to look into could be interesting.

对，我再补充一点。我觉得你可以看看 VLM 这个项目，可能会很有意思。比如，他们会在不同硬件上运行多个不同的模型，你可以稍微研究一下。我想它是开源的，CI 也是开源的，所以这是一个值得参考的例子。

### 00:57:44 - 00:58:03

Um I feel like we have answered this but I'll just quickly uh just just uh shout it out here. So somebody asked on the on the comment thread um is it uh is PyTorch supporting the newest CUDA? I think we're we're either on the newest CUDA or pretty close. And usually if you jump on nightlys, you can get the latest and greatest.

我觉得我们已经回答过这个问题，但我还是快速提一下。有人在评论区问，PyTorch 是否支持最新的 CUDA。我想我们要么已经用上了最新的 CUDA，要么离它很近。通常，如果你使用 nightlys，就能用上最新的版本。

### 00:58:03 - 00:58:10

Um, so Andre, you should your head when I said that. So I must have said something slightly wrong. So jump in here and correct me.

Andre，我刚才说那句话时，你“should your head”了。所以我肯定有哪里说得不太对。请你接着说，纠正一下。

### 00:58:10 - 00:58:50

So for 214, we're supporting up to CUDA 13.2. This is not the latest. The latest was um is 13.4 and it was released just like a week after we released PyTorch. So it's like we did not include it yet but it will be included in the next release of PyTorch and it is already in nightly and it has been in nightly for quite some time. So if you really want to have the latest CUDA download nightly and it should work uh for Linux only for Windows I think it's going to be enabled pretty soon maybe within one or two weeks.

对于 214，我们支持到 CUDA 13.2。这不是最新版本；最新的是 13.4，它是在我们发布 PyTorch 大约一周后才发布的，所以我们还没把它包含进来。但它会包含在下一个 PyTorch 版本中，而且已经在 nightly 里有一段时间了。所以，如果你特别想用最新的 CUDA，就下载 nightly，应该可以用，但目前只支持 Linux。至于 Windows，我想很快就会启用，可能在一两周内。

### 00:58:50 - 00:59:17

Awesome. Well I think that was a good note to end on. Um, I think we had uh some fun conver fun conversation. [snorts] Um, a little a little repetition in the questions, but I think that's sort of built into the way we do things. So, thank you uh to my experts for uh for for putting up with that. Um, uh, so let's uh just in terms of closing remarks, um, thanks as always have to start with the contributors across the community who put in the steady effort that makes PyTorch what it is.

太好了。我觉得用这个话题收尾很合适。我们聊得很愉快。问题有一点重复，但我觉得这也是我们的活动形式所决定的。所以，谢谢几位专家包容这一点。最后说几句感谢的话：首先，一如既往，要感谢社区里的贡献者，是他们持续付出的努力成就了今天的 PyTorch。

### 00:59:14 - 00:59:43

So 500 odd people uh contributed to this uh particular release and it's been a number near that for the last several releases. Uh so it's just huge community effort. Uh and you know what we're talking about here is that effort. Uh so uh so thanks to all those folks. Uh thanks also to the three of you uh here on on on on stage answering these questions. Uh many of which you were answering un completely unprepared because they they were audience questions [laughter] in.

这次发布有大约 500 人参与贡献，最近几个版本的贡献人数也都接近这个数字。这真的是整个社区共同努力的成果，而我们今天谈论的就是这些成果。所以感谢所有贡献者。也感谢台上的你们三位回答这些问题，其中很多都是观众提出的，你们事先完全没有准备（笑声）。

### 00:59:41 - 01:00:09

So, thank you for being brave uh and wise. Um so, uh and being generous with your time and your expertise. Uh thanks also to behind the scenes uh Basil and Jyn who were scrambling to to get the uh the the questions uh from the from the discussion thread into uh into uh the screen here. So, thank you guys for working your magic. I appreciate that. Uh and then also thanks to everybody who joined and asked questions because you made this a good conversation.

谢谢你们勇敢应对，也谢谢你们的见识，以及慷慨分享时间和专业知识。还要感谢幕后的 Basil 和 Jyn，他们一直忙着把讨论区里的问题整理到这里的屏幕上。谢谢你们施展本领，我很感激。也感谢每一位参与并提问的人，是你们促成了这场精彩的交流。

### 01:00:06 - 01:00:28

Uh we really appreciate it and keep building awesome things with PyTorch. Uh just a couple of resources uh for you to think about um a as we wrap up here. Um there are we released a blog with this release uh which has a lot of detailed uh details about the various different features. There's also the release notes. You should be able to find both of those really easily.

我们真的很感谢大家，也请继续用 PyTorch 构建出色的东西。结束之前，再给大家提几个资源。这次发布时，我们发表了一篇博客文章，详细介绍了各项功能；另外还有发布说明。这两份资料应该都很容易找到。

### 01:00:26 - 01:00:52

We may drop links to them here. I'm not I'm not sure whether that will happen or not. Um also the developer communities. So discuss.pyarch.org and devdiscuss.pyrush.org are great uh for asking really detailed questions like you guys uh did here. Um and you'll you'll you'll be able to get the the the experts uh um across the community to answer. Um also there's some events coming up that I want to make people aware of.

我们可能会在这里贴出链接，但我不确定最后会不会贴。另外还有开发者社区：discuss.pyarch.org 和 devdiscuss.pyrush.org 都很适合提出像大家今天这样非常具体的问题，社区里的专家会来回答。还有一些即将举行的活动，我想让大家知道。

### 01:00:50 - 01:01:12

Uh so PyTorch Conference North America will be in San Jose in just a little bit more than a month. Um so that's October 20th and 21st. I think all of us will be here. All of us here will be will be there. Um and um uh you know it's a great chance to to to learn a ton about PyTorch and say hello to to to fellow uh uh members of the community.

PyTorch Conference North America 再过一个多月就会在 San Jose 举行，时间是 October 20th 和 21st。我想我们都会去，在这里的所有人都会到场。这是一个深入了解 PyTorch、和社区其他成员见面打招呼的好机会。

### 01:01:11 - 01:01:35

Um I'm certainly looking forward to it and if you're um part of the global audience um there are a bunch of PyTorch, you know, we've been really proliferating uh global events uh this year and I imagine that will continue. Uh so we had an event in Paris in the spring which was a lot of fun. We just completed events uh well a while ago in India and then more recently in China and I think we have one planned in Japan for later this year.

我很期待。如果你身处世界其他地方，今年我们在各地举办的 PyTorch 活动越来越多，我想今后也会继续。春天我们在 Paris 举办了一场活动，非常有意思。不久前我们还在 India 办了活动，最近又在 China 办了一场。我想今年晚些时候还计划在 Japan 举办一场。

### 01:01:32 - 01:01:47

So um and then I think there are pie days and smaller events happening all the time in lots of different places. So uh look look at the piece.org website for uh for u events near you. So I think that's it

此外，我想很多地方还经常举办 pie days 和规模较小的活动。所以可以到 piece.org 网站上看看你附近有没有活动。我想就这些了。

### 01:01:47 - 01:01:57

and last but no there is one more thing last but not least thank you very much Chris for being a fantastic moderator. We could not have wished for a better moderator here and without you it would be a very different event. So, thank you.

最后……不，还有一件事。最后但同样重要的是，非常感谢 Chris，你是一位出色的主持人。我们不可能期待更好的主持人了；没有你，这场活动会完全不同。所以，谢谢你。

### 01:01:58 - 01:02:00

Thank you, Chris. Great job.

谢谢你，Chris。做得很棒。

### 01:02:00 - 01:02:00

Bye-bye.

再见。

### 01:02:00 - 01:02:02

Thank you.

谢谢。
