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

6.710–6.720  Okay. Uh hello everyone. Um and welcome

好的，大家好，欢迎。

6.720–10.150  Okay. Uh hello everyone. Um and welcome to our PyTorch uh 2.14

好的，大家好，欢迎参加我们的 PyTorch 2.14……

10.150–10.160  to our PyTorch uh 2.14

参加我们的 PyTorch 2.14……

10.160–15.030  to our PyTorch uh 2.14 uh release live uh Q&A. Um uh let me

参加我们的 PyTorch 2.14 版本发布直播问答。我先……

15.030–15.040  uh release live uh Q&A. Um uh let me

版本发布直播问答。我先……

15.040–17.269  uh release live uh Q&A. Um uh let me scroll back up to my uh my comments

版本发布直播问答。我先往上翻找我的评论。

17.269–17.279  scroll back up to my uh my comments

往上翻找我的评论。

17.279–20.390  scroll back up to my uh my comments here. Um this session will start with uh

往上翻找我的评论。这场活动将先……

20.390–20.400  here. Um this session will start with uh

这场活动将先……

20.400–22.950  here. Um this session will start with uh with introducing our experts um and then

这场活动将先介绍我们的专家，然后……

22.950–22.960  with introducing our experts um and then

先介绍我们的专家，然后……

22.960–24.550  with introducing our experts um and then a short overview of the key updates in

先介绍我们的专家，然后简要介绍……

24.550–24.560  a short overview of the key updates in

简要介绍主要更新……

24.560–28.150  a short overview of the key updates in PyTorch uh 21.14 followed by um the fun

简要介绍 PyTorch 21.14 的主要更新，然后进入有趣的……

28.150–28.160  PyTorch uh 21.14 followed by um the fun

PyTorch 21.14，接下来是有趣的……

28.160–30.710  PyTorch uh 21.14 followed by um the fun part which is an open Q&A covering your

PyTorch 21.14，接下来是开放问答，回答大家关于……

30.710–30.720  part which is an open Q&A covering your

接下来是开放问答，回答大家关于……

30.720–33.030  part which is an open Q&A covering your questions about this release. Um first

回答大家关于这次发布的问题。首先……

33.030–33.040  questions about this release. Um first

关于这次发布的问题。首先……

33.040–34.389  questions about this release. Um first I'll introduce myself. I'm I'm Chris

关于这次发布的问题。先介绍一下我自己，我是 Chris……

34.389–34.399  I'll introduce myself. I'm I'm Chris

我先介绍一下自己，我是 Chris……

34.399–35.830  I'll introduce myself. I'm I'm Chris Scott. I'm your moderator for today's

我先介绍一下自己，我是 Chris Scott，今天的主持人。

35.830–35.840  Scott. I'm your moderator for today's

Scott，今天由我主持……

35.840–37.030  Scott. I'm your moderator for today's webinar. I've been involved in the

Scott，今天由我主持这场线上研讨会。我一直参与……

37.030–37.040  webinar. I've been involved in the

这场线上研讨会。我一直参与……

37.040–39.430  webinar. I've been involved in the PyTorch community for about eight years

这场线上研讨会。我参与 PyTorch 社区已有大约八年。

39.430–39.440  PyTorch community for about eight years

我参与 PyTorch 社区已有大约八年。

39.440–41.190  PyTorch community for about eight years focusing on focusing on product and

我参与 PyTorch 社区已有大约八年，主要关注产品和……

41.190–41.200  focusing on focusing on product and

主要关注产品和……

41.200–43.830  focusing on focusing on product and marketing uh type angles. Um these days

主要关注产品和市场营销方面。现在……

43.830–43.840  marketing uh type angles. Um these days

市场营销方面。现在……

43.840–45.670  marketing uh type angles. Um these days I run a consulting practice uh focusing

市场营销方面。现在我经营一家咨询机构，专注于……

45.670–45.680  I run a consulting practice uh focusing

我经营一家咨询机构，专注于……

45.680–48.229  I run a consulting practice uh focusing on emerging AI technologies and I'm

我经营一家咨询机构，专注于新兴 AI 技术，也……

48.229–48.239  on emerging AI technologies and I'm

专注于新兴 AI 技术，也……

48.239–50.069  on emerging AI technologies and I'm delighted to welcome a panel of three

专注于新兴 AI 技术。我很高兴欢迎三位……

50.069–50.079  delighted to welcome a panel of three

很高兴欢迎三位……

50.079–52.389  delighted to welcome a panel of three absolute PyTorch experts. We have Andre

很高兴欢迎三位顶尖的 PyTorch 专家。我们请来了 Andre……

52.389–52.399  absolute PyTorch experts. We have Andre

顶尖的 PyTorch 专家。我们请来了 Andre……

52.399–55.990  absolute PyTorch experts. We have Andre Talman uh Joe Spizac and Nikita Schulga.

顶尖的 PyTorch 专家：Andre Talman、Joe Spizac 和 Nikita Schulga。

55.990–56.000  Talman uh Joe Spizac and Nikita Schulga.

他们是 Talman、Joe Spizac 和 Nikita Schulga。

56.000–57.430  Talman uh Joe Spizac and Nikita Schulga. Uh and they're here to answer your

Talman、Joe Spizac 和 Nikita Schulga。他们会回答大家的……

57.430–57.440  Uh and they're here to answer your

他们会回答大家的……

57.440–59.349  Uh and they're here to answer your questions about PyTorch and generally to

他们会回答大家关于 PyTorch 的问题，也会……

59.349–59.359  questions about PyTorch and generally to

关于 PyTorch 的问题，也会……

59.359–62.630  questions about PyTorch and generally to you know spread the word. Um so Andre is

回答有关 PyTorch 的问题，也借此介绍 PyTorch。Andre 是……

62.630–62.640  you know spread the word. Um so Andre is

也借此介绍 PyTorch。Andre 是……

62.640–64.310  you know spread the word. Um so Andre is a release engineer in the PyTorch Dev

也借此介绍 PyTorch。Andre 是 PyTorch Dev 团队的发布工程师。

64.310–64.320  a release engineer in the PyTorch Dev

PyTorch Dev 团队的发布工程师。

64.320–67.109  a release engineer in the PyTorch Dev Info team here at Meta or at Meta. I'm

他是 Meta 的 PyTorch Dev Info 团队发布工程师。我……

67.109–67.119  Info team here at Meta or at Meta. I'm

Meta 的信息团队，或者说在 Meta。我……

67.119–69.109  Info team here at Meta or at Meta. I'm sorry. I used to be at Meta so that just

Meta 的信息团队，或者说在 Meta。抱歉，我以前在 Meta 工作，所以……

69.109–69.119  sorry. I used to be at Meta so that just

抱歉，我以前在 Meta 工作，所以……

69.119–70.469  sorry. I used to be at Meta so that just rolls off my tongue. It's no longer

抱歉，我以前在 Meta 工作，所以顺口就这么说了。现在已经不是……

70.469–70.479  rolls off my tongue. It's no longer

顺口就这么说了。现在已经不是……

70.479–73.830  rolls off my tongue. It's no longer accurate. at Meta. Uh he drives uh the

顺口就这么说了。现在这么说已经不准确了。他在 Meta 负责……

73.830–73.840  accurate. at Meta. Uh he drives uh the

这么说已经不准确了。他在 Meta 负责……

73.840–75.429  accurate. at Meta. Uh he drives uh the release process and keeps uh the

这么说已经不准确了。他在 Meta 负责发布流程，并让……

75.429–75.439  release process and keeps uh the

发布流程，并让……

75.439–76.789  release process and keeps uh the continuous integration and continuous

发布流程，并让持续集成和持续……

76.789–76.799  continuous integration and continuous

持续集成和持续……

76.799–79.109  continuous integration and continuous development infrastructure humming. Uh

持续集成和持续开发的基础设施顺畅运转。

79.109–79.119  development infrastructure humming. Uh

开发基础设施顺畅运转。

79.119–83.109  development infrastructure humming. Uh Joe um is the uh VP of product uh and

开发基础设施顺畅运转。Joe 是产品副总裁，也是……

83.109–83.119  Joe um is the uh VP of product uh and

Joe 是产品副总裁，也是……

83.119–85.910  Joe um is the uh VP of product uh and head of open source at Reflection AI. Uh

Joe 是 Reflection AI 的产品副总裁兼开源负责人。

85.910–85.920  head of open source at Reflection AI. Uh

Reflection AI 的开源负责人。

85.920–87.749  head of open source at Reflection AI. Uh he's one of the PyTorch core maintainers

Reflection AI 的开源负责人。他也是 PyTorch 核心维护者之一。

87.749–87.759  he's one of the PyTorch core maintainers

他是 PyTorch 核心维护者之一。

87.759–89.590  he's one of the PyTorch core maintainers and among the strongest advocates that I

他是 PyTorch 核心维护者之一，也是我认识的最坚定的倡导者之一，

89.590–89.600  and among the strongest advocates that I

也是我认识的最坚定的倡导者之一，

89.600–91.670  and among the strongest advocates that I know of for open source, open models and

也是我认识的最坚定的倡导者之一，倡导开源、开放模型，以及……

91.670–91.680  know of for open source, open models and

倡导开源、开放模型，以及……

91.680–93.429  know of for open source, open models and open infrastructure across the AI

倡导在整个 AI 社区推广开源、开放模型和开放基础设施。

93.429–93.439  open infrastructure across the AI

在整个 AI 社区推广开放基础设施。

93.439–95.910  open infrastructure across the AI community. Uh and always a fun uh fun

在整个 AI 社区推广开放基础设施。而且听他说话总是很有意思。

95.910–95.920  community. Uh and always a fun uh fun

在社区里。而且听他说话总是很有意思。

95.920–98.230  community. Uh and always a fun uh fun person to to hear from. Nikita is a

在社区里。而且听他说话总是很有意思。Nikita 是……

98.230–98.240  person to to hear from. Nikita is a

听他说话总是很有意思。Nikita 是……

98.240–100.789  person to to hear from. Nikita is a PyTorch core maintainer uh and a key

听他说话总是很有意思。Nikita 是 PyTorch 核心维护者，也是重要的……

100.789–100.799  PyTorch core maintainer uh and a key

PyTorch 核心维护者，也是重要的……

100.799–102.870  PyTorch core maintainer uh and a key contributor and reviewer really active

PyTorch 核心维护者，也是重要的贡献者和审查者，活跃于……

102.870–102.880  contributor and reviewer really active

贡献者和审查者，活跃于……

102.880–106.069  contributor and reviewer really active across just a bunch of areas uh in in

贡献者和审查者，在许多领域都很活跃，包括……

106.069–106.079  across just a bunch of areas uh in in

在许多领域都很活跃，包括……

106.079–108.069  across just a bunch of areas uh in in the core area of PyTorch. So you

在许多领域都很活跃，包括 PyTorch 的核心领域。所以你……

108.069–108.079  the core area of PyTorch. So you

PyTorch 的核心领域。所以你……

108.079–109.590  the core area of PyTorch. So you probably see his name on on lots of

PyTorch 的核心领域。所以你可能在很多地方都见过他的名字。

109.590–109.600  probably see his name on on lots of

可能在很多地方都见过他的名字。

109.600–112.870  probably see his name on on lots of things. Um I'll give a brief overview

可能在很多地方都见过他的名字。我先简要介绍一下……

112.870–112.880  things. Um I'll give a brief overview

很多地方都见过他的名字。我先简要介绍一下……

112.880–114.550  things. Um I'll give a brief overview now in a in a moment here about the

很多地方都见过他的名字。接下来我会简要介绍一下……

114.550–114.560  now in a in a moment here about the

接下来我会简要介绍一下……

114.560–116.950  now in a in a moment here about the release uh for a few minutes and I'm

接下来我会花几分钟简要介绍这次发布，我也……

116.950–116.960  release uh for a few minutes and I'm

花几分钟介绍这次发布，我也……

116.960–118.550  release uh for a few minutes and I'm would encourage you if you're in the in

花几分钟介绍这次发布。我建议现场的各位……

118.550–118.560  would encourage you if you're in the in

我建议现场的各位……

118.560–120.550  would encourage you if you're in the in the audience uh to to use that time to

我建议现场的各位听众利用这段时间……

120.550–120.560  the audience uh to to use that time to

各位听众利用这段时间……

120.560–122.389  the audience uh to to use that time to think about what questions you might

各位听众可以利用这段时间想想自己有什么问题。

122.389–122.399  think about what questions you might

想想你可能有什么问题。

122.399–125.109  think about what questions you might have um for our experts. uh you can put

想想有什么问题想问我们的专家。你可以把问题发到……

125.109–125.119  have um for our experts. uh you can put

想问专家的问题，你可以发到……

125.119–126.870  have um for our experts. uh you can put those in LinkedIn and and the and the

想问专家的问题，可以发到 LinkedIn 和……

126.870–126.880  those in LinkedIn and and the and the

把问题发到 LinkedIn 和……

126.880–129.430  those in LinkedIn and and the and the YouTube tools. Uh we'll uh we'll cue

把问题发到 LinkedIn 和 YouTube 的提问工具。我们会把问题……

129.430–129.440  YouTube tools. Uh we'll uh we'll cue

通过 YouTube 的提问工具。我们会把问题……

129.440–132.309  YouTube tools. Uh we'll uh we'll cue them up for our experts to answer. We

通过 YouTube 的提问工具。我们会把问题交给专家回答。我们……

132.309–132.319  them up for our experts to answer. We

交给专家回答。我们……

132.319–133.750  them up for our experts to answer. We will we'll also have a few seed

交给专家回答。我们也准备了几个引导性问题……

133.750–133.760  will we'll also have a few seed

我们也准备了几个引导性问题……

133.760–136.470  will we'll also have a few seed questions up uh on our you know prepared

我们也事先准备了几个引导性问题……

136.470–136.480  questions up uh on our you know prepared

事先准备了一些问题……

136.480–137.750  questions up uh on our you know prepared uh based on ongoing community

事先准备了一些问题，依据的是社区正在进行的……

137.750–137.760  uh based on ongoing community

依据社区正在进行的……

137.760–139.510  uh based on ongoing community discussions which you can join by uh

依据社区正在进行的讨论，你也可以参与……

139.510–139.520  discussions which you can join by uh

这些讨论，你也可以通过……参与。

139.520–141.990  discussions which you can join by uh looking them up in uh discuss.pytor.org

这些讨论，你可以在 discuss.pytor.org 上找到并参与。

141.990–142.000  looking them up in uh discuss.pytor.org

到 discuss.pytor.org 查找这些讨论。

142.000–144.630  looking them up in uh discuss.pytor.org and dev discuss.pytor.org or hopping on

可以到 discuss.pytor.org 和 dev discuss.pytor.org 查找，或加入……

144.630–144.640  and dev discuss.pytor.org or hopping on

也可以到 dev discuss.pytor.org，或加入……

144.640–147.270  and dev discuss.pytor.org or hopping on PyTorch Slack. Um those are just here to

也可以到 dev discuss.pytor.org，或加入 PyTorch Slack。这些问题只是为了……

147.270–147.280  PyTorch Slack. Um those are just here to

加入 PyTorch Slack。这些问题只是为了……

147.280–148.790  PyTorch Slack. Um those are just here to to kind of keep things flowing and make

加入 PyTorch Slack。这些问题只是为了让讨论顺畅进行，并确保……

148.790–148.800  to kind of keep things flowing and make

让讨论顺畅进行，并确保……

148.800–150.229  to kind of keep things flowing and make sure we touch on some important topics

让讨论顺畅进行，确保我们谈到一些重要话题。

150.229–150.239  sure we touch on some important topics

确保我们谈到一些重要话题。

150.239–151.910  sure we touch on some important topics but it's more fun really to answer the

确保谈到一些重要话题，但回答……其实更有意思。

151.910–151.920  but it's more fun really to answer the

但回答……其实更有意思。

151.920–152.790  but it's more fun really to answer the questions that come in from the

但回答来自……的问题其实更有意思。

152.790–152.800  questions that come in from the

来自……的问题。

152.800–155.670  questions that come in from the community. So, please don't be shy.

社区提出的问题。所以请大家踊跃提问。

155.670–155.680  community. So, please don't be shy.

社区提出的问题。所以请大家踊跃提问。

155.680–158.070  community. So, please don't be shy. Okay, so an overview of the release. Uh,

社区提出的问题。所以请大家踊跃提问。好，下面概览一下这次发布。

158.070–158.080  Okay, so an overview of the release. Uh,

好，下面概览一下这次发布。

158.080–159.350  Okay, so an overview of the release. Uh, this actually seems to me to be a pretty

好，下面概览一下这次发布。我觉得这次发布相当……

159.350–159.360  this actually seems to me to be a pretty

我觉得这次发布相当……

159.360–160.630  this actually seems to me to be a pretty significant release despite, you know,

我觉得这次发布相当重要，尽管……

160.630–160.640  significant release despite, you know,

这次发布相当重要，尽管……

160.640–163.110  significant release despite, you know, the the numbering 21.14 makes it seem

这次发布相当重要，尽管 21.14 这个版本号让它看起来……

163.110–163.120  the the numbering 21.14 makes it seem

21.14 这个版本号让它看起来……

163.120–164.790  the the numbering 21.14 makes it seem small, but it's not really, I don't

21.14 这个版本号让它显得规模不大，但我觉得实际并非如此……

164.790–164.800  small, but it's not really, I don't

看起来规模不大，但我觉得实际并非如此……

164.800–167.830  small, but it's not really, I don't think, at all. Um, uh, it really moves

看起来规模不大，但我觉得完全不是这样。它确实推动了……

167.830–167.840  think, at all. Um, uh, it really moves

我觉得完全不是这样。它确实推动了……

167.840–170.150  think, at all. Um, uh, it really moves the needle on like compilation, uh, with

我觉得完全不是这样。它确实推动了编译方面的进展，带来了……

170.150–170.160  the needle on like compilation, uh, with

推动了编译方面的进展，带来了……

170.160–172.390  the needle on like compilation, uh, with some new concepts in compilation, fault

推动了编译方面的进展，涉及编译方面的新概念、容错……

172.390–172.400  some new concepts in compilation, fault

编译方面的一些新概念，以及容错……

172.400–175.190  some new concepts in compilation, fault tolerance, uh, and, uh, platform

编译方面的一些新概念、容错，以及平台方面的……

175.190–175.200  tolerance, uh, and, uh, platform

容错，还有平台

175.200–177.270  tolerance, uh, and, uh, platform support. As always, it's a huge

容错和平台支持。一如既往，这是一项庞大的

177.270–177.280  support. As always, it's a huge

支持。一如既往，这是一项庞大的

177.280–179.990  support. As always, it's a huge community left with almost 3,000 commits

支持。一如既往，社区贡献巨大，提交近3000次

179.990–180.000  community left with almost 3,000 commits

社区贡献巨大，提交近3000次

180.000–182.309  community left with almost 3,000 commits from almost 500 uh different uh

社区贡献巨大，提交近3000次，来自近500位

182.309–182.319  from almost 500 uh different uh

来自近500位不同的

182.319–185.190  from almost 500 uh different uh contributors, 487 this year or this

来自近500位贡献者，这次是487位

185.190–185.200  contributors, 487 this year or this

贡献者，这次是487位

185.200–187.750  contributors, 487 this year or this time. Um they made improvements all

贡献者，这次是487位。他们做了各方面的改进

187.750–187.760  time. Um they made improvements all

这次，他们做了各方面的改进

187.760–189.190  time. Um they made improvements all across PyTorch and I encourage folks to

这次，他们改进了PyTorch的方方面面。我也鼓励大家

189.190–189.200  across PyTorch and I encourage folks to

涉及PyTorch的方方面面。我也鼓励大家

189.200–190.790  across PyTorch and I encourage folks to look at the release release blog and the

涉及PyTorch的方方面面。我也鼓励大家查看发布博客和

190.790–190.800  look at the release release blog and the

查看发布博客和

190.800–193.270  look at the release release blog and the release notes for uh for more detail to

查看发布博客和发行说明，了解更多详情

193.270–193.280  release notes for uh for more detail to

发行说明里有更多详情

193.280–195.750  release notes for uh for more detail to supplement this uh this Q&A. A few of

发行说明里有更多详情，可补充这场问答。其中一些

195.750–195.760  supplement this uh this Q&A. A few of

可作为这场问答的补充。其中一些

195.760–197.750  supplement this uh this Q&A. A few of the most spec uh most significant

可作为这场问答的补充。其中最重要的几项

197.750–197.760  the most spec uh most significant

最重要的几项

197.760–201.589  the most spec uh most significant changes this time around are uh NV gym

本次最重要的变化包括NVGEMM

201.589–201.599  changes this time around are uh NV gym

本次变化包括NVGEMM

201.599–204.790  changes this time around are uh NV gym from Nvidia brings cute DSL uh generated

本次变化包括：NVIDIA的NVGEMM将CuTeDSL生成的

204.790–204.800  from Nvidia brings cute DSL uh generated

NVIDIA的NVGEMM将CuTeDSL生成的

204.800–206.949  from Nvidia brings cute DSL uh generated cutless kernels to inductor. So this is

NVIDIA的NVGEMM将CuTeDSL生成的CUTLASS内核引入Inductor。这是

206.949–206.959  cutless kernels to inductor. So this is

CUTLASS内核引入Inductor。这是

206.959–209.830  cutless kernels to inductor. So this is like a performance uh improvement uh for

CUTLASS内核引入Inductor。这能提升

209.830–209.840  like a performance uh improvement uh for

这能提升

209.840–213.190  like a performance uh improvement uh for uh modern uh GPUs uh with epilog fusion

这能提升现代GPU的性能，并支持尾处理融合

213.190–213.200  uh modern uh GPUs uh with epilog fusion

现代GPU上的尾处理融合

213.200–217.430  uh modern uh GPUs uh with epilog fusion scaled uh NV FP4 gems and grouped uh

现代GPU上的尾处理融合、缩放型NVFP4 GEMM和分组

217.430–217.440  scaled uh NV FP4 gems and grouped uh

缩放型NVFP4 GEMM和分组

217.440–220.070  scaled uh NV FP4 gems and grouped uh reduction epilogues autotuned alongside

缩放型NVFP4 GEMM和分组归约尾处理，并与

220.070–220.080  reduction epilogues autotuned alongside

归约尾处理，并与

220.080–222.869  reduction epilogues autotuned alongside Triton and ATIN.

归约尾处理，并与Triton和ATen一起自动调优。

222.869–222.879  Triton and ATIN.

与Triton和ATen一起自动调优。

222.879–225.670  Triton and ATIN. We have a preview of our uh some really

与Triton和ATen一起自动调优。我们还预览了

225.670–225.680  We have a preview of our uh some really

我们还预览了

225.680–228.070  We have a preview of our uh some really deep changes in our in our um uh

我们还预览了分布式后端的一些重大改动

228.070–228.080  deep changes in our in our um uh

分布式后端的一些重大改动

228.080–230.470  deep changes in our in our um uh distributed backend. So rewritten nickel

分布式后端的一些重大改动：重写的NCCL

230.470–230.480  distributed backend. So rewritten nickel

分布式后端：重写的NCCL

230.480–232.949  distributed backend. So rewritten nickel backend for PyTorch. This is ported from

分布式后端：为PyTorch重写的NCCL后端，移植自

232.949–232.959  backend for PyTorch. This is ported from

PyTorch的NCCL后端，移植自

232.959–234.789  backend for PyTorch. This is ported from sort of a an area where we've been doing

PyTorch的NCCL后端，移植自我们一直在做实验的

234.789–234.799  sort of a an area where we've been doing

我们一直在做实验的

234.799–236.949  sort of a an area where we've been doing some experimentation called torchcoms

我们一直在做实验的项目叫torchcomms

236.949–236.959  some experimentation called torchcoms

我们实验的项目叫torchcomms

236.959–238.949  some experimentation called torchcoms implementing uh the full collective

我们实验的项目叫torchcomms，它实现了完整的集合通信接口

238.949–238.959  implementing uh the full collective

实现完整的集合通信功能

238.959–241.670  implementing uh the full collective contract uh with non-blocking

实现完整的集合通信接口，包括非阻塞通信

241.670–241.680  contract uh with non-blocking

包括非阻塞通信的接口

241.680–243.830  contract uh with non-blocking communication communicators and advanced

包括非阻塞通信、通信器和高级功能的接口

243.830–243.840  communication communicators and advanced

通信、通信器和高级功能

243.840–245.830  communication communicators and advanced features such as fault tolerance and

通信、通信器，以及容错等高级功能

245.830–245.840  features such as fault tolerance and

容错等功能

245.840–247.910  features such as fault tolerance and windows uh designed as drag and drop

容错和窗口等功能，设计成可直接替换

247.910–247.920  windows uh designed as drag and drop

窗口，设计成可直接替换

247.920–250.390  windows uh designed as drag and drop replacements for existing nickel C10

窗口，设计成可直接替换现有的 NCCL C10

250.390–250.400  replacements for existing nickel C10

替换现有的 NCCL C10

250.400–253.110  replacements for existing nickel C10 backend. Um, so this I think is really

替换现有的 NCCL C10 后端。我觉得这确实

253.110–253.120  backend. Um, so this I think is really

后端。我觉得这确实

253.120–254.710  backend. Um, so this I think is really uh the fault tolerance I think is going

后端。我觉得容错确实会

254.710–254.720  uh the fault tolerance I think is going

我觉得容错会

254.720–256.710  uh the fault tolerance I think is going to be is going to be a big uh thing as

我觉得容错将是一件大事

256.710–256.720  to be is going to be a big uh thing as

将是一件大事

256.720–258.469  to be is going to be a big uh thing as we uh go forward and I think this is

随着我们继续推进，这将是一件大事，我觉得这

258.469–258.479  we uh go forward and I think this is

随着我们继续推进，我觉得这

258.479–261.189  we uh go forward and I think this is really laying really strong uh uh

随着我们继续推进，我觉得这确实打下了坚实的

261.189–261.199  really laying really strong uh uh

确实打下了坚实的

261.199–264.070  really laying really strong uh uh foundations for that. Uh fault tolerance

确实为此打下了坚实基础。容错

264.070–264.080  foundations for that. Uh fault tolerance

为此打下了基础。容错

264.080–265.990  foundations for that. Uh fault tolerance also becomes a first class citizen in

为此打下了基础。容错也成为

265.990–266.000  also becomes a first class citizen in

也成为一项核心能力

266.000–269.670  also becomes a first class citizen in the C10 uh D uh with in place process

也成为 c10d 的一项核心能力，包括原地进程

269.670–269.680  the C10 uh D uh with in place process

c10d 中的原地进程

269.680–271.510  the C10 uh D uh with in place process group reconfiguration, one-sided RMA

c10d 中的原地进程组重配置、单侧 RMA

271.510–271.520  group reconfiguration, one-sided RMA

进程组重配置、单侧 RMA

271.520–273.430  group reconfiguration, one-sided RMA windows and a flight recorder that works

进程组重配置、单侧 RMA 窗口，以及一个适用于

273.430–273.440  windows and a flight recorder that works

RMA 窗口，以及一个适用于

273.440–275.270  windows and a flight recorder that works for any backend rather than only for

RMA 窗口，以及一个适用于任何后端的飞行记录器

275.270–275.280  for any backend rather than only for

适用于任何后端，而不只是

275.280–277.990  for any backend rather than only for nickel. Uh Apple silicon also got a lot

适用于任何后端，而不只是 NCCL。Apple 芯片也增加了许多

277.990–278.000  nickel. Uh Apple silicon also got a lot

NCCL。Apple 芯片也增加了许多

278.000–279.990  nickel. Uh Apple silicon also got a lot more u functionality with more native

NCCL。Apple 芯片也增加了更多功能，支持更多原生

279.990–280.000  more u functionality with more native

更多功能，支持更多原生

280.000–283.350  more u functionality with more native linear uh algebra operators including ig

更多功能，支持更多原生线性代数算子，包括 eig

283.350–283.360  linear uh algebra operators including ig

线性代数算子，包括 eig

283.360–286.230  linear uh algebra operators including ig uh QR and Cholski alongside uh fast

线性代数算子，包括 eig、QR 和 Cholesky，以及快速

286.230–286.240  uh QR and Cholski alongside uh fast

QR 和 Cholesky，以及快速

286.240–288.870  uh QR and Cholski alongside uh fast production operations and further uh uh

QR 和 Cholesky，以及快速归约运算和更多

288.870–288.880  production operations and further uh uh

归约运算和更多

288.880–291.510  production operations and further uh uh MPS graph to metal uh kernel migrations.

归约运算，以及将更多 MPSGraph 算子迁移为 Metal 内核。

291.510–291.520  MPS graph to metal uh kernel migrations.

将 MPSGraph 算子迁移为 Metal 内核。

291.520–293.830  MPS graph to metal uh kernel migrations. So performance improvements uh in many

将 MPSGraph 算子迁移为 Metal 内核。因此，很多地方的性能都有提升

293.830–293.840  So performance improvements uh in many

因此，很多地方的性能都有提升

293.840–296.870  So performance improvements uh in many places on Apple um torch switch

因此，Apple 平台很多地方的性能都有提升。torch.switch

296.870–296.880  places on Apple um torch switch

Apple 平台的性能提升，以及 torch.switch

296.880–299.430  places on Apple um torch switch generalize uh the torch cond to

Apple 平台的性能提升。torch.switch 将 torch.cond 泛化为

299.430–299.440  generalize uh the torch cond to

把 torch.cond 扩展为

299.440–300.870  generalize uh the torch cond to multi-way branching. So this is like a

把 torch.cond 扩展为多路分支。这就像一种

300.870–300.880  multi-way branching. So this is like a

多路分支。这就像一种

300.880–302.390  multi-way branching. So this is like a language improvement in the way you can

多路分支。这是一种语言层面的改进，让你能

302.390–302.400  language improvement in the way you can

语言层面的改进，让你能

302.400–306.390  language improvement in the way you can think about um uh uh expressing uh

语言层面的改进，让你能思考如何表达

306.390–306.400  think about um uh uh expressing uh

思考如何表达

306.400–308.310  think about um uh uh expressing uh things that used to cause graph breaks

思考如何表达那些过去会导致计算图中断的写法

308.310–308.320  things that used to cause graph breaks

那些过去会导致计算图中断的写法

308.320–311.909  things that used to cause graph breaks in PyTorch. Uh with torch while loop uh

那些过去会导致 PyTorch 计算图中断的写法。借助 torch.while_loop

311.909–311.919  in PyTorch. Uh with torch while loop uh

在 PyTorch 中。借助 torch.while_loop

311.919–313.670  in PyTorch. Uh with torch while loop uh also can now be captured in a CUDA

在 PyTorch 中，torch.while_loop 现在也能被 CUDA 图捕获

313.670–313.680  also can now be captured in a CUDA

现在也能被 CUDA 图捕获

313.680–316.950  also can now be captured in a CUDA graph. Um declarative dynamic shapes uh

现在也能被 CUDA Graph 捕获。声明式动态形状

316.950–316.960  graph. Um declarative dynamic shapes uh

CUDA Graph。声明式动态形状

316.960–319.110  graph. Um declarative dynamic shapes uh are also uh added. This is another sort

CUDA Graph。声明式动态形状也已加入。这是另一种

319.110–319.120  are also uh added. This is another sort

也已加入。这是另一种

319.120–321.110  are also uh added. This is another sort of language feature. There's now a

也已加入。这是另一种语言特性。现在有了

321.110–321.120  of language feature. There's now a

语言特性。现在有了

321.120–324.070  of language feature. There's now a dynamic spec uh uh decorator at dynamic

语言特性。现在有了 @dynamic_spec 装饰器

324.070–324.080  dynamic spec uh uh decorator at dynamic

@dynamic_spec 装饰器

324.080–325.990  dynamic spec uh uh decorator at dynamic spec uh which is shared across torch

@dynamic_spec 装饰器，可供 torch 系列工具共用

325.990–326.000  spec uh which is shared across torch

@dynamic_spec 可供 torch 系列工具共用

326.000–328.790  spec uh which is shared across torch compile torch export and make.fx FX

@dynamic_spec 可供 torch.compile、torch.export 和 make_fx 共用

328.790–328.800  compile torch export and make.fx FX

torch.compile、torch.export 和 make_fx

328.800–330.870  compile torch export and make.fx FX experimental torch compiler uh support

torch.compile、torch.export 和 make_fx。实验性的 torch.compile 支持

330.870–330.880  experimental torch compiler uh support

实验性的 torch.compile 支持

330.880–332.870  experimental torch compiler uh support is also available for complex value

torch.compile 也开始试验性支持复数值

332.870–332.880  is also available for complex value

也支持复数值

332.880–335.830  is also available for complex value tensors. Uh this is an opt-in thing uh

也支持复数值张量。这项功能需主动启用

335.830–335.840  tensors. Uh this is an opt-in thing uh

张量。这项功能需主动启用

335.840–338.310  tensors. Uh this is an opt-in thing uh which decomposes uh supported complex

张量。这项功能需主动启用，会将受支持的复数

338.310–338.320  which decomposes uh supported complex

会将受支持的复数

338.320–339.670  which decomposes uh supported complex operations into real and imaginary

会将受支持的复数运算分解为实部和虚部

339.670–339.680  operations into real and imaginary

运算分解为实部和虚部

339.680–341.830  operations into real and imaginary computations enabling compiler backends

运算分解为实部和虚部的计算，让编译器后端

341.830–341.840  computations enabling compiler backends

计算，让编译器后端

341.840–344.469  computations enabling compiler backends to optimize complex number workloads

计算，让编译器后端优化复数计算任务

344.469–344.479  to optimize complex number workloads

优化复数计算任务

344.479–346.070  to optimize complex number workloads which are super important for science

优化复数计算任务，这对科学研究非常重要

346.070–346.080  which are super important for science

这对科学研究非常重要

346.080–347.749  which are super important for science and I think uh have been hard to do in

这对科学研究非常重要，而且我觉得

347.749–347.759  and I think uh have been hard to do in

而且我觉得

347.759–350.230  and I think uh have been hard to do in PyTorch for a long time. Um broader

而且我觉得这在 PyTorch 里长期都很难做到。更广泛的

350.230–350.240  PyTorch for a long time. Um broader

PyTorch 里长期都很难做到。更广泛的

350.240–352.790  PyTorch for a long time. Um broader platform support uh is you know we also

PyTorch 里长期都很难做到。平台支持方面，我们也

352.790–352.800  platform support uh is you know we also

平台支持方面，我们也

352.800–353.909  platform support uh is you know we also made improvements across a bunch of

平台支持方面，我们也改进了许多

353.909–353.919  made improvements across a bunch of

改进了许多

353.919–355.749  made improvements across a bunch of different platforms including Rockcom.

改进了包括 ROCm 在内的多个平台。

355.749–355.759  different platforms including Rockcom.

包括 Rockcom 在内的不同平台。

355.759–359.350  different platforms including Rockcom. There's now 7.14 wheels uh produced um

包括 Rockcom 在内的不同平台。现在已有 7.14 版 wheel 包。

359.350–359.360  There's now 7.14 wheels uh produced um

现在已有 7.14 版 wheel 包。

359.360–363.830  There's now 7.14 wheels uh produced um from the rock pip uh SDK and Intel

现在已有来自 rock pip SDK 和英特尔的 7.14 版 wheel 包。

363.830–363.840  from the rock pip uh SDK and Intel

来自 rock pip SDK 和英特尔。

363.840–366.230  from the rock pip uh SDK and Intel adding uh native graph capture for the

来自 rock pip SDK 和英特尔，并加入原生图捕获功能。

366.230–366.240  adding uh native graph capture for the

加入原生图捕获功能。

366.240–369.830  adding uh native graph capture for the XPU and inductor uh target inductor

为 XPU 和 Inductor 目标加入原生图捕获功能。

369.830–369.840  XPU and inductor uh target inductor

XPU 和 Inductor 目标，涉及 Inductor。

369.840–372.870  XPU and inductor uh target inductor targeting uh inductor uh functionality

XPU 和 Inductor 目标，重点是 Inductor 功能。

372.870–372.880  targeting uh inductor uh functionality

重点是 Inductor 功能。

372.880–374.790  targeting uh inductor uh functionality targeting the new Ruben platform from

重点是 Inductor 功能，并面向新的 Ruben 平台。

374.790–374.800  targeting the new Ruben platform from

面向新的 Ruben 平台。

374.800–377.830  targeting the new Ruben platform from Nvidia. So that was a mouthful but

面向英伟达新的 Ruben 平台。说起来有点长，不过……

377.830–377.840  Nvidia. So that was a mouthful but

英伟达。说起来有点长，不过……

377.840–379.590  Nvidia. So that was a mouthful but that's the sort of like highlights from

英伟达。说起来有点长，不过这些就是主要亮点。

379.590–379.600  that's the sort of like highlights from

这些就是主要亮点。

379.600–381.590  that's the sort of like highlights from this particular release. Having set that

这些就是此次发布的亮点。说到这里，

381.590–381.600  this particular release. Having set that

此次发布。说到这里，

381.600–384.629  this particular release. Having set that stage let's move to questions. Um so at

此次发布。说到这里，我们来回答问题。首先，

384.629–384.639  stage let's move to questions. Um so at

我们来回答问题。首先，

384.639–386.790  stage let's move to questions. Um so at the first question I think that we might

我们来回答问题。首先，我想第一个问题可能是：

386.790–386.800  the first question I think that we might

我想第一个问题可能是：

386.800–391.909  the first question I think that we might have

我想第一个问题可能会是：

395.430–395.440  will be what CUDA version does PyTorch

PyTorch 会随附哪个 CUDA 版本？

395.440–397.830  will be what CUDA version does PyTorch 2.14 ship

PyTorch 2.14 会随附哪个 CUDA 版本？

397.830–397.840  2.14 ship

2.14 会随附哪个版本？

397.840–400.309  2.14 ship uh and which one do I get by default and

2.14 会随附哪个版本？默认又会得到哪个？

400.309–400.319  uh and which one do I get by default and

默认会得到哪个版本？还有，

400.319–402.629  uh and which one do I get by default and what packaging changes for CUDA do we

默认会得到哪个版本？CUDA 的打包方式又会有哪些变化？

402.629–402.639  what packaging changes for CUDA do we

CUDA 的打包方式会有哪些变化？

402.639–406.094  what packaging changes for CUDA do we expect for the upcoming release? Andre

下次发布时，CUDA 的打包方式会有哪些变化？Andre，

406.094–406.104  expect for the upcoming release? Andre

下次发布时会有哪些变化？Andre，

406.104–406.150  expect for the upcoming release? Andre >> [snorts]

下次发布时会有哪些变化？Andre。（哼笑）

406.150–406.160  >> [snorts]

（哼笑）

406.160–408.710  >> [snorts] >> um yes so basically for the current

（哼笑）嗯，是的，当前……

408.710–408.720  >> um yes so basically for the current

嗯，是的，当前……

408.720–412.550  >> um yes so basically for the current 21.14 release uh we support exactly same

嗯，是的，当前 21.14 版支持与之前完全相同的……

412.550–412.560  21.14 release uh we support exactly same

21.14 版支持与之前完全相同的……

412.560–416.230  21.14 release uh we support exactly same cuda matrix as previous release 21.13 so

21.14 版支持与上一版 21.13 完全相同的 CUDA 版本矩阵，

416.230–416.240  cuda matrix as previous release 21.13 so

CUDA 版本矩阵与上一版 21.13 相同，

416.240–422.309  cuda matrix as previous release 21.13 so it's cuda 12.6 kuda 13.0 and cuda 13.2

CUDA 版本矩阵与上一版 21.13 相同，即 CUDA 12.6、13.0 和 13.2。

422.309–422.319  it's cuda 12.6 kuda 13.0 and cuda 13.2

即 CUDA 12.6、13.0 和 13.2。

422.319–426.469  it's cuda 12.6 kuda 13.0 and cuda 13.2 um cuda 13.0 is being published to pipy

即 CUDA 12.6、13.0 和 13.2。CUDA 13.0 已发布到 PyPI。

426.469–426.479  um cuda 13.0 is being published to pipy

CUDA 13.0 已发布到 PyPI。

426.479–431.029  um cuda 13.0 is being published to pipy um cuda 13.2 two becomes stable. Um it

CUDA 13.0 已发布到 PyPI，CUDA 13.2 则转为稳定版。

431.029–431.039  um cuda 13.2 two becomes stable. Um it

CUDA 13.2 转为稳定版；它在……

431.039–433.510  um cuda 13.2 two becomes stable. Um it for previous release it was prototype

CUDA 13.2 转为稳定版；在上一版中，它还是原型版。

433.510–433.520  for previous release it was prototype

在上一版中，它还是原型版。

433.520–435.350  for previous release it was prototype considered as prototype right now it's

在上一版中，它被视为原型版，现在则是……

435.350–435.360  considered as prototype right now it's

目前仍被视为原型。

435.360–440.710  considered as prototype right now it's stable. Um so um for the next uh also

目前仍被视为原型，但已经稳定。接下来……

440.710–440.720  stable. Um so um for the next uh also

已经稳定。接下来……

440.720–443.270  stable. Um so um for the next uh also for this release we included rocker

已经稳定。这次发布还加入了 ROCm……

443.270–443.280  for this release we included rocker

这次发布加入了 ROCm……

443.280–446.629  for this release we included rocker modifications. So rocom 714 wheels are

这次发布加入了 ROCm 的改动，因此 ROCm 7.1.4 的 wheel 包……

446.629–446.639  modifications. So rocom 714 wheels are

这些改动使 ROCm 7.1.4 的 wheel 包……

446.639–448.150  modifications. So rocom 714 wheels are published

这些改动使 ROCm 7.1.4 的 wheel 包得以发布。

448.150–448.160  published

已发布。

448.160–451.589  published um right now build with uh the rock uh

已发布，目前使用 ROCm……构建。

451.589–451.599  um right now build with uh the rock uh

目前使用 ROCm……构建。

451.599–453.510  um right now build with uh the rock uh pip SDK.

目前使用 ROCm pip SDK 构建。

453.510–453.520  pip SDK.

使用 pip SDK。

453.520–457.029  pip SDK. Um so for the next release 2.15 it's I

使用 pip SDK。至于下一个 2.15 版本，我……

457.029–457.039  Um so for the next release 2.15 it's I

至于下一个 2.15 版本，我……

457.039–458.710  Um so for the next release 2.15 it's I think where the lot of changes are

我认为下一个 2.15 版本会有很多变化。

458.710–458.720  think where the lot of changes are

我认为会有很多变化。

458.720–464.150  think where the lot of changes are coming um first of all 2 um 14 release

我认为会有很多变化。首先，2.14 版本……

464.150–464.160  coming um first of all 2 um 14 release

首先，2.14 版本……

464.160–466.469  coming um first of all 2 um 14 release was the last release that we released

首先，2.14 版本是我们最后一个发布……的版本。

466.469–466.479  was the last release that we released

是我们最后一个发布……的版本。

466.479–470.070  was the last release that we released CUDA 12.x wheels so 12.6 six for the

是我们最后一个发布 CUDA 12.x wheel 包的版本，也就是 CUDA 12.6。

470.070–470.080  CUDA 12.x wheels so 12.6 six for the

CUDA 12.x wheel 包，也就是 CUDA 12.6；接下来……

470.080–472.710  CUDA 12.x wheels so 12.6 six for the 2.15 release. We will uh remove those

CUDA 12.x wheel 包，也就是 CUDA 12.6。到了 2.15 版本，我们会移除这些包。

472.710–472.720  2.15 release. We will uh remove those

到了 2.15 版本，我们会移除这些包。

472.720–475.589  2.15 release. We will uh remove those wheels. There will no longer be shipped

到了 2.15 版本，我们会移除这些 wheel 包，不再随 PyTorch 提供。

475.589–475.599  wheels. There will no longer be shipped

这些 wheel 包将不再提供。

475.599–478.869  wheels. There will no longer be shipped with PyTorch. However, you are able you

这些 wheel 包将不再随 PyTorch 提供。不过，你仍然可以……

478.869–478.879  with PyTorch. However, you are able you

随 PyTorch 提供。不过，你仍然可以……

478.879–481.350  with PyTorch. However, you are able you will still be able to compile

随 PyTorch 提供。不过，你仍然可以自行编译。

481.350–481.360  will still be able to compile

仍然可以自行编译。

481.360–483.510  will still be able to compile with those manually

仍然可以手动用这些版本编译。

483.510–483.520  with those manually

手动用这些版本编译。

483.520–486.550  with those manually um because and we are keeping one CUDA

手动用这些版本编译，因为我们会保留一个 CUDA……

486.550–486.560  um because and we are keeping one CUDA

因为我们会保留一个 CUDA……

486.560–489.510  um because and we are keeping one CUDA 12.6 CI built on our system. So we are

因为我们会在系统中保留一个 CUDA 12.6 的 CI 构建，以便……

489.510–489.520  12.6 CI built on our system. So we are

我们会在系统中保留一个 CUDA 12.6 的 CI 构建，以便……

489.520–492.230  12.6 CI built on our system. So we are ensuring that it's still the case. So

我们会在系统中保留一个 CUDA 12.6 的 CI 构建，确保它仍然可用。

492.230–492.240  ensuring that it's still the case. So

确保它仍然可用。

492.240–495.510  ensuring that it's still the case. So there is no regression in this case. Um

确保它仍然可用，避免出现回归。

495.510–495.520  there is no regression in this case. Um

这方面不会出现回归。

495.520–498.950  there is no regression in this case. Um with CUDA 12.6 six support um uh that

这方面不会出现回归。至于 CUDA 12.6 支持……

498.950–498.960  with CUDA 12.6 six support um uh that

至于 CUDA 12.6 支持……

498.960–501.350  with CUDA 12.6 six support um uh that we're removing we're no longer be

随着我们移除 CUDA 12.6 支持，我们将不再……

501.350–501.360  we're removing we're no longer be

我们将不再……

501.360–503.830  we're removing we're no longer be supporting pre-built wheels for Maxwell,

我们将不再为 Maxwell 提供预编译 wheel 包，

503.830–503.840  supporting pre-built wheels for Maxwell,

不再为 Maxwell 提供预编译 wheel 包，

503.840–507.589  supporting pre-built wheels for Maxwell, Pascal and VA architectures.

不再为 Maxwell、Pascal 和 Volta 架构提供预编译 wheel 包。

507.589–507.599  Pascal and VA architectures.

Pascal 和 Volta 架构。

507.599–511.990  Pascal and VA architectures. Um so the next part uh it's CUDA 13.2

Pascal 和 Volta 架构。接下来是 CUDA 13.2。

511.990–512.000  Um so the next part uh it's CUDA 13.2

嗯，接下来是 CUDA 13.2。

512.000–514.469  Um so the next part uh it's CUDA 13.2 will be published as default CUDA

接下来，CUDA 13.2 将作为默认 CUDA 版本发布。

514.469–514.479  will be published as default CUDA

将作为默认 CUDA 版本发布。

514.479–518.230  will be published as default CUDA version on PIP and we will introduce

将在 pip 上作为默认 CUDA 版本发布，我们还会推出……

518.230–518.240  version on PIP and we will introduce

pip 上的版本，我们还会推出……

518.240–522.949  version on PIP and we will introduce CUDA 13.4 four as a prototype uh CUDA

pip 上的版本；我们还会推出 CUDA 13.4 的原型版本。

522.949–522.959  CUDA 13.4 four as a prototype uh CUDA

CUDA 13.4 的原型版本。

522.959–525.269  CUDA 13.4 four as a prototype uh CUDA version with support for ver Rubin

CUDA 13.4 的原型版本，将支持 Vera Rubin 架构。

525.269–525.279  version with support for ver Rubin

支持 Vera Rubin 架构的版本。

525.279–527.990  version with support for ver Rubin architecture. Um I think this covers

支持 Vera Rubin 架构的版本。嗯，我想这涵盖了……

527.990–528.000  architecture. Um I think this covers

架构。嗯，我想这涵盖了……

528.000–530.230  architecture. Um I think this covers pretty much it in terms of cuda and a

架构。嗯，我想 CUDA 方面基本就这些了，还有……

530.230–530.240  pretty much it in terms of cuda and a

CUDA 方面基本就这些了，还有……

530.240–532.230  pretty much it in terms of cuda and a little bit of rocom. Please

CUDA 方面基本就这些了，ROCm 也讲了一点。请……

532.230–532.240  little bit of rocom. Please

ROCm 也讲了一点。请……

532.240–534.550  little bit of rocom. Please >> if I can make a small comment there's a

ROCm 也讲了一点。请……我补充一句，数字和版本很多。

534.550–534.560  >> if I can make a small comment there's a

我补充一句，数字和版本很多。

534.560–537.269  >> if I can make a small comment there's a like a lot of numbers lot of versions

我补充一句，数字和版本确实很多。

537.269–537.279  like a lot of numbers lot of versions

数字和版本确实很多。

537.279–540.310  like a lot of numbers lot of versions please track the pytor issues and dev

数字和版本确实很多，请关注 PyTorch 的议题和 dev-discuss。

540.310–540.320  please track the pytor issues and dev

请关注 PyTorch 的议题和 dev-discuss。

540.320–542.150  please track the pytor issues and dev discuss where we publish all those

请关注 PyTorch 的议题和 dev-discuss，我们会在那里公布各版本。

542.150–542.160  discuss where we publish all those

我们会在那里公布各版本。

542.160–544.150  discuss where we publish all those versions because it's they hard to

我们会在那里公布各版本，因为这些版本很难……

544.150–544.160  versions because it's they hard to

这些版本很难……

544.160–545.990  versions because it's they hard to remember and sometimes we have to pivot

这些版本很难记，而且有时我们得调整计划。

545.990–546.000  remember and sometimes we have to pivot

很难记，而且有时我们得调整计划。

546.000–547.509  remember and sometimes we have to pivot depending on some new hardware

很难记，而且有时我们得根据新硬件……

547.509–547.519  depending on some new hardware

得根据新硬件……

547.519–549.670  depending on some new hardware capabilities or whatever but that's the

得根据新硬件的能力等情况调整，但这就是……

549.670–549.680  capabilities or whatever but that's the

硬件能力等情况，不过这就是……

549.680–552.150  capabilities or whatever but that's the current plans. Thank you.

硬件能力等情况，不过这就是目前的计划。谢谢。

552.150–552.160  current plans. Thank you.

目前的计划。谢谢。

552.160–553.990  current plans. Thank you. Yeah, and just to add, there's a release

目前的计划。谢谢。对了，补充一下，有一份发布文件……

553.990–554.000  Yeah, and just to add, there's a release

对了，补充一下，有一份发布文件……

554.000–556.550  Yeah, and just to add, there's a release MD file in the PyTorch PyTorch where the

对了，补充一下，PyTorch 仓库里有一份 release.md 文件。

556.550–556.560  MD file in the PyTorch PyTorch where the

PyTorch 仓库里的 release.md 文件……

556.560–558.550  MD file in the PyTorch PyTorch where the current matrix is published and yeah,

PyTorch 仓库里的 release.md 文件公布了当前的版本矩阵。

558.550–558.560  current matrix is published and yeah,

那里公布了当前的版本矩阵。

558.560–560.230  current matrix is published and yeah, you can go ahead and take a look there

那里公布了当前的版本矩阵，大家可以去看看。

560.230–560.240  you can go ahead and take a look there

大家可以去那里看看。

560.240–565.509  you can go ahead and take a look there and ask any questions.

大家可以去那里看看，有问题也可以问。

567.990–568.000  >> Okay. Um, do we have a next question

好的。嗯，下一个问题……

568.000–571.811  >> Okay. Um, do we have a next question queued up?

好的。嗯，还有下一个问题吗？

572.710–572.720  [snorts]

[哼笑]

572.720–575.190  [snorts] Okay. Um, thank you, uh, Michael for

[哼笑] 好的，感谢 Michael 提问。

575.190–575.200  Okay. Um, thank you, uh, Michael for

好的，感谢 Michael 提问。

575.200–577.110  Okay. Um, thank you, uh, Michael for asking this question. With Agentic RL

好的，感谢 Michael 提问。随着 Agentic RL……

577.110–577.120  asking this question. With Agentic RL

感谢提问。随着 Agentic RL……

577.120–579.350  asking this question. With Agentic RL becoming a bigger part of training, um

感谢提问。随着 Agentic RL 在训练中越来越重要，嗯……

579.350–579.360  becoming a bigger part of training, um

正逐渐成为训练中更重要的一部分，嗯。

579.360–581.110  becoming a bigger part of training, um are you all seeing these workloads

正逐渐成为训练中更重要的一部分。嗯，你们是否发现这些工作负载

581.110–581.120  are you all seeing these workloads

你们是否发现这些工作负载

581.120–582.070  are you all seeing these workloads create different infrastructure

你们是否发现这些工作负载带来了不同的基础设施

582.070–582.080  create different infrastructure

带来了不同的基础设施

582.080–583.350  create different infrastructure challenges than traditional model

带来了与传统模型训练不同的基础设施挑战

583.350–583.360  challenges than traditional model

与传统模型训练不同的挑战

583.360–585.829  challenges than traditional model training? Anything PyTorch needs still

与传统模型训练不同的挑战？PyTorch 在这方面还有什么

585.829–585.839  training? Anything PyTorch needs still

训练？PyTorch 在这方面还有什么

585.839–587.430  training? Anything PyTorch needs still needs to get better at there? I'm

训练？PyTorch 在这方面还有什么需要改进的？我

587.430–587.440  needs to get better at there? I'm

需要改进的？我

587.440–589.269  needs to get better at there? I'm actually going to um direct this one to

需要改进的？我想先把这个问题抛给

589.269–589.279  actually going to um direct this one to

我想先把这个问题抛给

589.279–591.509  actually going to um direct this one to Joe first and then Andre and Nikita, you

我想先请 Joe 回答，然后 Andre 和 Nikita，你们

591.509–591.519  Joe first and then Andre and Nikita, you

先请 Joe 回答，然后 Andre 和 Nikita，你们

591.519–593.910  Joe first and then Andre and Nikita, you guys can jump in after. Um Joe, I know

Joe 先来，Andre 和 Nikita 随后补充。嗯，Joe，我知道

593.910–593.920  guys can jump in after. Um Joe, I know

你们随后补充。嗯，Joe，我知道

593.920–596.870  guys can jump in after. Um Joe, I know you care a lot about agentic RL. So

你们随后补充。Joe，我知道你很关注智能体强化学习。所以

596.870–596.880  you care a lot about agentic RL. So

你很关注智能体强化学习。所以

596.880–598.150  you care a lot about agentic RL. So >> yeah,

你很关注智能体强化学习。所以。对。

598.150–598.160  >> yeah,

对。

598.160–600.550  >> yeah, >> I mean I think um you know if you look

对。我觉得，你看

600.550–600.560  >> I mean I think um you know if you look

我觉得，你看

600.560–602.710  >> I mean I think um you know if you look at it's it's pretty interesting the the

我觉得，你看，这很有意思，现在的

602.710–602.720  at it's it's pretty interesting the the

这很有意思，现在的

602.720–604.150  at it's it's pretty interesting the the models coming out these days are coming

这很有意思，现在的模型推出得

604.150–604.160  models coming out these days are coming

现在的模型推出得

604.160–606.550  models coming out these days are coming out fast and furious and I think there

现在的模型推出得又快又多，我觉得

606.550–606.560  out fast and furious and I think there

又快又多，我觉得

606.560–608.310  out fast and furious and I think there there was at least a for a while a

又快又多，我觉得至少有一段时间

608.310–608.320  there was at least a for a while a

至少有一段时间

608.320–609.509  there was at least a for a while a perception that people are are

至少有一段时间，大家觉得人们在

609.509–609.519  perception that people are are

大家觉得人们在

609.519–611.509  perception that people are are pre-training models at an increased pace

大家觉得人们在加快模型预训练的速度

611.509–611.519  pre-training models at an increased pace

加快模型预训练的速度

611.519–612.949  pre-training models at an increased pace right but I think a lot of the gains

加快模型预训练的速度，对吧？但我觉得很多进展

612.949–612.959  right but I think a lot of the gains

对吧？但我觉得很多进展

612.959–614.710  right but I think a lot of the gains that that we've seen actually you know

对吧？但我觉得我们看到的很多进展其实

614.710–614.720  that that we've seen actually you know

我们看到的很多进展其实

614.720–616.630  that that we've seen actually you know some of the Chinese models for example

我们看到的很多进展，其实比如一些中国模型

616.630–616.640  some of the Chinese models for example

比如一些中国模型

616.640–618.470  some of the Chinese models for example that are coming out and and versioned

比如一些中国模型不断推出、快速迭代

618.470–618.480  that are coming out and and versioned

不断推出、快速迭代

618.480–620.550  that are coming out and and versioned really fast and the Quen models a lot of

这些模型迭代得很快，Qwen 模型的很多进展

620.550–620.560  really fast and the Quen models a lot of

迭代得很快，Qwen 模型的很多进展

620.560–623.269  really fast and the Quen models a lot of it is actually due to RL uh and you know

其实得益于 RL，而且

623.269–623.279  it is actually due to RL uh and you know

其实得益于 RL，而且

623.279–624.710  it is actually due to RL uh and you know I think one of the things that we've

其实得益于 RL，而且我觉得我们

624.710–624.720  I think one of the things that we've

我觉得我们

624.720–626.790  I think one of the things that we've learned over the is you know if you

我觉得我们逐渐认识到，你知道，如果你

626.790–626.800  learned over the is you know if you

学到的是，如果你……

626.800–629.509  learned over the is you know if you build a really strong pre-trained base

学到的是，如果你构建了强大的预训练基座

629.509–629.519  build a really strong pre-trained base

构建强大的预训练基座

629.519–632.310  build a really strong pre-trained base um to build on you can actually RL over

有了强大的预训练基座，就能在此基础上做强化学习

632.310–632.320  um to build on you can actually RL over

在这个基础上，你就能做强化学习

632.320–633.509  um to build on you can actually RL over it and you can actually throw more

在这个基础上，你就能做强化学习，还能投入更多

633.509–633.519  it and you can actually throw more

还能在上面投入更多

633.519–635.350  it and you can actually throw more compute on it and I think the deepseek

还能投入更多算力。我认为 DeepSeek

635.350–635.360  compute on it and I think the deepseek

投入更多算力。我认为 DeepSeek

635.360–637.110  compute on it and I think the deepseek paper actually taught us a lot of

投入更多算力。我认为 DeepSeek 的论文

637.110–637.120  paper actually taught us a lot of

这篇论文确实让我们学到了很多

637.120–639.110  paper actually taught us a lot of lessons that kind of confirms for the

这篇论文确实让我们学到了很多，也印证了

639.110–639.120  lessons that kind of confirms for the

这些经验也印证了

639.120–641.990  lessons that kind of confirms for the the bitter lesson pill people among us

这些经验也印证了我们这些信奉“苦涩教训”的人

641.990–642.000  the bitter lesson pill people among us

我们这些信奉“苦涩教训”的人

642.000–643.430  the bitter lesson pill people among us including myself that you know if you

我们这些信奉“苦涩教训”的人，包括我自己，都认为如果你

643.430–643.440  including myself that you know if you

包括我自己，都认为如果你

643.440–646.310  including myself that you know if you continue to throw more compute at stable

包括我自己，都认为如果你继续给稳定的强化学习投入更多算力

646.310–646.320  continue to throw more compute at stable

继续给稳定的强化学习投入更多算力

646.320–648.550  continue to throw more compute at stable stable RL using that word stable very

继续给稳定的强化学习投入更多算力。我说“稳定”这个词时

648.550–648.560  stable RL using that word stable very

说“稳定的强化学习”时，我对“稳定”这个词

648.560–651.590  stable RL using that word stable very very uh carefully purposefully uh then

说“稳定的强化学习”时，我非常谨慎、刻意地用了“稳定”这个词。那么

651.590–651.600  very uh carefully purposefully uh then

我是非常谨慎、刻意地这么说的。那么

651.600–653.350  very uh carefully purposefully uh then you can actually extend capabilities of

我是非常谨慎、刻意地这么说的。那么你确实能持续提升

653.350–653.360  you can actually extend capabilities of

你确实能持续提升

653.360–656.550  you can actually extend capabilities of models over time. And so I think like

你确实能不断提升模型的能力。所以我觉得

656.550–656.560  models over time. And so I think like

模型的能力能不断提升。所以我觉得

656.560–659.350  models over time. And so I think like you know uh in the earlier days I guess

模型的能力能不断提升。所以我觉得，早些时候

659.350–659.360  you know uh in the earlier days I guess

我想，在早些时候

659.360–661.670  you know uh in the earlier days I guess of RL um and and we started this project

我想，在强化学习发展的早期，我们启动了一个项目

661.670–661.680  of RL um and and we started this project

在强化学习发展的早期，我们启动了一个项目

661.680–664.150  of RL um and and we started this project called OpenM last year when I was

在强化学习发展的早期，我们启动了一个叫 OpenM 的项目。那是去年

664.150–664.160  called OpenM last year when I was

叫 OpenM 的项目。那是去年

664.160–666.230  called OpenM last year when I was actually almost one year ago exactly uh

叫 OpenM 的项目。准确说，差不多正好是一年前

666.230–666.240  actually almost one year ago exactly uh

准确说，差不多正好是一年前

666.240–668.710  actually almost one year ago exactly uh at the PyTorch conference we uh launched

准确说，差不多正好是一年前，我们在 PyTorch 大会上发布了它

668.710–668.720  at the PyTorch conference we uh launched

我们在 PyTorch 大会上发布了它

668.720–669.990  at the PyTorch conference we uh launched that project and it continues to grow

我们在 PyTorch 大会上发布了这个项目，它还在持续发展

669.990–670.000  that project and it continues to grow

这个项目还在持续发展

670.000–672.069  that project and it continues to grow and I think one the whole goal of it was

这个项目还在持续发展。我想，它的总体目标是

672.069–672.079  and I think one the whole goal of it was

我想，它的总体目标是

672.079–673.910  and I think one the whole goal of it was to democratize the use of reinforcement

我想，它的总体目标是让强化学习的使用更加普及

673.910–673.920  to democratize the use of reinforcement

让强化学习的使用更加普及

673.920–676.389  to democratize the use of reinforcement learning um because it's it's a lot

让强化学习的使用更加普及，因为现在

676.389–676.399  learning um because it's it's a lot

因为现在做这件事

676.399–678.389  learning um because it's it's a lot easier to kind of create an environment

因为现在创建一个环境容易多了

678.389–678.399  easier to kind of create an environment

创建一个环境容易多了

678.399–680.550  easier to kind of create an environment create your tasks um and then be able to

创建一个环境、设计自己的任务都容易多了，然后你就能

680.550–680.560  create your tasks um and then be able to

设计自己的任务，然后你就能

680.560–682.870  create your tasks um and then be able to to set up you know especially today with

设计自己的任务，然后就能完成配置，尤其是现在有了

682.870–682.880  to set up you know especially today with

搭建起来，尤其是如今有了

682.880–685.110  to set up you know especially today with all the tools available Nemo RL

尤其是如今有了 NeMo RL 等各种工具，搭建起来更容易

685.110–685.120  all the tools available Nemo RL

有了 NeMo RL 等各种工具

685.120–687.670  all the tools available Nemo RL TRL unsloth, you can actually uh use

有了 NeMo RL、TRL、Unsloth 等工具，你就能用

687.670–687.680  TRL unsloth, you can actually uh use

有了 TRL 和 Unsloth，你就能用

687.680–690.710  TRL unsloth, you can actually uh use reinforcement learning um you know way

有了 TRL 和 Unsloth，你就能用强化学习，方式也

690.710–690.720  reinforcement learning um you know way

强化学习也变得

690.720–691.990  reinforcement learning um you know way easier than you could probably like five

强化学习比五年前容易得多

691.990–692.000  easier than you could probably like five

比五年前容易得多

692.000–694.069  easier than you could probably like five or 10 years ago in the earlier days um

比五年或十年前的早期容易得多

694.069–694.079  or 10 years ago in the earlier days um

比十年前的早期容易得多

694.079–696.150  or 10 years ago in the earlier days um and actually add capabilities to your

比十年前的早期容易得多，还能给模型增加能力

696.150–696.160  and actually add capabilities to your

还能给模型增加能力

696.160–698.150  and actually add capabilities to your models bas basically be able to postrain

还能给模型增加能力，基本上就是给模型做后训练

698.150–698.160  models bas basically be able to postrain

基本上就是给模型做后训练

698.160–699.750  models bas basically be able to postrain using reinforcement learning for openw

基本上就是用强化学习对开放权重模型做后训练

699.750–699.760  using reinforcement learning for openw

用强化学习处理开放权重模型

699.760–701.190  using reinforcement learning for openw weight models and I think that's

用强化学习处理开放权重模型，我认为这

701.190–701.200  weight models and I think that's

开放权重模型，我认为这

701.200–702.150  weight models and I think that's something we want to continue to

开放权重模型，我认为这也是我们想继续

702.150–702.160  something we want to continue to

这也是我们想继续做的事

702.160–704.630  something we want to continue to obviously de democratize and and enable

我们显然想继续普及这项技术，让更多开发者用上

704.630–704.640  obviously de democratize and and enable

普及这项技术，让更多开发者用上

704.640–706.630  obviously de democratize and and enable with developers um and so like Daniel

让更多开发者用上。比如 Daniel

706.630–706.640  with developers um and so like Daniel

开发者也能用上。比如 Daniel

706.640–708.630  with developers um and so like Daniel from Unsloth continues to bear that that

Unsloth 的 Daniel 一直在为这件事努力

708.630–708.640  from Unsloth continues to bear that that

Unsloth 的 Daniel 一直在

708.640–710.949  from Unsloth continues to bear that that flag for many uh and obviously we

Unsloth 的 Daniel 一直在为许多人扛起这面旗帜，我们当然也

710.949–710.959  flag for many uh and obviously we

为许多人扛起这面旗帜，我们当然也

710.959–712.630  flag for many uh and obviously we contribute uh significantly to open

为许多人扛起这面旗帜，我们也为开源项目做出了重要贡献

712.630–712.640  contribute uh significantly to open

我们也为开源项目做出了重要贡献

712.640–714.470  contribute uh significantly to open project and that fits nicely into the

我们为开源项目做出了重要贡献，这也很好地融入了

714.470–714.480  project and that fits nicely into the

这些项目也很好地融入了

714.480–716.790  project and that fits nicely into the PyTorch ecosystem and extends it as

这些项目也很好地融入了 PyTorch 生态，并扩展了它

716.790–716.800  PyTorch ecosystem and extends it as

PyTorch 生态也因此得到扩展

716.800–718.870  PyTorch ecosystem and extends it as people use PyTorch and other tools to uh

随着人们使用 PyTorch 和其他工具，它也得到扩展

718.870–718.880  people use PyTorch and other tools to uh

人们用 PyTorch 和其他工具

718.880–721.190  people use PyTorch and other tools to uh to to to customize their models for

人们用 PyTorch 和其他工具，根据任务定制模型

721.190–721.200  to to to customize their models for

根据任务定制模型

721.200–723.829  to to to customize their models for their tasks.

根据各自的任务定制模型。

723.829–723.839  their tasks.

各自的任务。

723.839–725.829  their tasks. >> But fundamentally, PyTorch has a lot of

各自的任务。说到底，PyTorch 已经具备很多

725.829–725.839  >> But fundamentally, PyTorch has a lot of

说到底，PyTorch 已经具备很多

725.839–728.389  >> But fundamentally, PyTorch has a lot of the things that that folks need. Um we

说到底，PyTorch 已经具备大家需要的很多东西。我们

728.389–728.399  the things that that folks need. Um we

大家需要的很多东西。我们

728.399–729.670  the things that that folks need. Um we need these sort of layered packages like

大家需要的很多东西。我们还需要这类分层软件包，比如

729.670–729.680  need these sort of layered packages like

我们还需要这类分层软件包，比如

729.680–731.590  need these sort of layered packages like OpenM but but PyTorch itself. Are there

我们还需要 OpenM 这样的分层软件包，但 PyTorch 本身呢？

731.590–731.600  OpenM but but PyTorch itself. Are there

OpenM 这样的软件包之外，PyTorch 本身呢？

731.600–734.150  OpenM but but PyTorch itself. Are there any changes that you see uh you see

OpenM 这样的软件包之外，PyTorch 本身还有哪些变化是你预见到的？

734.150–734.160  any changes that you see uh you see

你觉得，呃，你觉得有哪些改动

734.160–736.710  any changes that you see uh you see needed now or in the future in PyTorch?

你觉得 PyTorch 现在或将来需要做哪些改动？

736.710–736.720  needed now or in the future in PyTorch?

PyTorch 现在或将来需要做哪些改动？

736.720–738.949  needed now or in the future in PyTorch? >> Well, I think

PyTorch 现在或将来需要做哪些改动？>>嗯，我觉得

738.949–738.959  >> Well, I think

>>嗯，我觉得

738.959–740.150  >> Well, I think Oh, sorry. Go for it.

>>嗯，我觉得……哦，抱歉，你先说。

740.150–740.160  Oh, sorry. Go for it.

哦，抱歉，你先说。

740.160–741.750  Oh, sorry. Go for it. >> No, no, go ahead. I want to hear your

哦，抱歉，你先说。>>不不，你先说，我想听听你的

741.750–741.760  >> No, no, go ahead. I want to hear your

>>不不，你先说，我想听听你的

741.760–743.430  >> No, no, go ahead. I want to hear your point of view. on the infrastructure

>>不不，你先说，我想听听你的看法。基础设施方面

743.430–743.440  point of view. on the infrastructure

看法。基础设施方面

743.440–746.150  point of view. on the infrastructure side like uh the nickel 2 and like the

看法。基础设施方面，比如 NCCL 2，还有

746.150–746.160  side like uh the nickel 2 and like the

基础设施方面，比如 NCCL 2，还有

746.160–747.829  side like uh the nickel 2 and like the safer nickel is one of the things that

基础设施方面，比如 NCCL 2，还有更安全的 NCCL，这是

747.829–747.839  safer nickel is one of the things that

更安全的 NCCL，这是

747.839–750.230  safer nickel is one of the things that is constantly painful for post training

更安全的 NCCL 一直是后训练中让人头疼的问题

750.230–750.240  is constantly painful for post training

一直是后训练中让人头疼的问题

750.240–752.310  is constantly painful for post training workflows when you have to restart. So

后训练流程一旦需要重启，就很麻烦。所以

752.310–752.320  workflows when you have to restart. So

流程一旦需要重启，就很麻烦。所以

752.320–753.829  workflows when you have to restart. So that would be an infrastructure part but

流程一旦需要重启，就很麻烦。所以这属于基础设施层面，但

753.829–753.839  that would be an infrastructure part but

这属于基础设施层面，但

753.839–755.590  that would be an infrastructure part but also we have as Joel Joe said a lot of

这属于基础设施层面，但正如 Joel Joe 所说，我们也有很多

755.590–755.600  also we have as Joel Joe said a lot of

正如 Joel Joe 所说，我们也有很多

755.600–758.069  also we have as Joel Joe said a lot of great partnership. I encourage people to

正如 Joel Joe 所说，我们也有很多很棒的合作。我建议大家

758.069–758.079  great partnership. I encourage people to

很棒的合作。我建议大家

758.079–761.110  great partnership. I encourage people to check out well u maybe not encourage but

很棒的合作。我建议大家看看……嗯，也不算建议吧，但

761.110–761.120  check out well u maybe not encourage but

去看看……嗯，也不算建议吧，但

761.120–763.030  check out well u maybe not encourage but there are two frameworks that exist in

去看看……嗯，也不算建议吧，但有两个框架

763.030–763.040  there are two frameworks that exist in

有两个框架

763.040–764.629  there are two frameworks that exist in the pytor side. There is a torch titan

PyTorch 这边有两个框架，其中一个是 TorchTitan

764.629–764.639  the pytor side. There is a torch titan

PyTorch 这边有 TorchTitan

764.639–767.269  the pytor side. There is a torch titan and torch forge that kind of give you

PyTorch 这边有 TorchTitan 和 TorchForge，它们提供

767.269–767.279  and torch forge that kind of give you

TorchForge 也会提供

767.279–769.190  and torch forge that kind of give you some recipes on the post training and I

TorchForge 也会提供一些后训练方案，我

769.190–769.200  some recipes on the post training and I

一些后训练方案，我

769.200–770.550  some recipes on the post training and I think you should kind of direct your

一些后训练方案，我觉得你应该把

770.550–770.560  think you should kind of direct your

我觉得你应该把

770.560–772.230  think you should kind of direct your attention more towards those

我觉得你应该多关注这些

772.230–772.240  attention more towards those

多关注这些

772.240–773.750  attention more towards those repositories when you're asking a

多关注这些代码仓库，尤其是当你问

773.750–773.760  repositories when you're asking a

这些代码仓库，尤其是当你问

773.760–775.350  repositories when you're asking a specific questions about the frameworks

这些代码仓库，尤其是当你问框架的具体问题时

775.350–775.360  specific questions about the frameworks

框架的具体问题时

775.360–777.910  specific questions about the frameworks rather about the like harness rather

框架的具体问题时，更像是在问配套工具，而

777.910–777.920  rather about the like harness rather

更像是在问配套工具，而

777.920–780.629  rather about the like harness rather than like a training framework itself

更像是在问配套工具，而非训练框架本身

780.629–780.639  than like a training framework itself

非训练框架本身

780.639–782.550  than like a training framework itself but support for MFP4 is also very

非训练框架本身，不过对 MFP4 的支持也很

782.550–782.560  but support for MFP4 is also very

不过对 MFP4 的支持也很

782.560–785.110  but support for MFP4 is also very important for post training pipelines so

不过对 MFP4 的支持对后训练流程也很重要，所以

785.110–785.120  important for post training pipelines so

这对后训练流程很重要，所以

785.120–786.470  important for post training pipelines so like all the infrastructures that is

这对后训练流程很重要，所以所需的基础设施

786.470–786.480  like all the infrastructures that is

所需的基础设施

786.480–789.269  like all the infrastructures that is needed is there but Joe to continue

所需的基础设施都已到位。不过 Joe，请接着说。

789.269–789.279  needed is there but Joe to continue

都已到位。不过 Joe，请接着说。

789.279–790.949  needed is there but Joe to continue >> no I I think you're spot on I think

都已到位。不过 Joe，请接着说。>> 不，我觉得你说得很对。我觉得

790.949–790.959  >> no I I think you're spot on I think

>> 不，我觉得你说得很对。我觉得

790.959–793.110  >> no I I think you're spot on I think there you know infra I mean we're we're

>> 不，我觉得你说得很对。我觉得，基础设施这块，我们

793.110–793.120  there you know infra I mean we're we're

基础设施这块，我们

793.120–795.910  there you know infra I mean we're we're using a pretty large large amount of

基础设施这块，我们用了相当多的

795.910–795.920  using a pretty large large amount of

用了相当多的

795.920–798.069  using a pretty large large amount of compute to do RL here in reflection. We

我们在 Reflection 用了大量算力做 RL。我们

798.069–798.079  compute to do RL here in reflection. We

用算力在 Reflection 做 RL。我们

798.079–800.230  compute to do RL here in reflection. We obviously did a lot of RL meta. Um I

用算力在 Reflection 做 RL。我们在 Meta 显然也做了大量 RL。嗯，我

800.230–800.240  obviously did a lot of RL meta. Um I

我们在 Meta 显然也做了大量 RL。嗯，我

800.240–801.990  obviously did a lot of RL meta. Um I think infrastructure is the challenge. I

我们在 Meta 显然也做了大量 RL。我觉得基础设施是个挑战。我

801.990–802.000  think infrastructure is the challenge. I

觉得基础设施是个挑战。我

802.000–803.190  think infrastructure is the challenge. I think there there's a lot of work I

觉得基础设施是个挑战。我觉得还有很多工作要做，我

803.190–803.200  think there there's a lot of work I

觉得还有很多工作要做，我

803.200–806.069  think there there's a lot of work I think to um you know to to scale it up

觉得还有很多工作要做，才能把规模扩大

806.069–806.079  think to um you know to to scale it up

才能把规模扩大

806.079–807.509  think to um you know to to scale it up to make it stable. I think I'm just

才能把规模扩大并保持稳定。我只是

807.509–807.519  to make it stable. I think I'm just

并保持稳定。我只是

807.519–809.670  to make it stable. I think I'm just looking at even like I'm I can't really

并保持稳定。我只是想，就连我现在也没法

809.670–809.680  looking at even like I'm I can't really

就连我现在也没法

809.680–810.710  looking at even like I'm I can't really use examples of what we're doing

就连我现在也没法拿我们正在做的事举例

810.710–810.720  use examples of what we're doing

拿我们正在做的事举例

810.720–811.829  use examples of what we're doing internally yet because we're still

拿我们内部正在做的事举例，因为我们还

811.829–811.839  internally yet because we're still

因为我们还

811.839–814.069  internally yet because we're still pretty salty. But like even when we were

因为我们还挺保密的。但即使当时我们

814.069–814.079  pretty salty. But like even when we were

还挺保密的。但即使当时我们

814.079–816.150  pretty salty. But like even when we were uh dealing with like fault tolerance in

还挺保密的。但即使当时我们在处理容错问题

816.150–816.160  uh dealing with like fault tolerance in

在处理容错问题

816.160–817.990  uh dealing with like fault tolerance in like the llama days, you know, most of

在 Llama 那段时期处理容错问题时，大部分

817.990–818.000  like the llama days, you know, most of

在 Llama 那段时期，大部分

818.000–819.750  like the llama days, you know, most of our uh the faults that we found even in

在 Llama 那段时期，我们发现的大部分故障，甚至

819.750–819.760  our uh the faults that we found even in

我们发现的大部分故障，甚至

819.760–821.509  our uh the faults that we found even in pre-training or even in post- training

我们发现的大部分故障，即使在预训练或后训练阶段

821.509–821.519  pre-training or even in post- training

在预训练甚至后训练阶段

821.519–824.470  pre-training or even in post- training came from obviously GPUs failing over.

在预训练甚至后训练阶段，显然都来自 GPU 故障。

824.470–824.480  came from obviously GPUs failing over.

显然都来自 GPU 故障。

824.480–827.670  came from obviously GPUs failing over. um you know we had uh yeah we I think

显然都来自 GPU 故障。嗯，我们当时……对，我想

827.670–827.680  um you know we had uh yeah we I think

嗯，我们当时……对，我想

827.680–829.590  um you know we had uh yeah we I think what something like 30% of of those

嗯，我们当时……对，我想，大约 30% 的

829.590–829.600  what something like 30% of of those

大约 30% 的

829.600–832.150  what something like 30% of of those failures or we had close to like 400 and

大约 30% 的故障，或者说，我们有 400 多块

832.150–832.160  failures or we had close to like 400 and

故障，或者说，我们有 400 多块

832.160–834.629  failures or we had close to like 400 and something GPUs out of the 16,000 GPUs we

故障，或者说，训练用的 16,000 块 GPU 中有 400 多块

834.629–834.639  something GPUs out of the 16,000 GPUs we

训练用的 16,000 块 GPU 中有 400 多块

834.639–836.550  something GPUs out of the 16,000 GPUs we were training on for the large model

用于训练大模型的 16,000 块 GPU 中有 400 多块

836.550–836.560  were training on for the large model

用于训练大模型的那些任务

836.560–837.910  were training on for the large model were failing. We wrote that into the

用于训练大模型的那些任务失败了。我们把这点写进了

837.910–837.920  were failing. We wrote that into the

失败了。我们把这点写进了

837.920–839.189  were failing. We wrote that into the paper. You can kind of look at the graph

失败了。我们把这点写进了论文。看看图就能发现

839.189–839.199  paper. You can kind of look at the graph

论文。看看图就能发现

839.199–841.269  paper. You can kind of look at the graph there. I think like infrastructure

论文。看看图就能发现。我觉得基础设施

841.269–841.279  there. I think like infrastructure

我觉得基础设施

841.279–843.990  there. I think like infrastructure stability and um and fall tolerance and

我觉得基础设施的稳定性和容错能力

843.990–844.000  stability and um and fall tolerance and

稳定性和容错能力

844.000–845.269  stability and um and fall tolerance and I think we'll get to that in some of the

稳定性和容错能力，我想后面的问题会谈到

845.269–845.279  I think we'll get to that in some of the

我想后面的问题会谈到

845.279–847.269  I think we'll get to that in some of the other questions but that's probably one

我想后面的问题会谈到，但这可能是

847.269–847.279  other questions but that's probably one

后面的问题会谈到，但这可能是

847.279–848.550  other questions but that's probably one of the biggest biggest things and I

后面的问题会谈到，但这可能是最大的难题之一。我

848.550–848.560  of the biggest biggest things and I

最大的难题之一。我

848.560–849.750  of the biggest biggest things and I think that's one of the reasons why like

最大的难题之一。我觉得这也是

849.750–849.760  think that's one of the reasons why like

我觉得这也是

849.760–851.509  think that's one of the reasons why like Tinker is actually like pretty

我觉得这也是 Tinker 挺有意思的原因之一，

851.509–851.519  Tinker is actually like pretty

Tinker 其实挺有意思，

851.519–853.189  Tinker is actually like pretty interesting from from thinking machines.

Thinking Machines 推出的 Tinker 其实挺有意思。

853.189–853.199  interesting from from thinking machines.

Thinking Machines 推出的产品挺有意思。

853.199–855.110  interesting from from thinking machines. It's like a, you know, not not to to

Thinking Machines 推出的产品挺有意思。我不是想

855.110–855.120  It's like a, you know, not not to to

我不是想

855.120–857.509  It's like a, you know, not not to to plug the his product here, but like it

我不是想在这里给他的产品打广告，但它

857.509–857.519  plug the his product here, but like it

在这里给他的产品打广告，但它

857.519–859.910  plug the his product here, but like it it actually does um it's actually a

在这里给他的产品打广告，但它确实做到了，它是个

859.910–859.920  it actually does um it's actually a

它确实做到了，它是个

859.920–862.069  it actually does um it's actually a really uh cool API that actually

它确实做到了，它是个很棒的 API，能

862.069–862.079  really uh cool API that actually

很棒的 API，能

862.079–863.910  really uh cool API that actually abstracts that away. And I think that's

很棒的 API，能把那些复杂问题封装起来。我觉得

863.910–863.920  abstracts that away. And I think that's

把那些复杂问题封装起来。我觉得

863.920–865.590  abstracts that away. And I think that's for those especially the academics I've

把那些复杂问题封装起来。我觉得，对我接触过的学者来说，

865.590–865.600  for those especially the academics I've

对我接触过的学者来说，尤其如此。

865.600–867.670  for those especially the academics I've talked to and researchers like that is

对我接触过的学者和研究人员来说，这

867.670–867.680  talked to and researchers like that is

对我接触过的研究人员来说，这

867.680–869.430  talked to and researchers like that is by far the hardest part. And obviously

对我接触过的研究人员来说，这无疑是最难的部分。当然，

869.430–869.440  by far the hardest part. And obviously

这无疑是最难的部分。当然，

869.440–871.030  by far the hardest part. And obviously it's powered by PyTorch which is really

这无疑是最难的部分。当然，它由 PyTorch 驱动，这很好。

871.030–871.040  it's powered by PyTorch which is really

它由 PyTorch 驱动，这很好。

871.040–873.350  it's powered by PyTorch which is really great. Um but that that is like the the

它由 PyTorch 驱动，这很好。但

873.350–873.360  great. Um but that that is like the the

这很好。但

873.360–874.710  great. Um but that that is like the the thing that that needs to be solved for

这很好。但这正是需要解决的

874.710–874.720  thing that that needs to be solved for

这正是需要解决的

874.720–876.069  thing that that needs to be solved for this to be something that's widespread

这正是需要解决的问题，解决了才能

876.069–876.079  this to be something that's widespread

才能让它得到广泛

876.079–878.710  this to be something that's widespread used. So

才能让它得到广泛应用。所以

878.710–878.720  used. So

得到应用。所以

878.720–880.550  used. So >> awesome. I think that's a pretty good

所以，太好了。我觉得这个回答

880.550–880.560  >> awesome. I think that's a pretty good

太好了。我觉得这个回答

880.560–883.189  >> awesome. I think that's a pretty good pretty solid answer to that question.

太好了。我觉得这个问题回答得很充分。

883.189–883.199  pretty solid answer to that question.

这个问题答得挺到位。

883.199–887.110  pretty solid answer to that question. Um, next up, um, why didn't you add LLM?

这个问题答得挺到位。接下来，为什么没有加入 LLM？

887.110–887.120  Um, next up, um, why didn't you add LLM?

接下来，为什么没有加入 LLM？

887.120–890.150  Um, next up, um, why didn't you add LLM? Why don't didn't you add LLM support

接下来，为什么没有加入 LLM？为什么没有加入 LLM 支持？

890.150–890.160  Why don't didn't you add LLM support

为什么没有加入 LLM 支持？

890.160–892.230  Why don't didn't you add LLM support into the UI so people can talk to an

为什么不在 UI 中加入 LLM 支持，让大家能和

892.230–892.240  into the UI so people can talk to an

在 UI 中，让大家能和

892.240–893.670  into the UI so people can talk to an agent instead of reading through the

在 UI 中，让大家能和智能体交谈，而不用翻阅

893.670–893.680  agent instead of reading through the

智能体交谈，而不用翻阅

893.680–896.150  agent instead of reading through the documentation?

智能体交谈，而不用翻阅文档？

896.150–896.160  documentation?

文档？

896.160–899.315  documentation? Uh, who wants to answer this? Uh,

文档？谁想回答这个问题？

899.315–899.325  Uh, who wants to answer this? Uh,

谁想回答这个问题？

899.325–899.829  Uh, who wants to answer this? Uh, [laughter]

谁想回答这个问题？[笑声]

899.829–899.839  [laughter]

[笑声]

899.839–901.509  [laughter] >> I'm not sure what UI we're talking

[笑声] 我不太确定说的是哪个 UI。

901.509–901.519  >> I'm not sure what UI we're talking

我不太确定说的是哪个 UI。

901.519–903.110  >> I'm not sure what UI we're talking about, but yes, like you can ask

我不太确定说的是哪个 UI。不过，确实可以向智能体提问。

903.110–903.120  about, but yes, like you can ask

不过，确实可以提问。

903.120–905.430  about, but yes, like you can ask questions to an agent uh on the

不过，确实可以在网站上向智能体提问。

905.430–905.440  questions to an agent uh on the

可以在网站上向智能体提问。

905.440–907.910  questions to an agent uh on the pytor.org, work but there's also kind of

可以在 pytorch.org 上向智能体提问，但也有一些

907.910–907.920  pytor.org, work but there's also kind of

pytorch.org 上可以用，但也有一些

907.920–910.389  pytor.org, work but there's also kind of a pragmatic approach that um like

pytorch.org 上可以用，但也有一些实际考量。

910.389–910.399  a pragmatic approach that um like

实际考量是，

910.399–913.350  a pragmatic approach that um like running LM for public consumption is not

实际考量是，向公众提供 LM 服务

913.350–913.360  running LM for public consumption is not

向公众提供 LM 服务

913.360–915.670  running LM for public consumption is not cheap and via foundation right so there

向公众提供 LM 服务并不便宜；我们是基金会，对吧，所以

915.670–915.680  cheap and via foundation right so there

并不便宜；我们是基金会，对吧，所以

915.680–918.230  cheap and via foundation right so there are lots of open LLMs that are very very

并不便宜；我们是基金会，对吧。市面上有很多开放的 LLM，

918.230–918.240  are lots of open LLMs that are very very

市面上有很多开放的 LLM，

918.240–919.670  are lots of open LLMs that are very very good about answering questions about

市面上有很多开放的 LLM，很擅长回答

919.670–919.680  good about answering questions about

很擅长回答

919.680–921.990  good about answering questions about PyTorch documentation and we partnered

很擅长回答关于 PyTorch 文档的问题。我们也曾合作

921.990–922.000  PyTorch documentation and we partnered

关于 PyTorch 文档的问题。我们也曾合作

922.000–923.990  PyTorch documentation and we partnered with one of the providers early on so

我们早期和其中一家服务商合作过，所以

923.990–924.000  with one of the providers early on so

我们早期和其中一家服务商合作过，所以

924.000–926.710  with one of the providers early on so you can ask PyTorch questions that I

我们早期和其中一家服务商合作过，所以你可以问 PyTorch 相关问题。我

926.710–926.720  you can ask PyTorch questions that I

你可以问 PyTorch 相关问题。我

926.720–928.310  you can ask PyTorch questions that I don't remember job probably have more

我不记得了，Job 可能更了解

928.310–928.320  don't remember job probably have more

不记得了，Job 可能更了解

928.320–929.910  don't remember job probably have more upto-ate details about the partnership

不记得了，Job 可能更清楚合作的最新情况，

929.910–929.920  upto-ate details about the partnership

合作的最新情况，

929.920–934.150  upto-ate details about the partnership like who is the provider but yeah

合作的最新情况，比如服务商是谁。总之，

934.150–934.160  like who is the provider but yeah

比如服务商是谁。总之，

934.160–936.629  like who is the provider but yeah >> yeah go to collab and talk an agent,

比如服务商是谁。对，去 Colab 和智能体聊就行。

936.629–936.639  >> yeah go to collab and talk an agent,

对，去 Colab 和智能体聊就行。

936.639–937.829  >> yeah go to collab and talk an agent, right? And it answers your questions

对，去 Colab 和智能体聊就行，对吧？它会回答你的问题。

937.829–937.839  right? And it answers your questions

对吧？它会回答你的问题。

937.839–939.670  right? And it answers your questions about PyTorch.

对吧？它会回答你关于 PyTorch 的问题。

939.670–939.680  about PyTorch.

关于 PyTorch。

939.680–940.310  about PyTorch. >> Actually,

关于 PyTorch。>> 其实，

940.310–940.320  >> Actually,

其实，

940.320–941.829  >> Actually, >> I would like to add Go ahead.

其实，>> 我想补充一点。请说。

941.829–941.839  >> I would like to add Go ahead.

我想补充一点。请说。

941.839–943.269  >> I would like to add Go ahead. >> Go for it, Andre.

我想补充一点。请说。>> Andre，你说。

943.269–943.279  >> Go for it, Andre.

Andre，你说。

943.279–945.590  >> Go for it, Andre. >> So, basically, yes. Um, go ahead and

Andre，你说。>> 对，基本上是这样。你可以

945.590–945.600  >> So, basically, yes. Um, go ahead and

对，基本上是这样。你可以

945.600–947.750  >> So, basically, yes. Um, go ahead and create an issue for us describing

对，基本上是这样。你可以提个问题，说明

947.750–947.760  create an issue for us describing

提个问题，说明

947.760–950.790  create an issue for us describing exactly where you want to add the LLM

提个问题，说明你具体想在哪里加入 LLM

950.790–950.800  exactly where you want to add the LLM

你具体想在哪里加入 LLM

950.800–953.110  exactly where you want to add the LLM and maybe we can look into it. Yes,

你具体想在哪里加入 LLM，我们可以研究一下。对，

953.110–953.120  and maybe we can look into it. Yes,

我们可以研究一下。对，

953.120–955.030  and maybe we can look into it. Yes, because we have extensive documentation

我们可以研究一下。对，因为我们有大量文档

955.030–955.040  because we have extensive documentation

因为我们有大量文档

955.040–958.629  because we have extensive documentation and extensive um uh basically release

因为我们有大量文档，还有详尽的

958.629–958.639  and extensive um uh basically release

还有详尽的版本更新

958.639–960.389  and extensive um uh basically release notes and there there are multiple

还有详尽的版本更新说明，也有多个

960.389–960.399  notes and there there are multiple

说明，也有多个

960.399–962.710  notes and there there are multiple sources of documentation. So I think if

说明，也有多个文档来源。所以我觉得，如果

962.710–962.720  sources of documentation. So I think if

文档来源。所以我觉得，如果

962.720–965.269  sources of documentation. So I think if you're more direct about it what you

文档来源。所以我觉得，如果你能更明确地说说

965.269–965.279  you're more direct about it what you

你能更明确地说说

965.279–967.829  you're more direct about it what you want to do maybe we can come come up

你能更明确地说说想做什么，我们或许能想出

967.829–967.839  want to do maybe we can come come up

想做什么，我们或许能想出

967.839–969.749  want to do maybe we can come come up with the solutions for this.

想做什么，我们或许能想出解决办法。

969.749–969.759  with the solutions for this.

解决办法。

969.759–971.350  with the solutions for this. >> Yeah and an issue on GitHub is the right

解决办法。>> 对，在 GitHub 上提个问题是

971.350–971.360  >> Yeah and an issue on GitHub is the right

对，在 GitHub 上提个问题是

971.360–972.710  >> Yeah and an issue on GitHub is the right way to do that. Go ahead Joe.

对，在 GitHub 上提个问题是合适的。Joe，你说。

972.710–972.720  way to do that. Go ahead Joe.

合适的。Joe，你说。

972.720–974.949  way to do that. Go ahead Joe. >> Yeah. No, I was just gonna say I mean in

合适的。Joe，你说。>> 对，我只是想说，

974.949–974.959  >> Yeah. No, I was just gonna say I mean in

对，我只是想说，

974.959–977.110  >> Yeah. No, I was just gonna say I mean in the if the the the search obviously we

对，我只是想说，搜索方面，我们显然

977.110–977.120  the if the the the search obviously we

搜索方面，我们显然

977.120–979.030  the if the the the search obviously we use kind of an LM based search now I

搜索方面，我们显然已经用了某种基于 LM 的搜索。我

979.030–979.040  use kind of an LM based search now I

已经用了某种基于 LM 的搜索。我

979.040–980.870  use kind of an LM based search now I think there and I think you know you can

已经用了某种基于 LM 的搜索。我觉得你也可以

980.870–980.880  think there and I think you know you can

我觉得你也可以

980.880–983.829  think there and I think you know you can pragmatically um

我觉得你也可以务实一点，

983.829–983.839  pragmatically um

务实一点，

983.839–986.310  pragmatically um you can pragmatically use like Gemini as

务实一点，比如用 Gemini

986.310–986.320  you can pragmatically use like Gemini as

比如用 Gemini

986.320–987.670  you can pragmatically use like Gemini as well and I think all of our

比如也可以用 Gemini。我觉得我们所有的

987.670–987.680  well and I think all of our

我觉得我们所有的

987.680–989.829  well and I think all of our documentation is in distribution right

我觉得我们所有的文档都包含在发行版里，对吧？

989.829–989.839  documentation is in distribution right

文档都包含在发行版里，对吧？

989.839–992.870  documentation is in distribution right and uh it's it's well well indexed um

文档都包含在发行版里，对吧？而且索引做得很好。

992.870–992.880  and uh it's it's well well indexed um

而且索引做得很好。

992.880–994.550  and uh it's it's well well indexed um and so you know I go into Google and and

而且索引做得很好。所以我会打开 Google，

994.550–994.560  and so you know I go into Google and and

所以我会打开 Google，

994.560–996.389  and so you know I go into Google and and and actually Gemini does a really great

所以我会打开 Google。其实 Gemini 很擅长

996.389–996.399  and actually Gemini does a really great

其实 Gemini 很擅长

996.399–998.150  and actually Gemini does a really great job of like of scouring our

其实 Gemini 很擅长查找我们的

998.150–998.160  job of like of scouring our

查找我们的

998.160–1000.389  job of like of scouring our documentation and and distilling things

查找我们的文档，并提炼内容，

1000.389–1000.399  documentation and and distilling things

文档，并提炼内容，

1000.399–1002.470  documentation and and distilling things down to what I need. So, um, we don't

文档，并提炼出我需要的信息。所以我们

1002.470–1002.480  down to what I need. So, um, we don't

提炼出我需要的信息。所以我们

1002.480–1003.910  down to what I need. So, um, we don't actually even need to do a lot embedded

提炼出我需要的信息。所以我们甚至不需要在网站里嵌入太多功能，

1003.910–1003.920  actually even need to do a lot embedded

甚至不需要嵌入太多功能，

1003.920–1005.189  actually even need to do a lot embedded in our site, but obviously make sure our

甚至不需要在网站里嵌入太多功能，但显然要确保我们的

1005.189–1005.199  in our site, but obviously make sure our

在网站里嵌入太多功能，但显然要确保我们的

1005.199–1008.150  in our site, but obviously make sure our docs are updated. Um, and and I mean, we

在网站里嵌入太多功能，但显然要确保文档及时更新。我是说，我们

1008.150–1008.160  docs are updated. Um, and and I mean, we

文档及时更新。我是说，我们

1008.160–1009.910  docs are updated. Um, and and I mean, we like there's obviously the last year

文档及时更新。我是说，过去一年我们显然

1009.910–1009.920  like there's obviously the last year

过去一年我们显然

1009.920–1011.829  like there's obviously the last year we've been trending towards how do you

过去一年我们显然一直在思考如何

1011.829–1011.839  we've been trending towards how do you

我们一直在思考如何

1011.839–1013.590  we've been trending towards how do you build things in a way obviously are good

我们一直在思考如何构建既方便

1013.590–1013.600  build things in a way obviously are good

构建既方便

1013.600–1015.509  build things in a way obviously are good for humans, but also like consumable by

构建既方便人类使用，也方便

1015.509–1015.519  for humans, but also like consumable by

人类使用，也方便

1015.519–1017.509  for humans, but also like consumable by LMS as well. And so, I think the team

人类使用，也方便 LMS 读取的内容。所以我觉得团队

1017.509–1017.519  LMS as well. And so, I think the team

LMS 也能读取。所以我觉得团队

1017.519–1019.189  LMS as well. And so, I think the team has done a great job kind of getting

LMS 也能读取。所以我觉得团队做得很好，已经让

1019.189–1019.199  has done a great job kind of getting

团队做得很好，已经让

1019.199–1020.710  has done a great job kind of getting things in a place where machines can

团队做得很好，已经让内容变得便于机器

1020.710–1020.720  things in a place where machines can

内容变得便于机器

1020.720–1022.790  things in a place where machines can read things um along with the humans.

内容变得既便于机器读取，也便于人阅读。

1022.790–1022.800  read things um along with the humans.

既便于机器读取，也便于人阅读。

1022.800–1027.909  read things um along with the humans. So, um, but we continue to improve.

既便于机器读取，也便于人阅读。不过我们还在继续改进。

1031.750–1031.760  >> Sounds great. Uh, next question. Um,

听起来很棒。下一个问题。

1031.760–1035.669  >> Sounds great. Uh, next question. Um, Demila B uh asks, "What is the what is

听起来很棒。下一个问题。Demila B 问：“目前

1035.669–1035.679  Demila B uh asks, "What is the what is

Demila B 问：“目前

1035.679–1037.270  Demila B uh asks, "What is the what is still the biggest unresolved limitation

Demila B 问：“目前最大的未解决局限

1037.270–1037.280  still the biggest unresolved limitation

最大的未解决局限

1037.280–1039.750  still the biggest unresolved limitation in PyTorch today for training large

如今在 PyTorch 中训练大型

1039.750–1039.760  in PyTorch today for training large

如今在 PyTorch 中训练大型

1039.760–1042.870  in PyTorch today for training large dynamic GNN's especially when the graph

如今在 PyTorch 中训练大型动态 GNN，尤其当图的

1042.870–1042.880  dynamic GNN's especially when the graph

动态 GNN，尤其当图的

1042.880–1046.390  dynamic GNN's especially when the graph topology say changes at every step?"

动态 GNN，尤其当图的拓扑结构每一步都变化时，是什么？”

1046.390–1046.400  topology say changes at every step?"

拓扑结构每一步都变化时，是什么？”

1046.400–1049.750  topology say changes at every step?" That's a great question. Um, who here

拓扑结构每一步都变化时，是什么？”好问题。这里谁

1049.750–1049.760  That's a great question. Um, who here

好问题。这里谁

1049.760–1052.310  That's a great question. Um, who here feels most confident talking about graph

好问题。这里谁最有把握谈谈图

1052.310–1052.320  feels most confident talking about graph

最有把握谈谈图

1052.320–1055.190  feels most confident talking about graph neural networks?

最有把握谈谈图神经网络？

1055.190–1055.200  neural networks?

神经网络？

1055.200–1056.870  neural networks? Joe, is that an area you've you've

神经网络？Joe，这方面你有没有……

1056.870–1056.880  Joe, is that an area you've you've

Joe，这方面你有没有……

1056.880–1059.430  Joe, is that an area you've you've looked into or not so much?

Joe，这方面你研究过吗？

1059.430–1059.440  looked into or not so much?

研究过吗？

1059.440–1062.630  looked into or not so much? I'm probably not the GNN expert here. We

研究过吗？我可能不是这里的 GNN 专家。我们……

1062.630–1062.640  I'm probably not the GNN expert here. We

我可能不是这里的 GNN 专家。我们……

1062.640–1064.630  I'm probably not the GNN expert here. We have used GNN's in the past for

我可能不是这里的 GNN 专家。我们以前用过 GNN，来处理……

1064.630–1064.640  have used GNN's in the past for

以前用过 GNN，来处理……

1064.640–1067.190  have used GNN's in the past for interesting kind of scientific

以前用过 GNN，来处理一些有意思的科学……

1067.190–1067.200  interesting kind of scientific

一些有意思的科学……

1067.200–1070.310  interesting kind of scientific use cases. Um especially in the

一些有意思的科学应用，尤其是在……

1070.310–1070.320  use cases. Um especially in the

应用，尤其是在……

1070.320–1073.590  use cases. Um especially in the chemistry work we did uh with with uh

应用，尤其是在我们与……合作开展的化学研究中。

1073.590–1073.600  chemistry work we did uh with with uh

我们与……合作开展的化学研究。

1073.600–1076.549  chemistry work we did uh with with uh with open chemistry. Um and there's

我们与 Open Chemistry 合作开展的化学研究。而且……

1076.549–1076.559  with open chemistry. Um and there's

与 Open Chemistry 合作。而且……

1076.559–1079.590  with open chemistry. Um and there's they're certainly useful um for those

与 Open Chemistry 合作。GNN 对这类应用确实很有用，

1079.590–1079.600  they're certainly useful um for those

GNN 对这类应用确实很有用，

1079.600–1081.190  they're certainly useful um for those types because you have very complex data

GNN 对这类应用确实很有用，因为数据非常复杂，

1081.190–1081.200  types because you have very complex data

因为数据非常复杂，

1081.200–1083.190  types because you have very complex data that you're you're trying to embed that

因为你要嵌入模型的数据非常复杂，

1083.190–1083.200  that you're you're trying to embed that

你要把这些……

1083.200–1085.510  that you're you're trying to embed that information into the model. So, a lot of

你要把这些信息嵌入模型。所以涉及很多……

1085.510–1085.520  information into the model. So, a lot of

信息嵌入模型。所以涉及很多……

1085.520–1087.029  information into the model. So, a lot of different modalities, but I'm I'm

信息嵌入模型。所以涉及很多不同模态，但我……

1087.029–1087.039  different modalities, but I'm I'm

不同模态，但我……

1087.039–1089.669  different modalities, but I'm I'm probably far from a GNN expert.

不同模态，但我远称不上 GNN 专家。

1089.669–1089.679  probably far from a GNN expert.

远称不上 GNN 专家。

1089.679–1090.230  probably far from a GNN expert. >> Yeah.

远称不上 GNN 专家。对。

1090.230–1090.240  >> Yeah.

对。

1090.240–1093.270  >> Yeah. >> Yeah. I think the good answer is the one

对。对，我觉得比较好的回答是……

1093.270–1093.280  >> Yeah. I think the good answer is the one

对，我觉得比较好的回答是……

1093.280–1094.950  >> Yeah. I think the good answer is the one that Andrew made to the previous

对，我觉得 Andrew 对上一个问题的回答很好。

1094.950–1094.960  that Andrew made to the previous

Andrew 对上一个问题的回答。

1094.960–1096.230  that Andrew made to the previous question like, you know, if you have a

Andrew 对上一个问题的回答是：如果你有……

1096.230–1096.240  question like, you know, if you have a

如果你有……

1096.240–1098.150  question like, you know, if you have a particular like I think you should

如果你有具体问题，我觉得你应该……

1098.150–1098.160  particular like I think you should

有具体问题的话，我觉得你应该……

1098.160–1099.590  particular like I think you should probably file an issue against PyTorch

有具体问题的话，最好向 PyTorch……

1099.590–1099.600  probably file an issue against PyTorch

最好向 PyTorch……提交问题反馈。

1099.600–1101.750  probably file an issue against PyTorch Geometric and if they say that you know

最好向 PyTorch Geometric 提交问题反馈。如果他们说……

1101.750–1101.760  Geometric and if they say that you know

向 PyTorch Geometric 反馈。如果他们说……

1101.760–1103.190  Geometric and if they say that you know that there's a fundamental issue in

如果他们说 PyTorch 存在根本性问题，

1103.190–1103.200  that there's a fundamental issue in

存在根本性问题，

1103.200–1105.029  that there's a fundamental issue in PyTorch, you should file an issue

如果 PyTorch 存在根本性问题，你应该提交问题反馈……

1105.029–1105.039  PyTorch, you should file an issue

你应该向 PyTorch 提交问题反馈……

1105.039–1106.630  PyTorch, you should file an issue against us because we don't know until

你应该向我们提交问题反馈，因为我们只有……

1106.630–1106.640  against us because we don't know until

向我们反馈，因为我们只有……

1106.640–1109.510  against us because we don't know until we like hear about it. But like I would

向我们反馈，因为你不说，我们就不知道。不过我会……

1109.510–1109.520  we like hear about it. But like I would

我们很想听听。不过我觉得……

1109.520–1113.270  we like hear about it. But like I would say GNN's is not the most hot topic in

我们很想听听。不过我觉得，GNN 不是最热门的话题，

1113.270–1113.280  say GNN's is not the most hot topic in

我觉得 GNN 不是最热门的话题，

1113.280–1115.750  say GNN's is not the most hot topic in uh artificial intelligence in the last

我觉得 GNN 不是过去一段时间人工智能领域最热门的话题，

1115.750–1115.760  uh artificial intelligence in the last

过去一段时间，人工智能领域……

1115.760–1119.029  uh artificial intelligence in the last 12 months and so we don't like have a

过去 12 个月，人工智能领域……所以我们没有

1119.029–1119.039  12 months and so we don't like have a

过去 12 个月，所以我们没有

1119.039–1121.830  12 months and so we don't like have a particular focus on those

过去 12 个月，所以我们没有特别关注这类模型。

1121.830–1121.840  particular focus on those

特别关注这类模型，

1121.840–1123.350  particular focus on those but they should work and there shouldn't

特别关注这类模型，但它们应该能运行，也不该

1123.350–1123.360  but they should work and there shouldn't

但它们应该能运行，也不该

1123.360–1125.270  but they should work and there shouldn't be any limitations other than inherent

但它们应该能运行，除了架构本身的限制，不该有其他限制。

1125.270–1125.280  be any limitations other than inherent

除了架构本身的限制，不该有其他限制。

1125.280–1126.789  be any limitations other than inherent limitations in the architecture that

除了架构本身固有的限制，不该有其他限制，比如

1126.789–1126.799  limitations in the architecture that

架构本身的限制，比如

1126.799–1128.950  limitations in the architecture that like post training is somewhat slow

架构本身的限制，比如后训练有点慢，

1128.950–1128.960  like post training is somewhat slow

后训练有点慢，

1128.960–1131.029  like post training is somewhat slow because your graph propagation like

后训练有点慢，因为图上的信息传播，

1131.029–1131.039  because your graph propagation like

因为图上的信息传播，

1131.039–1132.950  because your graph propagation like backward propagation across the graph

因为图上的信息传播，比如沿图反向传播，

1132.950–1132.960  backward propagation across the graph

沿图反向传播，

1132.960–1134.310  backward propagation across the graph like introduce lots and lots of

沿图反向传播，会产生大量

1134.310–1134.320  like introduce lots and lots of

会产生大量

1134.320–1135.750  like introduce lots and lots of gradients which are somewhat unstable

会产生大量不太稳定的梯度，

1135.750–1135.760  gradients which are somewhat unstable

不太稳定的梯度，

1135.760–1137.750  gradients which are somewhat unstable and you need to like be very careful

梯度不太稳定，所以你得格外小心，

1137.750–1137.760  and you need to like be very careful

所以你得格外小心，

1137.760–1139.029  and you need to like be very careful because the trading electron rate should

所以你得格外小心，因为训练学习率得

1139.029–1139.039  because the trading electron rate should

因为训练学习率得

1139.039–1143.110  because the trading electron rate should be very low but that's not new. [snorts]

因为训练学习率得非常低。不过这也不是什么新问题。（轻哼）

1146.950–1146.960  >> Okay, thank you folks for taking a swing

好的，谢谢各位尝试回答

1146.960–1151.590  >> Okay, thank you folks for taking a swing at a challenging question.

好的，谢谢各位尝试回答这个难题。

1154.390–1154.400  >> Um Adam's uh question is next. Um how

接下来是 Adam 的问题：

1154.400–1157.430  >> Um Adam's uh question is next. Um how does an inductor choose between kublos

接下来是 Adam 的问题：Inductor 如何在 cuBLAS、

1157.430–1157.440  does an inductor choose between kublos

Inductor 如何在 cuBLAS、

1157.440–1159.830  does an inductor choose between kublos cutless and cute DSL kernels and does

Inductor 如何在 cuBLAS、CUTLASS 和 CuTe DSL 内核之间选择？

1159.830–1159.840  cutless and cute DSL kernels and does

CUTLASS 和 CuTe DSL 内核之间如何选择？

1159.840–1161.830  cutless and cute DSL kernels and does the user have any flexibility here? Is

CUTLASS 和 CuTe DSL 内核之间如何选择？用户能否自行调整？

1161.830–1161.840  the user have any flexibility here? Is

用户能否自行调整？这个

1161.840–1165.110  the user have any flexibility here? Is this cho is this choice exposed um uh

用户能否自行调整？这个选择是否开放给用户？

1165.110–1165.120  this cho is this choice exposed um uh

这个选择是否开放给用户？

1165.120–1168.390  this cho is this choice exposed um uh for example in FX uh layer?

这个选择是否开放给用户？比如在 FX 层？

1168.390–1168.400  for example in FX uh layer?

比如在 FX 层？

1168.400–1170.390  for example in FX uh layer? >> So I can probably answer this one. Well

比如在 FX 层？我大概可以回答这个问题。

1170.390–1170.400  >> So I can probably answer this one. Well

我大概可以回答这个问题。嗯，

1170.400–1174.390  >> So I can probably answer this one. Well like inductor tries to do the best perf

我大概可以回答这个问题。Inductor 会尽量取得最佳性能，

1174.390–1174.400  like inductor tries to do the best perf

Inductor 会尽量取得最佳性能，

1174.400–1176.789  like inductor tries to do the best perf right and it like tries to select the

Inductor 会尽量取得最佳性能，并尝试选出

1176.789–1176.799  right and it like tries to select the

并尝试选出

1176.799–1179.430  right and it like tries to select the ones that it believes will be better. Uh

并尝试选出它认为性能更好的内核。

1179.430–1179.440  ones that it believes will be better. Uh

它认为更好的那些。嗯

1179.440–1181.110  ones that it believes will be better. Uh you can also do max autotune when it

它认为更好的那些。嗯，你也可以使用最大自动调优，当它……

1181.110–1181.120  you can also do max autotune when it

你也可以使用最大自动调优，当它……

1181.120–1182.870  you can also do max autotune when it should just dutifully iterate over all

你也可以使用最大自动调优，让它逐一遍历所有……

1182.870–1182.880  should just dutifully iterate over all

逐一遍历所有……

1182.880–1184.470  should just dutifully iterate over all the options and try to pick the one that

逐一遍历所有选项，并尝试选出……

1184.470–1184.480  the options and try to pick the one that

这些选项，并尝试选出……

1184.480–1185.909  the options and try to pick the one that works for you. Also there is a hardware

这些选项，并尝试选出适合你的方案。另外还有硬件……

1185.909–1185.919  works for you. Also there is a hardware

适合你的方案。另外还有硬件……

1185.919–1188.230  works for you. Also there is a hardware limitations and there should be knobs in

适合你的方案。另外还有硬件限制，应该有一些开关……

1188.230–1188.240  limitations and there should be knobs in

硬件限制，应该有一些开关……

1188.240–1190.150  limitations and there should be knobs in the inductor to say use or doesn't use

硬件限制，Inductor 里应该有开关来指定是否使用……

1190.150–1190.160  the inductor to say use or doesn't use

在 Inductor 里指定是否使用……

1190.160–1193.029  the inductor to say use or doesn't use certain backends but I don't think there

在 Inductor 里指定是否使用某些后端，但我觉得……

1193.029–1193.039  certain backends but I don't think there

某些后端，但我觉得……

1193.039–1195.909  certain backends but I don't think there is a very easy way to express it through

某些后端，但我觉得没有很简便的方法……

1195.909–1195.919  is a very easy way to express it through

没有很简便的方法……

1195.919–1198.470  is a very easy way to express it through the FX graph but you can suggest this

没有很简便的方法通过 FX 图表达，不过你可以把这点……

1198.470–1198.480  the FX graph but you can suggest this

通过 FX 图表达，不过你可以把这点……

1198.480–1200.150  the FX graph but you can suggest this one as an improvement. So once again

通过 FX 图表达，不过你可以把这点作为改进建议。所以还是那句话……

1200.150–1200.160  one as an improvement. So once again

作为改进建议。所以还是那句话……

1200.160–1202.310  one as an improvement. So once again file an issue saying that you want this

作为改进建议。所以还是那句话：提个 issue，说明你想要这个功能……

1202.310–1202.320  file an issue saying that you want this

提个 issue，说明你想要这个功能……

1202.320–1204.950  file an issue saying that you want this one but I kind of want you to think

提个 issue，说明你想要这个功能，但我希望你也想想……

1204.950–1204.960  one but I kind of want you to think

但我希望你也想想……

1204.960–1207.350  one but I kind of want you to think about it like why do you want to be

但我希望你也想想，为什么要……

1207.350–1207.360  about it like why do you want to be

想想为什么要……

1207.360–1210.310  about it like why do you want to be explicit like don't we want to converge

想想为什么要明确指定它。我们难道不希望统一到……

1210.310–1210.320  explicit like don't we want to converge

明确指定它。我们难道不希望统一到……

1210.320–1213.350  explicit like don't we want to converge to one model and I suspect that cute DSL

明确指定它。我们难道不希望统一到一种模型吗？我猜 CuTe DSL……

1213.350–1213.360  to one model and I suspect that cute DSL

统一到一种模型吗？我猜 CuTe DSL……

1213.360–1218.150  to one model and I suspect that cute DSL is probably something we kind of leaning

统一到一种模型吗？我猜 CuTe DSL 可能是我们……

1218.150–1218.160  is probably something we kind of leaning

可能是我们……

1218.160–1219.830  is probably something we kind of leaning towards on the most modern architectures

可能是我们在最新架构上倾向采用的方向……

1219.830–1219.840  towards on the most modern architectures

在最新架构上倾向采用的方向……

1219.840–1220.950  towards on the most modern architectures like blackfields and probably have

在 Blackwell 等最新架构上倾向采用的方向，而且可能……

1220.950–1220.960  like blackfields and probably have

比如 Blackwell，而且可能……

1220.960–1225.669  like blackfields and probably have common verbs but I think like kublast

比如 Blackwell，可能还有一些共通的操作，但我觉得像 cuBLAS……

1225.669–1225.679  common verbs but I think like kublast

有一些共通的操作，但我觉得像 cuBLAS……

1225.679–1226.950  common verbs but I think like kublast will probably give you a much better

有一些共通的操作，但我觉得 cuBLAS 可能会带来好得多的……

1226.950–1226.960  will probably give you a much better

可能会带来好得多的……

1226.960–1229.909  will probably give you a much better perf on and cutless on an architectures

可能在某些架构上，cuBLAS 和 CUTLASS 的性能会好得多……

1229.909–1229.919  perf on and cutless on an architectures

在某些架构上，cuBLAS 和 CUTLASS 的性能……

1229.919–1234.870  perf on and cutless on an architectures because there's kind of a cut off there.

在某些架构上，cuBLAS 和 CUTLASS 的性能会更好，因为这里有个分界点。

1236.950–1236.960  >> Okay, any other uh responses on this

好，关于这个问题还有其他回应吗？

1236.960–1240.390  >> Okay, any other uh responses on this one? Okay, let's move to the next one.

好，关于这个问题还有其他回应吗？那我们进入下一个问题。

1240.390–1240.400  one? Okay, let's move to the next one.

那我们进入下一个问题。

1240.400–1242.230  one? Okay, let's move to the next one. We have I love that there are so many

那我们进入下一个问题。观众提了这么多……

1242.230–1242.240  We have I love that there are so many

观众提了这么多……

1242.240–1243.990  We have I love that there are so many audience questions. So, thank you folks

观众提了这么多问题，我很高兴。谢谢大家。

1243.990–1244.000  audience questions. So, thank you folks

观众提问。谢谢大家。

1244.000–1246.149  audience questions. So, thank you folks for putting those in. Uh I think that's

观众提问。谢谢大家提交问题。我觉得这

1246.149–1246.159  for putting those in. Uh I think that's

谢谢大家提交问题。我觉得这

1246.159–1249.430  for putting those in. Uh I think that's great. Okay. Um we have somebody from

谢谢大家提交问题。我觉得这很好。好，有位来自

1249.430–1249.440  great. Okay. Um we have somebody from

很好。好，有位来自

1249.440–1251.909  great. Okay. Um we have somebody from LinkedIn who asked uh which PyTorch 20

很好。好，有位来自 LinkedIn 的人问，PyTorch 20

1251.909–1251.919  LinkedIn who asked uh which PyTorch 20

LinkedIn 上有人问，PyTorch 20

1251.919–1254.549  LinkedIn who asked uh which PyTorch 20 uh 21.14 uh feature gives the biggest

LinkedIn 上有人问，PyTorch 20、21.14 的哪个功能

1254.549–1254.559  uh 21.14 uh feature gives the biggest

21.14 的哪个功能带来的

1254.559–1256.470  uh 21.14 uh feature gives the biggest performance boost for production LLM

21.14 的哪个功能最能提升生产环境中的 LLM

1256.470–1256.480  performance boost for production LLM

最能提升生产环境中的 LLM

1256.480–1259.350  performance boost for production LLM inference and how easy is it uh to

最能提升生产环境中的 LLM 推理性能？采用起来又有多容易？

1259.350–1259.360  inference and how easy is it uh to

LLM 推理性能？采用起来又有多容易？

1259.360–1263.190  inference and how easy is it uh to adopt? Um

LLM 推理性能？采用起来又有多容易？嗯。

1263.190–1263.200  adopt? Um

采用起来？嗯。

1263.200–1264.950  adopt? Um I don't know that we did an ablation

采用起来？嗯。我不确定我们是否做过消融实验。

1264.950–1264.960  I don't know that we did an ablation

我不确定我们是否做过消融实验。

1264.960–1266.630  I don't know that we did an ablation study of the various different features

我不确定我们是否对各项功能做过消融实验。

1266.630–1266.640  study of the various different features

对各项功能做消融实验。

1266.640–1270.549  study of the various different features but does anybody have a um have a

对各项功能做消融实验。不过有人有什么

1270.549–1270.559  but does anybody have a um have a

不过有人有什么

1270.559–1274.549  but does anybody have a um have a thought here? I suspect cudgraphs uh

不过有人有什么想法吗？我猜可能是 CUDA Graphs。

1274.549–1274.559  thought here? I suspect cudgraphs uh

有什么想法吗？我猜可能是 CUDA Graphs。

1274.559–1276.789  thought here? I suspect cudgraphs uh ability to trace more functions, avoid

有什么想法吗？我猜可能是 CUDA Graphs 能追踪更多函数，避免

1276.789–1276.799  ability to trace more functions, avoid

能追踪更多函数，避免

1276.799–1280.149  ability to trace more functions, avoid cudograph breaks and ability to like

能追踪更多函数，避免 CUDA Graph 中断，还能

1280.149–1280.159  cudograph breaks and ability to like

避免 CUDA Graph 中断，还能

1280.159–1283.430  cudograph breaks and ability to like trace coms uh would be the one maybe

避免 CUDA Graph 中断，还能追踪通信操作。可能是这个。

1283.430–1283.440  trace coms uh would be the one maybe

追踪通信操作。可能是这个。

1283.440–1285.270  trace coms uh would be the one maybe some of the torch compiler features but

追踪通信操作。也可能是 Torch 编译器的某些功能，不过

1285.270–1285.280  some of the torch compiler features but

也可能是 Torch 编译器的某些功能，不过

1285.280–1286.870  some of the torch compiler features but that depends on the frameworks that

也可能是 Torch 编译器的某些功能，不过这取决于你使用的

1286.870–1286.880  that depends on the frameworks that

这取决于你使用的

1286.880–1289.669  that depends on the frameworks that you're using for your LLM inference,

这取决于你用什么框架做 LLM 推理，

1289.669–1289.679  you're using for your LLM inference,

你用什么框架做 LLM 推理，

1289.679–1291.270  you're using for your LLM inference, right?

你用什么框架做 LLM 推理，对吧？

1291.270–1291.280  right?

对吧？

1291.280–1293.270  right? And they should be pretty easy to adopt.

对吧？这些功能应该很容易采用。

1293.270–1293.280  And they should be pretty easy to adopt.

这些功能应该很容易采用。

1293.280–1294.710  And they should be pretty easy to adopt. So like lots of those features were

这些功能应该很容易采用。很多功能都是

1294.710–1294.720  So like lots of those features were

很多功能都是

1294.720–1298.390  So like lots of those features were driven by like asks from users such as

很多功能都是应用户需求而开发的，比如

1298.390–1298.400  driven by like asks from users such as

应用户需求而开发的，比如

1298.400–1302.230  driven by like asks from users such as like VLM or SJAN. So I hope that they

应 VLM 或 SJAN 等用户的需求而开发。所以我希望他们

1302.230–1302.240  like VLM or SJAN. So I hope that they

比如 VLM 或 SJAN。所以我希望他们

1302.240–1303.750  like VLM or SJAN. So I hope that they will be incorporating those features

比如 VLM 或 SJAN。所以我希望他们能把这些功能

1303.750–1303.760  will be incorporating those features

能把这些功能

1303.760–1307.110  will be incorporating those features into their frameworks and NVJMS is also

能把这些功能整合进他们的框架。NVJMS 也是

1307.110–1307.120  into their frameworks and NVJMS is also

整合进他们的框架。NVJMS 也是

1307.120–1310.070  into their frameworks and NVJMS is also a great one that probably important for

整合进他们的框架。NVJMS 也很棒，可能对……很重要。

1310.070–1310.080  a great one that probably important for

这点很好，可能对……很重要

1310.080–1313.669  a great one that probably important for performance.

这点很好，可能对性能很重要。

1314.950–1314.960  >> Okay.

好的。

1314.960–1318.390  >> Okay. >> Yeah, I think NVJ specifically on black

好的。我觉得 NVJ 尤其是在 Black 上……

1318.390–1318.400  >> Yeah, I think NVJ specifically on black

我觉得 NVJ 尤其是在 Black 上……

1318.400–1320.470  >> Yeah, I think NVJ specifically on black with low precision I think it's

我觉得 NVJ 尤其是在 Black 上使用低精度时……

1320.470–1320.480  with low precision I think it's

使用低精度时，我觉得这……

1320.480–1325.190  with low precision I think it's important. Yeah.

使用低精度时，我觉得这很重要。对。

1329.110–1329.120  >> Okay. Um the next one is from Asam. Uh,

好的。下一个问题来自 Asam。

1329.120–1330.630  >> Okay. Um the next one is from Asam. Uh, and I might be mispronouncing there. I

好的。下一个问题来自 Asam。我可能念错了名字，

1330.630–1330.640  and I might be mispronouncing there. I

我可能念错了名字，

1330.640–1333.110  and I might be mispronouncing there. I apologize if I did. Um, PyTorch, uh,

如果念错了，抱歉。PyTorch……

1333.110–1333.120  apologize if I did. Um, PyTorch, uh,

如果念错了，抱歉。PyTorch……

1333.120–1335.669  apologize if I did. Um, PyTorch, uh, 21.14 makes fault tolerance a first

如果念错了，抱歉。PyTorch 21.14 将容错提升为一等……

1335.669–1335.679  21.14 makes fault tolerance a first

21.14 将容错提升为一等……

1335.679–1338.230  21.14 makes fault tolerance a first class CTN concept. Does the new model

21.14 将容错提升为一等 CTN 概念。新模型能否……

1338.230–1338.240  class CTN concept. Does the new model

一等 CTN 概念。新模型能否……

1338.240–1340.230  class CTN concept. Does the new model allow a training job to recover from a

一等 CTN 概念。新模型能否让训练任务从……

1340.230–1340.240  allow a training job to recover from a

让训练任务从……恢复，

1340.240–1341.669  allow a training job to recover from a failed rank without reconstructing the

让训练任务从某个 rank 故障中恢复，而无须重建……

1341.669–1341.679  failed rank without reconstructing the

某个 rank 故障中恢复，而无须重建……

1341.679–1343.669  failed rank without reconstructing the entire distributed process group? And

某个 rank 故障中恢复，而无须重建整个分布式进程组？另外，

1343.669–1343.679  entire distributed process group? And

整个分布式进程组？另外，

1343.679–1345.590  entire distributed process group? And what state is expected to remain valid

整个分布式进程组？另外，恢复后哪些状态预计仍然有效？

1345.590–1345.600  what state is expected to remain valid

哪些状态预计仍然有效？

1345.600–1349.750  what state is expected to remain valid after recovery?

恢复后哪些状态预计仍然有效？

1352.230–1352.240  Do we have any anyone here who's expert

这里有谁熟悉……

1352.240–1361.095  Do we have any anyone here who's expert in the new um torchcom's model?

这里有谁熟悉新的 torchcom 模型吗？

1361.110–1361.120  >> [clears throat]

（清嗓子）

1361.120–1362.789  >> [clears throat] >> None of us are.

（清嗓子）我们都不熟悉。

1362.789–1362.799  >> None of us are.

我们都不熟悉。

1362.799–1364.070  >> None of us are. >> Okay. Not me.

我们都不熟悉。好的，我也不熟悉。

1364.070–1364.080  >> Okay. Not me.

好的，我也不熟悉。

1364.080–1365.350  >> Okay. Not me. >> I think we're I think we're going to

好的，我也不熟悉。我想我们得……

1365.350–1365.360  >> I think we're I think we're going to

我想我们得……

1365.360–1368.789  >> I think we're I think we're going to have to refer this question to um uh to

我想我们得把这个问题……

1368.789–1368.799  have to refer this question to um uh to

得把这个问题……

1368.799–1370.789  have to refer this question to um uh to the discussion. So this would be a great

得把这个问题转到讨论区。这会是个很好的……

1370.789–1370.799  the discussion. So this would be a great

转到讨论区。这会是个很好的……

1370.799–1372.870  the discussion. So this would be a great question for example for discuss

转到讨论区。比如，这个问题很适合在 discuss 上提。

1372.870–1372.880  question for example for discuss

比如，这个问题很适合在 discuss 上提。

1372.880–1376.549  question for example for discuss uh.pytorch or dev discuss. Um so I think

比如在 discuss.pytorch 或 dev discuss 上提。我想，

1376.549–1376.559  uh.pytorch or dev discuss. Um so I think

在 discuss.pytorch 或 dev discuss 上提。我想，

1376.559–1379.270  uh.pytorch or dev discuss. Um so I think uh the the specific uh details about

在 discuss.pytorch 或 dev discuss 上提。我想，具体细节……

1379.270–1379.280  uh the the specific uh details about

具体细节……

1379.280–1381.430  uh the the specific uh details about like what part of the state uh would be

具体来说，状态的哪些部分……

1381.430–1381.440  like what part of the state uh would be

状态的哪些部分……

1381.440–1383.590  like what part of the state uh would be expected to remain valid uh is the kind

状态的哪些部分预计仍然有效，这类问题……

1383.590–1383.600  expected to remain valid uh is the kind

预计仍然有效，这类问题……

1383.600–1385.909  expected to remain valid uh is the kind of thing one of the distributed uh uh

预计仍然有效，这类问题得由分布式领域的……

1385.909–1385.919  of thing one of the distributed uh uh

关于这件事，分布式领域的一位……

1385.919–1388.630  of thing one of the distributed uh uh engineers would be really strong

关于这件事，一位分布式工程师应该很在行。

1388.630–1388.640  engineers would be really strong

工程师应该很在行。

1388.640–1389.909  engineers would be really strong rice question.

工程师应该很在行。好问题。

1389.909–1389.919  rice question.

好问题。

1389.919–1390.390  rice question. >> Yeah.

好问题。>> 对。

1390.390–1390.400  >> Yeah.

>> 对。

1390.400–1392.149  >> Yeah. >> Yep. Exactly. But I want to just flag

>> 对。>> 没错。不过我想提醒一下

1392.149–1392.159  >> Yep. Exactly. But I want to just flag

>> 没错。不过我想提醒一下

1392.159–1393.830  >> Yep. Exactly. But I want to just flag that this is experimental support and we

>> 没错。不过我想提醒一下，这项支持仍属实验性，我们

1393.830–1393.840  that this is experimental support and we

这项支持仍属实验性，我们

1393.840–1396.149  that this is experimental support and we indeed like seeking your feedback. So

这项支持仍属实验性，我们也很想听到你的反馈。所以

1396.149–1396.159  indeed like seeking your feedback. So

也很想听到你的反馈。所以

1396.159–1398.630  indeed like seeking your feedback. So please try to obtain into this like

也很想听到你的反馈。所以请试着启用这些功能，

1398.630–1398.640  please try to obtain into this like

请试着启用这些功能，

1398.640–1400.710  please try to obtain into this like nickel ch and other CTN default

请试着启用这些功能，比如 NCCL 和其他 c10d

1400.710–1400.720  nickel ch and other CTN default

NCCL 和其他 c10d

1400.720–1402.870  nickel ch and other CTN default tolerance features and let us know if

NCCL 和其他 c10d 容错功能，并告诉我们

1402.870–1402.880  tolerance features and let us know if

容错功能，并告诉我们

1402.880–1405.270  tolerance features and let us know if they work for you or not.

容错功能，并告诉我们它们是否适合你的场景。

1405.270–1405.280  they work for you or not.

它们是否适合你的场景。

1405.280–1408.070  they work for you or not. But yes, like a new model allows for a

它们是否适合你的场景。不过，新模型确实能让

1408.070–1408.080  But yes, like a new model allows for a

不过，新模型确实能让

1408.080–1410.630  But yes, like a new model allows for a more graceful recovery, but it doesn't

不过，新模型确实能让恢复过程更平稳，但它不能

1410.630–1410.640  more graceful recovery, but it doesn't

恢复过程更平稳，但它不能

1410.640–1412.310  more graceful recovery, but it doesn't guarantee that it will work out of the

恢复过程更平稳，但它不能保证功能

1412.310–1412.320  guarantee that it will work out of the

保证功能

1412.320–1414.870  guarantee that it will work out of the box. You probably need to like adjust

保证功能开箱即用。你可能需要调整

1414.870–1414.880  box. You probably need to like adjust

开箱即用。你可能需要调整

1414.880–1418.630  box. You probably need to like adjust your training pipeline to uh like take

开箱即用。你可能需要调整训练流程，才能

1418.630–1418.640  your training pipeline to uh like take

训练流程，才能

1418.640–1420.390  your training pipeline to uh like take benefit of that and like recognize this

训练流程，才能利用这一点，并将这种

1420.390–1420.400  benefit of that and like recognize this

利用这一点，并将这种

1420.400–1423.029  benefit of that and like recognize this error as recoverable. But yes, sounds

利用这一点，并将这种错误识别为可恢复错误。不过，

1423.029–1423.039  error as recoverable. But yes, sounds

错误识别为可恢复错误。不过，

1423.039–1427.909  error as recoverable. But yes, sounds like a great discuss question.

错误识别为可恢复错误。不过，这确实是个值得讨论的好问题。

1430.070–1430.080  >> Next question.

>> 下一个问题。

1430.080–1434.549  >> Next question. Yes, it's the um uh Shiva Shush um uh

>> 下一个问题。是 Shiva Shush 的问题，

1434.549–1434.559  Yes, it's the um uh Shiva Shush um uh

是 Shiva Shush 的问题，

1434.559–1437.830  Yes, it's the um uh Shiva Shush um uh again pronunciation caveat on Apple

是 Shiva Shush 的问题，名字读音可能有误：关于 Apple

1437.830–1437.840  again pronunciation caveat on Apple

名字读音可能有误：关于 Apple

1437.840–1440.390  again pronunciation caveat on Apple silicon uh 21.14 adds uh native linear

名字读音可能有误：在 Apple Silicon 上，21.14 增加了原生线性

1440.390–1440.400  silicon uh 21.14 adds uh native linear

在 Apple Silicon 上，21.14 增加了原生线性

1440.400–1442.630  silicon uh 21.14 adds uh native linear algebra and more metal kernels uh for

在 Apple Silicon 上，21.14 增加了原生线性代数支持和更多用于

1442.630–1442.640  algebra and more metal kernels uh for

代数支持和更多用于

1442.640–1445.029  algebra and more metal kernels uh for LLM decode on unified memory. Which path

代数支持和更多用于统一内存上大语言模型解码的 Metal 内核。哪条路径

1445.029–1445.039  LLM decode on unified memory. Which path

统一内存上大语言模型解码的 Metal 内核。哪条路径

1445.039–1447.669  LLM decode on unified memory. Which path actually moves the needle uh versus 2.13

统一内存上大语言模型解码的 Metal 内核。与 2.13 相比，哪条路径真正带来了提升：

1447.669–1447.679  actually moves the needle uh versus 2.13

与 2.13 相比，哪条路径真正带来了提升：

1447.679–1449.830  actually moves the needle uh versus 2.13 the new metal kernels or inductor

与 2.13 相比，哪条路径真正带来了提升：新的 Metal 内核，还是 Inductor？

1449.830–1449.840  the new metal kernels or inductor

新的 Metal 内核或 Inductor

1449.840–1452.549  the new metal kernels or inductor picking up better epilogs? Any measured

新的 Metal 内核或 Inductor 用上了更好的尾部计算？有测量过

1452.549–1452.559  picking up better epilogs? Any measured

用上了更好的尾部计算？有测量过

1452.559–1455.110  picking up better epilogs? Any measured tokens per second deltas um on the M

用上了更好的尾部计算？有测量过 M 系列上每秒 token 数的变化吗？

1455.110–1455.120  tokens per second deltas um on the M

M 系列上每秒 token 数的变化

1455.120–1457.990  tokens per second deltas um on the M series or fixed to code loop? And I I

M 系列上每秒 token 数的变化，还是只针对代码循环？我，我

1457.990–1458.000  series or fixed to code loop? And I I

系列，还是只针对代码循环？我，我

1458.000–1459.269  series or fixed to code loop? And I I believe Nikita, you've been really

系列，还是只针对代码循环？我想 Nikita，你之前一直

1459.269–1459.279  believe Nikita, you've been really

我想 Nikita，你之前一直

1459.279–1460.710  believe Nikita, you've been really active in the past with some of the

我想 Nikita，你之前一直积极参与一些

1460.710–1460.720  active in the past with some of the

之前积极参与一些

1460.720–1462.789  active in the past with some of the Apple changes. So I'm gonna toss this

之前积极参与一些 Apple 方面的改动。这个问题我就交给

1462.789–1462.799  Apple changes. So I'm gonna toss this

Apple 方面的改动。这个问题我就交给

1462.799–1464.230  Apple changes. So I'm gonna toss this one to you.

Apple 方面的改动。这个问题就交给你了。

1464.230–1464.240  one to you.

交给你了。

1464.240–1465.750  one to you. >> Sure. Well, first of all, we don't

交给你了。>> 好的。首先，我们没有

1465.750–1465.760  >> Sure. Well, first of all, we don't

>> 好的。首先，我们没有

1465.760–1467.990  >> Sure. Well, first of all, we don't specifically measure and also it's very

>> 好的。首先，我们没有专门测量过，而且这也很

1467.990–1468.000  specifically measure and also it's very

专门测量过，而且这也很

1468.000–1470.470  specifically measure and also it's very subjective because like different M

专门测量过，而且这也很难一概而论，因为不同的 M

1470.470–1470.480  subjective because like different M

难一概而论，因为不同的 M

1470.480–1472.310  subjective because like different M series have a different behavior. But

难一概而论，因为不同的 M 系列表现不同。不过

1472.310–1472.320  series have a different behavior. But

系列表现不同。不过

1472.320–1474.149  series have a different behavior. But there is one feature that was introduced

系列表现不同。不过，214 中引入了一项功能

1474.149–1474.159  there is one feature that was introduced

214 中引入了一项功能

1474.159–1478.149  there is one feature that was introduced in 214. we use uh like a new apple

214 中引入了一项功能。我们用了 Apple 的一项新

1478.149–1478.159  in 214. we use uh like a new apple

在 214 中。我们用了 Apple 的一项新

1478.159–1480.310  in 214. we use uh like a new apple silicon or like metal framework feature

在 214 中。我们用了一项新的 Apple Silicon 或 Metal 框架功能

1480.310–1480.320  silicon or like metal framework feature

Apple Silicon 或 Metal 框架功能

1480.320–1481.750  silicon or like metal framework feature called metal performance primitives

Apple Silicon 或 Metal 框架功能，叫作 Metal Performance Primitives

1481.750–1481.760  called metal performance primitives

叫作 Metal Performance Primitives

1481.760–1485.029  called metal performance primitives which is similar to uh I don't know

叫作 Metal Performance Primitives，它有点类似于，嗯，我也说不准

1485.029–1485.039  which is similar to uh I don't know

它有点类似于，嗯，我也说不准

1485.039–1487.669  which is similar to uh I don't know probably to cut in a sense so it's very

它有点类似于，嗯，我也说不准，某种意义上可能类似 CUDA，所以它很

1487.669–1487.679  probably to cut in a sense so it's very

某种意义上可能类似 CUDA，所以它很

1487.679–1489.669  probably to cut in a sense so it's very different it allows one to express

某种意义上可能类似 CUDA，但它很不一样：它允许你表达

1489.669–1489.679  different it allows one to express

不一样：它允许你表达

1489.679–1491.909  different it allows one to express matrix multiplication it's done in eager

不一样：它允许你表达矩阵乘法；运算在即时模式下

1491.909–1491.919  matrix multiplication it's done in eager

矩阵乘法；运算在即时模式下

1491.919–1495.029  matrix multiplication it's done in eager mode and it can in some cases produce

矩阵乘法；运算在即时模式下进行，有时能带来

1495.029–1495.039  mode and it can in some cases produce

模式下进行，有时能带来

1495.039–1497.029  mode and it can in some cases produce just of this particular operation like

模式下进行，有时仅就这个运算而言，能带来

1497.029–1497.039  just of this particular operation like

仅就这个运算而言，能带来

1497.039–1499.750  just of this particular operation like 3x or even sometimes measured 5x but

仅就这个运算而言，能带来 3 倍提升，有时甚至测到 5 倍，但

1499.750–1499.760  3x or even sometimes measured 5x but

3 倍提升，有时甚至测到 5 倍，但

1499.760–1501.350  3x or even sometimes measured 5x but this is more like a synthetic benchmark

3 倍提升，有时甚至测到 5 倍，但这更像是合成基准测试的结果

1501.350–1501.360  this is more like a synthetic benchmark

这更像是合成基准测试的结果

1501.360–1503.430  this is more like a synthetic benchmark not an end to end one performance boost

这更像是合成基准测试的结果，不是端到端的性能提升

1503.430–1503.440  not an end to end one performance boost

不是端到端的性能提升

1503.440–1506.149  not an end to end one performance boost so you should give it tries there is no

不是端到端的性能提升，所以你可以试试；目前没有

1506.149–1506.159  so you should give it tries there is no

所以你应该试试，没有……

1506.159–1508.149  so you should give it tries there is no inductor peing better epilogues

所以你应该试试；Inductor 不一定能生成更好的尾声代码。

1508.149–1508.159  inductor peing better epilogues

Inductor 生成更好的尾声代码

1508.159–1511.430  inductor peing better epilogues necessarily for Apple silicon and yes we

Inductor 在 Apple 芯片上不一定能生成更好的尾声代码。对，我们……

1511.430–1511.440  necessarily for Apple silicon and yes we

对于 Apple 芯片未必如此。对，我们……

1511.440–1512.950  necessarily for Apple silicon and yes we we don't do measurements because again

对于 Apple 芯片未必如此。对，我们没有做测量，因为……

1512.950–1512.960  we don't do measurements because again

我们没有做测量，因为……

1512.960–1514.630  we don't do measurements because again it's kind of you cannot measure on the

我们没有做测量，因为你没法只测……

1514.630–1514.640  it's kind of you cannot measure on the

因为你没法只测……

1514.640–1516.390  it's kind of you cannot measure on the pietorch alone you need to measure on a

因为你没法只测 PyTorch，还得测某个……

1516.390–1516.400  pietorch alone you need to measure on a

光测 PyTorch 不够，还得测某个……

1516.400–1518.470  pietorch alone you need to measure on a particular framework but you can

光测 PyTorch 不够，还得测具体框架。不过你可以……

1518.470–1518.480  particular framework but you can

具体框架。不过你可以……

1518.480–1521.750  particular framework but you can probably check those numbers at hugging

具体框架。不过你或许能在 Hugging……

1521.750–1521.760  probably check those numbers at hugging

你或许能在 Hugging……查到那些数据

1521.760–1523.990  probably check those numbers at hugging face transformers on one of their um

你或许能在 Hugging Face Transformers 的某个……查到那些数据

1523.990–1524.000  face transformers on one of their um

Face Transformers 的某个……

1524.000–1525.590  face transformers on one of their um frameworks if this is what you're using

如果你用的是 Hugging Face Transformers 的某个框架……

1525.590–1525.600  frameworks if this is what you're using

如果你用的是这个框架……

1525.600–1532.070  frameworks if this is what you're using for your inference pipelines

如果你的推理流程用的是这个框架。

1537.350–1537.360  >> okay Um yes, let's do so. Uh Shibash had

>> 好，那就这样。Shibash 刚才……

1537.360–1538.950  >> okay Um yes, let's do so. Uh Shibash had three several questions in a row, so I

>> 好，那就这样。Shibash 连着问了好几个问题，所以我……

1538.950–1538.960  three several questions in a row, so I

连着问了好几个问题，所以我……

1538.960–1539.909  three several questions in a row, so I don't know if we'll get to all of them,

连着问了好几个问题，所以我不确定能否全部回答。

1539.909–1539.919  don't know if we'll get to all of them,

我不确定能否全部回答。

1539.919–1541.222  don't know if we'll get to all of them, but let's take this next one here.

我不确定能否全部回答，不过先看下一个。

1541.222–1541.232  but let's take this next one here.

不过先看下一个。

1541.232–1543.190  but let's take this next one here. [snorts] Uh the declarative uh dynamic

不过先看下一个。[轻哼] 声明式动态……

1543.190–1543.200  [snorts] Uh the declarative uh dynamic

[轻哼] 声明式动态……

1543.200–1545.029  [snorts] Uh the declarative uh dynamic spec looks useful for Asian workloads

[轻哼] 声明式动态规格似乎适合亚洲工作负载

1545.029–1545.039  spec looks useful for Asian workloads

这种规格似乎适合亚洲工作负载

1545.039–1547.510  spec looks useful for Asian workloads where context length um changes

这种规格似乎适合上下文长度会变化的亚洲工作负载

1547.510–1547.520  where context length um changes

这类工作负载的上下文长度会变化

1547.520–1550.070  where context length um changes midsession. Uh if the live shape set

上下文长度会在会话中途变化。如果实际出现的形状集合……

1550.070–1550.080  midsession. Uh if the live shape set

会在会话中途变化。如果实际出现的形状集合……

1550.080–1552.870  midsession. Uh if the live shape set expands beyond the declared spec, does

会在会话中途变化。如果实际出现的形状集合超出声明的规格，会……

1552.870–1552.880  expands beyond the declared spec, does

如果超出声明的规格，会……

1552.880–1554.870  expands beyond the declared spec, does inductor recompile quietly or can you

如果超出声明的规格，Inductor 会悄悄重新编译，还是可以……

1554.870–1554.880  inductor recompile quietly or can you

Inductor 会悄悄重新编译，还是可以……

1554.880–1557.350  inductor recompile quietly or can you make it a hard fail uh so that you can

Inductor 会悄悄重新编译，还是可以让它直接报错，以便……

1557.350–1557.360  make it a hard fail uh so that you can

让它直接报错，以便……

1557.360–1561.029  make it a hard fail uh so that you can have more predictable latency? Um does

让它直接报错，让延迟更可预测？

1561.029–1561.039  have more predictable latency? Um does

让延迟更可预测？那么……

1561.039–1564.870  have more predictable latency? Um does anybody know the answer to this one? Um

让延迟更可预测？有人知道这个问题的答案吗？

1564.870–1564.880  anybody know the answer to this one? Um

有人知道这个问题的答案吗？

1564.880–1567.190  anybody know the answer to this one? Um >> I don't know on top of my head but

有人知道这个问题的答案吗？>> 我一时想不起来，不过……

1567.190–1567.200  >> I don't know on top of my head but

>> 我一时想不起来，不过……

1567.200–1569.110  >> I don't know on top of my head but knowing the development patterns of the

>> 我一时想不起来，不过按 Inductor 的开发习惯……

1569.110–1569.120  knowing the development patterns of the

按它的开发习惯……

1569.120–1571.750  knowing the development patterns of the inductor I suspect there should be an op

按 Inductor 的开发习惯，我猜应该有个选项。

1571.750–1571.760  inductor I suspect there should be an op

我觉得 Inductor 应该有个操作

1571.760–1574.390  inductor I suspect there should be an op to make it fail

我觉得 Inductor 应该有个操作，能让它失败

1574.390–1574.400  to make it fail

让它失败

1574.400–1576.870  to make it fail >> but there also should be a knob like

让它失败。但也应该有个开关

1576.870–1576.880  >> but there also should be a knob like

但也应该有个开关

1576.880–1579.830  >> but there also should be a knob like that was a default orch compile behavior

但也应该有个开关。那曾是 torch.compile 的默认行为

1579.830–1579.840  that was a default orch compile behavior

那曾是 torch.compile 的默认行为

1579.840–1581.430  that was a default orch compile behavior for quite some time that like if it

很长一段时间里，那都是 torch.compile 的默认行为：如果它

1581.430–1581.440  for quite some time that like if it

很长一段时间里，如果它

1581.440–1582.950  for quite some time that like if it fails to do something instead of ering

没能完成某项操作，它不会报错

1582.950–1582.960  fails to do something instead of ering

没能完成某项操作，它不会报错

1582.960–1585.190  fails to do something instead of ering out it just falls back to either or like

没能完成某项操作时，它不会报错，而是回退

1585.190–1585.200  out it just falls back to either or like

而是回退

1585.200–1587.029  out it just falls back to either or like to the previous behavior. So there are

而是回退到之前的行为。所以两种

1587.029–1587.039  to the previous behavior. So there are

回退到之前的行为。所以两种

1587.039–1588.870  to the previous behavior. So there are both nodes exist but you should ask this

回退到之前的行为。所以两种模式都存在，但你应该问问

1588.870–1588.880  both nodes exist but you should ask this

两种模式都存在，但你应该问问

1588.880–1591.350  both nodes exist but you should ask this question and de discuss. Um it's kind of

两种模式都存在，但你应该提出这个问题，讨论一下。嗯，这有点

1591.350–1591.360  question and de discuss. Um it's kind of

提出这个问题，讨论一下。嗯，这有点

1591.360–1594.230  question and de discuss. Um it's kind of yeah hard to know all the specifics.

提出这个问题，讨论一下。嗯，所有细节确实很难弄清楚。

1594.230–1594.240  yeah hard to know all the specifics.

对，所有细节确实很难弄清楚。

1594.240–1597.029  yeah hard to know all the specifics. >> Yeah I feel I I feel I feel like the if

对，所有细节确实很难弄清楚。对，我觉得，如果

1597.029–1597.039  >> Yeah I feel I I feel I feel like the if

对，我觉得，如果

1597.039–1598.789  >> Yeah I feel I I feel I feel like the if we don't already have a knob to do this

对，我觉得，如果我们还没有这样的开关

1598.789–1598.799  we don't already have a knob to do this

如果我们还没有这样的开关

1598.799–1600.390  we don't already have a knob to do this this would be a knob that that the team

如果我们还没有这样的开关，团队应该会接受

1600.390–1600.400  this would be a knob that that the team

团队应该会接受这个开关

1600.400–1603.029  this would be a knob that that the team would likely receive. Well um because

团队应该会接受这个开关。嗯，因为

1603.029–1603.039  would likely receive. Well um because

应该会接受。嗯，因为

1603.039–1605.510  would likely receive. Well um because that that latency uh predictability is

应该会接受。嗯，因为延迟的可预测性

1605.510–1605.520  that that latency uh predictability is

因为延迟的可预测性

1605.520–1606.870  that that latency uh predictability is is pretty important to a lot of use

因为延迟的可预测性对很多使用场景

1606.870–1606.880  is pretty important to a lot of use

对很多使用场景都很重要

1606.880–1609.909  is pretty important to a lot of use cases. Okay. Um let's go ahead and move

对很多使用场景都很重要。好，我们继续

1609.909–1609.919  cases. Okay. Um let's go ahead and move

好，我们继续

1609.919–1613.029  cases. Okay. Um let's go ahead and move on to the to the next uh question in the

好，我们继续看下一个问题

1613.029–1613.039  on to the to the next uh question in the

看列表里的下一个问题

1613.039–1616.950  on to the to the next uh question in the list here. Um, how will dropping a CUDA

看列表里的下一个问题。取消提供 CUDA

1616.950–1616.960  list here. Um, how will dropping a CUDA

列表里的问题：取消提供 CUDA

1616.960–1620.470  list here. Um, how will dropping a CUDA 12 wheels in 2.15 affect downstream uh

列表里的问题：在 2.15 中取消提供 CUDA 12 的 wheel 包，会怎样影响下游

1620.470–1620.480  12 wheels in 2.15 affect downstream uh

在 2.15 中取消提供 CUDA 12 的 wheel 包，会怎样影响下游

1620.480–1624.310  12 wheels in 2.15 affect downstream uh C++ libr uh libraries uh or binaries for

在 2.15 中取消提供 CUDA 12 的 wheel 包，会怎样影响下游的 C++ 库或二进制文件

1624.310–1624.320  C++ libr uh libraries uh or binaries for

下游的 C++ 库或二进制文件

1624.320–1626.549  C++ libr uh libraries uh or binaries for for projects such as to sharp, but I

下游的 C++ 库或二进制文件，比如 TorchSharp 这样的项目

1626.549–1626.559  for projects such as to sharp, but I

比如 TorchSharp 这样的项目，不过我

1626.559–1629.190  for projects such as to sharp, but I know there's there's others. Um, Andre,

比如 TorchSharp 这样的项目，不过我知道还有其他项目。嗯，Andre，

1629.190–1629.200  know there's there's others. Um, Andre,

我知道还有其他项目。嗯，Andre，

1629.200–1630.390  know there's there's others. Um, Andre, do you want to take this one?

我知道还有其他项目。嗯，Andre，这个问题你来回答吗？

1630.390–1630.400  do you want to take this one?

这个问题你来回答吗？

1630.400–1632.470  do you want to take this one? >> Yeah. Yes. Uh, so basically I think

这个问题你来回答吗？对。嗯，基本上我觉得

1632.470–1632.480  >> Yeah. Yes. Uh, so basically I think

对。嗯，我觉得基本上

1632.480–1635.110  >> Yeah. Yes. Uh, so basically I think right now it would be good time for you

对。嗯，我觉得现在是个好时机

1635.110–1635.120  right now it would be good time for you

现在是个好时机

1635.120–1637.590  right now it would be good time for you to start planning to migrating for coder

现在是时候开始规划迁移到 CUDA 13 了

1637.590–1637.600  to start planning to migrating for coder

开始规划迁移到 CUDA

1637.600–1640.549  to start planning to migrating for coder 13 and probably the best version would

开始规划迁移到 CUDA 13，最佳版本可能是

1640.549–1640.559  13 and probably the best version would

13，最佳版本可能是

1640.559–1643.590  13 and probably the best version would be 13.2 two since this is going to be

13，最佳版本可能是 13.2.2，因为它将是

1643.590–1643.600  be 13.2 two since this is going to be

13.2.2，因为它将是

1643.600–1647.350  be 13.2 two since this is going to be our um stable version for the next 2.15

13.2.2，因为它将是下一个 2.15 版本的

1647.350–1647.360  our um stable version for the next 2.15

我们下一个 2.15 版本的稳定版

1647.360–1650.950  our um stable version for the next 2.15 release. Um and yeah, I think it's and

我们下一个 2.15 版本的稳定版。嗯，我觉得

1650.950–1650.960  release. Um and yeah, I think it's and

版本。嗯，我觉得

1650.960–1654.390  release. Um and yeah, I think it's and if you still want to use CUDA2.x, you

版本。嗯，如果你还想用 CUDA 2.x，

1654.390–1654.400  if you still want to use CUDA2.x, you

如果你还想用 CUDA 2.x，

1654.400–1656.950  if you still want to use CUDA2.x, you should be able to compile yourself from

如果你还想用 CUDA 2.x，应该可以自行从源码

1656.950–1656.960  should be able to compile yourself from

应该可以自行从源码编译

1656.960–1659.350  should be able to compile yourself from the source. So this is should still be

应该可以自行从源码编译。所以这应该仍然

1659.350–1659.360  the source. So this is should still be

从源码编译。所以这应该仍然

1659.360–1662.149  the source. So this is should still be supported. So you can take a look at how

从源码编译。所以这应该仍受支持。你可以看看

1662.149–1662.159  supported. So you can take a look at how

仍受支持。你可以看看

1662.159–1664.630  supported. So you can take a look at how we compile liptor binaries and basically

仍受支持。你可以看看我们如何编译 libtorch 二进制文件，然后

1664.630–1664.640  we compile liptor binaries and basically

我们如何编译 libtorch 二进制文件，然后

1664.640–1666.950  we compile liptor binaries and basically duplicate the workflow and compile it

我们如何编译 libtorch 二进制文件，再照着流程自己编译

1666.950–1666.960  duplicate the workflow and compile it

照着流程自己编译

1666.960–1670.310  duplicate the workflow and compile it yourself on your workers. um that that

照着流程在你的工作机上自行编译。这

1670.310–1670.320  yourself on your workers. um that that

在你的工作机上自行编译。这

1670.320–1672.710  yourself on your workers. um that that could be an option for short term like

在你的工作机上自行编译。短期内这

1672.710–1672.720  could be an option for short term like

短期内这也是个办法，比如

1672.720–1676.470  could be an option for short term like if you cannot migrate uh faster um I

短期内这也是个办法，比如你暂时没法更快地迁移。

1676.470–1676.480  if you cannot migrate uh faster um I

如果你暂时没法更快地迁移，我

1676.480–1678.070  if you cannot migrate uh faster um I don't know maybe Nikita wants to chime

如果你暂时没法更快地迁移，我也不确定，也许 Nikita 想

1678.070–1678.080  don't know maybe Nikita wants to chime

我也不确定，也许 Nikita 想

1678.080–1679.909  don't know maybe Nikita wants to chime in here any additional

我也不确定，也许 Nikita 想补充几句

1679.909–1679.919  in here any additional

补充几句

1679.919–1685.190  in here any additional >> um sure I guess like uh uh two

还有什么补充吗？嗯，可以，我想说两点

1685.190–1685.200  >> um sure I guess like uh uh two

嗯，可以，我想说两点

1685.200–1687.669  >> um sure I guess like uh uh two question like two two points one is that

嗯，可以，我想说两点。第一点是

1687.669–1687.679  question like two two points one is that

我想说两点。第一点是

1687.679–1690.149  question like two two points one is that uh like if you're on an older GPU

第一点是，如果你用的是较老的 GPU，

1690.149–1690.159  uh like if you're on an older GPU

如果你用的是较老的 GPU，

1690.159–1691.909  uh like if you're on an older GPU unfortunately CUDA 13 doesn't support

如果你用的是较老的 GPU，很遗憾，CUDA 13 不支持

1691.909–1691.919  unfortunately CUDA 13 doesn't support

很遗憾，CUDA 13 不支持

1691.919–1694.389  unfortunately CUDA 13 doesn't support some of them so you have to stay on the

很遗憾，CUDA 13 不支持其中一些型号，所以你得继续用

1694.389–1694.399  some of them so you have to stay on the

其中一些型号，所以你得继续用

1694.399–1696.470  some of them so you have to stay on the older PyTorch versions but also one

其中一些型号，所以你得继续用旧版 PyTorch。另外，

1696.470–1696.480  older PyTorch versions but also one

旧版 PyTorch。另外，

1696.480–1698.710  older PyTorch versions but also one should not expect any performance

旧版 PyTorch。另外，也别指望有任何性能

1698.710–1698.720  should not expect any performance

别指望有任何性能提升

1698.720–1703.029  should not expect any performance uh gains or uh like new features for

别指望有任何性能提升或新功能

1703.029–1703.039  uh gains or uh like new features for

旧硬件的性能提升或新功能……

1703.039–1704.870  uh gains or uh like new features for this older hardware with new releases.

新版本给旧硬件带来的性能提升或新功能。

1704.870–1704.880  this older hardware with new releases.

新版本为旧硬件提供的功能。

1704.880–1706.630  this older hardware with new releases. So like there is no reason for you to

新版本为旧硬件提供的功能。所以你没有理由……

1706.630–1706.640  So like there is no reason for you to

所以你没有理由……

1706.640–1709.750  So like there is no reason for you to upgrade. You can continue using 214 uh

所以你没必要升级，旧项目可以继续用 2.14。

1709.750–1709.760  upgrade. You can continue using 214 uh

升级。你可以继续用 2.14……

1709.760–1713.430  upgrade. You can continue using 214 uh for your older like pipelines

升级。旧的流水线可以继续用 2.14。

1713.430–1713.440  for your older like pipelines

用于你以前的流水线。

1713.440–1715.350  for your older like pipelines and TPU binaries will still be there,

用于你以前的流水线，TPU 二进制包也还在。

1715.350–1715.360  and TPU binaries will still be there,

TPU 二进制包也还在。

1715.360–1717.590  and TPU binaries will still be there, right? I don't know how much to sharp is

TPU 二进制包也还在，对吧？我不太清楚 TorchSharp 的情况。

1717.590–1717.600  right? I don't know how much to sharp is

对吧？我不太清楚 TorchSharp 的情况。

1717.600–1719.750  right? I don't know how much to sharp is indeed about like I guess it's most

对吧？我不太清楚 TorchSharp 的实际情况，我猜它主要……

1719.750–1719.760  indeed about like I guess it's most

实际上，我猜它主要……

1719.760–1721.909  indeed about like I guess it's most frequently used on Windows and I don't

实际上，我猜它主要用在 Windows 上，但我不……

1721.909–1721.919  frequently used on Windows and I don't

常用在 Windows 上，但我不……

1721.919–1726.630  frequently used on Windows and I don't know how much of that is like

常用在 Windows 上，但我不知道其中有多少……

1726.630–1726.640  know how much of that is like

不知道其中有多少……

1726.640–1729.110  know how much of that is like relevant to people on Windows like using

不知道这对 Windows 用户有多大关系，比如使用……

1729.110–1729.120  relevant to people on Windows like using

对使用这些功能的 Windows 用户有多大关系……

1729.120–1730.950  relevant to people on Windows like using blackville features like like VFP4

对使用 Blackwell 特性的 Windows 用户有多大关系，比如 NVFP4。

1730.950–1730.960  blackville features like like VFP4

Blackwell 的特性，比如 NVFP4。

1730.960–1735.110  blackville features like like VFP4 because like not available. So use 214

Blackwell 的特性，比如 NVFP4，因为这些功能还用不了。所以用 2.14。

1735.110–1735.120  because like not available. So use 214

因为这些功能还用不了，所以用 2.14。

1735.120–1739.350  because like not available. So use 214 but also yeah like CUDA 12 was released

因为这些功能还用不了，所以用 2.14。不过 CUDA 12 也已经发布……

1739.350–1739.360  but also yeah like CUDA 12 was released

不过 CUDA 12 也已经发布……

1739.360–1741.750  but also yeah like CUDA 12 was released five years ago if I'm not mistaken. So

不过，如果我没记错，CUDA 12 是五年前发布的。所以……

1741.750–1741.760  five years ago if I'm not mistaken. So

如果我没记错，是五年前发布的。所以……

1741.760–1743.590  five years ago if I'm not mistaken. So it's it's a good time to upgrade like

如果我没记错，是五年前发布的。所以现在也该升级了。

1743.590–1743.600  it's it's a good time to upgrade like

现在也该升级了。

1743.600–1745.110  it's it's a good time to upgrade like ask yourself why do you need to be on

现在也该升级了。想想自己为什么还需要用……

1745.110–1745.120  ask yourself why do you need to be on

想想自己为什么还需要用……

1745.120–1748.549  ask yourself why do you need to be on CUDA 12 in 2026 or like heading towards

想想为什么到了 2026 年，你还需要用 CUDA 12，甚至快到……

1748.549–1748.559  CUDA 12 in 2026 or like heading towards

到了 2026 年，甚至快到 2027 年，还在用 CUDA 12。

1748.559–1751.480  CUDA 12 in 2026 or like heading towards 2027

到了 2026 年，甚至快到 2027 年，还在用 CUDA 12。

1751.480–1751.490  2027

2027 年。

1751.490–1751.990  2027 [snorts]

2027 年。（哼笑）

1751.990–1752.000  [snorts]

（哼笑）

1752.000–1753.510  [snorts] >> though I' though I've known people who

（哼笑）不过，我也认识一些人……

1753.510–1753.520  >> though I' though I've known people who

不过，我也认识一些人……

1753.520–1755.110  >> though I' though I've known people who have stayed on old CUDA versions for a

不过，我也认识一些人，一直沿用旧版 CUDA……

1755.110–1755.120  have stayed on old CUDA versions for a

一直沿用旧版 CUDA……

1755.120–1758.630  have stayed on old CUDA versions for a long time but but yes

一直沿用旧版 CUDA 很多年。不过，确实。

1758.630–1758.640  long time but but yes

很多年。不过，确实。

1758.640–1760.310  long time but but yes okay

很多年。不过，确实。好。

1760.310–1760.320  okay

好。

1760.320–1763.269  okay next question

好，下一个问题。

1763.269–1763.279  next question

下一个问题。

1763.279–1765.110  next question um okay so this one is a simple one and

下一个问题。嗯，好，这个问题很简单……

1765.110–1765.120  um okay so this one is a simple one and

嗯，好，这个问题很简单，

1765.120–1768.070  um okay so this one is a simple one and this uh came from the the community uh

嗯，好，这个问题很简单，来自社区。

1768.070–1768.080  this uh came from the the community uh

这个问题来自社区。

1768.080–1770.070  this uh came from the the community uh questions ahead of time uh if you're on

这是社区提前提交的问题。如果你用的是……

1770.070–1770.080  questions ahead of time uh if you're on

提前提交的问题：如果你用的是……

1770.080–1775.190  questions ahead of time uh if you're on AMD what change for uh rockcomm in 2.14.

提前提交的问题：如果你用 AMD，ROCm 在 PyTorch 2.14 中有什么变化？

1775.190–1775.200  AMD what change for uh rockcomm in 2.14.

如果用 AMD，ROCm 在 PyTorch 2.14 中有什么变化？

1775.200–1777.190  AMD what change for uh rockcomm in 2.14. Andre, do you want to talk to this one?

如果用 AMD，ROCm 在 PyTorch 2.14 中有什么变化？Andre，你来回答好吗？

1777.190–1777.200  Andre, do you want to talk to this one?

Andre，你来回答好吗？

1777.200–1779.990  Andre, do you want to talk to this one? >> Yes. Yes. So, basically for romcom

Andre，你来回答好吗？好的。关于 ROCm，

1779.990–1780.000  >> Yes. Yes. So, basically for romcom

好的。关于 ROCm，

1780.000–1783.190  >> Yes. Yes. So, basically for romcom specifically, uh we removed the rocom

好的。具体来说，我们移除了 ROCm……

1783.190–1783.200  specifically, uh we removed the rocom

具体来说，我们移除了 ROCm……

1783.200–1787.110  specifically, uh we removed the rocom 7.1 support. Uh no wheels are no longer

具体来说，我们移除了对 ROCm 7.1 的支持，不再提供相应的安装包。

1787.110–1787.120  7.1 support. Uh no wheels are no longer

不再支持 ROCm 7.1，也不再提供相应的安装包。

1787.120–1792.549  7.1 support. Uh no wheels are no longer shipped with 214 PyTorch. So, um uh if

PyTorch 2.14 不再提供相应的安装包。所以，如果……

1792.549–1792.559  shipped with 214 PyTorch. So, um uh if

PyTorch 2.14 不再提供相应的安装包。所以，如果……

1792.559–1795.190  shipped with 214 PyTorch. So, um uh if you're on roam 7.1, I think you need to

PyTorch 2.14 不再提供相应的安装包。如果你用的是 ROCm 7.1，我想你需要……

1795.190–1795.200  you're on roam 7.1, I think you need to

如果你用的是 ROCm 7.1，我想你需要……

1795.200–1799.990  you're on roam 7.1, I think you need to migrate to either Rocomm 7.2 or 714. Uh

如果你用的是 ROCm 7.1，需要迁移到 ROCm 7.2 或 714。

1799.990–1800.000  migrate to either Rocomm 7.2 or 714. Uh

迁移到 ROCm 7.2 或 714。

1800.000–1802.310  migrate to either Rocomm 7.2 or 714. Uh those are two versions that are

迁移到 ROCm 7.2 或 714。这两个版本……

1802.310–1802.320  those are two versions that are

这两个版本……

1802.320–1806.630  those are two versions that are supported currently by 214 PyTorch. Um

这两个版本目前都受 PyTorch 2.14 支持。

1806.630–1806.640  supported currently by 214 PyTorch. Um

目前都受 PyTorch 2.14 支持。

1806.640–1810.230  supported currently by 214 PyTorch. Um 714 is the latest one and it is using uh

目前都受 PyTorch 2.14 支持。714 是最新版本，它使用……

1810.230–1810.240  714 is the latest one and it is using uh

714 是最新版本，它使用……

1810.240–1814.710  714 is the latest one and it is using uh the rock pdk 7.2 it's the same version

714 是最新版本，使用 ROCm PDK 7.2；它与……版本相同。

1814.710–1814.720  the rock pdk 7.2 it's the same version

ROCm PDK 7.2，与……版本相同。

1814.720–1816.789  the rock pdk 7.2 it's the same version that we also shipped with previous

ROCm PDK 7.2，与我们上一版……所用的版本相同。

1816.789–1816.799  that we also shipped with previous

与我们上一版……所用的版本相同。

1816.799–1819.669  that we also shipped with previous release of torch. So it's uh more

与上一版 PyTorch 所用的版本相同，所以它更……

1819.669–1819.679  release of torch. So it's uh more

与上一版 PyTorch 所用的版本相同，所以它更……

1819.679–1822.789  release of torch. So it's uh more similar to what rocom 7.1 was. So I

与上一版 PyTorch 所用的版本相同，所以更接近 ROCm 7.1。

1822.789–1822.799  similar to what rocom 7.1 was. So I

更接近 ROCm 7.1。所以我……

1822.799–1824.950  similar to what rocom 7.1 was. So I think you so you have a choice

更接近 ROCm 7.1。所以我觉得你可以选择。

1824.950–1824.960  think you so you have a choice

我觉得你可以选择。

1824.960–1826.870  think you so you have a choice basically. I think the best way is to

总之，你可以选择。我觉得最好……

1826.870–1826.880  basically. I think the best way is to

总之，我觉得最好……

1826.880–1831.190  basically. I think the best way is to migrate straight to 714. Um but yeah, I

总之，我觉得最好直接迁移到 714。不过……

1831.190–1831.200  migrate straight to 714. Um but yeah, I

直接迁移到 714。不过……

1831.200–1835.321  migrate straight to 714. Um but yeah, I think that's about it. Yes.

直接迁移到 714。嗯，我想就这些。

1835.321–1835.331  think that's about it. Yes.

我想就这些。对。

1835.331–1837.510  think that's about it. Yes. [snorts]

我想就这些。对。［轻哼］

1837.830–1837.840  >> Okay.

好的。

1837.840–1839.590  >> Okay. >> And there are some advanced features

好的。另外还有一些高级功能……

1839.590–1839.600  >> And there are some advanced features

另外还有一些高级功能……

1839.600–1842.549  >> And there are some advanced features which are available in 714 like he file

另外，714 还提供一些高级功能，比如 HIP 文件……

1842.549–1842.559  which are available in 714 like he file

714 提供的功能，比如 HIP 文件……

1842.559–1844.870  which are available in 714 like he file which is similar to coup file like

714 提供的功能，比如 HIP 文件，类似于 CUDA 文件……

1844.870–1844.880  which is similar to coup file like

这类似于 cuFile，比如

1844.880–1847.990  which is similar to coup file like direct storage access from GPUs and you

这类似于 cuFile，比如 GPU 直接访问存储，你还

1847.990–1848.000  direct storage access from GPUs and you

GPU 直接访问存储，你还

1848.000–1850.470  direct storage access from GPUs and you can use more hip solver functions for

GPU 直接访问存储，你还可以使用更多 hipSOLVER 函数，比如

1850.470–1850.480  can use more hip solver functions for

可以使用更多 hipSOLVER 函数，比如

1850.480–1854.630  can use more hip solver functions for example torch and alf uses that one.

可以使用更多 hipSOLVER 函数，比如 Torch 和 ALF 就用到了它。

1854.630–1854.640  example torch and alf uses that one.

比如 Torch 和 ALF 就用到了它。

1854.640–1856.950  example torch and alf uses that one. >> Yes. And the packaging is actually a lot

比如 Torch 和 ALF 就用到了它。>> 对，而且打包方式也改进了很多

1856.950–1856.960  >> Yes. And the packaging is actually a lot

>> 对，而且打包方式也改进了很多

1856.960–1863.269  >> Yes. And the packaging is actually a lot better and a lot more modern for 714.

>> 对，714 的打包方式改进了很多，也更现代化了。

1865.029–1865.039  >> Okay.

>> 好的。

1865.039–1867.830  >> Okay. Um I think our next question is going to

>> 好的。我想下一个问题会

1867.830–1867.840  Um I think our next question is going to

我想下一个问题会

1867.840–1869.750  Um I think our next question is going to be kind of a little bit of a broader

我想下一个问题会稍微宽泛一些

1869.750–1869.760  be kind of a little bit of a broader

稍微宽泛一些

1869.760–1872.950  be kind of a little bit of a broader question. Um uh so it seems like the

是个稍微宽泛一些的问题。看起来

1872.950–1872.960  question. Um uh so it seems like the

是个问题。看起来

1872.960–1875.830  question. Um uh so it seems like the diversity in the Okay, here we are. Uh

是个问题。看起来多样性……好，找到了。

1875.830–1875.840  diversity in the Okay, here we are. Uh

多样性……好，找到了。

1875.840–1877.190  diversity in the Okay, here we are. Uh it seems like the diversity in hardware

多样性……好，找到了。最近硬件种类似乎

1877.190–1877.200  it seems like the diversity in hardware

最近硬件种类似乎

1877.200–1879.590  it seems like the diversity in hardware in the AI world is exploding lately. At

最近 AI 领域的硬件种类似乎在迅速增加。至少

1879.590–1879.600  in the AI world is exploding lately. At

AI 领域的硬件种类似乎在迅速增加。至少

1879.600–1882.470  in the AI world is exploding lately. At least that's what I see on on on uh uh

AI 领域的硬件种类似乎在迅速增加。至少我看到的是这样，

1882.470–1882.480  least that's what I see on on on uh uh

至少我看到的是这样，

1882.480–1886.470  least that's what I see on on on uh uh in my news passing by. uh my uh screen

至少我每天刷到的新闻给我这种感觉，屏幕上

1886.470–1886.480  in my news passing by. uh my uh screen

我刷到的新闻，屏幕上

1886.480–1888.549  in my news passing by. uh my uh screen every day. Uh how do we think about

每天都能看到。我们该如何看待

1888.549–1888.559  every day. Uh how do we think about

每天都能看到。我们该如何看待

1888.559–1891.269  every day. Uh how do we think about platform support um within the PyTorch

每天都能看到。我们该如何看待 PyTorch 社区的

1891.269–1891.279  platform support um within the PyTorch

PyTorch 社区的平台支持

1891.279–1893.350  platform support um within the PyTorch community? I'd love to hear from from

PyTorch 社区的平台支持？我想先听听

1893.350–1893.360  community? I'd love to hear from from

社区的平台支持？我想先听听

1893.360–1895.590  community? I'd love to hear from from Joe on this and then I'm sure Andrea and

社区的平台支持？我想先听听 Joe 的看法，然后 Andrea 和

1895.590–1895.600  Joe on this and then I'm sure Andrea and

Joe 的看法，然后 Andrea 和

1895.600–1897.190  Joe on this and then I'm sure Andrea and Nikita may also want to chime in with

Joe 的看法，然后 Andrea 和 Nikita 可能也想补充

1897.190–1897.200  Nikita may also want to chime in with

Nikita 可能也想补充

1897.200–1898.149  Nikita may also want to chime in with details.

Nikita 可能也想补充一些细节。

1898.149–1898.159  details.

一些细节。

1898.159–1899.669  details. >> Yeah,

一些细节。>> 对，

1899.669–1899.679  >> Yeah,

>> 对，

1899.679–1902.549  >> Yeah, >> I mean we know we've been I'm coming at

>> 对，>> 我是说，我们知道我们一直……我从

1902.549–1902.559  >> I mean we know we've been I'm coming at

>> 我是说，我们知道我们一直……我从

1902.559–1905.430  >> I mean we know we've been I'm coming at Harbor 180 for a long time with PyTorch.

>> 我是说，我们知道我们一直……我从 Harbor 180 的角度关注 PyTorch 已经很久了。

1905.430–1905.440  Harbor 180 for a long time with PyTorch.

从 Harbor 180 的角度关注 PyTorch 已经很久了。

1905.440–1907.190  Harbor 180 for a long time with PyTorch. I think we've been trying uh to support

从 Harbor 180 的角度关注 PyTorch 已经很久了。我想我们一直在努力支持

1907.190–1907.200  I think we've been trying uh to support

我想我们一直在努力支持

1907.200–1909.110  I think we've been trying uh to support a lot of different backends. We there's

我想我们一直在努力支持许多不同的后端。基金会里还有

1909.110–1909.120  a lot of different backends. We there's

许多不同的后端。基金会里还有

1909.120–1911.750  a lot of different backends. We there's working groups in the in the foundation.

许多不同的后端。基金会里还有相关工作组。

1911.750–1911.760  working groups in the in the foundation.

基金会里的工作组。

1911.760–1915.750  working groups in the in the foundation. I think you know in some ways like you

基金会里的工作组。我觉得，从某些方面来说，你……

1915.750–1915.760  I think you know in some ways like you

我觉得，从某些方面来说，你……

1915.760–1917.110  I think you know in some ways like you know there's a couple different angles.

我觉得这件事可以从几个不同角度来看。

1917.110–1917.120  know there's a couple different angles.

这件事可以从几个不同角度来看。

1917.120–1918.549  know there's a couple different angles. One obviously we wanted to have a

这件事可以从几个不同角度来看。首先，我们显然希望有……

1918.549–1918.559  One obviously we wanted to have a

首先，我们显然希望有……

1918.559–1921.190  One obviously we wanted to have a diverse set of of hardware vendors and

多元的硬件厂商，以及……

1921.190–1921.200  diverse set of of hardware vendors and

多元的硬件厂商，以及……

1921.200–1922.870  diverse set of of hardware vendors and and a hardware ecosystem to give choice

多元的硬件厂商和硬件生态，让开发者有更多选择。

1922.870–1922.880  and a hardware ecosystem to give choice

还有一个硬件生态，让开发者有更多选择。

1922.880–1925.269  and a hardware ecosystem to give choice to developers because not just at the

还有一个硬件生态，让开发者有更多选择，因为这不只是云端的事。

1925.269–1925.279  to developers because not just at the

让开发者有更多选择，因为这不只是云端的事。

1925.279–1926.389  to developers because not just at the cloud. I think everyone sort of thinks

让开发者有更多选择，因为这不只是云端的事。我觉得大家往往认为……

1926.389–1926.399  cloud. I think everyone sort of thinks

云端。我觉得大家往往认为……

1926.399–1927.909  cloud. I think everyone sort of thinks of like the cloud side is like the

云端。我觉得大家往往认为，云端才是……

1927.909–1927.919  of like the cloud side is like the

云端才是……

1927.919–1929.909  of like the cloud side is like the default but like running on device

云端才是默认选择，但模型也可以在设备上运行。

1929.909–1929.919  default but like running on device

默认选择，但模型也可以在设备上运行。

1929.919–1931.909  default but like running on device running you know um within wearables

默认选择，但模型也可以在设备上、在可穿戴设备中运行。

1931.909–1931.919  running you know um within wearables

也可以在可穿戴设备中运行。

1931.919–1934.310  running you know um within wearables like there's there's like a lot going on

也可以在可穿戴设备中运行。这里面有很多变化。

1934.310–1934.320  like there's there's like a lot going on

这里面有很多变化。

1934.320–1935.909  like there's there's like a lot going on in different hardware architectures and

不同的硬件架构都有很多发展。

1935.909–1935.919  in different hardware architectures and

不同的硬件架构，以及……

1935.919–1938.070  in different hardware architectures and that's why this like compiler and kernel

不同的硬件架构，这也让编译器和内核……

1938.070–1938.080  that's why this like compiler and kernel

这也让编译器和内核……

1938.080–1940.470  that's why this like compiler and kernel ecosystems is so diverse. I think that's

这也让编译器和内核生态如此多样。我觉得……

1940.470–1940.480  ecosystems is so diverse. I think that's

生态如此多样。我觉得……

1940.480–1942.549  ecosystems is so diverse. I think that's like one piece. I think pragmatically

生态如此多样。我觉得这是一个原因。实际来看……

1942.549–1942.559  like one piece. I think pragmatically

这是一个原因。实际来看……

1942.559–1945.029  like one piece. I think pragmatically though like like compute is scarce these

这是一个原因。不过现实是，如今算力紧缺。

1945.029–1945.039  though like like compute is scarce these

不过现实是，如今算力紧缺。

1945.039–1947.750  though like like compute is scarce these days and you know being able to run on

不过现实是，如今算力紧缺，能够在不同平台上运行……

1947.750–1947.760  days and you know being able to run on

如今算力紧缺，能够在不同平台上运行……

1947.760–1949.430  days and you know being able to run on Nvidia, being able to run on AMD, being

如今算力紧缺，能用 Nvidia，也能用 AMD，还能……

1949.430–1949.440  Nvidia, being able to run on AMD, being

能用 Nvidia，也能用 AMD，还能……

1949.440–1950.950  Nvidia, being able to run on AMD, being able to run on TPUs. I think it's

能用 Nvidia，也能用 AMD，还能用 TPU。我觉得……

1950.950–1950.960  able to run on TPUs. I think it's

还能用 TPU。我觉得……

1950.960–1954.789  able to run on TPUs. I think it's actually um you know things have like

还能用 TPU。我觉得，其实情况是……

1954.789–1954.799  actually um you know things have like

其实情况是……

1954.799–1956.710  actually um you know things have like you know big labs or folks that that

其实，大型实验室或者其他……

1956.710–1956.720  you know big labs or folks that that

大型实验室或者其他……

1956.720–1958.950  you know big labs or folks that that need a lot of compute have have had to

大型实验室，或者需要大量算力的人，都不得不……

1958.950–1958.960  need a lot of compute have have had to

需要大量算力的人，都不得不……

1958.960–1961.350  need a lot of compute have have had to learn how to run on on you know a lot of

需要大量算力的人，都不得不学会在多种平台上运行。

1961.350–1961.360  learn how to run on on you know a lot of

学会在多种平台上运行。

1961.360–1963.190  learn how to run on on you know a lot of different architectures. I think like if

学会在多种不同架构上运行。我觉得，如果……

1963.190–1963.200  different architectures. I think like if

不同的架构。我觉得，如果……

1963.200–1964.950  different architectures. I think like if I look at anthropic they run on Nvidia,

不同的架构。比如 Anthropic，他们会用 Nvidia，

1964.950–1964.960  I look at anthropic they run on Nvidia,

我看 Anthropic，他们用 NVIDIA。

1964.960–1967.669  I look at anthropic they run on Nvidia, they run on Tranium, they run on TPUs.

我看 Anthropic，他们用 NVIDIA，也用 Trainium 和 TPU。

1967.669–1967.679  they run on Tranium, they run on TPUs.

他们用 Trainium，也用 TPU。

1967.679–1970.630  they run on Tranium, they run on TPUs. Um even in like different workflows like

他们用 Trainium，也用 TPU。嗯，甚至在不同的工作流程里，比如……

1970.630–1970.640  Um even in like different workflows like

嗯，甚至在不同的工作流程里，比如……

1970.640–1971.909  Um even in like different workflows like I know we're talking a lot about

嗯，甚至在不同的工作流程里，比如……我知道我们一直在谈……

1971.909–1971.919  I know we're talking a lot about

我知道我们一直在谈……

1971.919–1974.950  I know we're talking a lot about training typically but you know training

我知道我们通常主要在谈训练，但你看，训练……

1974.950–1974.960  training typically but you know training

通常主要在谈训练，但你看，训练……

1974.960–1976.710  training typically but you know training RL like we're talking about running on

通常主要在谈训练，但做强化学习时，我们说的是运行在……

1976.710–1976.720  RL like we're talking about running on

比如强化学习，我们说的是运行在……

1976.720–1979.269  RL like we're talking about running on GPUs. We run environments in CPUs. So

比如强化学习，模型跑在 GPU 上，环境跑在 CPU 上。所以……

1979.269–1979.279  GPUs. We run environments in CPUs. So

模型跑在 GPU 上，环境跑在 CPU 上。所以……

1979.279–1980.789  GPUs. We run environments in CPUs. So all of a sudden you see the Intel stock

模型跑在 GPU 上，环境跑在 CPU 上。所以最近你突然看到英特尔股价……

1980.789–1980.799  all of a sudden you see the Intel stock

你突然看到英特尔股价……

1980.799–1982.389  all of a sudden you see the Intel stock up these days because they're selling a

你突然看到英特尔股价最近上涨，因为他们卖出了……

1982.389–1982.399  up these days because they're selling a

最近上涨，因为他们卖出了……

1982.399–1985.590  up these days because they're selling a lot more CPUs for RL which is good. Um

最近上涨，因为他们为强化学习卖出了更多 CPU，这是好事。嗯……

1985.590–1985.600  lot more CPUs for RL which is good. Um

为强化学习卖出了更多 CPU，这是好事。嗯……

1985.600–1988.149  lot more CPUs for RL which is good. Um you know when you you run reinforcement

为强化学习卖出了更多 CPU，这是好事。嗯，你知道，运行强化……

1988.149–1988.159  you know when you you run reinforcement

你知道，运行强化……

1988.159–1989.110  you know when you you run reinforcement learning you're running a lot of

你知道，运行强化学习时，会进行大量……

1989.110–1989.120  learning you're running a lot of

学习时，会进行大量……

1989.120–1991.350  learning you're running a lot of inference. um at inference time you're

学习时，会进行大量推理。嗯，在推理阶段，你会……

1991.350–1991.360  inference. um at inference time you're

推理。嗯，在推理阶段，你会……

1991.360–1993.029  inference. um at inference time you're running a ton of inference obviously

推理。嗯，在推理阶段，你显然要进行海量推理。

1993.029–1993.039  running a ton of inference obviously

显然要进行海量推理。

1993.039–1995.190  running a ton of inference obviously like you're doing a lot of compute time

显然要进行海量推理，也就是要投入大量计算资源……

1995.190–1995.200  like you're doing a lot of compute time

也就是要投入大量计算资源……

1995.200–1997.110  like you're doing a lot of compute time inference time compute um scaling that

比如推理阶段的计算资源，要在……

1997.110–1997.120  inference time compute um scaling that

推理阶段的计算资源，要在……

1997.120–1999.430  inference time compute um scaling that up in different axes um and so I think

推理阶段的计算资源，要从不同维度扩大规模。所以我觉得……

1999.430–1999.440  up in different axes um and so I think

从不同维度扩大规模。所以我觉得……

1999.440–2001.669  up in different axes um and so I think like you know you might run that on on

从不同维度扩大规模。所以我觉得，你可能会把它跑在……

2001.669–2001.679  like you know you might run that on on

你可能会把它跑在……

2001.679–2003.350  like you know you might run that on on different architecture you might run AMD

你可能会把它跑在不同的架构上，比如 AMD……

2003.350–2003.360  different architecture you might run AMD

不同的架构，比如 AMD……

2003.360–2004.789  different architecture you might run AMD you might run on on on different things

不同的架构，比如 AMD，也可能跑在其他硬件上……

2004.789–2004.799  you might run on on on different things

也可能跑在其他硬件上……

2004.799–2006.310  you might run on on on different things so I think the hardware diversity is

也可能跑在其他硬件上。所以我认为，硬件多样性……

2006.310–2006.320  so I think the hardware diversity is

所以我认为，硬件多样性……

2006.320–2008.230  so I think the hardware diversity is like is great for developers and I think

所以我认为，硬件多样性对开发者很有利，而且我觉得……

2008.230–2008.240  like is great for developers and I think

对开发者很有利，而且我觉得……

2008.240–2010.549  like is great for developers and I think that even going back like a year or

对开发者很有利，而且我觉得，即使回到大约一年前……

2010.549–2010.559  that even going back like a year or

即使回到大约一年前……

2010.559–2012.549  that even going back like a year or longer we've been talking about this and

即使回到一年前甚至更早，我们就在谈这个……

2012.549–2012.559  longer we've been talking about this and

甚至更早，我们就在谈这个……

2012.559–2014.470  longer we've been talking about this and um for PyTorch even back in we've been

甚至更早，我们就在谈这个。嗯，PyTorch 方面，我们早在……

2014.470–2014.480  um for PyTorch even back in we've been

嗯，PyTorch 方面，我们早在……

2014.480–2017.590  um for PyTorch even back in we've been working with AMD since what 2018 um I'm

嗯，PyTorch 方面，我们从 2018 年左右就开始和 AMD 合作了，我……

2017.590–2017.600  working with AMD since what 2018 um I'm

大概从2018年起就和AMD合作，嗯，我……

2017.600–2019.750  working with AMD since what 2018 um I'm so you remember those days right So, I I

大概从2018年起就和AMD合作。嗯，你还记得那时候吧？所以我……

2019.750–2019.760  so you remember those days right So, I I

你还记得那时候吧？所以我……

2019.760–2021.110  so you remember those days right So, I I think it's it's I I wouldn't even say

你还记得那时候吧？所以我觉得，甚至不能说……

2021.110–2021.120  think it's it's I I wouldn't even say

我觉得，甚至不能说……

2021.120–2022.870  think it's it's I I wouldn't even say it's exploding lately. I actually think

我甚至不能说它是最近才爆发的。其实我觉得……

2022.870–2022.880  it's exploding lately. I actually think

它是最近才爆发的。其实我觉得……

2022.880–2024.710  it's exploding lately. I actually think it's been exploding for a long time and

它早就蓬勃发展了，而且……

2024.710–2024.720  it's been exploding for a long time and

它早就蓬勃发展了，而且……

2024.720–2027.750  it's been exploding for a long time and I think um we just continue to grow and

它早就蓬勃发展了，而且我觉得我们还在持续增长……

2027.750–2027.760  I think um we just continue to grow and

我觉得我们还在持续增长，而且……

2027.760–2029.029  I think um we just continue to grow and obviously Nvidia gets a lot of the

我觉得我们还在持续增长。当然，Nvidia吸引了很多……

2029.029–2029.039  obviously Nvidia gets a lot of the

当然，Nvidia吸引了很多……

2029.039–2031.509  obviously Nvidia gets a lot of the headlines as they as they well deserve.

当然，Nvidia占据了很多头条，这也是实至名归。

2031.509–2031.519  headlines as they as they well deserve.

占据头条也是他们实至名归。

2031.519–2032.950  headlines as they as they well deserve. Um but there's definitely a lot of

占据头条也是他们实至名归。不过，确实还有很多……

2032.950–2032.960  Um but there's definitely a lot of

不过，确实还有很多……

2032.960–2034.389  Um but there's definitely a lot of architectures out there that are are

不过，确实还需要很多其他架构……

2034.389–2034.399  architectures out there that are are

还有很多其他架构……

2034.399–2037.909  architectures out there that are are needed and um and it's just going to

这些架构都有其用武之地，而且还会……

2037.909–2037.919  needed and um and it's just going to

这些架构都有其用武之地，而且还会……

2037.919–2039.350  needed and um and it's just going to continue to kind of grow in different

这些架构都有其用武之地，而且会继续向不同方向发展。

2039.350–2039.360  continue to kind of grow in different

继续向不同方向发展。

2039.360–2041.669  continue to kind of grow in different directions. Um, and I think we're going

继续向不同方向发展。我想我们会……

2041.669–2041.679  directions. Um, and I think we're going

向不同方向发展。我想我们会……

2041.679–2043.590  directions. Um, and I think we're going to get inference is obviously much more

向不同方向发展。我想，推理显然越来越……

2043.590–2043.600  to get inference is obviously much more

推理显然越来越……

2043.600–2046.070  to get inference is obviously much more important these days for many. Um, and

如今对很多人来说，推理显然重要得多。而且……

2046.070–2046.080  important these days for many. Um, and

如今对很多人来说，这重要得多。而且……

2046.080–2047.830  important these days for many. Um, and it has to be much more efficient to be

如今对很多人来说，这重要得多，而且它必须更高效，才能……

2047.830–2047.840  it has to be much more efficient to be

它必须更高效，才能……

2047.840–2049.430  it has to be much more efficient to be able to scale and that's just going to

它必须更高效，才能实现规模化，而这又会……

2049.430–2049.440  able to scale and that's just going to

实现规模化，而这又会……

2049.440–2050.550  able to scale and that's just going to necessitate different types of

实现规模化，而这又需要不同类型的……

2050.550–2050.560  necessitate different types of

需要不同类型的……

2050.560–2052.149  necessitate different types of architectures. Disagregated inference

需要不同类型的架构。比如分离式推理……

2052.149–2052.159  architectures. Disagregated inference

架构。比如分离式推理……

2052.159–2053.430  architectures. Disagregated inference for example where you separate pre-fill

比如分离式推理，就是把预填充……

2053.430–2053.440  for example where you separate pre-fill

比如把预填充……

2053.440–2055.909  for example where you separate pre-fill and decode um is a major trend right now

比如把预填充和解码分开，这是目前的一大趋势。

2055.909–2055.919  and decode um is a major trend right now

把解码分开，是目前的一大趋势。

2055.919–2056.950  and decode um is a major trend right now and has been for the last several

把解码分开，是目前的一大趋势，而且过去几个月一直如此。

2056.950–2056.960  and has been for the last several

过去几个月一直如此。

2056.960–2059.270  and has been for the last several months. Um, so I think that's it's going

过去几个月一直如此。所以我觉得，这方面还会……

2059.270–2059.280  months. Um, so I think that's it's going

所以我觉得，这方面还会……

2059.280–2060.950  months. Um, so I think that's it's going to continue to innovate obviously where

所以我觉得，显然还会继续创新……

2060.950–2060.960  to continue to innovate obviously where

显然还会继续创新……

2060.960–2062.790  to continue to innovate obviously where the workloads are going to be. So would

显然会随着工作负载的发展继续创新。所以我也想……

2062.790–2062.800  the workloads are going to be. So would

随着工作负载的发展继续创新。所以我也想……

2062.800–2065.909  the workloads are going to be. So would love others to to chime in though.

不过，我也很想听听其他人的看法。

2065.909–2065.919  love others to to chime in though.

不过也想听听其他人的看法。

2065.919–2070.230  love others to to chime in though. >> Yeah. Um, Nikita and Andre, do you want

不过也想听听其他人的看法。对，Nikita 和 Andre，你们想……

2070.230–2070.240  >> Yeah. Um, Nikita and Andre, do you want

对，Nikita 和 Andre，你们想……

2070.240–2071.990  >> Yeah. Um, Nikita and Andre, do you want to say anything about distributed or

对，Nikita 和 Andre，你们想谈谈分布式，或者……

2071.990–2072.000  to say anything about distributed or

谈谈分布式，或者……

2072.000–2074.710  to say anything about distributed or sorry about uh heterogeneous hardware?

谈谈分布式，抱歉，是异构硬件？

2074.710–2074.720  sorry about uh heterogeneous hardware?

抱歉，是异构硬件？

2074.720–2077.990  sorry about uh heterogeneous hardware? >> Uh, sure. Um, yes. Uh, basically on

抱歉，是异构硬件？可以。基本上……

2077.990–2078.000  >> Uh, sure. Um, yes. Uh, basically on

可以。基本上……

2078.000–2080.230  >> Uh, sure. Um, yes. Uh, basically on PyTorch, PyTorch specifically, we've

可以。具体到 PyTorch，我们……

2080.230–2080.240  PyTorch, PyTorch specifically, we've

具体到 PyTorch，我们……

2080.240–2083.829  PyTorch, PyTorch specifically, we've been working on the project called CRCR,

具体到 PyTorch，我们一直在做一个叫 CRCR 的项目。

2083.829–2083.839  been working on the project called CRCR,

一直在做一个叫 CRCR 的项目。

2083.839–2087.030  been working on the project called CRCR, it's a cross repository CI relay. So, it

一直在做一个叫 CRCR 的项目，它是一个跨仓库 CI 中继。

2087.030–2087.040  it's a cross repository CI relay. So, it

它是一个跨仓库 CI 中继，可以让……

2087.040–2089.349  it's a cross repository CI relay. So, it lets partner back ends to run their own

它是一个跨仓库 CI 中继，可以让合作伙伴的后端自行运行……

2089.349–2089.359  lets partner back ends to run their own

让合作伙伴的后端自行运行……

2089.359–2092.470  lets partner back ends to run their own CI against PyTorch changes. So on the

让合作伙伴的后端针对 PyTorch 的改动运行自己的 CI。

2092.470–2092.480  CI against PyTorch changes. So on the

针对 PyTorch 的改动运行 CI，然后……

2092.480–2096.470  CI against PyTorch changes. So on the PyTorch PRS and report result back to us

针对 PyTorch 的 PR 运行 CI，并把结果反馈给我们。

2096.470–2096.480  PyTorch PRS and report result back to us

针对 PyTorch 的 PR，并把结果反馈给我们。

2096.480–2100.310  PyTorch PRS and report result back to us um on the PyTorch HUD page. Uh so we

针对 PyTorch 的 PR，并把结果反馈到 PyTorch HUD 页面。

2100.310–2100.320  um on the PyTorch HUD page. Uh so we

在 PyTorch HUD 页面上。所以我们……

2100.320–2103.430  um on the PyTorch HUD page. Uh so we developed this with the partners um in

在 PyTorch HUD 页面上。过去几个月，我们与合作伙伴……

2103.430–2103.440  developed this with the partners um in

过去几个月，我们与合作伙伴……

2103.440–2106.550  developed this with the partners um in the past few months and basically it's

过去几个月，我们与合作伙伴一起开发了它，基本上……

2106.550–2106.560  the past few months and basically it's

过去几个月开发了它，基本上……

2106.560–2109.349  the past few months and basically it's pretty much released right now for this

过去几个月开发了它，现在基本已经发布，用于这次……

2109.349–2109.359  pretty much released right now for this

现在基本已经发布，用于这次……

2109.359–2114.230  pretty much released right now for this um 2015 um 2014 release and this is a

现在基本已经发布，用于这次 2015、2014 版本发布。

2114.230–2114.240  um 2015 um 2014 release and this is a

用于 2015、2014 版本发布。这也是……

2114.240–2116.310  um 2015 um 2014 release and this is a new feature in the Hut. So if you go to

用于 2015、2014 版本发布，也是 HUD 的一项新功能。如果你……

2116.310–2116.320  new feature in the Hut. So if you go to

这是 HUD 的一项新功能。如果你……

2116.320–2119.910  new feature in the Hut. So if you go to our hot page, there's the CRCR

这是 HUD 的一项新功能。如果你打开我们的 HUD 页面，会看到 CRCR……

2119.910–2119.920  our hot page, there's the CRCR

打开我们的 HUD 页面，会看到 CRCR……

2119.920–2122.550  our hot page, there's the CRCR button. You can go and access this to

打开我们的 HUD 页面，会看到 CRCR 按钮。你可以点击它……

2122.550–2122.560  button. You can go and access this to

点击这个按钮，你就可以……

2122.560–2125.190  button. You can go and access this to have a like a preview. Uh so basically

点击这个按钮，就可以预览一下。基本上……

2125.190–2125.200  have a like a preview. Uh so basically

可以预览一下。基本上……

2125.200–2127.990  have a like a preview. Uh so basically it's the way it's um configured. It's um

可以预览一下。它的配置方式是……

2127.990–2128.000  it's the way it's um configured. It's um

它的配置方式是……

2128.000–2130.790  it's the way it's um configured. It's um the four levels of access. So level one

它的配置方式分为四个接入级别。第一级……

2130.790–2130.800  the four levels of access. So level one

分为四个接入级别。第一级……

2130.800–2133.670  the four levels of access. So level one is on boarding. So events are forwarded

分为四个接入级别。第一级是接入阶段，事件会转发……

2133.670–2133.680  is on boarding. So events are forwarded

第一级是接入阶段，事件会转发……

2133.680–2138.150  is on boarding. So events are forwarded to downstream uh repositories and um

第一级是接入阶段，事件会转发到下游仓库。

2138.150–2138.160  to downstream uh repositories and um

转发到下游仓库，而……

2138.160–2140.630  to downstream uh repositories and um upstream receive no feedback. So PyTorch

转发到下游仓库，上游不会收到反馈。所以 PyTorch……

2140.630–2140.640  upstream receive no feedback. So PyTorch

上游不会收到反馈。所以 PyTorch……

2140.640–2143.430  upstream receive no feedback. So PyTorch we don't know what our

上游不会收到反馈。所以在 PyTorch 这边，我们不知道我们的……

2143.430–2143.440  we don't know what our

我们不知道我们的……

2143.440–2146.790  we don't know what our tests are running on the downstream

我们不知道我们的测试在下游仓库中的运行情况。

2146.790–2146.800  tests are running on the downstream

测试正在下游仓库运行。

2146.800–2150.150  tests are running on the downstream repos but tests are happening. So L2 is

测试正在下游仓库运行，但我们知道测试确实在进行。所以 L2 是……

2150.150–2150.160  repos but tests are happening. So L2 is

仓库，但测试确实在进行。所以 L2 是……

2150.160–2153.109  repos but tests are happening. So L2 is observation. It's when we actually uh

仓库，但测试确实在进行。所以 L2 是观察阶段，也就是我们真正……

2153.109–2153.119  observation. It's when we actually uh

观察阶段，也就是我们真正……

2153.119–2156.470  observation. It's when we actually uh PyTorch receives the results

观察阶段，也就是 PyTorch 真正收到结果的时候。

2156.470–2156.480  PyTorch receives the results

PyTorch 收到结果。

2156.480–2160.069  PyTorch receives the results uh of the downstream um tests and able

PyTorch 收到下游测试的结果，并且能够……

2160.069–2160.079  uh of the downstream um tests and able

下游测试的结果，并且能够……

2160.079–2162.550  uh of the downstream um tests and able to visualize and see what are the

下游测试的结果，并且能够直观看到有哪些……

2162.550–2162.560  to visualize and see what are the

直观看到有哪些……

2162.560–2165.589  to visualize and see what are the failures and this helps like um uh

直观看到有哪些失败，这有助于……

2165.589–2165.599  failures and this helps like um uh

失败，这有助于……

2165.599–2171.190  failures and this helps like um uh observation uh and uh um L3 it's stable

失败，这有助于观察。L3 是稳定阶段。

2171.190–2171.200  observation uh and uh um L3 it's stable

观察，而 L3 是稳定阶段。

2171.200–2175.109  observation uh and uh um L3 it's stable um it's where we are currently at um

观察，而 L3 是稳定阶段。我们目前就处于这个阶段。

2175.109–2175.119  um it's where we are currently at um

我们目前就处于这个阶段。

2175.119–2178.069  um it's where we are currently at um it's non-blocking run check on PR so

我们目前就处于这个阶段。PR 上的检查不会阻断流程，所以……

2178.069–2178.079  it's non-blocking run check on PR so

PR 上的检查不会阻断流程，所以……

2178.079–2181.670  it's non-blocking run check on PR so it's basically you apply um flow out of

PR 上的检查不会阻断流程，所以基本上就是将……

2181.670–2181.680  it's basically you apply um flow out of

基本上就是将……

2181.680–2185.030  it's basically you apply um flow out of tree lab out CR flow CRCR label to the

基本上就是把 flow out of tree lab out CR flow CRCR 标签加到……

2185.030–2185.040  tree lab out CR flow CRCR label to the

把 tree lab out CR flow CRCR 标签加到……

2185.040–2187.829  tree lab out CR flow CRCR label to the PR and the results from the uh

给 PR 加上 tree lab out CR flow CRCR 标签，以及来自……的结果

2187.829–2187.839  PR and the results from the uh

PR 和来自……的结果

2187.839–2190.150  PR and the results from the uh downstream repositories appear on your

PR 和下游仓库的结果会显示在你的……

2190.150–2190.160  downstream repositories appear on your

下游仓库会显示在你的……

2190.160–2194.310  downstream repositories appear on your PyTorch PR and um

下游仓库会显示在你的 PyTorch PR 上，嗯

2194.310–2194.320  PyTorch PR and um

PyTorch PR 上，嗯

2194.320–2197.510  PyTorch PR and um currently we're working with Nindia

PyTorch PR 上；目前我们正与 Nindia 合作

2197.510–2197.520  currently we're working with Nindia

目前我们正与 Nindia 合作

2197.520–2201.990  currently we're working with Nindia Windowsci Red Hat IBM Spire Ascent NPU

目前我们正与 Nindia、Windowsci、Red Hat、IBM、Spire 和 Ascent NPU 合作

2201.990–2202.000  Windowsci Red Hat IBM Spire Ascent NPU

包括 Windowsci、Red Hat、IBM、Spire 和 Ascent NPU

2202.000–2206.055  Windowsci Red Hat IBM Spire Ascent NPU Google TPU and VLM repos for that. So

包括 Windowsci、Red Hat、IBM、Spire、Ascent NPU、Google TPU 和 VLM 仓库。所以

2206.055–2206.065  Google TPU and VLM repos for that. So

还有 Google TPU 和 VLM 仓库。所以

2206.065–2208.550  Google TPU and VLM repos for that. So [clears throat] this will help with um

为此有 Google TPU 和 VLM 的代码仓库。所以，[清嗓]这将有助于……

2208.550–2208.560  [clears throat] this will help with um

[清嗓]这将有助于……

2208.560–2211.109  [clears throat] this will help with um onboarding different um vendors,

[清嗓]这将有助于接入不同的供应商，

2211.109–2211.119  onboarding different um vendors,

接入不同的供应商，

2211.119–2213.270  onboarding different um vendors, different architectures and different

接入不同的供应商、不同的架构，以及不同的

2213.270–2213.280  different architectures and different

不同的架构，以及不同的

2213.280–2217.190  different architectures and different repositories uh and uh maintainers of

不同的架构、不同的代码仓库，以及这些仓库的维护者

2217.190–2217.200  repositories uh and uh maintainers of

代码仓库，以及这些仓库的维护者

2217.200–2219.109  repositories uh and uh maintainers of those repositories should be able to

代码仓库，而这些仓库的维护者应该能够

2219.109–2219.119  those repositories should be able to

这些仓库的维护者应该能够

2219.119–2222.069  those repositories should be able to onboard very quickly and get results

这些仓库的维护者应该能够快速接入，并很快得到结果

2222.069–2222.079  onboard very quickly and get results

快速接入，并很快得到结果

2222.079–2225.510  onboard very quickly and get results testing for PyTorch pretty quickly and

快速接入，开展 PyTorch 测试并很快得到结果，而且

2225.510–2225.520  testing for PyTorch pretty quickly and

很快就能为 PyTorch 开展测试，并且……

2225.520–2228.069  testing for PyTorch pretty quickly and supported on PR testing and on nightly

很快就能为 PyTorch 开展测试，支持 PR 测试和夜间测试

2228.069–2228.079  supported on PR testing and on nightly

支持 PR 测试和夜间测试

2228.079–2230.310  supported on PR testing and on nightly testing. So there are kind of two modes

支持 PR 测试和夜间测试。所以有两种测试模式

2230.310–2230.320  testing. So there are kind of two modes

测试。所以有两种模式

2230.320–2233.030  testing. So there are kind of two modes for this and yeah I think this is a very

测试。所以这里有两种模式，我觉得这非常……

2233.030–2233.040  for this and yeah I think this is a very

对于这件事，我觉得这非常……

2233.040–2235.750  for this and yeah I think this is a very uh interesting uh part of the PyTorch

对于这件事，我觉得这是 PyTorch 项目中很有意思的……

2235.750–2235.760  uh interesting uh part of the PyTorch

PyTorch 项目中很有意思的一部分

2235.760–2237.510  uh interesting uh part of the PyTorch project that we're currently developing

PyTorch 项目中很有意思的一部分，我们正在开发

2237.510–2237.520  project that we're currently developing

我们正在开发的项目

2237.520–2240.069  project that we're currently developing and going to develop for

我们正在开发的项目，今后也会继续开发

2240.069–2240.079  and going to develop for

今后也会继续开发

2240.079–2241.589  and going to develop for um

今后也会继续开发，嗯……

2241.589–2241.599  um

嗯……

2241.599–2242.870  um I don't know maybe Nikita

嗯，我不确定，也许 Nikita……

2242.870–2242.880  I don't know maybe Nikita

我不确定，也许 Nikita……

2242.880–2246.069  I don't know maybe Nikita >> yeah and maybe uh like one more yeah one

我不确定，也许 Nikita……对，也许还有一点

2246.069–2246.079  >> yeah and maybe uh like one more yeah one

对，也许还有一点

2246.079–2247.670  >> yeah and maybe uh like one more yeah one more thing that like on the API side

对，还有一点是关于 API 的

2247.670–2247.680  more thing that like on the API side

还有一点是关于 API 的

2247.680–2249.430  more thing that like on the API side there's a torch accelerate API which

API 方面有个 torch accelerate API，它……

2249.430–2249.440  there's a torch accelerate API which

有个 torch accelerate API，它……

2249.440–2251.990  there's a torch accelerate API which tries to abstract away different devices

有个 torch accelerate API，旨在屏蔽不同设备的差异

2251.990–2252.000  tries to abstract away different devices

旨在屏蔽不同设备的差异

2252.000–2254.230  tries to abstract away different devices different names different capabilities

旨在屏蔽不同设备在名称和能力上的差异

2254.230–2254.240  different names different capabilities

不同的名称和能力

2254.240–2256.710  different names different capabilities so you can write once for accelerated

不同的名称和能力，这样加速计算的代码只需写一次

2256.710–2256.720  so you can write once for accelerated

这样加速计算的代码只需写一次

2256.720–2259.510  so you can write once for accelerated compute and it should on not I don't

这样加速计算的代码只需写一次，应该就能在……

2259.510–2259.520  compute and it should on not I don't

计算代码应该能在……不，我……

2259.520–2261.829  compute and it should on not I don't know I don't want to repeat the Java

计算代码应该能在……算了，我不想照搬 Java 的那句口号

2261.829–2261.839  know I don't want to repeat the Java

我不想照搬 Java 的那句口号

2261.839–2264.390  know I don't want to repeat the Java everywhere but it will run on multiple

我不想照搬 Java 那句“到处运行”的口号，但它能在多种……

2264.390–2264.400  everywhere but it will run on multiple

到处运行，不过它能在多种……

2264.400–2266.870  everywhere but it will run on multiple compute and we try to like align all our

它能在多种计算设备上运行，我们也在努力统一所有……

2266.870–2266.880  compute and we try to like align all our

计算设备上运行，我们也在努力统一所有……

2266.880–2269.030  compute and we try to like align all our operator implementation this is why CRCR

我们也在努力统一所有算子的实现，所以才有 CRCR

2269.030–2269.040  operator implementation this is why CRCR

算子的实现，所以才有 CRCR

2269.040–2272.069  operator implementation this is why CRCR is there is that uh like vendors can

算子的实现。设立 CRCR 是为了让厂商……

2272.069–2272.079  is there is that uh like vendors can

是为了让厂商……

2272.079–2274.710  is there is that uh like vendors can make sure that like their hardware

是为了让厂商确保自己的硬件……

2274.710–2274.720  make sure that like their hardware

确保自己的硬件……

2274.720–2278.470  make sure that like their hardware report the same results as we expect it

确保自己的硬件给出符合预期的结果

2278.470–2278.480  report the same results as we expect it

给出符合预期的结果

2278.480–2280.870  report the same results as we expect it would and developers will have fewer

给出符合预期的结果，开发者也会少遇到……

2280.870–2280.880  would and developers will have fewer

开发者也会少遇到……

2280.880–2282.950  would and developers will have fewer surprises when they use this like

开发者使用这种机制时，也会少遇到意外

2282.950–2282.960  surprises when they use this like

使用这种机制时少遇到意外

2282.960–2285.910  surprises when they use this like abstraction to dispatch their job to

用这种抽象层分发任务时，也会少遇到意外

2285.910–2285.920  abstraction to dispatch their job to

用于将任务分派给……的抽象层

2285.920–2287.750  abstraction to dispatch their job to accelerator available but yes it is very

用于将任务分派给可用加速器的抽象层。不过没错，这非常……

2287.750–2287.760  accelerator available but yes it is very

有可用的加速器，不过没错，这非常……

2287.760–2290.310  accelerator available but yes it is very important because you need to be on more

有可用的加速器，不过没错，这很重要，因为你需要支持更多……

2290.310–2290.320  important because you need to be on more

这很重要，因为你需要支持更多……

2290.320–2293.510  important because you need to be on more platforms.

这很重要，因为你需要支持更多平台。

2295.109–2295.119  >> Yeah, I think I think I mean I I think

>> 对，我觉得，我的意思是……

2295.119–2296.470  >> Yeah, I think I think I mean I I think this whole area is really fascinating

>> 对，我觉得，PyTorch 的这一块工作真的很有意思。

2296.470–2296.480  this whole area is really fascinating

这一块工作真的很有意思。

2296.480–2300.230  this whole area is really fascinating within PyTorch because like uh like how

PyTorch 的这一块工作真的很有意思，因为，呃，该怎么说……

2300.230–2300.240  within PyTorch because like uh like how

在 PyTorch 中，因为，呃，该怎么……

2300.240–2302.310  within PyTorch because like uh like how do you how do you have like a consistent

在 PyTorch 中，你该如何提供一致的……

2302.310–2302.320  do you how do you have like a consistent

你该如何提供一致的……

2302.320–2303.990  do you how do you have like a consistent experience across the such different

你该如何在如此不同的……之间提供一致体验？

2303.990–2304.000  experience across the such different

在如此不同的……之间提供一致体验。

2304.000–2307.510  experience across the such different hardware and then how do we do QA? How

如何在各种不同硬件上提供一致体验？然后我们怎么做 QA？又该怎么……

2307.510–2307.520  hardware and then how do we do QA? How

面对各种硬件，我们怎么做 QA？又该怎么……

2307.520–2309.190  hardware and then how do we do QA? How do we deal with like new hardware

面对各种硬件，我们怎么做 QA？又该如何应对新硬件……

2309.190–2309.200  do we deal with like new hardware

我们该如何应对新硬件……

2309.200–2310.950  do we deal with like new hardware emerging which may or may not be in the

我们该如何应对不断出现的新硬件？它们可能……

2310.950–2310.960  emerging which may or may not be in the

不断出现的新硬件，可能……

2310.960–2313.030  emerging which may or may not be in the cloud for us to test? and how do we do

不断出现的新硬件，可能在云端供我们测试，也可能不在。我们又该怎么……

2313.030–2313.040  cloud for us to test? and how do we do

云端能否供我们测试？我们又该怎么……

2313.040–2315.430  cloud for us to test? and how do we do like the the the the

云端能否供我们测试？还有，那个……该怎么做？

2315.430–2315.440  like the the the the

比如，那个……

2315.440–2318.069  like the the the the um the the the continuous integration

比如，那个持续集成……

2318.069–2318.079  um the the the continuous integration

嗯，持续集成……

2318.079–2320.870  um the the the continuous integration testing in a distributed fashion. Um so

嗯，如何以分布式方式进行持续集成测试。所以……

2320.870–2320.880  testing in a distributed fashion. Um so

以分布式方式进行测试。所以……

2320.880–2323.349  testing in a distributed fashion. Um so so this is an area where like just a lot

以分布式方式进行测试。所以，这方面投入了大量……

2323.349–2323.359  so this is an area where like just a lot

所以，这方面投入了大量……

2323.359–2324.630  so this is an area where like just a lot of effort has gone and I think we've

所以，这方面投入了大量精力，我觉得我们已经……

2324.630–2324.640  of effort has gone and I think we've

投入了大量精力，我觉得我们已经……

2324.640–2326.069  of effort has gone and I think we've just gotten like this part of PyTorch

投入了大量精力，我觉得 PyTorch 的这部分……

2326.069–2326.079  just gotten like this part of PyTorch

PyTorch 的这部分现在……

2326.079–2327.589  just gotten like this part of PyTorch has gotten a lot more mature. So I

PyTorch 的这部分成熟了很多。所以我……

2327.589–2327.599  has gotten a lot more mature. So I

成熟了很多。所以我……

2327.599–2329.829  has gotten a lot more mature. So I imagine for hardware vendors it's way

成熟了很多。所以我想，对硬件厂商来说……

2329.829–2329.839  imagine for hardware vendors it's way

我想，对硬件厂商来说……

2329.839–2332.069  imagine for hardware vendors it's way easier to sort of contemplate getting

我想，对硬件厂商来说，如今考虑让平台获得支持容易多了。

2332.069–2332.079  easier to sort of contemplate getting

考虑让平台获得支持容易多了。

2332.079–2333.589  easier to sort of contemplate getting their their platform properly supported

如今要让他们的平台得到妥善支持，容易多了。

2333.589–2333.599  their their platform properly supported

让他们的平台得到妥善支持。

2333.599–2335.510  their their platform properly supported in PyTorch now than it was you know two

如今要让他们的平台在 PyTorch 中得到妥善支持，比……

2335.510–2335.520  in PyTorch now than it was you know two

如今在 PyTorch 中实现支持，比两……

2335.520–2337.829  in PyTorch now than it was you know two years ago three years ago. So okay I

如今在 PyTorch 中实现支持，比两三年前容易多了。好，我……

2337.829–2337.839  years ago three years ago. So okay I

两三年前。好，我……

2337.839–2338.790  years ago three years ago. So okay I think

两三年前。好，我觉得……

2338.790–2338.800  think

我觉得……

2338.800–2341.270  think >> want to add sorry last last point to

我觉得……>> 抱歉，我想再补充最后一点……

2341.270–2341.280  >> want to add sorry last last point to

>> 抱歉，我想再补充最后一点……

2341.280–2343.190  >> want to add sorry last last point to this and it's the way it's developed

欠，我想再补充最后一点：这与它的开发方式有关。

2343.190–2343.200  this and it's the way it's developed

这与它的开发方式有关。

2343.200–2345.910  this and it's the way it's developed right now it's a lot more secure than it

按现在的开发方式，它比以前安全得多。

2345.910–2345.920  right now it's a lot more secure than it

现在它比以前安全得多。

2345.920–2348.630  right now it's a lot more secure than it used to be. So there's no um sharing of

现在它比以前安全得多，所以不用共享……

2348.630–2348.640  used to be. So there's no um sharing of

以前没这么安全。所以现在不用共享……

2348.640–2350.950  used to be. So there's no um sharing of the keys nothing like that. So on

不用共享密钥，也不需要类似操作。所以在……

2350.950–2350.960  the keys nothing like that. So on

密钥之类的都不用共享。所以在……

2350.960–2353.109  the keys nothing like that. So on boarding and from infrastructure point

密钥之类的都不用共享。从接入和基础设施角度看……

2353.109–2353.119  boarding and from infrastructure point

从接入和基础设施角度看……

2353.119–2356.630  boarding and from infrastructure point of view it's very fast and secure and uh

从接入和基础设施角度看，整个过程快速、安全，而且……

2356.630–2356.640  of view it's very fast and secure and uh

从这个角度看，整个过程快速、安全，而且……

2356.640–2358.550  of view it's very fast and secure and uh uh very straightforward. We set up

整个过程快速、安全，也很简单。我们设置了……

2358.550–2358.560  uh very straightforward. We set up

也很简单。我们设置了……

2358.560–2361.292  uh very straightforward. We set up specific repository CRC test when

我们设置了针对特定仓库的 CRC 测试，届时……

2361.292–2361.302  specific repository CRC test when

针对特定仓库的 CRC 测试，届时……

2361.302–2362.390  specific repository CRC test when [clears throat] you can go and look at

针对特定仓库的 CRC 测试。你可以去看看……

2362.390–2362.400  [clears throat] you can go and look at

[清嗓] 你可以去看看……

2362.400–2364.950  [clears throat] you can go and look at the on boarding guide and can start on

[清嗓] 你可以查看接入指南，然后开始……

2364.950–2364.960  the on boarding guide and can start on

查看接入指南，然后就可以开始……

2364.960–2366.470  the on boarding guide and can start on boarding.

查看接入指南，然后就可以开始接入。

2366.470–2366.480  boarding.

接入。

2366.480–2370.390  boarding. >> Yeah. Awesome. Okay. I think we have a

接入。>> 对，太好了。好，我想我们还有……

2370.390–2370.400  >> Yeah. Awesome. Okay. I think we have a

>> 对，太好了。好，我想我们还有……

2370.400–2373.750  >> Yeah. Awesome. Okay. I think we have a next question lined up.

>> 对，太好了。好，下一个问题已经准备好了。

2373.750–2373.760  next question lined up.

下一个问题已经准备好了。

2373.760–2375.829  next question lined up. Okay. And this again thank you

下一个问题已经准备好了。好，也再次感谢……

2375.829–2375.839  Okay. And this again thank you

好，也再次感谢……

2375.839–2378.870  Okay. And this again thank you Shivashish uh for asking uh three

好，也再次感谢 Shivashish 提出了三个……

2378.870–2378.880  Shivashish uh for asking uh three

Shivashish 提出了三个……

2378.880–2381.270  Shivashish uh for asking uh three interesting and challenging questions uh

Shivashish 提出了三个有趣又有挑战性的问题……

2381.270–2381.280  interesting and challenging questions uh

有趣又有挑战性的问题……

2381.280–2383.510  interesting and challenging questions uh uh for us to answer here. Uh so for

这些问题很有趣，也很有挑战性，我们来回答一下。那么……

2383.510–2383.520  uh for us to answer here. Uh so for

我们来回答一下。那么，对于……

2383.520–2385.670  uh for us to answer here. Uh so for draft then verify loops uh which would

那么，对于先生成再验证的循环，它可能……

2385.670–2385.680  draft then verify loops uh which would

先生成再验证的循环，可能……

2385.680–2387.589  draft then verify loops uh which would be sort of speculative decoding style

先生成再验证的循环，有点像推测解码……

2387.589–2387.599  be sort of speculative decoding style

有点像推测解码……

2387.599–2391.030  be sort of speculative decoding style style is could a graph capture um in the

有点像推测解码。那么，图捕获能否……

2391.030–2391.040  style is could a graph capture um in the

那么，图捕获能否……

2391.040–2394.310  style is could a graph capture um in the uh torch while loop u uh mode usable on

图捕获能否在 torch while loop 模式下用于……

2394.310–2394.320  uh torch while loop u uh mode usable on

torch while loop 模式能否用于……

2394.320–2396.630  uh torch while loop u uh mode usable on the verify step in 214 or does that

torch while loop 模式能否用于 214 中的验证步骤，还是……

2396.630–2396.640  the verify step in 214 or does that

214 中的验证步骤，还是……

2396.640–2399.030  the verify step in 214 or does that control uh flow still force a graph

214 中的验证步骤，还是这种控制流仍会导致计算图……

2399.030–2399.040  control uh flow still force a graph

这种控制流仍会导致计算图……

2399.040–2403.109  control uh flow still force a graph break um do we do we know the answer to

这种控制流仍会导致计算图中断吗？我们知道……

2403.109–2403.119  break um do we do we know the answer to

计算图中断吗？我们知道……

2403.119–2404.390  break um do we do we know the answer to that among the

计算图中断吗？我们当中有人知道答案吗？

2404.390–2404.400  that among the

其中……

2404.400–2406.630  that among the >> I think generically you cannot answer

其中……我觉得笼统地说，没法回答。

2406.630–2406.640  >> I think generically you cannot answer

我觉得笼统地说，没法回答。

2406.640–2408.310  >> I think generically you cannot answer this question because it depends on the

我觉得笼统地说，没法回答这个问题，因为这取决于……

2408.310–2408.320  this question because it depends on the

这个问题，因为这取决于……

2408.320–2411.510  this question because it depends on the workflow it should be usable and that's

这个问题，因为这取决于工作流程；它应该能用，而这正是……

2411.510–2411.520  workflow it should be usable and that's

工作流程；它应该能用，而这正是……

2411.520–2414.550  workflow it should be usable and that's uh what the new um like abstractions

工作流程；它应该能用，而这正是新的抽象机制……

2414.550–2414.560  uh what the new um like abstractions

新的抽象机制……

2414.560–2416.950  uh what the new um like abstractions that we talked about in the highlights

我们在亮点部分提到的新抽象机制……

2416.950–2416.960  that we talked about in the highlights

我们在亮点部分提到的……

2416.960–2418.790  that we talked about in the highlights are for but there could be some cases

我们在亮点部分提到的就是为此设计的，但也可能有些情况……

2418.790–2418.800  are for but there could be some cases

就是为此设计的，但也可能有些情况……

2418.800–2421.589  are for but there could be some cases when it's not but we you already have uh

就是为此设计的，但也可能有些情况无法使用。不过你已经有……

2421.589–2421.599  when it's not but we you already have uh

有些情况无法使用。不过你已经有……

2421.599–2424.630  when it's not but we you already have uh toggles to test that you can just wrap

有些情况无法使用。不过你已经有开关可以测试，只需用它包住……

2424.630–2424.640  toggles to test that you can just wrap

有开关可以测试，只需用它包住……

2424.640–2427.430  toggles to test that you can just wrap your code with like aborton graph break

有开关可以测试，只需用类似 abort on graph break 的功能包住代码。

2427.430–2427.440  your code with like aborton graph break

用类似 abort on graph break 的功能包住代码。

2427.440–2429.430  your code with like aborton graph break and then you know that you produced a

用类似 abort on graph break 的功能包住代码，就能知道是否产生了……

2429.430–2429.440  and then you know that you produced a

就能知道是否产生了……

2429.440–2431.829  and then you know that you produced a graph break or not but it's kind of I I

就能知道是否产生了图中断。不过我觉得……

2431.829–2431.839  graph break or not but it's kind of I I

是否发生图中断。不过我觉得……

2431.839–2433.349  graph break or not but it's kind of I I don't think there's one one answer that

是否发生图中断。不过我觉得没有一个……

2433.349–2433.359  don't think there's one one answer that

我觉得没有一个……

2433.359–2436.630  don't think there's one one answer that fits them all like we try to make those

我觉得没有一个适用于所有情况的答案。我们会尽量让这些功能……

2436.630–2436.640  fits them all like we try to make those

适用于所有情况。我们会尽量让这些功能……

2436.640–2439.270  fits them all like we try to make those not cause graph breaks but depending on

适用于所有情况。我们会尽量避免这些功能造成图中断，但这取决于……

2439.270–2439.280  not cause graph breaks but depending on

避免造成图中断，但这取决于……

2439.280–2441.030  not cause graph breaks but depending on the CUDA version you're on the hardware

避免造成图中断，但这取决于你使用的 CUDA 版本、硬件……

2441.030–2441.040  the CUDA version you're on the hardware

你使用的 CUDA 版本、硬件……

2441.040–2443.190  the CUDA version you're on the hardware capabilities is a Python version and so

你使用的 CUDA 版本、硬件能力、Python 版本等……

2443.190–2443.200  capabilities is a Python version and so

硬件能力、Python 版本等……

2443.200–2444.870  capabilities is a Python version and so on and so forth. It can cause graph

硬件能力、Python 版本等因素。它可能造成图中断。

2444.870–2444.880  on and so forth. It can cause graph

等等。它可能造成图中断。

2444.880–2448.790  on and so forth. It can cause graph break. So please check if it doesn't

等等。它可能造成图中断。所以请检查是否发生了……

2448.790–2448.800  break. So please check if it doesn't

图中断。所以请检查是否发生了……

2448.800–2451.109  break. So please check if it doesn't create great if it does file an issue

图中断。如果没有发生，那很好；如果发生了，请提交 issue。

2451.109–2451.119  create great if it does file an issue

没有发生就好；如果发生了，请提交 issue。

2451.119–2453.430  create great if it does file an issue but we try to eliminate as graph breaks

没有发生就好；如果发生了，请提交 issue。我们会尽量消除图中断。

2453.430–2453.440  but we try to eliminate as graph breaks

我们会尽量消除图中断。

2453.440–2455.510  but we try to eliminate as graph breaks as much as possible because it slows

我们会尽量消除图中断，因为它会拖慢……

2455.510–2455.520  as much as possible because it slows

因为它会拖慢……

2455.520–2457.190  as much as possible because it slows down

因为它会拖慢运行速度。

2457.190–2457.200  down

拖慢运行速度。

2457.200–2459.829  down your pipelines.

拖慢你的处理流程。

2459.829–2459.839  your pipelines.

你的处理流程。

2459.839–2462.710  your pipelines. >> Okay, awesome. Um, can we jump next to

你的处理流程。好的，太棒了。接下来可以看……

2462.710–2462.720  >> Okay, awesome. Um, can we jump next to

好的，太棒了。接下来可以看……

2462.720–2467.030  >> Okay, awesome. Um, can we jump next to the question about Envy Gym uh Q5? Uh,

好的，太棒了。接下来可以看关于 Envy Gym 的 Q5 问题吗？

2467.030–2467.040  the question about Envy Gym uh Q5? Uh,

关于 NVGEMM 的 Q5 问题？呃，

2467.040–2471.589  the question about Envy Gym uh Q5? Uh, Basil,

关于 NVGEMM 的 Q5 问题？呃，Basil，

2473.349–2473.359  >> sorry to do that verbally rather than in

抱歉，我直接口头说，没在……

2473.359–2483.430  >> sorry to do that verbally rather than in the chat.

抱歉，我直接口头说，没在聊天里发。

2485.349–2485.359  So, this is one of the key things that

这是我们强调的一个重点。

2485.359–2486.710  So, this is one of the key things that we highlighted at the be at the

这是我们一开始强调的一个重点。

2486.710–2486.720  we highlighted at the be at the

我们一开始就强调过。

2486.720–2487.829  we highlighted at the be at the beginning and I wanted to make sure we

我们一开始就强调过，我想确保大家……

2487.829–2487.839  beginning and I wanted to make sure we

一开始我就想确保大家……

2487.839–2489.510  beginning and I wanted to make sure we we took a moment to talk about it among

一开始我就想确保我们在答疑中花点时间聊聊。

2489.510–2489.520  we took a moment to talk about it among

我们在答疑中花点时间聊聊。

2489.520–2492.710  we took a moment to talk about it among the questions. Uh so what is NVGEM? Um

我们在答疑中花点时间聊聊：NVGEMM 是什么？

2492.710–2492.720  the questions. Uh so what is NVGEM? Um

问题是：NVGEMM 是什么？

2492.720–2494.309  the questions. Uh so what is NVGEM? Um kind of a generic question and how do I

问题是：NVGEMM 是什么？这个问题有点宽泛，还有怎么……

2494.309–2494.319  kind of a generic question and how do I

这个问题有点宽泛，还有怎么……

2494.319–2498.630  kind of a generic question and how do I turn it on? Um uh so uh Nikita do you

这个问题有点宽泛，还有怎么启用它？Nikita，你……

2498.630–2498.640  turn it on? Um uh so uh Nikita do you

怎么启用它？Nikita，你……

2498.640–2500.550  turn it on? Um uh so uh Nikita do you want to jump on this or Andre?

怎么启用它？Nikita 或 Andre，谁来回答？

2500.550–2500.560  want to jump on this or Andre?

Nikita 或 Andre，谁来回答？

2500.560–2502.950  want to jump on this or Andre? >> Sure. But we already somewhat covered it

Nikita 或 Andre，谁来回答？当然，不过我们刚才已经讲过一些了。

2502.950–2502.960  >> Sure. But we already somewhat covered it

当然，不过我们刚才已经讲过一些了。

2502.960–2504.550  >> Sure. But we already somewhat covered it when the other user asked a question

当然，不过刚才另一位用户提问时，我们已经讲过一些了。

2504.550–2504.560  when the other user asked a question

刚才另一位用户提问时……

2504.560–2508.230  when the other user asked a question about different uh good gen backends and

刚才另一位用户问到了不同的代码生成后端。

2508.230–2508.240  about different uh good gen backends and

关于不同的代码生成后端。

2508.240–2510.390  about different uh good gen backends and I said that there is a like you can't do

关于不同的代码生成后端，我说过，这个没法……

2510.390–2510.400  I said that there is a like you can't do

我说过，这个没法……

2510.400–2512.710  I said that there is a like you can't do it over FX graph but there is a toggles

我说过，没法通过 FX 图来做，但有相应的开关。

2512.710–2512.720  it over FX graph but there is a toggles

没法通过 FX 图来做，但有相应的开关。

2512.720–2516.309  it over FX graph but there is a toggles and specifically for NVJ there's max to

没法通过 FX 图来做，但有相应的开关。具体到 NVGEMM，还有……

2516.309–2516.319  and specifically for NVJ there's max to

具体到 NVGEMM，还有……

2516.319–2517.910  and specifically for NVJ there's max to tune gem backends

具体到 NVGEMM，有 max_autotune_gemm_backends 配置。

2517.910–2517.920  tune gem backends

GEMM 自动调优后端列表。

2517.920–2520.390  tune gem backends >> list and if you add NVJM here so it's

这个列表里加入 NVGEMM 就行。

2520.390–2520.400  >> list and if you add NVJM here so it's

把 NVGEMM 加进列表后，它……

2520.400–2522.470  >> list and if you add NVJM here so it's not enabled by default but then it will

把 NVGEMM 加进列表后，虽然它默认未启用，但这样就会……

2522.470–2522.480  not enabled by default but then it will

它默认未启用，但这样就会……

2522.480–2524.470  not enabled by default but then it will be enabled and it will allow some more

它默认未启用；加进去后就会启用，还能支持更多……

2524.470–2524.480  be enabled and it will allow some more

启用后还能支持更多……

2524.480–2528.069  be enabled and it will allow some more advanced epilog fusion and yeah NVJM is

启用后还能支持更高级的尾部融合。NVGEMM 是……

2528.069–2528.079  advanced epilog fusion and yeah NVJM is

更高级的尾部融合。NVGEMM 是……

2528.079–2532.309  advanced epilog fusion and yeah NVJM is um QGSL gem back end that based on

更高级的尾部融合。NVGEMM 是基于 CuTeDSL 的 GEMM 后端。

2532.309–2532.319  um QGSL gem back end that based on

它是基于 CuTeDSL 的 GEMM 后端，又基于……

2532.319–2534.230  um QGSL gem back end that based on cutless

它是基于 CUTLASS 的 CuTeDSL GEMM 后端。

2534.230–2534.240  cutless

基于 CUTLASS。

2534.240–2537.109  cutless and it allows you to like yeah fuse

基于 CUTLASS，还能让你融合……

2537.109–2537.119  and it allows you to like yeah fuse

还能让你融合……

2537.119–2539.270  and it allows you to like yeah fuse mules and fusions and get generate even

还能融合乘法等操作，生成更快的……

2539.270–2539.280  mules and fusions and get generate even

融合乘法等操作，生成更快的……

2539.280–2543.349  mules and fusions and get generate even faster uh yeah uh kernels specifically

融合乘法等操作，生成更快的内核。

2543.349–2543.359  faster uh yeah uh kernels specifically

更快，嗯，对，特别是内核。

2543.359–2545.109  faster uh yeah uh kernels specifically if you do low precision data types like

内核会更快，嗯，对，尤其是用低精度数据类型时，比如……

2545.109–2545.119  if you do low precision data types like

如果用低精度数据类型，比如……

2545.119–2547.589  if you do low precision data types like in VFP4 but it requires you to have an

比如 VFP4，但这要求你安装……

2547.589–2547.599  in VFP4 but it requires you to have an

用 VFP4 的话，你还需要安装……

2547.599–2550.069  in VFP4 but it requires you to have an Nvidia cutless DSL installed on your

用 VFP4 的话，你的系统必须装有 NVIDIA CUTLASS DSL……

2550.069–2550.079  Nvidia cutless DSL installed on your

你的系统得装有 NVIDIA CUTLASS DSL……

2550.079–2553.670  Nvidia cutless DSL installed on your system and blackwell as well so like you

你的系统得装有 NVIDIA CUTLASS DSL，还得有 Blackwell，所以……

2553.670–2553.680  system and blackwell as well so like you

系统还得有 Blackwell，所以……

2553.680–2555.910  system and blackwell as well so like you unfortunately cannot use 44 [laughter]

系统还得有 Blackwell，所以很遗憾，你用不了 44（笑声）。

2555.910–2555.920  unfortunately cannot use 44 [laughter]

很遗憾，用不了 44（笑声）。

2555.920–2562.230  unfortunately cannot use 44 [laughter] on your ampers or hoppers

很遗憾，Ampere 或 Hopper 上用不了 44（笑声）。

2565.670–2565.680  Okay, I see a question from Jose Silva

好，我看到 Jose Silva 的一个问题。

2565.680–2570.150  Okay, I see a question from Jose Silva on the on the chat. Um,

好，我看到 Jose Silva 在聊天区提了个问题。

2570.150–2570.160  on the on the chat. Um,

在聊天区。嗯。

2570.160–2572.870  on the on the chat. Um, okay, awesome. Thank you, Basil. Um, if

在聊天区。好，太好了。谢谢你，Basil。如果……

2572.870–2572.880  okay, awesome. Thank you, Basil. Um, if

好，太好了。谢谢你，Basil。如果……

2572.880–2575.349  okay, awesome. Thank you, Basil. Um, if I train a vision model on CUDA and

好，太好了。谢谢你，Basil。如果我在 CUDA 上训练视觉模型，然后……

2575.349–2575.359  I train a vision model on CUDA and

我在 CUDA 上训练视觉模型，然后……

2575.359–2578.309  I train a vision model on CUDA and export it using AOT inductor to C++

我在 CUDA 上训练视觉模型，再用 AOT Inductor 把它导出为 C++……

2578.309–2578.319  export it using AOT inductor to C++

用 AOT Inductor 把它导出为 C++……

2578.319–2581.750  export it using AOT inductor to C++ binaries for a machine running an Intel

用 AOT Inductor 把它导出为 C++ 二进制文件，供搭载 Intel……

2581.750–2581.760  binaries for a machine running an Intel

二进制文件，供搭载 Intel……

2581.760–2585.190  binaries for a machine running an Intel Iris iGPU. Um, is dynamic shape and

二进制文件，供搭载 Intel Iris iGPU 的机器运行。那么动态形状和……

2585.190–2585.200  Iris iGPU. Um, is dynamic shape and

Intel Iris iGPU。那么动态形状和……

2585.200–2586.950  Iris iGPU. Um, is dynamic shape and anti-quantization supported out of the

Intel Iris iGPU。那么动态形状和反量化能否直接……

2586.950–2586.960  anti-quantization supported out of the

反量化能否直接……

2586.960–2589.589  anti-quantization supported out of the box on those targets? On those XPU

反量化在这些目标平台上能否开箱即用？在这些 XPU……

2589.589–2589.599  box on those targets? On those XPU

在这些目标平台上能否开箱即用？这些 XPU……

2589.599–2591.854  box on those targets? On those XPU targets?

在这些目标平台上能否开箱即用？这些 XPU 目标平台呢？

2591.854–2591.864  targets?

目标平台呢？

2591.864–2593.030  targets? [laughter]

目标平台呢？（笑声）

2593.030–2593.040  [laughter]

（笑声）

2593.040–2594.870  [laughter] >> That's quite a specific question.

（笑声）>> 这个问题很具体。

2594.870–2594.880  >> That's quite a specific question.

>> 这个问题很具体。

2594.880–2596.390  >> That's quite a specific question. >> I was just going to say I don't know

>> 这个问题很具体。>> 我正想说，我不知道……

2596.390–2596.400  >> I was just going to say I don't know

>> 我正想说，我不知道……

2596.400–2599.059  >> I was just going to say I don't know that case is covered in rci. Uh,

>> 我正想说，我不知道 RCI 有没有覆盖这种情况。呃……

2599.059–2599.069  that case is covered in rci. Uh,

RCI 有没有覆盖这种情况。呃……

2599.069–2600.150  that case is covered in rci. Uh, [laughter]

RCI 有没有覆盖这种情况。呃，（笑声）

2600.150–2600.160  [laughter]

（笑声）

2600.160–2602.470  [laughter] >> it might be covered in rci, but maybe

（笑声）>> RCI 可能覆盖了，但也许……

2602.470–2602.480  >> it might be covered in rci, but maybe

>> RCI 可能覆盖了，但也许……

2602.480–2605.510  >> it might be covered in rci, but maybe not intel Iris iGPU in particular. Like

>> RCI 可能覆盖了，但未必专门覆盖 Intel Iris iGPU。我觉得……

2605.510–2605.520  not intel Iris iGPU in particular. Like

未必专门覆盖 Intel Iris iGPU。我觉得……

2605.520–2608.069  not intel Iris iGPU in particular. Like I think they're trying to cover a more

未必专门覆盖 Intel Iris iGPU。我觉得他们想覆盖更……

2608.069–2608.079  I think they're trying to cover a more

我觉得他们想覆盖更……

2608.079–2611.430  I think they're trying to cover a more >> modern GPUs. Uh

我觉得他们想覆盖更……>> 新一些的 GPU。呃……

2611.430–2611.440  >> modern GPUs. Uh

>> 新一些的 GPU。呃……

2611.440–2615.349  >> modern GPUs. Uh but I guess you never know until you try

>> 新一些的 GPU。不过不试试也说不准。

2615.349–2615.359  but I guess you never know until you try

不过，不试试怎么知道呢。

2615.359–2617.910  but I guess you never know until you try and then you can probably file an issue

不过，不试试怎么知道呢，然后你大概可以提个 issue。

2617.910–2617.920  and then you can probably file an issue

然后你大概可以提个 issue。

2617.920–2618.870  and then you can probably file an issue if it doesn't work.

然后你大概可以提个 issue，如果行不通的话。

2618.870–2618.880  if it doesn't work.

如果行不通的话。

2618.880–2619.750  if it doesn't work. >> Yeah, file is

如果行不通的话。>> 对，提个 issue。

2619.750–2619.760  >> Yeah, file is

>> 对，提个 issue。

2619.760–2622.230  >> Yeah, file is >> exactly [laughter]

>> 对，提个 issue。>> 没错（笑）

2622.950–2622.960  >> help us out.

>> 帮我们一把。

2622.960–2624.950  >> help us out. >> And we do have Yeah, we do have support

>> 帮我们一把。>> 而且我们确实有，对，确实支持……

2624.950–2624.960  >> And we do have Yeah, we do have support

>> 而且我们确实有，对，确实支持……

2624.960–2627.910  >> And we do have Yeah, we do have support for XPU devices. So on both PyTorch and

>> 而且我们确实有，对，确实支持 XPU 设备。所以在 PyTorch 和……

2627.910–2627.920  for XPU devices. So on both PyTorch and

适用于 XPU 设备。所以在 PyTorch 和

2627.920–2631.510  for XPU devices. So on both PyTorch and torch vision. So uh if you have a PR in

适用于 XPU 设备。所以在 PyTorch 和 torchvision 上。如果你有个 PR 的

2631.510–2631.520  torch vision. So uh if you have a PR in

torchvision。所以，如果你有个 PR 的

2631.520–2634.150  torch vision. So uh if you have a PR in mind, you can go and open a PR against

torchvision。所以，如果你有提交 PR 的想法，可以向

2634.150–2634.160  mind, you can go and open a PR against

想法，就可以向

2634.160–2637.829  mind, you can go and open a PR against PyTorch Vision CI.

想法，就可以向 PyTorch Vision CI 提交 PR。

2637.829–2637.839  PyTorch Vision CI.

PyTorch Vision 持续集成（CI）。

2637.839–2640.230  PyTorch Vision CI. But like I guess if I read the question

PyTorch Vision 持续集成（CI）。不过我想，如果我没理解错问题，

2640.230–2640.240  But like I guess if I read the question

不过我想，如果我没理解错问题，

2640.240–2642.309  But like I guess if I read the question correctly, if you train model on CUDA,

不过我想，如果我没理解错问题，如果你在 CUDA 上训练模型，

2642.309–2642.319  correctly, if you train model on CUDA,

如果我没理解错问题，如果你在 CUDA 上训练模型，

2642.319–2644.470  correctly, if you train model on CUDA, can you export it for XPU? You need to

如果我没理解错问题，如果你在 CUDA 上训练模型，能导出给 XPU 用吗？你需要……

2644.470–2644.480  can you export it for XPU? You need to

能导出给 XPU 用吗？你需要……

2644.480–2646.069  can you export it for XPU? You need to make some changes to your model first,

能导出到 XPU 吗？你得先修改一下模型，

2646.069–2646.079  make some changes to your model first,

先修改一下模型，

2646.079–2648.230  make some changes to your model first, right? You need to like transfer your

先修改一下模型，对吧？你得把你的……

2648.230–2648.240  right? You need to like transfer your

对吧？你得把你的……

2648.240–2650.470  right? You need to like transfer your model to XPU, make sure that it works

对吧？你得把模型转到 XPU 上，确保它能运行，

2650.470–2650.480  model to XPU, make sure that it works

把模型转到 XPU 上，确保它能运行，

2650.480–2653.750  model to XPU, make sure that it works and then AOT inductor should work on XPU

把模型转到 XPU 上，确保它能运行，然后 AOT Inductor 应该就能在 XPU 上工作了，

2653.750–2653.760  and then AOT inductor should work on XPU

然后 AOT Inductor 应该就能在 XPU 上工作了，

2653.760–2655.589  and then AOT inductor should work on XPU same as it works on any other back ends.

然后 AOT Inductor 应该就能在 XPU 上工作了，和在其他后端上一样。

2655.589–2655.599  same as it works on any other back ends.

和在其他后端上一样。

2655.599–2658.470  same as it works on any other back ends. But I'm not even sure if int 8

和在其他后端上一样。但我甚至不确定 int8……

2658.470–2658.480  But I'm not even sure if int 8

但我甚至不确定 int8……

2658.480–2660.069  But I'm not even sure if int 8 contisation is supported through

但我甚至不确定 Inductor 是否支持 int8 量化，

2660.069–2660.079  contisation is supported through

是否支持量化，

2660.079–2664.390  contisation is supported through inductor even for CUDA. like I think

即使在 CUDA 上，Inductor 是否支持量化，我也不确定。

2664.390–2664.400  inductor even for CUDA. like I think

即使在 CUDA 上，Inductor 是否支持量化，我也不确定。

2664.400–2665.990  inductor even for CUDA. like I think that there are lots of like layers to

即使在 CUDA 上，Inductor 是否支持量化，我也不确定。这里面还有很多层问题，

2665.990–2666.000  that there are lots of like layers to

这里面还有很多层问题，

2666.000–2668.870  that there are lots of like layers to unload here. So please file an issue but

这里面还有很多层问题要弄清楚。所以请提个 issue，不过……

2668.870–2668.880  unload here. So please file an issue but

要弄清楚。所以请提个 issue，不过……

2668.880–2672.550  unload here. So please file an issue but I also kind of think why do you want an

要弄清楚。所以请提个 issue。不过我也在想，你为什么想要……

2672.550–2672.560  I also kind of think why do you want an

我也在想，你为什么想要……

2672.560–2678.309  I also kind of think why do you want an int 8 rather than like well int 8 and

我也在想，你为什么想用 int8，而不是……嗯，int8，还有……

2678.309–2678.319  int 8 rather than like well int 8 and

用 int8，而不是……嗯，int8，还有……

2678.319–2680.390  int 8 rather than like well int 8 and and check the spec if iGPU even support

用 int8，而不是……嗯，int8。另外，查查规格，看看 iGPU 到底支不支持。

2680.390–2680.400  and check the spec if iGPU even support

再查查规格，看 iGPU 是否支持。

2680.400–2683.190  and check the spec if iGPU even support an int8ate uh fast smart n I know that

再查查规格，看 iGPU 是否支持 INT8 快速运算。我知道……

2683.190–2683.200  an int8ate uh fast smart n I know that

INT8 快速运算。我知道……

2683.200–2685.750  an int8ate uh fast smart n I know that CPUs do but I'm not sure about GPUs in

INT8 快速运算。我知道 CPU 支持，但不确定 GPU 是否……

2685.750–2685.760  CPUs do but I'm not sure about GPUs in

CPU 支持，但我不确定 GPU 是否……

2685.760–2690.309  CPUs do but I'm not sure about GPUs in particular.

CPU 支持，但我不确定 GPU 是否支持。

2697.030–2697.040  Um

嗯。

2697.040–2701.829  Um I think Basil's pulling the one that we

我想 Basil 正在找我们……

2701.829–2701.839  I think Basil's pulling the one that we

我想 Basil 正在找我们……

2701.839–2705.910  I think Basil's pulling the one that we mentioned to him here.

我想 Basil 正在找我们刚才跟他提到的那个。

2709.510–2709.520  Maybe in the meantime we can do a small

趁这会儿，我们可以简单……

2709.520–2712.870  Maybe in the meantime we can do a small uh note about PyTorch release matrix. So

趁这会儿，我们可以简单说说 PyTorch 的版本对应关系。

2712.870–2712.880  uh note about PyTorch release matrix. So

说说 PyTorch 的版本对应关系。那么……

2712.880–2715.510  uh note about PyTorch release matrix. So one of the changes that is in 214 is

说说 PyTorch 的版本对应关系。214 中有一项改动是……

2715.510–2715.520  one of the changes that is in 214 is

214 中有一项改动是……

2715.520–2717.349  one of the changes that is in 214 is that we mandate that all of your

214 中有一项改动是，我们要求你的所有……

2717.349–2717.359  that we mandate that all of your

我们要求你的所有……

2717.359–2719.190  that we mandate that all of your extensions will be compiled with oh here

我们要求你的所有扩展都用……哦，找到了。

2719.190–2719.200  extensions will be compiled with oh here

扩展都用……哦，找到了。

2719.200–2721.510  extensions will be compiled with oh here you go C++ 20 by default. So if you're

扩展默认用 C++ 20 编译。所以如果你……

2721.510–2721.520  you go C++ 20 by default. So if you're

找到了，是默认用 C++ 20。所以如果你……

2721.520–2723.589  you go C++ 20 by default. So if you're using an older compiler, you really need

是默认用 C++ 20。所以如果你用的是旧编译器，就得……

2723.589–2723.599  using an older compiler, you really need

如果你用的是旧编译器，就得……

2723.599–2726.390  using an older compiler, you really need to update. It is 2026 and majority of

如果你用的是旧编译器，就得升级。现在是 2026 年，大多数……

2726.390–2726.400  to update. It is 2026 and majority of

就得升级。现在是 2026 年，大多数……

2726.400–2728.390  to update. It is 2026 and majority of the compiler should be able to support

就得升级。现在是 2026 年，大多数编译器都应该支持……

2728.390–2728.400  the compiler should be able to support

大多数编译器都应该支持……

2728.400–2732.150  the compiler should be able to support C++ 20. Uh but that's

大多数编译器都应该支持 C++ 20。不过这……

2732.150–2732.160  C++ 20. Uh but that's

C++ 20。不过这……

2732.160–2733.670  C++ 20. Uh but that's backward compatibility breaking change.

C++ 20。不过这是一项破坏向后兼容性的改动。

2733.670–2733.680  backward compatibility breaking change.

这是一项破坏向后兼容性的改动。

2733.680–2735.349  backward compatibility breaking change. And with that, I don't know, Andrew, do

这是一项破坏向后兼容性的改动。说到这儿，Andrew，你要不要……

2735.349–2735.359  And with that, I don't know, Andrew, do

说到这儿，Andrew，你要不要……

2735.359–2736.710  And with that, I don't know, Andrew, do you want to Oh, Chris, do you want to

说到这儿，Andrew，你要不要……哦，Chris，你要不要……

2736.710–2736.720  you want to Oh, Chris, do you want to

你要不要……哦，Chris，你要不要……

2736.720–2737.750  you want to Oh, Chris, do you want to read the question first?

你要不要……哦，Chris，你要先读一下问题吗？

2737.750–2737.760  read the question first?

先读一下问题吗？

2737.760–2739.270  read the question first? >> Yeah. Yeah. Yeah. So, so this this

先读一下问题吗？对，对。那么这个……

2739.270–2739.280  >> Yeah. Yeah. Yeah. So, so this this

对，对。那么这个……

2739.280–2740.550  >> Yeah. Yeah. Yeah. So, so this this question which I think was going in the

对，对。我觉得这个问题和……

2740.550–2740.560  question which I think was going in the

我觉得这个问题和……

2740.560–2742.790  question which I think was going in the same direction that that Nikita was was

我觉得这个问题和 Nikita 想的方向一样……

2742.790–2742.800  same direction that that Nikita was was

和 Nikita 想的方向一样……

2742.800–2744.950  same direction that that Nikita was was thinking is do I still have to upgrade

和 Nikita 想的方向一样：我还需要每次都升级……

2744.950–2744.960  thinking is do I still have to upgrade

问题是，我还需要每次都升级……

2744.960–2747.109  thinking is do I still have to upgrade um torch vision every time I upgrade

问题是，每次升级 PyTorch，我还得升级 torchvision 吗？

2747.109–2747.119  um torch vision every time I upgrade

每次升级 PyTorch，我还得升级 torchvision 吗？

2747.119–2748.309  um torch vision every time I upgrade grade pietorch which I know was a

每次升级 PyTorch，我还得升级 torchvision 吗？我知道这以前……

2748.309–2748.319  grade pietorch which I know was a

升级 PyTorch 时，我知道这以前……

2748.319–2749.990  grade pietorch which I know was a headache that people uh had to deal with

升级 PyTorch 时，我知道这以前一直让大家很头疼。

2749.990–2750.000  headache that people uh had to deal with

大家不得不面对的麻烦

2750.000–2752.790  headache that people uh had to deal with for a very long time that um Andre do

大家长期以来不得不面对的麻烦。Andre，你……

2752.790–2752.800  for a very long time that um Andre do

长期以来的麻烦。Andre，你……

2752.800–2754.069  for a very long time that um Andre do you want to respond to this?

大家长期以来都要面对这个麻烦。Andre，你想回应一下吗？

2754.069–2754.079  you want to respond to this?

你想回应一下吗？

2754.079–2758.150  you want to respond to this? >> Yes. Yes. Um so torch vision uh 0.29 is

是的。torchvision 0.29 与……

2758.150–2758.160  >> Yes. Yes. Um so torch vision uh 0.29 is

是的。torchvision 0.29 与……

2758.160–2763.510  >> Yes. Yes. Um so torch vision uh 0.29 is a stable uh with respect to 21.14. So it

是的。torchvision 0.29 与 21.14 兼容，所以它……

2763.510–2763.520  a stable uh with respect to 21.14. So it

与 21.14 兼容，所以它……

2763.520–2765.430  a stable uh with respect to 21.14. So it will keep working with later version

与 21.14 兼容，也能继续支持后续版本。

2765.430–2765.440  will keep working with later version

也能继续支持后续版本。

2765.440–2769.109  will keep working with later version 2.15 2.16 and later. Uh the only catch

也能继续支持 2.15、2.16 及更高版本。唯一要注意的是……

2769.109–2769.119  2.15 2.16 and later. Uh the only catch

2.15、2.16 及更高版本。唯一要注意的是……

2769.119–2771.349  2.15 2.16 and later. Uh the only catch here if you need to um introduce new

2.15、2.16 及更高版本。唯一要注意的是，如果要引入新的……

2771.349–2771.359  here if you need to um introduce new

如果要引入新的……

2771.359–2774.950  here if you need to um introduce new CUDA version you may need to download

如果要引入新的 CUDA 版本，可能需要下载……

2774.950–2774.960  CUDA version you may need to download

如果要用新的 CUDA 版本，可能需要下载……

2774.960–2777.829  CUDA version you may need to download um version of torch vision compiled with

如果要用新的 CUDA 版本，可能需要下载对应版本编译的 torchvision。

2777.829–2777.839  um version of torch vision compiled with

用对应版本编译的 torchvision。

2777.839–2782.069  um version of torch vision compiled with this CUDA version. Um but yes um like as

用这个 CUDA 版本编译的 torchvision。不过，是的……

2782.069–2782.079  this CUDA version. Um but yes um like as

这个 CUDA 版本。不过，是的……

2782.079–2783.910  this CUDA version. Um but yes um like as um

这个 CUDA 版本。不过，基本上……

2783.910–2783.920  um

嗯。

2783.920–2786.710  um as a basic you no longer need matching

基本上，你不再需要匹配版本的……

2786.710–2786.720  as a basic you no longer need matching

基本上，你不再需要匹配版本的……

2786.720–2788.870  as a basic you no longer need matching torch vision to install every torch

基本上，每次安装新版 torch，都不再需要匹配版本的 torchvision。

2788.870–2788.880  torch vision to install every torch

每次安装新版 torch，都不再需要匹配版本的 torchvision。

2788.880–2793.190  torch vision to install every torch upgrade and uh in the future we may not

每次升级 torch，都不再需要安装匹配版本的 torchvision。以后我们可能也不会……

2793.190–2793.200  upgrade and uh in the future we may not

升级之后，以后我们可能也不会……

2793.200–2795.349  upgrade and uh in the future we may not do torch vision at the same time as

升级之后，以后我们可能不会再同时发布 torchvision 和……

2795.349–2795.359  do torch vision at the same time as

不会再同时发布 torchvision 和……

2795.359–2799.030  do torch vision at the same time as torch. So the releases could be um a

不会再同时发布 torchvision 和 torch，所以两个版本的发布可能会……

2799.030–2799.040  torch. So the releases could be um a

和 torch。所以两个版本的发布可能会……

2799.040–2802.069  torch. So the releases could be um a little bit decoupled because of this. So

和 torch。所以两个版本的发布可以稍微错开。

2802.069–2802.079  little bit decoupled because of this. So

因此，两个版本的发布可以稍微错开。

2802.079–2804.630  little bit decoupled because of this. So I think it's something good for us. It's

因此，两个版本的发布可以稍微错开。我觉得这对我们是件好事。

2804.630–2804.640  I think it's something good for us. It's

我觉得这对我们是件好事。

2804.640–2808.950  I think it's something good for us. It's a good news.

我觉得这对我们是件好事，是个好消息。

2810.710–2810.720  >> Just next next question, please.

请看下一个问题。

2810.720–2813.829  >> Just next next question, please. >> Next question.

请看下一个问题。下一个问题。

2813.829–2813.839  >> Next question.

下一个问题。

2813.839–2816.470  >> Next question. >> Okay. So, what is new for writing

下一个问题。好的，编写……有什么新进展？

2816.470–2816.480  >> Okay. So, what is new for writing

好的，编写……有什么新进展？

2816.480–2820.309  >> Okay. So, what is new for writing dynamic models that still compile? Well,

好的，编写仍能编译的动态模型有什么新进展？

2820.309–2820.319  dynamic models that still compile? Well,

仍能编译的动态模型有什么新进展？

2820.319–2823.270  dynamic models that still compile? Well, um Andre, do you want to take this one

仍能编译的动态模型有什么新进展？Andre，你要回答这个问题吗？

2823.270–2823.280  um Andre, do you want to take this one

Andre，你要回答这个问题吗？

2823.280–2826.390  um Andre, do you want to take this one as as as well?

Andre，这个问题也请你回答一下？

2826.390–2826.400  as as as well?

也请你回答一下？

2826.400–2828.470  as as as well? >> I think we answered this question kind

也请你回答一下？我觉得这个问题我们算是回答过了……

2828.470–2828.480  >> I think we answered this question kind

我想这个问题我们算是回答过了

2828.480–2830.069  >> I think we answered this question kind of twice already. There was a question

我想这个问题我们已经回答过两次了。还有个问题

2830.069–2830.079  of twice already. There was a question

已经回答过两次了。还有个问题

2830.079–2831.829  of twice already. There was a question separate. Like if we want to highlight

已经回答过两次了。还有个单独的问题。比如，如果要重点介绍

2831.829–2831.839  separate. Like if we want to highlight

单独的问题。比如，如果要重点介绍

2831.839–2834.710  separate. Like if we want to highlight the features there is a torch switch uh

单独的问题。比如，如果要重点介绍这些功能，有 torch switch

2834.710–2834.720  the features there is a torch switch uh

这些功能有 torch switch

2834.720–2836.470  the features there is a torch switch uh torch while loop that is captured across

这些功能有 torch switch，还有 torch while loop，它能跨

2836.470–2836.480  torch while loop that is captured across

torch while loop，它能跨

2836.480–2839.589  torch while loop that is captured across codraph and dynamic spec and people

torch while loop，它能跨 CUDA Graph 和动态规格捕获。大家

2839.589–2839.599  codraph and dynamic spec and people

CUDA Graph 和动态规格。大家

2839.599–2841.190  codraph and dynamic spec and people already asked about dynamic spec and

CUDA Graph 和动态规格。大家已经问过动态规格

2841.190–2841.200  already asked about dynamic spec and

已经问过动态规格

2841.200–2844.069  already asked about dynamic spec and whether it will cause inductor to fail

已经问过动态规格，以及它是否会导致 Inductor 失败

2844.069–2844.079  whether it will cause inductor to fail

它是否会导致 Inductor 失败

2844.079–2845.750  whether it will cause inductor to fail when it goes above like an upper

它是否会在超过某个上限时导致 Inductor 失败

2845.750–2845.760  when it goes above like an upper

在超过某个上限时

2845.760–2848.230  when it goes above like an upper boundary. People talked about uh asked

在超过某个上限时。大家还问过

2848.230–2848.240  boundary. People talked about uh asked

上限。大家还问过

2848.240–2849.829  boundary. People talked about uh asked about specialization but yes those are

上限。大家还问过特化的问题。对，这些就是

2849.829–2849.839  about specialization but yes those are

特化的问题。对，这些就是

2849.839–2852.790  about specialization but yes those are the features that you should try and

特化的问题。对，这些就是你应该尝试的功能

2852.790–2852.800  the features that you should try and

你应该尝试的功能

2852.800–2855.990  the features that you should try and like uh they will improve your VSC to

你应该尝试这些功能，它们会提升你的 VSC

2855.990–2856.000  like uh they will improve your VSC to

它们会提升你的 VSC

2856.000–2858.950  like uh they will improve your VSC to compile model with dynamically shaped

它们会提升你的 VSC，让它能编译输入形状动态变化的模型

2858.950–2858.960  compile model with dynamically shaped

编译输入形状动态变化的模型

2858.960–2861.750  compile model with dynamically shaped inputs like you will have fun um

编译输入形状动态变化的模型，你会得到

2861.750–2861.760  inputs like you will have fun um

输入形状动态变化的模型，你会得到

2861.760–2863.430  inputs like you will have fun um compiled artifacts that works for

对于形状动态变化的输入，你会得到适用的编译产物

2863.430–2863.440  compiled artifacts that works for

适用的编译产物

2863.440–2865.750  compiled artifacts that works for multiple shapes. So once again it's in

适用于多种形状的编译产物。所以再说一遍，就是

2865.750–2865.760  multiple shapes. So once again it's in

适用于多种形状。所以再说一遍，就是

2865.760–2868.870  multiple shapes. So once again it's in the torch switch [laughter] torch loop

适用于多种形状。所以再说一遍，就是 torch switch（笑）、torch loop

2868.870–2868.880  the torch switch [laughter] torch loop

就是 torch switch（笑）、torch loop

2868.880–2871.750  the torch switch [laughter] torch loop and dynamics back.

就是 torch switch（笑）、torch loop 和动态规格。

2871.750–2871.760  and dynamics back.

还有动态规格。

2871.760–2874.630  and dynamics back. >> Uh can we jump to question 14 in our

还有动态规格。>> 呃，我们能跳到准备好的第 14 个问题吗

2874.630–2874.640  >> Uh can we jump to question 14 in our

>> 呃，我们能跳到准备好的第 14 个问题吗

2874.640–2878.150  >> Uh can we jump to question 14 in our prepared ones? Um

>> 呃，我们能跳到准备好的第 14 个问题吗？

2878.150–2878.160  prepared ones? Um

准备好的问题吗？

2878.160–2882.230  prepared ones? Um this is the everpresent Torch script uh

准备好的问题吗？这是那个总会出现的 TorchScript 话题

2882.230–2882.240  this is the everpresent Torch script uh

这是那个总会出现的 TorchScript 话题

2882.240–2884.150  this is the everpresent Torch script uh topic [snorts]

这是那个总会出现的 TorchScript 话题（笑）

2884.150–2884.160  topic [snorts]

这个话题（笑）

2884.160–2885.430  topic [snorts] trying to touch on things that we

这个话题（笑），尽量谈谈我们

2885.430–2885.440  trying to touch on things that we

尽量谈谈我们

2885.440–2887.750  trying to touch on things that we haven't already discussed. Okay. So I

尽量谈谈我们还没讨论过的内容。好，我

2887.750–2887.760  haven't already discussed. Okay. So I

还没讨论过的内容。好，我

2887.760–2889.430  haven't already discussed. Okay. So I still use Torch script. Should I be

还没讨论过的内容。好，我还在用 TorchScript。我是不是该

2889.430–2889.440  still use Torch script. Should I be

还在用 TorchScript，我该……

2889.440–2891.910  still use Torch script. Should I be worried? Um Nikita, do you want to

我还在用 TorchScript，需要担心吗？呃，Nikita，你想……

2891.910–2891.920  worried? Um Nikita, do you want to

担心吗？呃，Nikita，你想……

2891.920–2893.510  worried? Um Nikita, do you want to answer this one?

担心吗？呃，Nikita，你想回答这个问题吗？

2893.510–2893.520  answer this one?

回答这个问题吗？

2893.520–2896.230  answer this one? Yes, we've been saying that to script

回答这个问题吗？是的，我们一直说 TorchScript……

2896.230–2896.240  Yes, we've been saying that to script

是的，我们一直说 TorchScript……

2896.240–2900.550  Yes, we've been saying that to script has been deprecated since I think 211 or

是的，我们一直说 TorchScript 大概从 211 或……起就已弃用。

2900.550–2900.560  has been deprecated since I think 211 or

我记得是从 211 或……起就已弃用。

2900.560–2903.750  has been deprecated since I think 211 or 212 and it has not been really working

我记得是从 211 或 212 起就已弃用，而且一直不太能用。

2903.750–2903.760  212 and it has not been really working

212，而且一直不太能用。

2903.760–2905.990  212 and it has not been really working even like you cannot use script on you

212，而且一直不太能用，甚至你都没法用 Script……

2905.990–2906.000  even like you cannot use script on you

甚至你都没法用 Script……

2906.000–2908.230  even like you cannot use script on you cannot trace the store script on Python

甚至你没法在 Python 上做追踪或用 Script……

2908.230–2908.240  cannot trace the store script on Python

没法在 Python 上做追踪或用 Script……

2908.240–2911.109  cannot trace the store script on Python 314 and I'm not sure what is the status

在 Python 3.14 上没法做追踪或用 Script，我也不清楚……

2911.109–2911.119  314 and I'm not sure what is the status

Python 3.14，我也不清楚……

2911.119–2913.990  314 and I'm not sure what is the status of 312 or 313 like it's just being

Python 3.14；至于 3.12 或 3.13，我也不清楚是什么情况。它已经……

2913.990–2914.000  of 312 or 313 like it's just being

至于 3.12 或 3.13，它已经……

2914.000–2915.750  of 312 or 313 like it's just being deprecated there are bugs there are

至于 3.12 或 3.13，它已经弃用，还有各种问题……

2915.750–2915.760  deprecated there are bugs there are

已经弃用，还有各种问题……

2915.760–2918.230  deprecated there are bugs there are security problems it supports less and

已经弃用，有漏洞和安全问题，支持的功能也越来越少……

2918.230–2918.240  security problems it supports less and

有安全问题，支持的功能也越来越少……

2918.240–2919.990  security problems it supports less and less operators I don't think you can

有安全问题，支持的算子越来越少，我觉得你已经没法……

2919.990–2920.000  less operators I don't think you can

支持的算子越来越少，我觉得你没法……

2920.000–2923.190  less operators I don't think you can capture anything that support

支持的算子越来越少，我觉得你没法捕获任何支持……

2923.190–2923.200  capture anything that support

捕获任何支持……

2923.200–2925.270  capture anything that support flotate or whatever. So don't use to

捕获任何支持 float8 之类功能的东西。所以别再用……

2925.270–2925.280  flotate or whatever. So don't use to

float8 之类的功能。所以别再用……

2925.280–2928.069  flotate or whatever. So don't use to script anymore. So AOTI expert is a

float8 之类的功能。所以别再用 TorchScript 了。AOTI 导出已经是个……

2928.069–2928.079  script anymore. So AOTI expert is a

别再用 TorchScript 了。AOTI 导出已经是个……

2928.079–2932.470  script anymore. So AOTI expert is a pretty uh like mature framework at this

别再用 TorchScript 了。AOTI 导出现在已经相当成熟……

2932.470–2932.480  pretty uh like mature framework at this

现在已经相当成熟……

2932.480–2934.390  pretty uh like mature framework at this point. Use this one. If you need to use

现在已经相当成熟了。用它吧。如果你需要用……

2934.390–2934.400  point. Use this one. If you need to use

成熟了。用它吧。如果你需要用……

2934.400–2936.230  point. Use this one. If you need to use script, you can use an older PyTorch but

用它吧。如果你必须用 Script，可以用旧版 PyTorch，但……

2936.230–2936.240  script, you can use an older PyTorch but

如果必须用 Script，可以用旧版 PyTorch，但……

2936.240–2938.470  script, you can use an older PyTorch but like be aware of the limitation. Be like

如果必须用 Script，可以用旧版 PyTorch，但要了解它的限制。

2938.470–2938.480  like be aware of the limitation. Be like

要了解它的限制。另外……

2938.480–2940.630  like be aware of the limitation. Be like we will be happy if there is a community

要了解它的限制。如果有社区维护者愿意……

2940.630–2940.640  we will be happy if there is a community

如果有社区维护者愿意……

2940.640–2943.190  we will be happy if there is a community maintainer who want to like look into

如果有社区维护者愿意接手维护……

2943.190–2943.200  maintainer who want to like look into

愿意接手维护……

2943.200–2946.069  maintainer who want to like look into the torch script but we are not actively

愿意接手维护 TorchScript，我们会很高兴，但我们目前没有积极……

2946.069–2946.079  the torch script but we are not actively

TorchScript，但我们目前没有积极……

2946.079–2948.309  the torch script but we are not actively working on new features or have anyone

TorchScript，但我们目前没有开发新功能，也没有人……

2948.309–2948.319  working on new features or have anyone

开发新功能，也没有人……

2948.319–2950.230  working on new features or have anyone who actively like triages in common

开发新功能，也没有人定期处理常见……

2950.230–2950.240  who actively like triages in common

定期处理常见……

2950.240–2952.549  who actively like triages in common issues. So unless they are security

定期处理常见问题。所以，除非涉及安全问题……

2952.549–2952.559  issues. So unless they are security

问题。所以，除非是安全

2952.559–2956.950  issues. So unless they are security related they will probably stay as is

问题。所以，除非涉及安全，否则大概会保持原样。

2956.950–2956.960  related they will probably stay as is

相关，否则大概会保持原样。

2956.960–2958.950  related they will probably stay as is and craft and script is not a security

相关，否则大概会保持原样；craft 和 script 不属于安全

2958.950–2958.960  and craft and script is not a security

而 craft 和 script 不属于安全

2958.960–2962.790  and craft and script is not a security issue unfortunately.

而 craft 和 script 很遗憾不属于安全问题。

2965.270–2965.280  >> Okay we have I think we have one left

好，我想我们还剩一个

2965.280–2969.190  >> Okay we have I think we have one left one more remaining um prepared question.

好，我想我们还剩最后一个事先准备的问题。

2969.190–2969.200  one more remaining um prepared question.

还剩最后一个事先准备的问题。

2969.200–2970.950  one more remaining um prepared question. Um

还剩最后一个事先准备的问题。嗯，

2970.950–2970.960  Um

嗯。

2970.960–2973.430  Um >> we switch to community ones or whatever.

嗯。我们要转到社区提问之类的吗？

2973.430–2973.440  >> we switch to community ones or whatever.

我们要转到社区提问之类的吗？

2973.440–2975.270  >> we switch to community ones or whatever. >> No I

我们要转到社区提问之类的吗？不，我……

2975.270–2975.280  >> No I

不，我……

2975.280–2976.549  >> No I okay I think we have one more prepared

不，我想我们还有一个事先准备的

2976.549–2976.559  okay I think we have one more prepared

好，我想我们还有一个事先准备的

2976.559–2977.990  okay I think we have one more prepared one. If there are community ones let's

好，我想我们还有一个事先准备的问题。如果有社区提问，我们

2977.990–2978.000  one. If there are community ones let's

问题。如果有社区提问，我们

2978.000–2979.430  one. If there are community ones let's go back and make sure we catch them

问题。如果有社区提问，我们回头确认一下，

2979.430–2979.440  go back and make sure we catch them

回头确认一下，别漏掉。

2979.440–2982.150  go back and make sure we catch them before we end. Um, so I hear that uh

回头确认一下，结束前别漏掉。嗯，我听说

2982.150–2982.160  before we end. Um, so I hear that uh

结束前别漏掉。嗯，我听说

2982.160–2986.069  before we end. Um, so I hear that uh 21.14 supports Python 315. Can I just

结束前别漏掉。嗯，我听说 2.14 支持 Python 3.15。我能直接

2986.069–2986.079  21.14 supports Python 315. Can I just

2.14 支持 Python 3.15。我能直接

2986.079–2988.470  21.14 supports Python 315. Can I just pip install Torch on it? Um, Andre, do

2.14 支持 Python 3.15。我能直接用 pip install torch 安装吗？Andre，

2988.470–2988.480  pip install Torch on it? Um, Andre, do

用 pip install torch 安装吗？Andre，你

2988.480–2989.910  pip install Torch on it? Um, Andre, do you want to take this one?

用 pip install torch 安装吗？Andre，你来回答这个问题好吗？

2989.910–2989.920  you want to take this one?

你来回答这个问题好吗？

2989.920–2992.309  you want to take this one? >> Um, yes. Uh, so basically the short

你来回答这个问题好吗？嗯，可以。简单

2992.309–2992.319  >> Um, yes. Uh, so basically the short

嗯，可以。简单来说，

2992.319–2995.990  >> Um, yes. Uh, so basically the short answer, we do support 315. Uh, but it's

嗯，可以。简单来说，我们确实支持 Python 3.15，但目前

2995.990–2996.000  answer, we do support 315. Uh, but it's

来说，我们确实支持 Python 3.15，但目前

2996.000–2999.109  answer, we do support 315. Uh, but it's only available via download pytorch.org.

来说，我们确实支持 Python 3.15，但目前只能从 download.pytorch.org 下载。

2999.109–2999.119  only available via download pytorch.org.

只能从 download.pytorch.org 下载。

2999.119–3002.630  only available via download pytorch.org. So pip install torch will not work with

只能从 download.pytorch.org 下载。所以直接运行 pip install torch

3002.630–3002.640  So pip install torch will not work with

所以直接运行 pip install torch 无法安装

3002.640–3007.349  So pip install torch will not work with 315 for this release 2014. Um

所以在 2.14 版本中，无法通过 pip install torch 安装 Python 3.15 对应的版本。

3007.349–3007.359  315 for this release 2014. Um

在 2.14 版本中，Python 3.15 对应的版本还不行。嗯，

3007.359–3009.910  315 for this release 2014. Um having said that in the future uh we are

在 2.14 版本中，Python 3.15 对应的版本还不行。不过将来我们

3009.910–3009.920  having said that in the future uh we are

不过将来我们

3009.920–3013.510  having said that in the future uh we are planning on um uh publishing for the

不过将来我们计划发布

3013.510–3013.520  planning on um uh publishing for the

计划发布

3013.520–3016.710  planning on um uh publishing for the next release 2.15 we're publishing uh

计划在下一个 2.15 版本发布

3016.710–3016.720  next release 2.15 we're publishing uh

下一个 2.15 版本会发布

3016.720–3021.349  next release 2.15 we're publishing uh 315 Python to the pipy as a default one.

下一个 2.15 版本会默认将 Python 3.15 对应的包发布到 PyPI。

3021.349–3021.359  315 Python to the pipy as a default one.

默认将 Python 3.15 对应的包发布到 PyPI。

3021.359–3025.030  315 Python to the pipy as a default one. We also be duplicating Python 3.10 for

默认将 Python 3.15 对应的包发布到 PyPI。我们还会为 Python 3.10 同时构建一份，

3025.030–3025.040  We also be duplicating Python 3.10 for

我们还会为 Python 3.10 同时构建一份，

3025.040–3026.630  We also be duplicating Python 3.10 for the next release. I think it's very

下一个版本也会为 Python 3.10 同时构建一份。我觉得这很

3026.630–3026.640  the next release. I think it's very

下个版本。我觉得这很……

3026.640–3029.829  the next release. I think it's very important if you're still running 3.10

下个版本。如果你还在用 3.10，这一点很重要。

3029.829–3029.839  important if you're still running 3.10

如果你还在用 3.10，这一点很重要。

3029.839–3031.829  important if you're still running 3.10 for next release it will not be

如果你还在用 3.10，下个版本将不再……

3031.829–3031.839  for next release it will not be

下个版本将不再……

3031.839–3035.349  for next release it will not be supported. no wheels published and um

下个版本将不再支持，也不会发布 wheel 包，嗯……

3035.349–3035.359  supported. no wheels published and um

不再支持，也不会发布 wheel 包，嗯……

3035.359–3037.349  supported. no wheels published and um compiling from source will not be

不再支持，也不会发布 wheel 包，而且也无法从源码编译……

3037.349–3037.359  compiling from source will not be

也无法从源码编译……

3037.359–3040.069  compiling from source will not be possible. So we'll be migrating or

也无法从源码编译。所以我们会迁移，或者……

3040.069–3040.079  possible. So we'll be migrating or

所以我们会迁移，或者……

3040.079–3042.710  possible. So we'll be migrating or supporting minimum 311.

所以我们会迁移，或者将最低支持版本设为 3.11。

3042.710–3042.720  supporting minimum 311.

最低支持版本是 3.11。

3042.720–3046.470  supporting minimum 311. Um but yes for the current version 310

最低支持版本是 3.11。不过当前版本仍支持 3.10……

3046.470–3046.480  Um but yes for the current version 310

不过当前版本仍支持 3.10……

3046.480–3051.109  Um but yes for the current version 310 3.15 including uh free threaded 3.15T is

不过当前版本支持 3.10 到 3.15，包括自由线程版 3.15T。

3051.109–3051.119  3.15 including uh free threaded 3.15T is

3.15，包括自由线程版 3.15T，都……

3051.119–3054.069  3.15 including uh free threaded 3.15T is are supported. However, one more call

3.15，包括自由线程版 3.15T，都受支持。不过还要说明一点……

3054.069–3054.079  are supported. However, one more call

都受支持。不过还要说明一点……

3054.079–3056.390  are supported. However, one more call out is that torch compile is still not

都受支持。不过还要说明一点：torch.compile 仍未……

3056.390–3056.400  out is that torch compile is still not

还要说明一点：torch.compile 仍未……

3056.400–3060.230  out is that torch compile is still not enabled for 315 and 315T.

还要说明一点：torch.compile 尚未对 3.15 和 3.15T 启用。

3060.230–3060.240  enabled for 315 and 315T.

尚未对 3.15 和 3.15T 启用。

3060.240–3063.829  enabled for 315 and 315T. Um yes I think that's that's about uh

尚未对 3.15 和 3.15T 启用。嗯，我想差不多就是这些。

3063.829–3063.839  Um yes I think that's that's about uh

嗯，我想差不多就是这些。

3063.839–3066.470  Um yes I think that's that's about uh covers it. I don't know if anybody else

嗯，我想这差不多说全了。不知道其他人是否……

3066.470–3066.480  covers it. I don't know if anybody else

差不多说全了。不知道其他人是否……

3066.480–3070.710  covers it. I don't know if anybody else have any additional thoughts.

差不多说全了。不知道其他人还有没有补充。

3070.710–3070.720  have any additional thoughts.

还有没有补充？

3070.720–3073.109  have any additional thoughts. Oh, maybe like kind of a reason why we

还有没有补充？哦，也许该说说我们为什么……

3073.109–3073.119  Oh, maybe like kind of a reason why we

哦，也许该说说我们为什么……

3073.119–3075.670  Oh, maybe like kind of a reason why we don't support 310 anymore because uh it

哦，也许该解释一下为什么不再支持 3.10，因为它……

3075.670–3075.680  don't support 310 anymore because uh it

不再支持 3.10，是因为它……

3075.680–3079.030  don't support 310 anymore because uh it reached its end of life per um like uh

不再支持 3.10，是因为它已经到了生命周期终点……

3079.030–3079.040  reached its end of life per um like uh

它已经到了生命周期终点……

3079.040–3081.190  reached its end of life per um like uh Python end of life policy for the

按照 Python 的版本生命周期政策，它已经到了生命周期终点……

3081.190–3081.200  Python end of life policy for the

按照 Python 的版本生命周期政策……

3081.200–3084.549  Python end of life policy for the releases and we also will be like so we

按照 Python 的版本生命周期政策，我们也会……

3084.549–3084.559  releases and we also will be like so we

我们也会……

3084.559–3086.390  releases and we also will be like so we constantly reviewing the supported

我们会持续审查受支持的……

3086.390–3086.400  constantly reviewing the supported

持续审查受支持的……

3086.400–3088.390  constantly reviewing the supported platforms and unfortunately retires the

我们持续审查受支持的平台，遗憾的是会淘汰……

3088.390–3088.400  platforms and unfortunately retires the

平台，遗憾的是会淘汰……

3088.400–3091.670  platforms and unfortunately retires the ones that like go out of support so we

遗憾的是，不再受支持的平台会被淘汰，所以我们……

3091.670–3091.680  ones that like go out of support so we

不再受支持的平台会被淘汰，所以我们……

3091.680–3093.109  ones that like go out of support so we probably also be stopping supporting

不再受支持的平台会被淘汰，所以我们可能也会停止支持……

3093.109–3093.119  probably also be stopping supporting

可能也会停止支持……

3093.119–3095.750  probably also be stopping supporting like Mac OS I forget what was released

可能也会停止支持某个 macOS 版本，我忘了三年前发布的是哪一版……

3095.750–3095.760  like Mac OS I forget what was released

比如三年前发布的 macOS 版本，我忘了具体是哪一版……

3095.760–3097.829  like Mac OS I forget what was released from three years ago but the typical and

比如三年前发布的 macOS 版本，我忘了具体是哪一版，但通常……

3097.829–3097.839  from three years ago but the typical and

三年前的，但通常……

3097.839–3099.670  from three years ago but the typical and life cycle for those releases are also

三年前的，但这些版本的正常生命周期也是……

3099.670–3099.680  life cycle for those releases are also

这些版本的生命周期也是……

3099.680–3102.470  life cycle for those releases are also three So we will be returning that one

这些版本的生命周期也是三年，所以我们也会停止支持那个版本。

3102.470–3102.480  three So we will be returning that one

三年，所以我们也会停止支持那个版本。

3102.480–3105.589  three So we will be returning that one as well. And again CUDA 12 that was

三年，所以我们也会停止支持那个版本。另外，刚才提到的 CUDA 12……

3105.589–3105.599  as well. And again CUDA 12 that was

另外，刚才提到的 CUDA 12……

3105.599–3107.270  as well. And again CUDA 12 that was mentioned before. Let's repeat it again

另外，刚才提到的 CUDA 12，我们再说一遍……

3107.270–3107.280  mentioned before. Let's repeat it again

刚才提到过，我们再说一遍……

3107.280–3109.270  mentioned before. Let's repeat it again that CUDA 12 support is going away. So

刚才提到过，我们再说一遍：CUDA 12 的支持即将结束。

3109.270–3109.280  that CUDA 12 support is going away. So

CUDA 12 的支持即将结束。所以……

3109.280–3110.630  that CUDA 12 support is going away. So you should still be able to build from

CUDA 12 的支持即将结束。不过你应该仍能从源码……

3110.630–3110.640  you should still be able to build from

你应该仍能从源码……

3110.640–3113.270  you should still be able to build from source. We try not to break that thing.

你应该仍能从源码构建。我们会尽量保证这条路可行。

3113.270–3113.280  source. We try not to break that thing.

从源码构建。我们会尽量保证这条路可行。

3113.280–3115.270  source. We try not to break that thing. And I don't think we will actively kind

从源码构建。我们会尽量保证这条路可行。我想我们也不会主动……

3115.270–3115.280  And I don't think we will actively kind

我想我们也不会主动……

3115.280–3118.309  And I don't think we will actively kind of break 310 but we will not be done

我想我们不会主动让 3.10 无法使用，但我们不会再……

3118.309–3118.319  of break 310 but we will not be done

让 3.10 无法使用，但我们不会再……

3118.319–3121.349  of break 310 but we will not be done like testing it. So it will get broken

让 3.10 无法使用，但我们不会再测试它，所以它迟早会出问题。

3121.349–3121.359  like testing it. So it will get broken

测试它，所以它迟早会出问题。

3121.359–3123.270  like testing it. So it will get broken eventually.

测试它，所以它迟早会出问题。

3123.270–3123.280  eventually.

迟早会出问题。

3123.280–3124.630  eventually. Cool.

迟早会出问题。好。

3124.630–3124.640  Cool.

好。

3124.640–3126.390  Cool. >> Okay. I think we have a couple more

好。>> 好的，我想我们还有几个……

3126.390–3126.400  >> Okay. I think we have a couple more

>> 好的，我想我们还有几个……

3126.400–3129.910  >> Okay. I think we have a couple more questions lined up. Um, so, uh, Brian

>> 好的，我想我们还有几个问题。Brian……

3129.910–3129.920  questions lined up. Um, so, uh, Brian

还有几个问题。Brian……

3129.920–3131.990  questions lined up. Um, so, uh, Brian asks, "How can I use PyTorch to create

还有几个问题。Brian 问：“怎样用 PyTorch 创建……

3131.990–3132.000  asks, "How can I use PyTorch to create

问：“怎样用 PyTorch 创建……

3132.000–3134.069  asks, "How can I use PyTorch to create my own LLMs?"

问：“怎样用 PyTorch 创建自己的大语言模型？”

3134.069–3134.079  my own LLMs?"

自己的大语言模型？”

3134.079–3137.510  my own LLMs?" Anyone [laughter] want to?

自己的大语言模型？”有人想答吗？［笑声］

3137.510–3137.520  Anyone [laughter] want to?

有人想答吗？［笑声］

3137.520–3139.510  Anyone [laughter] want to? >> There are lots of great tutorials. I

有人想答吗？［笑声］>> 有很多很棒的教程。我……

3139.510–3139.520  >> There are lots of great tutorials. I

>> 有很多很棒的教程。我……

3139.520–3141.910  >> There are lots of great tutorials. I want to say that like that tell how you

>> 有很多很棒的教程。我想说，它们会教你怎么……

3141.910–3141.920  want to say that like that tell how you

我想说，它们会教你怎么……

3141.920–3143.109  want to say that like that tell how you can write

我想说，它们会教你怎么编写……

3143.109–3143.119  can write

编写……

3143.119–3145.349  can write >> like LLM training and inference like

编写……>> 比如大语言模型的训练和推理……

3145.349–3145.359  >> like LLM training and inference like

>> 比如大语言模型的训练和推理……

3145.359–3149.030  >> like LLM training and inference like Andrew Corp's uh like tiny story will be

>> 比如大语言模型的训练和推理，Andrew Corp 的 TinyStory……

3149.030–3149.040  Andrew Corp's uh like tiny story will be

Andrew Corp 的 TinyStory……

3149.040–3150.790  Andrew Corp's uh like tiny story will be like a good one if you want to get

Andrew Corp 的 TinyStory 就很适合想要……

3150.790–3150.800  like a good one if you want to get

很适合想要……

3150.800–3153.510  like a good one if you want to get familiar with LLMs. But if you want to

很适合想要了解大语言模型的人。但如果你想……

3153.510–3153.520  familiar with LLMs. But if you want to

了解大语言模型。但如果你想……

3153.520–3155.670  familiar with LLMs. But if you want to train your own [laughter] like I don't

了解大语言模型。但如果你想训练自己的模型，［笑声］我就不……

3155.670–3155.680  train your own [laughter] like I don't

自己训练一个（笑），比如我也不……

3155.680–3158.150  train your own [laughter] like I don't know uh

自己训练一个（笑），比如我也不知道，呃……

3158.150–3158.160  know uh

知道，呃……

3158.160–3161.430  know uh leading model using just PyTorch you

知道，呃，只用 PyTorch 训练顶尖模型，你……

3161.430–3161.440  leading model using just PyTorch you

只用 PyTorch 训练顶尖模型，你……

3161.440–3163.109  leading model using just PyTorch you need a pretty substantial hardware

只用 PyTorch 训练顶尖模型，你需要投入相当多的硬件。

3163.109–3163.119  need a pretty substantial hardware

你需要投入相当多的硬件。

3163.119–3165.589  need a pretty substantial hardware investments uh which might be a bigger

需要投入相当多的硬件，呃，这可能是更大的……

3165.589–3165.599  investments uh which might be a bigger

硬件投入，呃，这可能是更大的……

3165.599–3166.470  investments uh which might be a bigger blocker

硬件投入，呃，这可能是更大的障碍。

3166.470–3166.480  blocker

障碍。

3166.480–3169.349  blocker >> and also you need a lot of data right so

障碍。>> 而且还需要大量数据，对吧？所以……

3169.349–3169.359  >> and also you need a lot of data right so

>> 而且还需要大量数据，对吧？所以……

3169.359–3172.470  >> and also you need a lot of data right so like I think the bottleneck is not your

>> 而且还需要大量数据，对吧？所以我觉得瓶颈不在于你的……

3172.470–3172.480  like I think the bottleneck is not your

我觉得瓶颈不在于你的……

3172.480–3175.030  like I think the bottleneck is not your framework but your infrastructure and

我觉得瓶颈不在于你的框架，而在于基础设施和……

3175.030–3175.040  framework but your infrastructure and

框架，而在于基础设施和……

3175.040–3176.950  framework but your infrastructure and your data

框架，而在于基础设施和数据。

3176.950–3176.960  your data

你的数据。

3176.960–3179.270  your data >> yeah I was going to say

你的数据。>> 对，我正想说。

3179.270–3179.280  >> yeah I was going to say

>> 对，我正想说。

3179.280–3180.790  >> yeah I was going to say >> there there are some folks out there

>> 对，我正想说。>> 确实有人在……

3180.790–3180.800  >> there there are some folks out there

>> 确实有人在……

3180.800–3182.710  >> there there are some folks out there creating opensource models models that

>> 确实有人在开发开源模型，这些模型……

3182.710–3182.720  creating opensource models models that

开发开源模型，这些模型……

3182.720–3183.990  creating opensource models models that you could then maybe [laughter]

开发开源模型，之后你或许可以（笑）……

3183.990–3184.000  you could then maybe [laughter]

之后你或许可以（笑）……

3184.000–3187.030  you could then maybe [laughter] refine. I don't know, Joe.

之后你或许可以（笑）微调一下。我也说不准，Joe。

3187.030–3187.040  refine. I don't know, Joe.

微调一下。我也说不准，Joe。

3187.040–3190.390  refine. I don't know, Joe. >> Hopefully soon. Yeah, hopefully. Uh when

微调一下。我也说不准，Joe。>> 希望很快吧。对，希望如此。呃，等……

3190.390–3190.400  >> Hopefully soon. Yeah, hopefully. Uh when

>> 希望很快吧。对，希望如此。呃，等……

3190.400–3192.069  >> Hopefully soon. Yeah, hopefully. Uh when those models models are hard to build, I

>> 希望很快吧。对，希望如此。呃，那些模型很难做，我……

3192.069–3192.079  those models models are hard to build, I

那些模型很难做，我……

3192.079–3195.190  those models models are hard to build, I have to say. Um no, I I I think like

那些模型很难做，我得承认。嗯，不，我觉得……

3195.190–3195.200  have to say. Um no, I I I think like

得承认。嗯，不，我觉得……

3195.200–3196.549  have to say. Um no, I I I think like building models from scratch, like

得承认。嗯，不，我觉得从零开始构建模型，比如……

3196.549–3196.559  building models from scratch, like

从零开始构建模型，比如……

3196.559–3198.470  building models from scratch, like really good ones obviously is expensive.

从零开始构建模型，尤其是好模型，显然很烧钱。

3198.470–3198.480  really good ones obviously is expensive.

尤其是好模型，显然很烧钱。

3198.480–3199.990  really good ones obviously is expensive. I think, you know, to Nikita's point,

尤其是好模型，显然很烧钱。我觉得，正如 Nikita 所说，

3199.990–3200.000  I think, you know, to Nikita's point,

我觉得，正如 Nikita 所说，

3200.000–3202.309  I think, you know, to Nikita's point, there's a definitely hardware is is a

我觉得，正如 Nikita 所说，硬件确实是个……

3202.309–3202.319  there's a definitely hardware is is a

硬件确实是个……

3202.319–3204.470  there's a definitely hardware is is a big challenge. I think there there's

硬件确实是个大挑战。我觉得还有……

3204.470–3204.480  big challenge. I think there there's

大挑战。我觉得还有……

3204.480–3206.870  big challenge. I think there there's some really cool like I I'll I'll plug

大挑战。我觉得有些很棒的东西，我想推荐……

3206.870–3206.880  some really cool like I I'll I'll plug

有些很棒的东西，我想推荐……

3206.880–3208.790  some really cool like I I'll I'll plug Unslo because like Daniel and Michael

有些很棒的东西，我想推荐 Unslo，因为 Daniel 和 Michael……

3208.790–3208.800  Unslo because like Daniel and Michael

Unslo，因为 Daniel 和 Michael……

3208.800–3210.069  Unslo because like Daniel and Michael are really great and they have a great

Unslo，因为 Daniel 和 Michael 很出色，他们还有一个很棒的……

3210.069–3210.079  are really great and they have a great

真的很棒，而且他们有很好的……

3210.079–3212.390  are really great and they have a great ecosystem. They have great tools. Uh you

真的很棒，生态和工具都很出色。你……

3212.390–3212.400  ecosystem. They have great tools. Uh you

生态和工具都很出色。你……

3212.400–3214.630  ecosystem. They have great tools. Uh you could basically pick out, you know, open

生态和工具都很出色。你基本上可以挑选开源……

3214.630–3214.640  could basically pick out, you know, open

你基本上可以挑选开源……

3214.640–3216.790  could basically pick out, you know, open source models and you can use unsloth

你基本上可以挑选开源模型，再用 Unsloth

3216.790–3216.800  source models and you can use unsloth

开源模型，再用 Unsloth

3216.800–3220.309  source models and you can use unsloth and and and kind of like do RL and and

开源模型，再用 Unsloth 做 RL 等训练

3220.309–3220.319  and and and kind of like do RL and and

做 RL 等训练

3220.319–3221.910  and and and kind of like do RL and and SFT and and be able to do it really

做 RL 和 SFT，而且可以做得很快

3221.910–3221.920  SFT and and be able to do it really

做 SFT，而且可以做得很快

3221.920–3224.549  SFT and and be able to do it really quick like quickly and cheaply. Uh even

做 SFT，速度快，成本也低，甚至……

3224.549–3224.559  quick like quickly and cheaply. Uh even

速度快，成本也低，甚至……

3224.559–3226.870  quick like quickly and cheaply. Uh even in notebooks um like there's actually a

速度快，成本也低。甚至在 Notebook 里就能做，那里有……

3226.870–3226.880  in notebooks um like there's actually a

在 Notebook 里就能做，那里有……

3226.880–3228.950  in notebooks um like there's actually a ton of examples there. So uh but like

在 Notebook 里就能做，那里还有大量示例。不过……

3228.950–3228.960  ton of examples there. So uh but like

那里有大量示例。不过……

3228.960–3230.470  ton of examples there. So uh but like creating it from scratch is like quite

那里有大量示例。不过从零开始做就相当……

3230.470–3230.480  creating it from scratch is like quite

从零开始做就相当……

3230.480–3233.030  creating it from scratch is like quite expensive u quite involved. Uh we've

从零开始做成本很高，也很费工夫。我们……

3233.030–3233.040  expensive u quite involved. Uh we've

成本很高，也很费工夫。我们……

3233.040–3235.109  expensive u quite involved. Uh we've kind of um we're we're at the point now

成本很高，也很费工夫。我们现在已经到了……

3235.109–3235.119  kind of um we're we're at the point now

我们现在已经到了……

3235.119–3236.710  kind of um we're we're at the point now as a community where it's just it's like

我们这个社区现在已经到了这样一个阶段……

3236.710–3236.720  as a community where it's just it's like

我们这个社区现在的情况是……

3236.720–3239.030  as a community where it's just it's like prohibitively expensive to to do that as

我们这个社区现在的情况是，自己做的成本高得难以承受

3239.030–3239.040  prohibitively expensive to to do that as

自己做的成本高得难以承受

3239.040–3241.109  prohibitively expensive to to do that as an individual. Um, but that said,

个人根本负担不起这个成本。不过话说回来，

3241.109–3241.119  an individual. Um, but that said,

个人负担不起。不过话说回来，

3241.119–3242.710  an individual. Um, but that said, there's a ton of great open models out

个人负担不起。不过话说回来，优秀的开放模型有很多

3242.710–3242.720  there's a ton of great open models out

优秀的开放模型有很多

3242.720–3244.230  there's a ton of great open models out there. Um, and you can play with them

优秀的开放模型有很多，你可以拿来试试

3244.230–3244.240  there. Um, and you can play with them

你可以拿来试试

3244.240–3245.670  there. Um, and you can play with them and you can customize them and you can

你可以拿来试试，也可以定制

3245.670–3245.680  and you can customize them and you can

你可以定制它们，还可以

3245.680–3247.430  and you can customize them and you can run them locally now with Olana and

你可以定制它们，现在还能用 Ollama 在本地运行

3247.430–3247.440  run them locally now with Olana and

现在还能用 Ollama 在本地运行

3247.440–3249.510  run them locally now with Olana and other platforms. Um, so I think that's

现在还能用 Ollama 等平台在本地运行。所以我觉得……

3249.510–3249.520  other platforms. Um, so I think that's

还有其他平台。所以我觉得……

3249.520–3252.069  other platforms. Um, so I think that's like um, yeah, obviously Karpath's uh,

还有其他平台。所以我觉得，当然，Karpathy 的……

3252.069–3252.079  like um, yeah, obviously Karpath's uh,

当然，Karpathy 的……

3252.079–3254.549  like um, yeah, obviously Karpath's uh, project is is super cool as well. U,

当然，Karpathy 的项目也特别酷。

3254.549–3254.559  project is is super cool as well. U,

那个项目也特别酷。

3254.559–3256.069  project is is super cool as well. U, there's a lot going on there. Um, but I

那个项目也特别酷，相关进展很多。不过我……

3256.069–3256.079  there's a lot going on there. Um, but I

相关进展很多。不过我……

3256.079–3257.109  there's a lot going on there. Um, but I wouldn't worry about training from

相关进展很多。不过我不会担心从零开始训练

3257.109–3257.119  wouldn't worry about training from

我不会担心从零开始训练

3257.119–3258.150  wouldn't worry about training from scratch because there's just a ton to

我不会担心从零开始训练，因为现在有大量现成的东西

3258.150–3258.160  scratch because there's just a ton to

从零开始训练没必要，因为现在有大量现成的东西

3258.160–3259.510  scratch because there's just a ton to build on right now. It's just readily

因为现在有大量现成的东西可以作为基础，而且很容易获取

3259.510–3259.520  build on right now. It's just readily

现在就能用它来开发，现成的。

3259.520–3261.670  build on right now. It's just readily available.

现在就能用它来开发，随时可用。

3261.670–3261.680  available.

随时可用。

3261.680–3265.750  available. >> Yeah. Awesome. Uh, I think we had one or

随时可用。对，太好了。我想我们还有一两个……

3265.750–3265.760  >> Yeah. Awesome. Uh, I think we had one or

对，太好了。我想我们还有一两个……

3265.760–3268.549  >> Yeah. Awesome. Uh, I think we had one or two more questions. Um so uh this one is

对，太好了。我想我们还有一两个问题。这个问题来自……

3268.549–3268.559  two more questions. Um so uh this one is

还有两个问题。这个问题来自……

3268.559–3271.109  two more questions. Um so uh this one is from Manukumar.

还有两个问题。这个问题来自 Manukumar。

3271.109–3271.119  from Manukumar.

来自 Manukumar。

3271.119–3273.030  from Manukumar. Uh

来自 Manukumar。呃……

3273.030–3273.040  Uh

呃……

3273.040–3275.589  Uh are there any uh Google Summer of Code

呃，有没有 Google Summer of Code……

3275.589–3275.599  are there any uh Google Summer of Code

有没有 Google Summer of Code……

3275.599–3278.950  are there any uh Google Summer of Code uh projects uh focused around PyTorch? I

有没有聚焦 PyTorch 的 Google Summer of Code 项目？我……

3278.950–3278.960  uh projects uh focused around PyTorch? I

有没有聚焦 PyTorch 的项目？我……

3278.960–3281.109  uh projects uh focused around PyTorch? I don't know if there are. Does anybody

有没有聚焦 PyTorch 的项目？我不知道有没有。有人……

3281.109–3281.119  don't know if there are. Does anybody

我不知道有没有。有人……

3281.119–3285.190  don't know if there are. Does anybody know among this group?

我不知道有没有。在座有谁知道吗？

3285.190–3285.200  know among this group?

在座有谁知道吗？

3285.200–3287.190  know among this group? >> I don't know. But that's a good idea.

在座有谁知道吗？我不知道，但这主意不错。

3287.190–3287.200  >> I don't know. But that's a good idea.

我不知道，但这主意不错。

3287.200–3290.710  >> I don't know. But that's a good idea. Like I think

我不知道，但这主意不错。我觉得……

3294.390–3294.400  maybe pitch a few few ideas to Google.

或许可以向 Google 提几个想法。

3294.400–3295.990  maybe pitch a few few ideas to Google. But like as name suggest, it's a Google

或许可以向 Google 提几个想法。不过顾名思义，这是 Google……

3295.990–3296.000  But like as name suggest, it's a Google

不过顾名思义，这是 Google……

3296.000–3299.349  But like as name suggest, it's a Google summer of code. We don't run it. So if

不过顾名思义，这是 Google Summer of Code，不是我们主办的。所以如果……

3299.349–3299.359  summer of code. We don't run it. So if

Google Summer of Code 不是我们主办的。所以如果……

3299.359–3301.030  summer of code. We don't run it. So if somebody wants to organize it, that

Google Summer of Code 不是我们主办的。如果有人愿意组织，那……

3301.030–3301.040  somebody wants to organize it, that

如果有人愿意组织，那……

3301.040–3302.390  somebody wants to organize it, that sounds great. And like

如果有人愿意组织，那就太好了。而且……

3302.390–3302.400  sounds great. And like

那就太好了。而且……

3302.400–3302.870  sounds great. And like >> yeah,

那就太好了。而且……对。

3302.870–3302.880  >> yeah,

对。

3302.880–3305.030  >> yeah, >> if somebody

对。如果有人……

3305.030–3305.040  >> if somebody

如果有人……

3305.040–3307.510  >> if somebody >> Yeah, we'll be happy to support.

如果有人……对，我们很乐意支持。

3307.510–3307.520  >> Yeah, we'll be happy to support.

对，我们很乐意支持。

3307.520–3308.069  >> Yeah, we'll be happy to support. >> Yep.

对，我们很乐意支持。没错。

3308.069–3308.079  >> Yep.

没错。

3308.079–3310.230  >> Yep. >> We should we should reach out to our uh

没错。我们应该联系一下……

3310.230–3310.240  >> We should we should reach out to our uh

我们应该联系一下……

3310.240–3313.910  >> We should we should reach out to our uh our our Google um PyTorch Foundation. Uh

我们应该联系一下 Google 和 PyTorch Foundation 那边，呃……

3313.910–3313.920  our our Google um PyTorch Foundation. Uh

Google 和 PyTorch Foundation 那边，呃……

3313.920–3314.309  our our Google um PyTorch Foundation. Uh uh

Google 和 PyTorch Foundation 那边，呃……

3314.309–3314.319  uh

呃……

3314.319–3315.109  uh >> there you go.

呃……对，就是这样。

3315.109–3315.119  >> there you go.

对，就是这样。

3315.119–3318.630  >> there you go. >> And see what we can help.

对，就是这样。看看我们能帮上什么忙。

3318.630–3318.640  >> And see what we can help.

看看我们能帮上什么忙。

3318.640–3322.069  >> And see what we can help. >> Exactly. Okay. Awesome. Um, Basil, were

看看我们能帮上什么忙。没错。好，太棒了。呃，Basil，你是不是……

3322.069–3322.079  >> Exactly. Okay. Awesome. Um, Basil, were

没错。好，太棒了。Basil，还有……

3322.079–3323.910  >> Exactly. Okay. Awesome. Um, Basil, were there any more questions in the in the

没错。好，太棒了。Basil，评论区还有其他问题吗？

3323.910–3323.920  there any more questions in the in the

还有其他问题吗？

3323.920–3328.069  there any more questions in the in the comments? Oh, okay. Uh, no, I think

评论区还有其他问题吗？哦，好。我想没有了。

3328.069–3328.079  comments? Oh, okay. Uh, no, I think

评论区吗？哦，好。我想没有了。

3328.079–3329.510  comments? Oh, okay. Uh, no, I think Okay. Well, yeah, this is a slightly

评论区吗？哦，好。我想没有了。好，这个问题稍有不同。

3329.510–3329.520  Okay. Well, yeah, this is a slightly

好，这个问题稍有不同。

3329.520–3330.870  Okay. Well, yeah, this is a slightly different question. With the silicon

好，这个问题稍有不同。随着芯片市场……

3330.870–3330.880  different question. With the silicon

这个问题稍有不同。随着芯片……

3330.880–3332.950  different question. With the silicon market exploding, how is PyTorch

这个问题稍有不同。随着芯片市场蓬勃发展，PyTorch 如何……

3332.950–3332.960  market exploding, how is PyTorch

市场蓬勃发展，PyTorch 如何……

3332.960–3334.870  market exploding, how is PyTorch ensuring that the XPU, you know,

市场蓬勃发展，PyTorch 如何确保 XPU……

3334.870–3334.880  ensuring that the XPU, you know,

如何确保 XPU……

3334.880–3337.990  ensuring that the XPU, you know, torch.xpu can compile and run um models

如何确保 XPU，也就是 torch.xpu，能编译和运行模型……

3337.990–3338.000  torch.xpu can compile and run um models

torch.xpu 能编译和运行模型……

3338.000–3340.870  torch.xpu can compile and run um models across Nvidia, AMD, and Intel chips with

torch.xpu 能在 Nvidia、AMD 和 Intel 芯片上编译和运行模型，且……

3340.870–3340.880  across Nvidia, AMD, and Intel chips with

在 Nvidia、AMD 和 Intel 芯片上，且……

3340.880–3343.589  across Nvidia, AMD, and Intel chips with um zero performance loss.

在 Nvidia、AMD 和 Intel 芯片上运行，且性能零损失。

3343.589–3343.599  um zero performance loss.

性能零损失。

3343.599–3346.870  um zero performance loss. Um

性能零损失。嗯。

3349.430–3349.440  >> I think it's a loaded question but also

我觉得这个问题有点预设前提，而且……

3349.440–3351.349  >> I think it's a loaded question but also like uh it's very hard to define what

我觉得这个问题有点预设前提，而且很难界定什么是……

3351.349–3351.359  like uh it's very hard to define what

很难界定什么是……

3351.359–3352.870  like uh it's very hard to define what zero component

很难界定什么是“零组件”。

3352.870–3352.880  zero component

零组件。

3352.880–3353.589  zero component >> right

零组件。对。

3353.589–3353.599  >> right

对。

3353.599–3356.309  >> right >> don't think there is any like hardware

对。我觉得没有哪种硬件……

3356.309–3356.319  >> don't think there is any like hardware

我觉得没有哪种硬件……

3356.319–3359.670  >> don't think there is any like hardware spec that you can say well this this

我觉得没有哪项硬件规格能让你说，这个……

3359.670–3359.680  spec that you can say well this this

规格能让你说，这个……

3359.680–3361.510  spec that you can say well this this piece of silicon is exactly the same as

规格能让你说，这块芯片和……

3361.510–3361.520  piece of silicon is exactly the same as

这块芯片和……

3361.520–3363.430  piece of silicon is exactly the same as another piece of silicon and also like a

这块芯片和另一块完全一样，而且……

3363.430–3363.440  another piece of silicon and also like a

另一块芯片，而且……

3363.440–3364.549  another piece of silicon and also like a pure

另一块芯片，还有纯粹的……

3364.549–3364.559  pure

纯粹的……

3364.559–3368.230  pure >> uh like a feature parity but as Andrew

纯粹的……嗯，比如功能对等。但正如 Andrew……

3368.230–3368.240  >> uh like a feature parity but as Andrew

嗯，比如功能对等。但正如 Andrew……

3368.240–3370.950  >> uh like a feature parity but as Andrew say there is CRC and there is in general

嗯，比如功能对等。但正如 Andrew 所说，有 CRC，整体上也有……

3370.950–3370.960  say there is CRC and there is in general

所说，有 CRC，整体上也有……

3370.960–3373.109  say there is CRC and there is in general and you should direct

所说，有 CRC，整体上也有……你应该直接……

3373.109–3373.119  and you should direct

你应该直接……

3373.119–3375.030  and you should direct like project experience development

你应该直接关注，比如项目体验方面的开发……

3375.030–3375.040  like project experience development

比如项目体验方面的开发……

3375.040–3376.710  like project experience development partnership with Intel which is part of

比如项目体验方面的开发，以及与 Intel 的合作。Intel 是……

3376.710–3376.720  partnership with Intel which is part of

与 Intel 的合作。Intel 是……

3376.720–3379.750  partnership with Intel which is part of the PyTorch Foundation and they doing

与 Intel 的合作。Intel 是 PyTorch 基金会的成员，他们正在……

3379.750–3379.760  the PyTorch Foundation and they doing

PyTorch 基金会的成员，他们正在……

3379.760–3381.670  the PyTorch Foundation and they doing their best and community doing their

PyTorch 基金会的成员。他们和社区都在尽全力……

3381.670–3381.680  their best and community doing their

他们在尽力，社区也在尽力……

3381.680–3383.990  their best and community doing their best and what you can do to make sure

他们和社区都在尽力；你也可以采取措施确保……

3383.990–3384.000  best and what you can do to make sure

你也可以采取措施确保……

3384.000–3385.670  best and what you can do to make sure that it is the case if you notice that

你也可以采取措施确保这一点。如果你发现……

3385.670–3385.680  that it is the case if you notice that

确保这一点。如果你发现……

3385.680–3388.470  that it is the case if you notice that something was working on your black fill

某些功能在你的 Blackwell 上能运行……

3388.470–3388.480  something was working on your black fill

某些功能在你的 Blackwell 上能运行……

3388.480–3390.789  something was working on your black fill and doesn't work on XPU please file an

某些功能在你的 Blackwell 上能运行，却无法在 XPU 上运行，请提交……

3390.789–3390.799  and doesn't work on XPU please file an

却无法在 XPU 上运行，请提交……

3390.799–3392.390  and doesn't work on XPU please file an issue and I guess there will be people

却无法在 XPU 上运行，请提交问题反馈。我想会有人……

3392.390–3392.400  issue and I guess there will be people

提交问题反馈。我想会有人……

3392.400–3394.630  issue and I guess there will be people who happy to answer if this can be

提交问题反馈。我想会有人愿意回答，看看能否……

3394.630–3394.640  who happy to answer if this can be

会有人愿意回答，看看能否……

3394.640–3398.950  who happy to answer if this can be addressed in some way but yeah like

会有人愿意回答，看看能否解决。不过，嗯……

3398.950–3398.960  addressed in some way but yeah like

能否解决。不过，嗯……

3398.960–3402.230  addressed in some way but yeah like we we want heterogeneity we want uh like

能否解决。不过，我们希望支持异构硬件，也希望……

3402.230–3402.240  we we want heterogeneity we want uh like

我们希望支持异构硬件，也希望……

3402.240–3404.390  we we want heterogeneity we want uh like models to work. We talked about torch

我们希望支持异构硬件，也希望模型能运行。我们谈到了 torch……

3404.390–3404.400  models to work. We talked about torch

模型能运行。我们谈到了 torch……

3404.400–3406.710  models to work. We talked about torch accelerate. We talked about CRCR. We

模型能运行。我们谈到了 torch accelerate，也谈到了 CRCR。我们还……

3406.710–3406.720  accelerate. We talked about CRCR. We

我们谈到了 torch accelerate，也谈到了 CRCR。我们还……

3406.720–3408.789  accelerate. We talked about CRCR. We talked a lot about like a newer romcom.

我们谈到了 torch accelerate，也谈到了 CRCR，还谈了很多新版 ROCm 的话题。

3408.789–3408.799  talked a lot about like a newer romcom.

还谈了很多新版 ROCm 的话题。

3408.799–3410.470  talked a lot about like a newer romcom. So there are lots of initiative but I

还谈了很多新版 ROCm 的话题。所以有不少举措，但我……

3410.470–3410.480  So there are lots of initiative but I

所以有不少举措，但我……

3410.480–3412.950  So there are lots of initiative but I guess zero performance loss is just

所以有不少举措，但我觉得零性能损失实在……

3412.950–3412.960  guess zero performance loss is just

我觉得零性能损失实在……

3412.960–3415.510  guess zero performance loss is just unattainable and it's also I'm not sure

我觉得零性能损失无法实现，而且我也不确定……

3415.510–3415.520  unattainable and it's also I'm not sure

无法实现，而且我也不确定……

3415.520–3416.470  unattainable and it's also I'm not sure what it means by

无法实现，而且我也不确定这是什么意思。

3416.470–3416.480  what it means by

这是什么意思？

3416.480–3418.150  what it means by >> not well defined. It's not well defined

——没有明确定义，确实没有。

3418.150–3418.160  >> not well defined. It's not well defined

——没有明确定义，确实没有。

3418.160–3420.069  >> not well defined. It's not well defined because different hardware performs you

——没有明确定义，因为不同硬件的表现……

3420.069–3420.079  because different hardware performs you

因为不同硬件的表现……

3420.079–3422.870  because different hardware performs you know just just behaves differently. Um

因为不同硬件的表现本来就不一样。嗯……

3422.870–3422.880  know just just behaves differently. Um

表现本来就不一样。嗯……

3422.880–3424.470  know just just behaves differently. Um and we try and give you lots of knobs

表现本来就不一样。我们也尽量提供很多调节选项……

3424.470–3424.480  and we try and give you lots of knobs

我们也尽量提供很多调节选项……

3424.480–3427.109  and we try and give you lots of knobs and lots of capabilities but uh

我们也尽量提供很多调节选项和功能，但……

3427.109–3427.119  and lots of capabilities but uh

还有很多功能，但……

3427.119–3428.630  and lots of capabilities but uh >> we try to reduce overhead with every

还有很多功能，但……——我们每次发布都努力降低开销……

3428.630–3428.640  >> we try to reduce overhead with every

——我们每次发布都努力降低开销……

3428.640–3430.309  >> we try to reduce overhead with every release like the majority of the

——我们每次发布都努力降低开销。多数版本里……

3430.309–3430.319  release like the majority of the

多数版本里……

3430.319–3431.510  release like the majority of the releases the features the kind of

多数版本里的功能，那些……

3431.510–3431.520  releases the features the kind of

版本里的功能，那些……

3431.520–3433.030  releases the features the kind of unspoken features that we don't talk

版本里的功能，那些我们没怎么提的改进……

3433.030–3433.040  unspoken features that we don't talk

那些我们没怎么提的改进……

3433.040–3435.910  unspoken features that we don't talk about is we always try to bring a little

那些我们没怎么提的改进，就是我们一直努力一点点带来的……

3435.910–3435.920  about is we always try to bring a little

我们总想带来一点

3435.920–3439.829  about is we always try to bring a little bit of performance here and there.

我们总想在各处提升一点性能。

3439.829–3439.839  bit of performance here and there.

在各处提升一点性能。

3439.839–3441.829  bit of performance here and there. >> Yeah, just to add to it I think one of

在各处提升一点性能。对，我补充一下，我觉得有个

3441.829–3441.839  >> Yeah, just to add to it I think one of

对，我补充一下，我觉得有个

3441.839–3443.829  >> Yeah, just to add to it I think one of the interesting project maybe for you to

对，我补充一下，我觉得有个项目可能值得你

3443.829–3443.839  the interesting project maybe for you to

有个项目可能值得你

3443.839–3447.109  the interesting project maybe for you to look into is VLM. This is like they run

有个项目可能值得你看看，就是 VLM。它会运行

3447.109–3447.119  look into is VLM. This is like they run

看看 VLM。它会运行

3447.119–3449.270  look into is VLM. This is like they run multiple different models on different

看看 VLM。它会在不同硬件上运行多种

3449.270–3449.280  multiple different models on different

在不同硬件上运行多种

3449.280–3452.150  multiple different models on different hardware and like as an example so you

在不同硬件上运行多种模型。举个例子，你

3452.150–3452.160  hardware and like as an example so you

硬件。举个例子，你

3452.160–3453.910  hardware and like as an example so you can explore it a little bit and I think

硬件。举个例子，你可以稍微研究一下，我觉得

3453.910–3453.920  can explore it a little bit and I think

可以稍微研究一下，我觉得

3453.920–3456.390  can explore it a little bit and I think it's open source and the CI is open

可以稍微研究一下，我觉得它是开源的，CI 也是开放的

3456.390–3456.400  it's open source and the CI is open

它是开源的，CI 也是开放的

3456.400–3460.309  it's open source and the CI is open source so um just as an example to look

它是开源的，CI 也开源。所以，这只是一个值得

3460.309–3460.319  source so um just as an example to look

开源。所以，这只是一个值得

3460.319–3464.230  source so um just as an example to look into could be interesting.

开源。所以，这只是一个值得研究的例子。

3464.230–3464.240  into could be interesting.

研究一下可能挺有意思。

3464.240–3465.829  into could be interesting. >> Um I feel like we have answered this but

研究一下可能挺有意思。嗯，我觉得我们已经回答过这个问题，不过

3465.829–3465.839  >> Um I feel like we have answered this but

嗯，我觉得我们已经回答过这个问题，不过

3465.839–3468.150  >> Um I feel like we have answered this but I'll just quickly uh just just uh shout

嗯，我觉得我们已经回答过这个问题，不过我再简单提一下

3468.150–3468.160  I'll just quickly uh just just uh shout

我再简单提一下

3468.160–3469.910  I'll just quickly uh just just uh shout it out here. So somebody asked on the on

我再简单提一下。有人在

3469.910–3469.920  it out here. So somebody asked on the on

这个问题。有人在

3469.920–3472.630  it out here. So somebody asked on the on the comment thread um is it uh is

这个问题。有人在评论区问，

3472.630–3472.640  the comment thread um is it uh is

评论区问，

3472.640–3475.190  the comment thread um is it uh is PyTorch supporting the newest CUDA? I

评论区问，PyTorch 支持最新的 CUDA 吗？我

3475.190–3475.200  PyTorch supporting the newest CUDA? I

PyTorch 支持最新的 CUDA 吗？我

3475.200–3478.710  PyTorch supporting the newest CUDA? I think we're we're either on the newest

PyTorch 支持最新的 CUDA 吗？我想我们用的要么是最新版

3478.710–3478.720  think we're we're either on the newest

我想我们用的要么是最新版

3478.720–3480.549  think we're we're either on the newest CUDA or pretty close. And usually if you

我想我们用的要么是最新版 CUDA，要么非常接近。通常，如果你

3480.549–3480.559  CUDA or pretty close. And usually if you

CUDA，要么非常接近。通常，如果你

3480.559–3482.390  CUDA or pretty close. And usually if you jump on nightlys, you can get the latest

CUDA，要么非常接近。通常，如果你用每日构建版，就能用上最新的

3482.390–3482.400  jump on nightlys, you can get the latest

用每日构建版，就能用上最新的

3482.400–3483.750  jump on nightlys, you can get the latest and greatest.

用每日构建版，就能用上最新的功能。

3483.750–3483.760  and greatest.

最新的功能。

3483.760–3487.750  and greatest. >> Um, so Andre, you should your head when

最新的功能。嗯，Andre，我刚才说的时候，你摇了摇头

3487.750–3487.760  >> Um, so Andre, you should your head when

嗯，Andre，我刚才说的时候，你摇了摇头

3487.760–3488.870  >> Um, so Andre, you should your head when I said that. So I must have said

嗯，Andre，我刚才说的时候，你摇了摇头。所以我一定是说错了

3488.870–3488.880  I said that. So I must have said

所以我一定是说错了

3488.880–3489.990  I said that. So I must have said something slightly wrong. So jump in

所以我一定是说错了点什么。你来

3489.990–3490.000  something slightly wrong. So jump in

说错了点什么。你来

3490.000–3490.630  something slightly wrong. So jump in here and correct me.

说错了点什么。你来纠正我吧。

3490.630–3490.640  here and correct me.

来纠正我吧。

3490.640–3494.150  here and correct me. >> So for 214, we're supporting up to CUDA

来纠正我吧。关于 214，我们最高支持到 CUDA

3494.150–3494.160  >> So for 214, we're supporting up to CUDA

关于 214，我们最高支持到 CUDA

3494.160–3497.190  >> So for 214, we're supporting up to CUDA 13.2. This is not the latest. The latest

关于 214，我们最高支持到 CUDA 13.2。这不是最新版本。最新的是

3497.190–3497.200  13.2. This is not the latest. The latest

13.2。这个不是最新版。最新版是……

3497.200–3500.950  13.2. This is not the latest. The latest was um is 13.4 and it was released just

13.2。这个不是最新版。最新版是 13.4，刚发布不久……

3500.950–3500.960  was um is 13.4 and it was released just

是 13.4，刚发布不久……

3500.960–3503.670  was um is 13.4 and it was released just like a week after we released PyTorch.

是 13.4，在我们发布 PyTorch 大约一周后发布的。

3503.670–3503.680  like a week after we released PyTorch.

大约在我们发布 PyTorch 一周后。

3503.680–3505.589  like a week after we released PyTorch. So it's like we did not include it yet

大约在我们发布 PyTorch 一周后。所以我们还没把它加进去。

3505.589–3505.599  So it's like we did not include it yet

所以我们还没把它加进去。

3505.599–3507.349  So it's like we did not include it yet but it will be included in the next

所以我们还没把它加进去，但下一个版本会包含它。

3507.349–3507.359  but it will be included in the next

但下一个版本会包含它。

3507.359–3510.150  but it will be included in the next release of PyTorch and it is already in

但下一个 PyTorch 版本会包含它，而且它已经在……

3510.150–3510.160  release of PyTorch and it is already in

下一个 PyTorch 版本，而且它已经在……

3510.160–3512.870  release of PyTorch and it is already in nightly and it has been in nightly for

下一个 PyTorch 版本，而且它已经进入 nightly 版，已经有……

3512.870–3512.880  nightly and it has been in nightly for

nightly 版，而且已经在里面有……

3512.880–3515.430  nightly and it has been in nightly for quite some time. So if you really want

nightly 版，而且已经有一段时间了。所以如果你真想……

3515.430–3515.440  quite some time. So if you really want

已经有一段时间了。所以如果你真想……

3515.440–3517.910  quite some time. So if you really want to have the latest CUDA download nightly

已经有一段时间了。所以如果你想用最新的 CUDA，就下载 nightly 版。

3517.910–3517.920  to have the latest CUDA download nightly

想用最新的 CUDA，就下载 nightly 版。

3517.920–3520.950  to have the latest CUDA download nightly and it should work uh for Linux only for

想用最新的 CUDA，就下载 nightly 版。它应该能用，目前只支持 Linux；至于……

3520.950–3520.960  and it should work uh for Linux only for

它应该能用，目前只支持 Linux；至于……

3520.960–3523.430  and it should work uh for Linux only for Windows I think it's going to be enabled

它应该能用，目前只支持 Linux；Windows 我想很快也会支持。

3523.430–3523.440  Windows I think it's going to be enabled

Windows 我想很快也会支持。

3523.440–3526.069  Windows I think it's going to be enabled pretty soon maybe within one or two

Windows 我想很快也会支持，可能再过一两……

3526.069–3526.079  pretty soon maybe within one or two

可能再过一两……

3526.079–3529.109  pretty soon maybe within one or two weeks.

可能再过一两周。

3530.950–3530.960  >> Awesome. Well I think that was a good

>> 太好了。我觉得这很适合……

3530.960–3533.589  >> Awesome. Well I think that was a good note to end on. Um, I think we had uh

太好了。我觉得用这句话收尾挺好。嗯，我觉得我们……

3533.589–3533.599  note to end on. Um, I think we had uh

用这句话收尾挺好。嗯，我觉得我们……

3533.599–3535.720  note to end on. Um, I think we had uh some fun conver fun conversation.

用这句话收尾挺好。嗯，我觉得我们聊得很愉快。

3535.720–3535.730  some fun conver fun conversation.

聊得很愉快。

3535.730–3536.390  some fun conver fun conversation. [snorts]

聊得很愉快。［哼笑］

3536.390–3536.400  [snorts]

［哼笑］

3536.400–3538.390  [snorts] Um, a little a little repetition in the

［哼笑］嗯，问题有一点……

3538.390–3538.400  Um, a little a little repetition in the

嗯，问题有一点……

3538.400–3539.430  Um, a little a little repetition in the questions, but I think that's sort of

嗯，问题有一点重复，但我觉得这算是……

3539.430–3539.440  questions, but I think that's sort of

问题有一点重复，但我觉得这算是……

3539.440–3540.950  questions, but I think that's sort of built into the way we do things. So,

问题有一点重复，但这和我们的做法有关。所以，

3540.950–3540.960  built into the way we do things. So,

这和我们的做法有关。所以，

3540.960–3543.829  built into the way we do things. So, thank you uh to my experts for uh for

这和我们的做法有关。所以，感谢几位专家……

3543.829–3543.839  thank you uh to my experts for uh for

感谢几位专家……

3543.839–3547.109  thank you uh to my experts for uh for for putting up with that. Um, uh, so

感谢几位专家包容这些重复。嗯，那么……

3547.109–3547.119  for putting up with that. Um, uh, so

感谢你们包容这些重复。嗯，那么……

3547.119–3548.870  for putting up with that. Um, uh, so let's uh just in terms of closing

感谢你们包容这些重复。嗯，那么最后……

3548.870–3548.880  let's uh just in terms of closing

那么最后……

3548.880–3550.950  let's uh just in terms of closing remarks, um, thanks as always have to

那么最后说几句。和往常一样，先要感谢……

3550.950–3550.960  remarks, um, thanks as always have to

说几句。和往常一样，先要感谢……

3550.960–3552.470  remarks, um, thanks as always have to start with the contributors across the

说几句。和往常一样，先要感谢社区里的贡献者……

3552.470–3552.480  start with the contributors across the

先要感谢社区里的贡献者……

3552.480–3554.230  start with the contributors across the community who put in the steady effort

先要感谢社区里持续付出的贡献者……

3554.230–3554.240  community who put in the steady effort

社区里持续付出的贡献者……

3554.240–3557.190  community who put in the steady effort that makes PyTorch what it is. So 500

社区里持续付出的贡献者，是他们成就了今天的 PyTorch。所以，500……

3557.190–3557.200  that makes PyTorch what it is. So 500

正是这些成就了今天的 PyTorch。所以，500……

3557.200–3560.150  that makes PyTorch what it is. So 500 odd people uh contributed to this uh

正是这些成就了今天的 PyTorch。这次版本有 500 多人参与贡献。

3560.150–3560.160  odd people uh contributed to this uh

多人参与了这次版本的开发。

3560.160–3561.910  odd people uh contributed to this uh particular release and it's been a

多人参与了这次版本的开发，而这个数字……

3561.910–3561.920  particular release and it's been a

这次版本的开发，而这个数字……

3561.920–3563.670  particular release and it's been a number near that for the last several

这次版本的开发，过去几次也差不多是这个人数。

3563.670–3563.680  number near that for the last several

过去几次也差不多是这个人数。

3563.680–3565.910  number near that for the last several releases. Uh so it's just huge community

过去几次发布也是这个人数。这是庞大的社区共同努力的结果。

3565.910–3565.920  releases. Uh so it's just huge community

过去几次发布也是如此。这是庞大的社区共同努力的结果。

3565.920–3567.510  releases. Uh so it's just huge community effort. Uh and you know what we're

每次发布都凝聚了庞大社区的努力。我们……

3567.510–3567.520  effort. Uh and you know what we're

共同的努力。我们……

3567.520–3569.589  effort. Uh and you know what we're talking about here is that effort. Uh so

共同的努力。我们今天谈的正是这些努力。所以……

3569.589–3569.599  talking about here is that effort. Uh so

我们今天谈的正是这些努力。所以……

3569.599–3571.430  talking about here is that effort. Uh so uh so thanks to all those folks. Uh

我们今天谈的正是这些努力。感谢所有参与的人。

3571.430–3571.440  uh so thanks to all those folks. Uh

感谢所有参与的人。

3571.440–3573.589  uh so thanks to all those folks. Uh thanks also to the three of you uh here

感谢所有参与的人，也谢谢在场的三位。

3573.589–3573.599  thanks also to the three of you uh here

也谢谢在场的三位。

3573.599–3575.670  thanks also to the three of you uh here on on on on stage answering these

也谢谢台上的三位回答这些……

3575.670–3575.680  on on on on stage answering these

在台上回答这些……

3575.680–3577.829  on on on on stage answering these questions. Uh many of which you were

在台上回答这些问题，其中很多问题你们……

3577.829–3577.839  questions. Uh many of which you were

这些问题，其中很多问题你们……

3577.839–3579.750  questions. Uh many of which you were answering un completely unprepared

这些问题，其中很多是你们毫无准备就回答的。

3579.750–3579.760  answering un completely unprepared

你们毫无准备就作了回答。

3579.760–3581.030  answering un completely unprepared because they they were audience

你们毫无准备就作了回答，因为那都是观众……

3581.030–3581.040  because they they were audience

因为那都是观众……

3581.040–3583.109  because they they were audience questions [laughter] in. So, thank you

因为那都是观众提的问题（笑）。所以，谢谢……

3583.109–3583.119  questions [laughter] in. So, thank you

观众提的问题（笑）。所以，谢谢……

3583.119–3586.789  questions [laughter] in. So, thank you for being brave uh and wise. Um so, uh

观众提的问题（笑）。谢谢你们的勇气和智慧。

3586.789–3586.799  for being brave uh and wise. Um so, uh

谢谢你们的勇气和智慧。

3586.799–3588.069  for being brave uh and wise. Um so, uh and being generous with your time and

谢谢你们的勇气和智慧，也谢谢你们慷慨地分享时间和……

3588.069–3588.079  and being generous with your time and

也谢谢你们慷慨地分享时间和……

3588.079–3590.309  and being generous with your time and your expertise. Uh thanks also to behind

也谢谢你们慷慨地分享时间和专业知识。还要感谢幕后的……

3590.309–3590.319  your expertise. Uh thanks also to behind

专业知识。还要感谢幕后的……

3590.319–3592.470  your expertise. Uh thanks also to behind the scenes uh Basil and Jyn who were

专业知识。还要感谢幕后的 Basil 和 Jyn，他们……

3592.470–3592.480  the scenes uh Basil and Jyn who were

幕后的 Basil 和 Jyn，他们……

3592.480–3594.870  the scenes uh Basil and Jyn who were scrambling to to get the uh the the

幕后的 Basil 和 Jyn 一直忙着把……

3594.870–3594.880  scrambling to to get the uh the the

一直忙着把……

3594.880–3597.190  scrambling to to get the uh the the questions uh from the from the

一直忙着把讨论区里的问题……

3597.190–3597.200  questions uh from the from the

讨论区里的问题……

3597.200–3599.910  questions uh from the from the discussion thread into uh into uh the

把讨论区里的问题搬到……

3599.910–3599.920  discussion thread into uh into uh the

从讨论区搬到……

3599.920–3601.190  discussion thread into uh into uh the screen here. So, thank you guys for

把讨论区里的问题搬到这里的屏幕上。谢谢你们。

3601.190–3601.200  screen here. So, thank you guys for

搬到这里的屏幕上。谢谢你们。

3601.200–3603.349  screen here. So, thank you guys for working your magic. I appreciate that.

搬到这里的屏幕上。谢谢你们大显身手，真的很感激。

3603.349–3603.359  working your magic. I appreciate that.

谢谢你们大显身手，真的很感激。

3603.359–3605.430  working your magic. I appreciate that. Uh and then also thanks to everybody who

谢谢你们大显身手，真的很感激。也感谢所有……

3605.430–3605.440  Uh and then also thanks to everybody who

也感谢所有……

3605.440–3606.870  Uh and then also thanks to everybody who joined and asked questions because you

也感谢所有参与并提问的人，因为你们……

3606.870–3606.880  joined and asked questions because you

参与并提问的人，因为你们……

3606.880–3609.510  joined and asked questions because you made this a good conversation. Uh we

参与并提问的人，因为你们让这场交流如此精彩。我们……

3609.510–3609.520  made this a good conversation. Uh we

让这次交流很精彩。我们

3609.520–3610.950  made this a good conversation. Uh we really appreciate it and keep building

让这次交流很精彩。我们非常感谢，也请继续用 PyTorch 创造

3610.950–3610.960  really appreciate it and keep building

非常感谢，也请继续创造

3610.960–3613.270  really appreciate it and keep building awesome things with PyTorch. Uh just a

非常感谢，也请继续用 PyTorch 创造精彩的作品。还有

3613.270–3613.280  awesome things with PyTorch. Uh just a

用 PyTorch 创造精彩的作品。还有

3613.280–3614.950  awesome things with PyTorch. Uh just a couple of resources uh for you to think

用 PyTorch 创造精彩的作品。结束前还有几个资源供大家参考

3614.950–3614.960  couple of resources uh for you to think

还有几个资源供大家参考

3614.960–3618.390  couple of resources uh for you to think about um a as we wrap up here. Um there

结束前还有几个资源供大家参考

3618.390–3618.400  about um a as we wrap up here. Um there

结束前还有

3618.400–3621.430  about um a as we wrap up here. Um there are we released a blog with this release

结束前还有：我们随这次发布推出了一篇博客

3621.430–3621.440  are we released a blog with this release

我们随这次发布推出了一篇博客

3621.440–3623.030  are we released a blog with this release uh which has a lot of detailed uh

我们随这次发布推出了一篇博客，里面有很多详细内容

3623.030–3623.040  uh which has a lot of detailed uh

里面有很多详细内容

3623.040–3624.150  uh which has a lot of detailed uh details about the various different

里面详细介绍了各种

3624.150–3624.160  details about the various different

各种功能的细节

3624.160–3625.430  details about the various different features. There's also the release

各种功能的细节。还有发布说明

3625.430–3625.440  features. There's also the release

还有发布说明

3625.440–3626.789  features. There's also the release notes. You should be able to find both

还有发布说明。这两份资料应该都很容易找到

3626.789–3626.799  notes. You should be able to find both

这两份资料应该都很容易找到

3626.799–3628.789  notes. You should be able to find both of those really easily. We may drop

这两份资料应该都很容易找到。我们也许会附上

3628.789–3628.799  of those really easily. We may drop

这两份资料很容易找到。我们也许会附上

3628.799–3630.069  of those really easily. We may drop links to them here. I'm not I'm not sure

我们也许会在这里放上链接，我不太确定

3630.069–3630.079  links to them here. I'm not I'm not sure

这里的链接。我不太确定

3630.079–3632.230  links to them here. I'm not I'm not sure whether that will happen or not. Um also

这里的链接。我不确定会不会放。另外

3632.230–3632.240  whether that will happen or not. Um also

会不会放。另外

3632.240–3633.829  whether that will happen or not. Um also the developer communities. So

会不会放。另外还有开发者社区

3633.829–3633.839  the developer communities. So

开发者社区，比如

3633.839–3635.349  the developer communities. So discuss.pyarch.org

开发者社区，比如 discuss.pyarch.org

3635.349–3635.359  discuss.pyarch.org

社区网站 discuss.pyarch.org

3635.359–3637.190  discuss.pyarch.org and devdiscuss.pyrush.org

discuss.pyarch.org 和 devdiscuss.pyrush.org 这两个社区

3637.190–3637.200  and devdiscuss.pyrush.org

以及 devdiscuss.pyrush.org 社区

3637.200–3639.589  and devdiscuss.pyrush.org are great uh for asking really detailed

devdiscuss.pyrush.org 也很适合提出具体问题

3639.589–3639.599  are great uh for asking really detailed

很适合提出具体问题

3639.599–3641.910  are great uh for asking really detailed questions like you guys uh did here. Um

很适合提出像大家刚才那样具体的问题

3641.910–3641.920  questions like you guys uh did here. Um

像大家刚才提出的那些具体问题

3641.920–3643.270  questions like you guys uh did here. Um and you'll you'll you'll be able to get

像大家刚才提出的那些具体问题，也会有人

3643.270–3643.280  and you'll you'll you'll be able to get

也会有人来解答

3643.280–3645.910  and you'll you'll you'll be able to get the the the experts uh um across the

也会有社区里的专家

3645.910–3645.920  the the the experts uh um across the

社区里的专家

3645.920–3648.069  the the the experts uh um across the community to answer. Um also there's

社区里的专家来解答。另外还有

3648.069–3648.079  community to answer. Um also there's

专家来解答。另外还有

3648.079–3650.150  community to answer. Um also there's some events coming up that I want to

专家来解答。另外还有一些即将举行的活动

3650.150–3650.160  some events coming up that I want to

一些即将举行的活动

3650.160–3652.309  some events coming up that I want to make people aware of. Uh so PyTorch

还有一些即将举行的活动，想告诉大家。PyTorch

3652.309–3652.319  make people aware of. Uh so PyTorch

想告诉大家。PyTorch

3652.319–3654.069  make people aware of. Uh so PyTorch Conference North America will be in San

想告诉大家，PyTorch 北美大会将在圣何塞举行

3654.069–3654.079  Conference North America will be in San

PyTorch 北美大会将在圣何塞举行

3654.079–3655.990  Conference North America will be in San Jose in just a little bit more than a

北美大会将在圣何塞举行，距今只有一个多月

3655.990–3656.000  Jose in just a little bit more than a

圣何塞，距今只有一个多月

3656.000–3657.829  Jose in just a little bit more than a month. Um so that's October 20th and

圣何塞，距今只有一个多月。时间是 10 月 20 日和

3657.829–3657.839  month. Um so that's October 20th and

这个月。嗯，就是10月20日和

3657.839–3660.309  month. Um so that's October 20th and 21st. I think all of us will be here.

这个月。嗯，就是10月20日和21日。我想我们都会到场。

3660.309–3660.319  21st. I think all of us will be here.

21日。我想我们都会到场。

3660.319–3662.549  21st. I think all of us will be here. All of us here will be will be there. Um

21日。我想我们都会到场。我们在座的都会去。

3662.549–3662.559  All of us here will be will be there. Um

我们在座的都会去。

3662.559–3665.829  All of us here will be will be there. Um and um uh you know it's a great chance

我们在座的都会去。这是个很好的机会。

3665.829–3665.839  and um uh you know it's a great chance

这是个很好的机会。

3665.839–3668.549  and um uh you know it's a great chance to to to learn a ton about PyTorch and

这是个深入了解PyTorch的好机会，还能

3668.549–3668.559  to to to learn a ton about PyTorch and

深入了解PyTorch，还能

3668.559–3671.349  to to to learn a ton about PyTorch and say hello to to to fellow uh uh members

深入了解PyTorch，还能和社区伙伴打招呼。

3671.349–3671.359  say hello to to to fellow uh uh members

和社区伙伴打招呼。

3671.359–3672.950  say hello to to to fellow uh uh members of the community. Um I'm certainly

和社区伙伴打招呼。嗯，我当然

3672.950–3672.960  of the community. Um I'm certainly

社区里的伙伴。嗯，我当然

3672.960–3675.190  of the community. Um I'm certainly looking forward to it and if you're um

社区里的伙伴。嗯，我当然很期待。如果你是

3675.190–3675.200  looking forward to it and if you're um

很期待。如果你是

3675.200–3677.030  looking forward to it and if you're um part of the global audience um there are

很期待。如果你是全球观众，

3677.030–3677.040  part of the global audience um there are

全球观众，

3677.040–3678.710  part of the global audience um there are a bunch of PyTorch, you know, we've been

全球观众，各地还有不少PyTorch活动，我们一直在

3678.710–3678.720  a bunch of PyTorch, you know, we've been

各地还有不少PyTorch活动，我们一直在

3678.720–3680.950  a bunch of PyTorch, you know, we've been really proliferating uh global events uh

各地还有不少PyTorch活动，我们一直在全球举办越来越多的活动。

3680.950–3680.960  really proliferating uh global events uh

在全球举办越来越多的活动。

3680.960–3682.710  really proliferating uh global events uh this year and I imagine that will

今年我们在全球举办了越来越多的活动，我想这还会

3682.710–3682.720  this year and I imagine that will

今年，我想这还会

3682.720–3684.950  this year and I imagine that will continue. Uh so we had an event in Paris

今年，我想这还会继续。春天我们在巴黎办过一场活动，

3684.950–3684.960  continue. Uh so we had an event in Paris

这还会继续。春天我们在巴黎办过一场活动，

3684.960–3686.950  continue. Uh so we had an event in Paris in the spring which was a lot of fun. We

这还会继续。春天我们在巴黎办过一场活动，很有意思。我们

3686.950–3686.960  in the spring which was a lot of fun. We

春天在巴黎办的活动很有意思。我们

3686.960–3688.470  in the spring which was a lot of fun. We just completed events uh well a while

春天在巴黎办的活动很有意思。我们前段时间还办了几场活动，

3688.470–3688.480  just completed events uh well a while

我们前段时间还办了几场活动，

3688.480–3690.150  just completed events uh well a while ago in India and then more recently in

我们前段时间在印度办了活动，后来又在

3690.150–3690.160  ago in India and then more recently in

前段时间在印度，后来又在

3690.160–3692.150  ago in India and then more recently in China and I think we have one planned in

前段时间在印度、最近在中国办了活动。我想我们还计划在

3692.150–3692.160  China and I think we have one planned in

在中国办了活动。我想我们还计划在

3692.160–3695.589  China and I think we have one planned in Japan for later this year. So um and

在中国办了活动。我想我们还计划今年晚些时候在日本办一场。另外，

3695.589–3695.599  Japan for later this year. So um and

今年晚些时候在日本办一场。另外，

3695.599–3697.430  Japan for later this year. So um and then I think there are pie days and

今年晚些时候在日本办一场。另外，我想还有PyDays和

3697.430–3697.440  then I think there are pie days and

另外，我想还有PyDays和

3697.440–3699.349  then I think there are pie days and smaller events happening all the time in

另外，我想还有PyDays和小型活动，不断在

3699.349–3699.359  smaller events happening all the time in

小型活动不断在

3699.359–3701.589  smaller events happening all the time in lots of different places. So uh look

小型活动不断在各地举办。所以，去

3701.589–3701.599  lots of different places. So uh look

各地举办。所以，去

3701.599–3704.710  lots of different places. So uh look look at the piece.org website for uh for

各地举办。所以，去piece.org网站看看

3704.710–3704.720  look at the piece.org website for uh for

去piece.org网站看看

3704.720–3707.190  look at the piece.org website for uh for u events near you. So I think that's it

去piece.org网站查看你附近的活动。我想就这些。

3707.190–3707.200  u events near you. So I think that's it

你附近的活动。我想就这些。

3707.200–3709.670  u events near you. So I think that's it >> and last but no there is one more thing

你附近的活动。我想就这些。最后……不对，还有一件事。

3709.670–3709.680  >> and last but no there is one more thing

最后……不对，还有一件事。

3709.680–3710.950  >> and last but no there is one more thing last but not least thank you very much

最后……不对，还有一件事。最后，非常感谢

3710.950–3710.960  last but not least thank you very much

最后，非常感谢

3710.960–3712.710  last but not least thank you very much Chris for being a fantastic moderator.

最后，非常感谢Chris出色地主持了节目。

3712.710–3712.720  Chris for being a fantastic moderator.

感谢 Chris 担任出色的主持人。

3712.720–3713.990  Chris for being a fantastic moderator. We could not have wished for a better

感谢 Chris 担任出色的主持人。我们找不到更好的

3713.990–3714.000  We could not have wished for a better

我们找不到更好的

3714.000–3715.750  We could not have wished for a better moderator here and without you it would

我们找不到更好的主持人，没有你，这场活动就会

3715.750–3715.760  moderator here and without you it would

主持人，没有你，这场活动就会

3715.760–3717.109  moderator here and without you it would be a very different event. So, thank

主持人，没有你，这场活动就会大不一样。所以，感谢

3717.109–3717.119  be a very different event. So, thank

大不一样。所以，感谢

3717.119–3717.990  be a very different event. So, thank you.

大不一样。所以，谢谢你。

3717.990–3718.000  you.

谢谢你。

3718.000–3720.390  you. >> Thank you, Chris. Great job.

谢谢你。>> 谢谢你，Chris。做得真棒。

3720.390–3720.400  >> Thank you, Chris. Great job.

>> 谢谢你，Chris。做得真棒。

3720.400–3720.950  >> Thank you, Chris. Great job. >> Bye-bye.

>> 谢谢你，Chris。做得真棒。>> 再见。

3720.950–3720.960  >> Bye-bye.

>> 再见。

3720.960–3723.520  >> Bye-bye. >> Thank you.

>> 再见。>> 谢谢。
