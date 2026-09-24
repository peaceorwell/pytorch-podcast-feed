# From Research Project to Open Source Ecosystem

原文：[From Research Project to Open Source Ecosystem: Bring Your Academic PyTorch Project to PyTorchCon NA](https://pytorch.org/blog/from-research-project-to-open-source-ecosystem-bring-your-academic-pytorch-project-to-pytorchcon-na/)

## 摘要

本期讨论 PyTorch 基金会面向学术项目的“从学术走向采用”工作坊，帮助优秀研究代码成长为可持续的开源社区项目。节目介绍征集对象、评选标准、闪电演讲和导师办公时间，以及文档、治理、可复现性、安全和维护等常见缺口。符合条件的项目需要使用 PyTorch 或其他 PyTorch 基金会托管项目，并在 2026 年 9 月 30 日太平洋时间午夜前提交。核心信息是：项目不必完美成熟，只要有价值、有潜力，就值得申请帮助。

## 对话

**Ava:** What happens when a useful PyTorch research project outlives the paper that created it? That’s the question behind a new academic workshop at PyTorch Conference North America 2026.

**Brian:** Right. The workshop is trying to help projects make a difficult jump: from a research artifact to a sustainable open source project that a wider community can use and maintain.

**Ava:** So this is for university labs and student teams whose code works, but whose project may still depend on one graduating student or one person who knows the release process.

**Brian:** Exactly. The PyTorch Foundation OSPO and Academic Outreach Working Group wants to bridge that gap. OSPO means Open Source Program Office. The group is inviting academic PyTorch projects to apply for a Day 0 workshop focused on education and community adoption.

**Ava:** Let’s start with the basic opportunity. What does a selected team actually get?

**Brian:** Eight selected projects will give short lightning talks during the workshop. Each talk is five minutes, and presenters must attend in person. After that, the teams join a hands-on workshop and mentor office hours.

**Ava:** Office hours with whom?

**Brian:** With maintainers, open source program managers, people who support open source ecosystems in industry and research organizations, open source contributors, and members of the broader PyTorch community.

**Ava:** So it’s not just a showcase. Teams bring their real repositories and work on them with mentors.

**Brian:** That’s the idea. The article describes a move from show-and-tell to show-and-build. The goal is for each team to leave with a prioritized set of next steps.

**Ava:** What kind of projects are they looking for? Is this only about a new model architecture?

**Brian:** No. They want projects with technical value beyond one experiment or one paper. That could be a library or developer tool, a research framework for a specific domain, or a compiler, runtime, profiling, distributed training, inference, or optimization project.

**Ava:** That’s a pretty wide range.

**Brian:** It is. The call also includes work in computer vision, natural language processing, robotics, scientific computing, healthcare, responsible AI, and accessibility. Evaluation and benchmarking frameworks count too, as do reproducibility toolkits, research infrastructure, and educational tools.

**Ava:** And the project has to connect to the PyTorch ecosystem.

**Brian:** Yes. It needs to use PyTorch or another PyTorch Foundation hosted project. The article names DeepSpeed, Helion, Ray, Safetensors, and vLLM as examples of those hosted projects.

**Ava:** Could it be an implementation that accompanies a major research paper?

**Brian:** Yes, if it has a life beyond the paper. They also welcome projects that connect these projects with other parts of the open source AI stack, and they say they’re open to ideas they haven’t thought of yet.

**Ava:** That last part sounds encouraging. A team doesn’t have to force its work into a fixed category.

**Brian:** Right. The larger message is that you don’t need to have everything figured out.

**Ava:** That sounds almost too good to be true for an open source call. What does “not everything figured out” mean here?

**Brian:** Academic research software and sustainable community projects are often built under different conditions. Academic teams optimize for experimentation, publication, reproducibility, and research velocity. A sustainable project also needs documentation, releases, governance, security, contributor onboarding, testing, licensing, maintainer succession, and community operations.

**Ava:** Maintainer succession is a big one. A student may create the project, graduate, and then nobody knows how to keep it moving.

**Brian:** Exactly. The article lists that situation directly. Other examples include documentation that needs work, a repository that new contributors can’t navigate, an informal governance process, weak CI or packaging, or no CONTRIBUTING dot md, SECURITY dot md, or maintainer policy.

**Ava:** So a project can apply even if those pieces are missing?

**Brian:** Yes. Those gaps are part of the reason the workshop exists. Teams may also be unsure how to build a contributor community, how to turn users into contributors, or what would eventually be needed to become a PyTorch Ecosystem project.

**Ava:** Let’s unpack the workshop itself. What happens during the project showcases?

**Brian:** Selected teams explain the problem they’re solving, how they use or extend PyTorch Foundation hosted projects, and the academic context behind the work. They also talk about who uses the project today, why it could matter to the broader community, and what support or changes could help it reach the next stage.

**Ava:** Then the office hours get more practical.

**Brian:** Much more practical. Mentors may ask whether someone can understand what the project does, install it, run it, test it, and learn how to contribute. That’s what the article calls open source readiness.

**Ava:** I like that sequence. First, can a new user get started? Then, can a contributor join?

**Brian:** Yes. Governance is another topic. Who makes decisions? Who can merge changes? What happens when the original academic maintainers move on?

**Ava:** And reproducibility is broader than publishing a paper.

**Brian:** Right. Another researcher should be able to reproduce the results. That means datasets, benchmarks, evaluation methodology, dependencies, and artifacts need to be clearly documented.

**Ava:** What about the operational side?

**Brian:** They’ll discuss security and maintenance too: dependencies, vulnerabilities, releases, and supported PyTorch versions. They may also look at ecosystem positioning. What’s the project’s relationship with PyTorch? What unique problem does it solve? What would need to change before a future PyTorch Ecosystem application made sense?

**Ava:** That sounds like a roadmap conversation, not a pass-or-fail inspection.

**Brian:** That’s how the article presents it. The ambition is to identify gaps and turn them into actionable next steps for making a project easier to use, contribute to, maintain, and grow.

**Ava:** Who can submit? Only professors or official university labs?

**Brian:** No. The call includes university students and graduate researchers, professors and academic educators, research labs and institutes, research software engineers, Academic OSPOs, maintainers of research-originated open source projects, and academic communities building tools around the hosted projects.

**Ava:** And projects can come from anywhere in the world.

**Brian:** Yes. They don’t need to represent a large institution, have thousands of users, or already have a mature open source organization. The organizers care about the idea, its usefulness, its relationship with PyTorch, the work already happening around it, and its potential to grow beyond the original academic environment.

**Ava:** How will they evaluate submissions?

**Brian:** There are six criteria. Academic relevance: the project came from, substantially grew out of, or supports academic research. Ecosystem relevance: it uses, extends, or integrates with PyTorch or another hosted project. Open source readiness: there’s a public repository, an open source license, and enough documentation for mentors to review it.

**Ava:** What are the remaining three?

**Brian:** Potential impact, meaning it addresses a meaningful problem and could help users beyond the original research team. Team commitment, meaning at least one maintainer can attend and continue the agreed next steps afterward. And cohort diversity, so the final group represents different institutions, research areas, project stages, and geographic regions.

**Ava:** What should an applicant put in the submission?

**Brian:** Start with the repository and a short description. Explain where the project came from: a university, lab, research group, course, or academic collaboration. Then describe how it uses or extends PyTorch or its ecosystem, and who it helps, such as researchers, students, machine learning engineers, or a particular scientific community.

**Ava:** They also want evidence of what exists today.

**Brian:** Yes. Users, contributors, releases, papers, deployments, courses, benchmarks, or other signs of adoption are useful context. Then be honest about where you’re stuck: documentation, governance, packaging, testing, contributors, maintainers, visibility, reproducibility, or ecosystem readiness.

**Ava:** And the final question is what the workshop should help achieve.

**Brian:** Exactly. They’re not looking for perfectly polished applications. They’re looking for projects worth helping.

**Ava:** Let’s make the deadline clear. When does this call close?

**Brian:** September thirtieth, twenty twenty-six, at midnight Pacific Time. Registration for the event is separate from the PyTorch Conference and will open later that week.

**Ava:** So if you have a project that deserves a life beyond a paper, thesis, course, grant, or research group, this is a direct invitation.

**Brian:** Yes. The project should already be open source and connected to PyTorch or one of the other hosted projects, but it doesn’t need to be polished. The workshop is designed for the transition itself.

**Ava:** Let’s close with a three-point recap. First, this workshop helps academic PyTorch projects move toward sustainable community adoption.

**Brian:** Second, selected teams get five-minute lightning talks, hands-on work with mentors, and office hours covering readiness, governance, reproducibility, community, security, maintenance, and ecosystem positioning.

**Ava:** Third, the organizers value potential and commitment over perfect maturity. Submit the project, explain what exists, say where you’re stuck, and describe the next step you want help with.

**Brian:** And remember the deadline: September thirtieth, twenty twenty-six, midnight Pacific Time.

**Ava:** That’s it for today. If your academic PyTorch project deserves a future beyond the original lab, bring it to the workshop.

**Brian:** Thanks for listening. We’ll see you in the next episode.

## 术语

| Term | 释义 |
|---|---|
| PyTorch Foundation OSPO | PyTorch 基金会开源项目办公室 |
| Academic Outreach Working Group | 学术外展工作组 |
| sustainable open source project | 可持续的开源项目 |
| research artifact | 研究产物或研究代码成果 |
| lightning talk | 限时短演讲 |
| maintainer succession | 维护者交接与继任机制 |
| governance | 项目治理与决策机制 |
| reproducibility | 可复现性 |
| open source readiness | 项目是否已具备开源协作条件 |
| contributor onboarding | 引导新贡献者加入项目 |
| ecosystem positioning | 项目在生态中的定位及与 PyTorch 的关系 |
| research software engineer | 研究软件工程师 |
| Academic OSPO | 高校或学术机构的开源项目办公室 |
| cohort diversity | 入选项目群体的多样性 |

## 口语表达

| Phrase | 释义 |
|---|---|
| What happens when... | 当……发生时会怎样？ |
| That’s the idea. | 这就是核心想法。 |
| Let’s unpack... | 我们来拆解一下…… |
| You don’t need to have everything figured out. | 你不需要事先把一切都想清楚。 |
| That’s a big one. | 这一点很关键。 |
| Let’s make the deadline clear. | 我们把截止时间说清楚。 |
| Be honest about where you’re stuck. | 如实说明你卡在哪里。 |
| They’re looking for projects worth helping. | 他们寻找的是值得帮助的项目。 |
