# How Shopify Built a Continual Learning Loop with PyTorch and vLLM

原文：[How Shopify built a continual learning loop with PyTorch and vLLM](https://pytorch.org/blog/how-shopify-built-a-continual-learning-loop-with-pytorch-and-vllm/)

## 摘要

本期讨论 Shopify 如何把生产环境中的失败对话，持续转化为模型权重更新。流程从定义质量标准、校准评估器开始，再通过自动研究改进提示词和工具编排，最后用监督微调与 GRPO 优化模型参数。Shopify 还用 gist tokens 压缩系统提示词，并通过 vLLM 降低延迟、硬件需求和服务成本。GraphQL agent 的案例显示，服务成本降低百分之九十六，同时质量超过 frontier model。

## 对话

Ava: What if every failed production conversation could teach your model something by the next day?

Ava：如果每次生产环境中的失败对话，都能让模型第二天有所进步呢？

That’s the idea behind Shopify’s continual learning loop.

这就是 Shopify 持续学习闭环的思路。

Brian: And it’s practical, not just a research slogan.

Brian：而且这很实用，不只是研究口号。

Shopify says its GraphQL agent now beats the frontier-model baseline, while cutting serving cost by ninety-six percent.

Shopify 表示，其 GraphQL 智能体如今优于前沿模型基线，同时将推理服务成本降低了 96%。

Ava: That sounds like a big claim. Let’s start with the problem.

Ava：这说法很有分量。先说说问题出在哪儿。

Why wasn’t a frontier model enough?

为什么前沿模型还不够？

Brian: Frontier models are a fast way to launch.

Brian：用前沿模型可以快速推出产品。

A small team can put a useful product in front of users and learn from real usage.

小团队就能把有用的产品交给用户，并从实际使用中学习。

But scale changes the economics. They can be slow and expensive for every request.

但规模一大，成本账就变了。每个请求都可能又慢又贵。

Ava: And they’re general-purpose models, so they don’t really know Shopify’s product.

Ava：而且它们是通用模型，并不真正了解 Shopify 的产品。

Brian: Right. More importantly, they don’t learn from production on their own.

Brian：对。更关键的是，它们不会自行从生产环境中学习。

A rejected answer or a user correction doesn’t automatically improve the next answer.

答案被否定或用户作出纠正，并不会自动改善下一次回答。

Ava: So the knowledge accumulates around the model—in prompts, retrieval examples, routing rules, and harness code.

Ava：所以知识积累在模型周围，存在于提示词、检索示例、路由规则和智能体框架代码里。

Brian: Exactly. The deployed weights stay frozen.

Brian：没错。部署后的模型权重始终不变。

Shopify’s flywheel tries to compress that production experience into the continuous space of the model’s weights.

Shopify 的学习飞轮试图把这些生产经验压缩进模型权重的连续空间。

PyTorch provides the training foundation.

PyTorch 提供训练基础。

Ava: Before training anything, how do they decide what “better” means?

Ava：训练之前，他们怎么定义“更好”？

Brian: They define quality first.

Brian：他们先定义质量。

It starts as a rubric, which turns product requirements into scored criteria.

首先制定评分标准，把产品要求转化为可打分的指标。

For the GraphQL agent, the criteria include completeness, execution, response quality, and safety.

对 GraphQL 智能体来说，指标包括完整性、执行情况、回答质量和安全性。

Ava: So the rubric is basically a quality contract.

Ava：所以评分标准基本就是一份质量约定。

Brian: Yes. It gives every score a concrete meaning.

Brian：对。它让每个分数都有明确含义。

Annotators use it to turn conversations into ground truth.

标注员用它把对话转化为真实标注。

And the samples need to include randomly sampled traffic, not only carefully chosen examples.

样本还必须包含随机抽取的真实流量，不能只用精挑细选的例子。

Ava: Why does random traffic matter so much?

Ava：为什么随机流量这么重要？

Brian: Golden sets test cases you already know to look for.

Brian：黄金测试集检验的是你已经想到要测试的情况。

Random samples show what good and bad actually look like in production.

随机样本则展示生产环境中真实的好坏表现。

They reveal surprises.

它们能暴露意外问题。

Ava: How do they check whether the rubric is clear?

Ava：他们怎么检查评分标准是否清晰？

Brian: They ask their two best annotators or product experts to blindly label twenty-five random samples.

Brian：他们请两位最优秀的标注员或产品专家，对 25 个随机样本进行盲标。

Then they measure inter-annotator agreement with Cohen’s kappa, which measures agreement above chance.

然后用科恩 κ 系数衡量标注者一致性，也就是超出随机巧合的一致程度。

Ava: And if the kappa is low?

Ava：如果 κ 系数很低呢？

Brian: If it’s around zero point two, the rubric is ambiguous. The experts meet and iterate.

Brian：如果只有 0.2 左右，就说明评分标准有歧义。专家会一起讨论并修订。

If people who work on the product every day can’t agree, an LLM won’t magically agree either.

如果天天做这款产品的人都无法达成一致，大语言模型也不会凭空达成一致。

Ava: There’s also a ceiling here, right? Humans don’t agree perfectly.

Ava：这里还有个上限，对吧？人也不可能完全一致。

Brian: Right. The judge’s ceiling is human agreement. The goal isn’t a perfect judge.

Brian：对。评判模型的上限是人的一致程度。目标不是造出完美评判模型。

It’s a judge that matches humans about as well as humans match each other.

而是让它与人的判断一致到人与人之间的水平。

Ava: They also want detailed explanations, not just scores.

Ava：他们还需要详细解释，不能只有分数。

Brian: Yes. A score and one sentence aren’t enough for calibration.

Brian：对。一个分数加一句话，不足以用来校准。

They want the reason behind every score. That reasoning becomes valuable training data.

他们要知道每个分数背后的理由。这些推理过程会成为有价值的训练数据。

Ava: Then they calibrate the judge. What does that involve?

Ava：接下来要校准评判模型。具体怎么做？

Brian: The rubric is only the judge’s first prompt.

Brian：评分标准只是评判模型的第一版提示词。

Calibration turns it into a judge that can run over production data.

校准后，它才能用于评判生产数据。

Shopify uses DSPy and reflection-based optimizers such as GEPA and Agentic Context Engineering.

Shopify 使用 DSPy，以及 GEPA 和 Agentic Context Engineering 等基于反思的优化器。

Ava: Can you make those sound less mysterious?

Ava：能说得直白一点吗？

Brian: GEPA reflects on natural-language failure traces and evolves the prompt.

Brian：GEPA 会反思用自然语言记录的失败过程，并逐步改进提示词。

It keeps a Pareto frontier of candidates instead of greedily choosing one winner.

它保留候选方案的帕累托前沿，而不是贪心地选出唯一赢家。

ACE, or Agentic Context Engineering, builds a structured playbook through small edits.

ACE，也就是 Agentic Context Engineering，通过小幅修改构建结构化操作手册。

Ava: But an offline judge is still only a proxy.

Ava：但离线评判模型终究只是替代指标。

Brian: Exactly. They backtest it against previous A/B tests.

Brian：没错。他们会用过去的 A/B 测试结果来回测它。

The judge should recover the direction of known wins and losses in engagement, retention, or whatever behavior the product is designed to drive.

评判器应能判断已知改动对参与度、留存率或产品目标行为的正负影响。

Ava: And they run degradation tests too?

Ava：他们也做退化测试吗？

Brian: Yes.

Brian：是的。

They deliberately make one behavior worse and check whether the matching criterion falls.

他们故意让某项行为变差，再看对应的评分指标是否下降。

If the agent stops trying to fulfill the user’s goal, the goal-fulfillment score should drop specifically.

如果智能体不再努力完成用户目标，目标完成度这一项的分数就应下降。

Ava: Why use several small judges instead of one giant judge?

Ava：为什么用多个小评判器，而不是一个大评判器？

Brian: Focused judges are easier to interpret and trust.

Brian：各司其职的评判器更容易理解，也更可信。

Cramming every product behavior into one judge makes failures hard to understand.

把所有产品行为塞进一个评判器，会让问题难以理解。

Ava: Once the judge works, they improve the original frontier-powered application without changing model weights.

Ava：评判器做好后，他们在不改模型权重的情况下，优化原先由前沿模型驱动的应用。

Brian: That’s the baseline stage. They optimize prompts, tool definitions, and the harness.

Brian：这是基线阶段。他们优化提示词、工具定义和运行框架。

The harness is the whole application logic: dynamically assembled prompts, control loops, and orchestration spread across a codebase.

运行框架涵盖整个应用逻辑：动态组装的提示词、控制循环，以及分布在代码库各处的编排逻辑。

Ava: So ordinary prompt tuning only touches a small part of the system.

Ava：所以常规的提示词调优只涉及系统的一小部分。

Brian: Right. They treat it as an autoresearch problem.

Brian：没错。他们把这当作 autoresearch（自动化研究）问题。

An agent proposes a change to a prompt, a tool definition, or the harness.

智能体提出对提示词、工具定义或运行框架的修改。

It evaluates the change with the judge, keeps it if the score improves, and discards it otherwise.

它用评判器评估修改，分数提高就保留，否则丢弃。

Ava: Where do they configure that loop?

Ava：这个循环在哪里配置？

Brian: In one readable Markdown file: where to get data, which directories the agent may edit, which judge is the metric, which optimizer to use, and the propose-evaluate-keep-or-discard loop.

Brian：在一份易读的 Markdown 文件里：数据来源、智能体可编辑的目录、用哪个评判器作指标、用哪种优化器，以及提出、评估、保留或丢弃的循环。

Ava: What happens when those harness improvements plateau?

Ava：运行框架的改进进入平台期后呢？

Brian: Then they move from discrete artifacts to continuous parameter updates.

Brian：他们就从离散产物转向连续的参数更新。

They mine anonymized production traffic for hard negatives—conversations the judge correctly scores low.

他们从匿名化的线上流量中挖掘困难负例，也就是被评判器正确打低分的对话。

Ava: Across millions of merchants, that must create a huge variety of failures.

Ava：面对数百万商家，失败案例一定五花八门吧。

Brian: It does.

Brian：确实。

Partial context, ambiguous requests, business-specific workflows, tool failures, and many ways to express the same intent.

上下文不完整、请求含糊、各商家特有的工作流程、工具故障，以及同一意图的多种表达。

Instead of becoming isolated bug reports or Slack threads, the failures enter a self-healing pipeline.

这些失败案例不再只是零散的缺陷报告或 Slack 讨论，而是进入自我修复流程。

Ava: Walk me through one failure.

Ava：举一个失败案例，讲讲流程。

Brian: A panel of frontier reasoning models critiques it.

Brian：一组前沿推理模型会评析这个案例。

An arbiter merges the critiques into one repair instruction.

一个仲裁器将这些评析合并成一条修复指令。

That instruction is injected before the user’s turn, a technique sometimes called hinting.

这条指令会在用户这一轮发言前注入，有时称为 hinting（提示法）。

Ava: Then they replay the conversation?

Ava：然后重放对话？

Brian: Yes. They replay from that point and ask the judge to score it again.

Brian：对。从那一点开始重放，再让评判器打分。

If the repair passes, the replay becomes a reinforcement-learning trajectory, with the judge’s score as the reward.

如果修复通过，重放结果就成为一条强化学习轨迹，评判器的分数就是奖励。

Ava: And if it still fails?

Ava：如果还是失败呢？

Brian: It goes to human annotation through Toloka.

Brian：就通过 Toloka 交给人工标注。

Expert annotators correct the conversation and score it with the same rubric used to calibrate the judge.

专家标注员会修正对话，并用校准评判器时的同一套评分标准打分。

Ava: Training has two stages. First is supervised fine-tuning.

Ava：训练分两阶段。第一阶段是监督微调。

Brian: Correct. They distill healed trajectories into a smaller model.

Brian：对。他们把修复后的轨迹蒸馏到较小的模型中。

They train on complete trajectories, including the reasoning that produced them, not only the final answers.

训练用的是完整轨迹，包括产生结果的推理过程，而不只是最终答案。

That’s chain-of-thought distillation.

这就是思维链蒸馏。

Ava: So the smaller model can inherit behavior that answers alone wouldn’t teach.

Ava：所以小模型能学到光看答案学不到的行为。

Brian: Exactly. Second, they apply GRPO.

Brian：正是。第二阶段用 GRPO。

For each prompt, the model samples a group of responses, the judges score them, and GRPO reinforces the responses that perform best.

对于每个提示，模型采样一组回答，评判器逐个打分，GRPO 再强化表现最好的回答。

Ava: Supervised fine-tuning imitates successful trajectories, while GRPO optimizes directly against the quality definition.

Ava：监督微调模仿成功轨迹，GRPO 则直接按质量标准优化。

Brian: That’s the distinction. The pipeline runs daily.

Brian：对，区别就在这里。这套流程每天运行。

They add new trajectories, run a full-parameter fine-tune over new and previous data, and then repeat GRPO.

他们加入新轨迹，用新旧数据做一次全参数微调，然后再运行 GRPO。

Ava: Why keep old trajectories?

Ava：为什么要保留旧轨迹？

Brian: To limit drift and catastrophic forgetting across cycles.

Brian：为了减少各轮训练中的偏移和灾难性遗忘。

PyTorch distributes the training across GPUs using tensor, context, and data parallelism, which makes full-parameter fine-tuning practical at scale.

PyTorch 通过张量并行、上下文并行和数据并行，把训练分布到多块 GPU 上，让大规模全参数微调切实可行。

Ava: Now the model is better, but serving it can still be expensive.

Ava：现在模型更好了，但在线推理仍可能成本很高。

Brian: That’s where gist compression helps. The agent’s system prompt is long and static.

Brian：这时 gist 压缩就派上用场了。智能体的系统提示词很长，而且是固定的。

Because attention scales with sequence length, every generated token attends over that whole prefix.

由于注意力计算量随序列长度增长，每生成一个 token 都要关注整个前缀。

Ava: So the prompt is a fixed latency tax on every request.

Ava：所以提示词会给每个请求增加固定的延迟。

Brian: Exactly.

Brian：没错。

They run the same model as a teacher with the full prompt and as a student with a short sequence of learned gist tokens.

他们用同一个模型：输入完整提示词时作为教师模型，输入一小段学得的 gist token 时作为学生模型。

A custom PyTorch trainer learns the gist token embeddings while keeping model weights frozen.

定制的 PyTorch 训练器只学习 gist token 的嵌入，模型权重保持冻结。

Ava: And the student matches the teacher’s output distribution?

Ava：学生模型要拟合教师模型的输出分布？

Brian: Yes.

Brian：对。

The result is a handful of tokens that reproduce the prompt’s behavior at a fraction of the length, with no measured quality loss on the judge.

最终只用少量 token，就能以短得多的长度复现提示词的效果，而且评判器未测出质量下降。

Ava: Let’s look at the GraphQL agent itself.

Ava：来看看 GraphQL 智能体本身。

Brian: It serves up to two thousand requests per minute.

Brian：它每分钟最多处理 2000 个请求。

A merchant might ask which products are almost out of stock.

商家可能会问哪些商品快缺货了。

The agent writes and runs a query against Shopify’s Admin GraphQL API, then turns the result into plain language.

智能体通过 Shopify 的 Admin GraphQL API 编写并执行查询，再把结果转成通俗的回答。

Ava: What did the whole flywheel change?

Ava：整个飞轮带来了什么变化？

Brian: First, quality.

Brian：首先是质量。

Self-healing turns low-scoring production conversations into successful trajectories.

自我修复把评分低的线上对话转化为成功的执行轨迹。

Together, supervised fine-tuning and reinforcement learning let the specialized model surpass frontier-model performance.

监督微调加上强化学习，让这个专用模型的表现超过了前沿模型。

Ava: Second, cost.

Ava：其次是成本。

Brian: Serving the traffic on a frontier model could cost an estimated twenty-seven million dollars per year, based on average token costs.

Brian：按平均 token 成本估算，用前沿模型处理这些流量，每年可能要花 2700 万美元。

The fine-tuned model could be closer to one million dollars—a ninety-six percent reduction.

微调后的模型可能只需约 100 万美元，成本降低 96%。

Ava: That changes whether you can leave the feature on for every merchant.

Ava：这就决定了能否向每个商家持续开放这项功能。

Brian: Exactly. Third, speed.

Brian：正是。第三是速度。

Gisting compressed roughly six thousand prompt tokens to about one thousand five hundred learned gist tokens.

Gisting 把约 6000 个提示词 token 压缩成约 1500 个学得的 gist token。

Ava: What happened in the load test?

Ava：负载测试结果如何？

Brian: At three hundred fifty requests per minute, time-to-first-token fell about nineteen percent, and end-to-end latency fell about thirty-eight percent.

Brian：在每分钟 350 个请求时，首个 token 的响应时间缩短约 19%，端到端延迟降低约 38%。

Ava: And throughput improved too?

Ava：吞吐量也提高了？

Brian: About sixteen percent more requests per second and twelve percent more output tokens per second on identical GPUs.

Brian：使用相同的 GPU，每秒处理的请求数增加约 16%，每秒输出的 token 数增加约 12%。

That works out to roughly fourteen percent fewer GPUs for the same traffic.

这意味着处理同样流量，所需 GPU 可减少约 14%。

Ava: Are there limitations or open questions?

Ava：还有什么局限或待解的问题？

Brian: The article emphasizes that quality depends on the rubric and judge.

Brian：文章强调，质量取决于评分标准和评判器。

If those are wrong, the whole loop optimizes the wrong behavior.

如果它们有误，整个循环就会优化错误的行为。

Human annotation is still needed for failures the critics can’t repair.

对于评审模型无法修复的失败，仍需要人工标注。

Ava: And continual learning has to manage drift and forgetting.

Ava：持续学习还得应对漂移和遗忘。

Brian: Right. That’s why they train on accumulated data, not only the newest examples.

Brian：对。所以他们用积累的全部数据训练，而不只用最新样本。

The article also presents this as a production case study, so the exact gains may depend on the workload.

文章展示的是一个生产案例，具体收益可能因工作负载而异。

Ava: Let’s close with the three points I should remember. First?

Ava：最后总结三个要记住的要点。第一点？

Brian: Define and calibrate quality before optimizing.

Brian：优化之前，先定义并校准质量。

The rubric, human agreement, backtesting, and degradation tests make the judge trustworthy.

评分标准、与人工判断的一致性、回测和退化测试，共同保证评判器可信。

Ava: Second?

Ava：第二点？

Brian: Improve the frontier baseline in the harness, then turn real failures into trajectories and model-weight updates with supervised fine-tuning and GRPO.

Brian：先在测试框架中提升前沿模型的基线表现，再把真实失败转化为执行轨迹，并通过监督微调和 GRPO 更新模型权重。

Ava: Third?

Ava：第三点？

Brian: Compress the prompt for serving.

Brian：压缩用于线上服务的提示词。

Gist tokens, PyTorch training, and vLLM inference make the specialized model faster, cheaper, and easier to run at scale.

gist token、PyTorch 训练和 vLLM 推理，让专用模型更快、更便宜，也更容易规模化运行。

Ava: So the durable advantage isn’t one clever prompt.

Ava：所以持久的优势并非来自某一条巧妙的提示词。

It’s the loop that keeps turning production experience into better weights.

而是来自不断把线上经验转化为更优模型权重的循环。

Brian: Exactly. Thanks for listening. We’ll see you next time.

Brian：没错。感谢收听，下次见。

## 术语

| Term | 释义 |
|---|---|
| continual learning loop | 持续学习循环，把生产经验持续反馈到模型训练中 |
| frontier model | 前沿模型，能力强但通常通用、昂贵且权重冻结 |
| rubric | 评分标准，将产品要求转化为可评分的质量指标 |
| Cohen’s kappa | 衡量标注者之间超出随机水平的一致性的指标 |
| ground truth | 真实标签或基准答案，用于训练和评估 |
| GEPA | 基于反思的提示词进化优化器 |
| Agentic Context Engineering | 通过渐进式编辑构建结构化上下文手册的方法 |
| autoresearch | 由智能体提出、评估并保留或丢弃改动的自动研究循环 |
| hard negative | 评估器正确判为低分、能暴露模型弱点的困难样本 |
| hinting | 在用户输入前注入修复指令并重放对话的技术 |
| supervised fine-tuning | 使用标注轨迹进行监督式微调 |
| GRPO | 对一组采样回答评分并强化高分回答的强化学习方法 |
| gist token | 用于压缩长系统提示词的一小组学习型 token |
| vLLM | 基于 PyTorch 的推理引擎，支持连续批处理 |
| catastrophic forgetting | 模型学习新数据时遗忘旧能力的现象 |

## 口语表达

| Phrase | 释义 |
|---|---|
| What if every failed production conversation could teach your model something? | 如果每次生产失败对话都能教会模型一些东西呢？ |
| Let’s start with the problem. | 我们先从问题开始。 |
| Walk me through one failure. | 带我走一遍一个失败案例。 |
| That’s the distinction. | 区别就在这里。 |
| What happens when those improvements plateau? | 这些改进达到瓶颈后会怎样？ |
| It works out to roughly | 最终大约相当于 |
| Let’s close with the three points I should remember. | 最后总结我应该记住的三点。 |
| The durable advantage isn’t one clever prompt. | 真正持久的优势不是某个巧妙提示词。 |
