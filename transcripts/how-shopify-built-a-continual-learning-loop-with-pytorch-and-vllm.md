# How Shopify Built a Continual Learning Loop with PyTorch and vLLM

原文：[How Shopify built a continual learning loop with PyTorch and vLLM](https://pytorch.org/blog/how-shopify-built-a-continual-learning-loop-with-pytorch-and-vllm/)

## 摘要

本期讨论 Shopify 如何把生产环境中的失败对话，持续转化为模型权重更新。流程从定义质量标准、校准评估器开始，再通过自动研究改进提示词和工具编排，最后用监督微调与 GRPO 优化模型参数。Shopify 还用 gist tokens 压缩系统提示词，并通过 vLLM 降低延迟、硬件需求和服务成本。GraphQL agent 的案例显示，服务成本降低百分之九十六，同时质量超过 frontier model。

## 对话

**Ava:** What if every failed production conversation could teach your model something by the next day? That’s the idea behind Shopify’s continual learning loop.

**Brian:** And it’s practical, not just a research slogan. Shopify says its GraphQL agent now beats the frontier-model baseline, while cutting serving cost by ninety-six percent.

**Ava:** That sounds like a big claim. Let’s start with the problem. Why wasn’t a frontier model enough?

**Brian:** Frontier models are a fast way to launch. A small team can put a useful product in front of users and learn from real usage. But scale changes the economics. They can be slow and expensive for every request.

**Ava:** And they’re general-purpose models, so they don’t really know Shopify’s product.

**Brian:** Right. More importantly, they don’t learn from production on their own. A rejected answer or a user correction doesn’t automatically improve the next answer.

**Ava:** So the knowledge accumulates around the model—in prompts, retrieval examples, routing rules, and harness code.

**Brian:** Exactly. The deployed weights stay frozen. Shopify’s flywheel tries to compress that production experience into the continuous space of the model’s weights. PyTorch provides the training foundation.

**Ava:** Before training anything, how do they decide what “better” means?

**Brian:** They define quality first. It starts as a rubric, which turns product requirements into scored criteria. For the GraphQL agent, the criteria include completeness, execution, response quality, and safety.

**Ava:** So the rubric is basically a quality contract.

**Brian:** Yes. It gives every score a concrete meaning. Annotators use it to turn conversations into ground truth. And the samples need to include randomly sampled traffic, not only carefully chosen examples.

**Ava:** Why does random traffic matter so much?

**Brian:** Golden sets test cases you already know to look for. Random samples show what good and bad actually look like in production. They reveal surprises.

**Ava:** How do they check whether the rubric is clear?

**Brian:** They ask their two best annotators or product experts to blindly label twenty-five random samples. Then they measure inter-annotator agreement with Cohen’s kappa, which measures agreement above chance.

**Ava:** And if the kappa is low?

**Brian:** If it’s around zero point two, the rubric is ambiguous. The experts meet and iterate. If people who work on the product every day can’t agree, an LLM won’t magically agree either.

**Ava:** There’s also a ceiling here, right? Humans don’t agree perfectly.

**Brian:** Right. The judge’s ceiling is human agreement. The goal isn’t a perfect judge. It’s a judge that matches humans about as well as humans match each other.

**Ava:** They also want detailed explanations, not just scores.

**Brian:** Yes. A score and one sentence aren’t enough for calibration. They want the reason behind every score. That reasoning becomes valuable training data.

**Ava:** Then they calibrate the judge. What does that involve?

**Brian:** The rubric is only the judge’s first prompt. Calibration turns it into a judge that can run over production data. Shopify uses DSPy and reflection-based optimizers such as GEPA and Agentic Context Engineering.

**Ava:** Can you make those sound less mysterious?

**Brian:** GEPA reflects on natural-language failure traces and evolves the prompt. It keeps a Pareto frontier of candidates instead of greedily choosing one winner. ACE, or Agentic Context Engineering, builds a structured playbook through small edits.

**Ava:** But an offline judge is still only a proxy.

**Brian:** Exactly. They backtest it against previous A/B tests. The judge should recover the direction of known wins and losses in engagement, retention, or whatever behavior the product is designed to drive.

**Ava:** And they run degradation tests too?

**Brian:** Yes. They deliberately make one behavior worse and check whether the matching criterion falls. If the agent stops trying to fulfill the user’s goal, the goal-fulfillment score should drop specifically.

**Ava:** Why use several small judges instead of one giant judge?

**Brian:** Focused judges are easier to interpret and trust. Cramming every product behavior into one judge makes failures hard to understand.

**Ava:** Once the judge works, they improve the original frontier-powered application without changing model weights.

**Brian:** That’s the baseline stage. They optimize prompts, tool definitions, and the harness. The harness is the whole application logic: dynamically assembled prompts, control loops, and orchestration spread across a codebase.

**Ava:** So ordinary prompt tuning only touches a small part of the system.

**Brian:** Right. They treat it as an autoresearch problem. An agent proposes a change to a prompt, a tool definition, or the harness. It evaluates the change with the judge, keeps it if the score improves, and discards it otherwise.

**Ava:** Where do they configure that loop?

**Brian:** In one readable Markdown file: where to get data, which directories the agent may edit, which judge is the metric, which optimizer to use, and the propose-evaluate-keep-or-discard loop.

**Ava:** What happens when those harness improvements plateau?

**Brian:** Then they move from discrete artifacts to continuous parameter updates. They mine anonymized production traffic for hard negatives—conversations the judge correctly scores low.

**Ava:** Across millions of merchants, that must create a huge variety of failures.

**Brian:** It does. Partial context, ambiguous requests, business-specific workflows, tool failures, and many ways to express the same intent. Instead of becoming isolated bug reports or Slack threads, the failures enter a self-healing pipeline.

**Ava:** Walk me through one failure.

**Brian:** A panel of frontier reasoning models critiques it. An arbiter merges the critiques into one repair instruction. That instruction is injected before the user’s turn, a technique sometimes called hinting.

**Ava:** Then they replay the conversation?

**Brian:** Yes. They replay from that point and ask the judge to score it again. If the repair passes, the replay becomes a reinforcement-learning trajectory, with the judge’s score as the reward.

**Ava:** And if it still fails?

**Brian:** It goes to human annotation through Toloka. Expert annotators correct the conversation and score it with the same rubric used to calibrate the judge.

**Ava:** Training has two stages. First is supervised fine-tuning.

**Brian:** Correct. They distill healed trajectories into a smaller model. They train on complete trajectories, including the reasoning that produced them, not only the final answers. That’s chain-of-thought distillation.

**Ava:** So the smaller model can inherit behavior that answers alone wouldn’t teach.

**Brian:** Exactly. Second, they apply GRPO. For each prompt, the model samples a group of responses, the judges score them, and GRPO reinforces the responses that perform best.

**Ava:** Supervised fine-tuning imitates successful trajectories, while GRPO optimizes directly against the quality definition.

**Brian:** That’s the distinction. The pipeline runs daily. They add new trajectories, run a full-parameter fine-tune over new and previous data, and then repeat GRPO.

**Ava:** Why keep old trajectories?

**Brian:** To limit drift and catastrophic forgetting across cycles. PyTorch distributes the training across GPUs using tensor, context, and data parallelism, which makes full-parameter fine-tuning practical at scale.

**Ava:** Now the model is better, but serving it can still be expensive.

**Brian:** That’s where gist compression helps. The agent’s system prompt is long and static. Because attention scales with sequence length, every generated token attends over that whole prefix.

**Ava:** So the prompt is a fixed latency tax on every request.

**Brian:** Exactly. They run the same model as a teacher with the full prompt and as a student with a short sequence of learned gist tokens. A custom PyTorch trainer learns the gist token embeddings while keeping model weights frozen.

**Ava:** And the student matches the teacher’s output distribution?

**Brian:** Yes. The result is a handful of tokens that reproduce the prompt’s behavior at a fraction of the length, with no measured quality loss on the judge.

**Ava:** Let’s look at the GraphQL agent itself.

**Brian:** It serves up to two thousand requests per minute. A merchant might ask which products are almost out of stock. The agent writes and runs a query against Shopify’s Admin GraphQL API, then turns the result into plain language.

**Ava:** What did the whole flywheel change?

**Brian:** First, quality. Self-healing turns low-scoring production conversations into successful trajectories. Together, supervised fine-tuning and reinforcement learning let the specialized model surpass frontier-model performance.

**Ava:** Second, cost.

**Brian:** Serving the traffic on a frontier model could cost an estimated twenty-seven million dollars per year, based on average token costs. The fine-tuned model could be closer to one million dollars—a ninety-six percent reduction.

**Ava:** That changes whether you can leave the feature on for every merchant.

**Brian:** Exactly. Third, speed. Gisting compressed roughly six thousand prompt tokens to about one thousand five hundred learned gist tokens.

**Ava:** What happened in the load test?

**Brian:** At three hundred fifty requests per minute, time-to-first-token fell about nineteen percent, and end-to-end latency fell about thirty-eight percent.

**Ava:** And throughput improved too?

**Brian:** About sixteen percent more requests per second and twelve percent more output tokens per second on identical GPUs. That works out to roughly fourteen percent fewer GPUs for the same traffic.

**Ava:** Are there limitations or open questions?

**Brian:** The article emphasizes that quality depends on the rubric and judge. If those are wrong, the whole loop optimizes the wrong behavior. Human annotation is still needed for failures the critics can’t repair.

**Ava:** And continual learning has to manage drift and forgetting.

**Brian:** Right. That’s why they train on accumulated data, not only the newest examples. The article also presents this as a production case study, so the exact gains may depend on the workload.

**Ava:** Let’s close with the three points I should remember. First?

**Brian:** Define and calibrate quality before optimizing. The rubric, human agreement, backtesting, and degradation tests make the judge trustworthy.

**Ava:** Second?

**Brian:** Improve the frontier baseline in the harness, then turn real failures into trajectories and model-weight updates with supervised fine-tuning and GRPO.

**Ava:** Third?

**Brian:** Compress the prompt for serving. Gist tokens, PyTorch training, and vLLM inference make the specialized model faster, cheaper, and easier to run at scale.

**Ava:** So the durable advantage isn’t one clever prompt. It’s the loop that keeps turning production experience into better weights.

**Brian:** Exactly. Thanks for listening. We’ll see you next time.

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
