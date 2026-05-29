# 11 分钟汇报稿：让大模型在嘈杂反馈中自进化

> 对应 PPT：`PPT/PPT.pptx`  
> 建议时长：约 11 分钟  
> 讲述主线：基于 Two-Channel / A-OMP，把“审计校准”迁移到长期 Agent 的“记忆修正调度”新场景。

## 时间分配

| 部分 | 建议时长 | 对应内容 |
| --- | ---: | --- |
| 开场与问题 | 1 分 20 秒 | 标题、研究场景、研究问题 |
| 相关工作缺口 | 1 分 10 秒 | 现有记忆方法的问题 |
| 方案与方法 | 3 分 00 秒 | 记忆结构、A-OMP、OLE、系统闭环 |
| 实验验证 | 4 分 20 秒 | HumanEval、RLHF、ScienceWorld、ALFWorld |
| 总结与展望 | 1 分 10 秒 | 贡献、局限、未来工作 |

---

## 正式汇报稿

各位老师好，我是王泓宇。今天汇报的题目是：**让大模型在嘈杂反馈中自进化：面向异构反馈融合的双通道测试时学习与自进化记忆**。

这各项目的核心研究问题是：在长期任务执行中，LLM Agent 不仅要会积累经验，更要知道**什么时候应该相信已有经验，什么时候应该触发高成本反馈来修正记忆**。

这次工作的方法，是基于已有 Two-Channel / A-OMP 框架，把其中“cheap feedback 加 audited feedback”的思想迁移到一个新的场景：**长期任务 Agent 的自进化记忆系统**。

---

### 一、研究背景：长期 Agent 为什么需要自进化记忆

随着大模型从单轮问答走向长期 Agent 执行，系统越来越需要一种经验闭环能力。也就是说，Agent 完成一次任务之后，不能每次都从零开始，而应该把过去的任务、动作序列、反馈结果和失败经验沉淀到 memory 中，在后续相似任务里复用。

但是，记忆本身有双重作用。

一方面，好的记忆可以提升长期任务能力，让 Agent 更快找到有效策略；另一方面，错误记忆也会被不断检索和复用，变成偏差累积的通道。尤其是在多步任务里，一个错误动作模板如果被写入 memory，后续任务可能会反复受到它的影响。

因此，我认为长期 Agent 的关键问题不只是“如何存储经验”，而是：**如何在有限高质量反馈预算下，判断哪些经验可以直接复用，哪些经验必须被修正或重写。**

这里就自然出现了两类反馈。

第一类是低成本快反馈，比如 step-level reward、弱 judge、局部轨迹信号、语义反思和工具回显。它的优点是便宜、覆盖广、可以高频使用；缺点是噪声大，容易漂移。

第二类是高质量慢反馈，比如最终任务成败、环境回放、强教师判定、执行验证或人工审计。它更可靠，能够纠偏和重写记忆，但成本高、延迟大，不可能每一步都调用。

所以，本工作的核心矛盾就是：**cheap feedback 适合高频使用，但不够可靠；slow feedback 足够可靠，但不能高频使用。系统必须学会调度二者。**

---

### 二、现有方法缺口：大多解决“怎么存”，没有解决“什么时候改”

从相关工作看，现有长期记忆方法已经覆盖了不少方向。

比如 MemRL 这类方法会把历史经验组织成 Intent-Experience-Utility 结构，用非参数化情节记忆增强 Agent；Skill0 更强调把上下文中的技能经验内化到模型或策略中；ExpRAG、ClinK 这类方法把检索对象从零散日志提升到完整交互轨迹；AWM 则会从高频成功轨迹中抽象工作流或宏动作。

这些方法的共同价值是：它们都在尝试解决经验沉淀和经验复用问题。

但它们也存在一个共同缺口：**缺少低成本的在线记忆修正调度机制。**

换句话说，很多方法能回答“记忆应该怎么存”“经验应该怎么检索”，但没有很好回答“当前记忆是否可靠”“什么时候应该花费高质量反馈去重写它”。这也是导师建议里提到的重点：这项工作不能只做成文章复现，而要体现为基于已有工作解决一个新场景中的新问题。

所以我的切入点是：把 Two-Channel / A-OMP 中的审计调度思想，迁移到长期 Agent memory correction scheduling 中。

---

### 三、方案概述：把双通道反馈变成记忆修正调度

具体来说，我把系统抽象成一个闭环。

在每个任务到来时，Agent 首先根据当前任务输入，从 memory store 中检索相似经验。然后 cheap channel 会基于这些经验给出一个低成本候选动作序列或策略。接下来系统会根据当前任务状态、检索结果和记忆质量特征，判断是否需要调用 slow feedback。

如果风险不高，就直接接受 cheap channel 的结果，并将本次经验写入 memory；如果风险较高，就触发高成本 refinement，通过更可靠的反馈修正动作序列，并把修正后的模板写入 memory，用于后续任务复用。

这里的关键不是“每次都 refine”，而是“只在需要的时候 refine”。

在方法上，我参考 Two-Channel A-OMP。原始框架关注的是后训练外循环中的 evaluator drift：cheap proxy 提供高吞吐预测信号，audited feedback 提供低噪声校准信号。A-OMP 的理论结果说明，外循环的 stationarity gap 可以分解为三部分：一个优化瞬态项，加上 cheap hint error 和 audited lookahead error。

这个分解的意义是：高质量审计不是附加技巧，而是直接控制系统可达到的稳定性下限。

但原始 A-OMP 不能直接搬到记忆系统里，因为长期记忆有三个额外困难。

第一，记忆系统缺少显式的演化算子。它的状态变化来自检索、生成、反馈和写入共同作用，很难直接写成一个清晰的 F。

第二，记忆质量不可直接观测。一个 memory entry 是否覆盖当前任务、是否过期、是否冗余、是否会误导模型，都不是直接标签。

第三，缺少触发 refinement 的可靠准则。如果没有证书机制，只能用固定间隔或经验规则触发高成本反馈，这就很容易要么浪费预算，要么错过修正窗口。

为了解决这个问题，我在 ScienceWorld 和 ALFWorld 实验中引入了 OLE certificate，把它作为高成本 refinement 的风险阀门。

具体流程是：系统先从当前任务状态和 memory 状态中提取质量特征，例如 coverage、precision、redundancy、freshness，以及检索相似度、memory size、burst history 等，构造一个 12 维 OLE 特征向量。然后用 SGLD ensemble 估计均值和方差，计算 OLE-UCB。如果 UCB 超过阈值，就认为当前记忆状态存在风险，触发 refine burst；否则继续使用 cheap channel。

因此，在我的系统里，OLE 的角色不是替代 reward，而是判断：**当前记忆是否值得花高成本反馈去修正。**

---

### 四、实验设计：先验证双通道机制，再验证记忆场景迁移

实验分成两层。

第一层是预实验，包括 HumanEval 和 RLHF/SHP。它们的目的不是作为主要贡献，而是验证 Two-Channel / A-OMP 式审计调度机制确实有效。

第二层是主实验，包括 ScienceWorld 和 ALFWorld。它们才是这项工作的重点，用来验证把双通道审计思想迁移到长期 Agent 记忆场景之后，是否真的能在有限 LLM 调用预算下提升任务表现。

所有实验都比较五类调度方法。

`cheap_only` 是最低成本下界，只使用 cheap channel，不触发 refinement。  
`fixed_low` 是静态低预算方法，按照固定频率做少量 refinement。  
`exp3` 是无证书的 bandit 调度方法。  
`adaptive` 是本文主要方法，基于 certificate 或 OLE 风险信号触发高质量反馈。  
`oracle_high` 是高成本上界，每轮都使用高质量反馈或 refinement。

这个设置的核心问题是：adaptive 能不能在明显低于 oracle_high 的成本下，接近甚至超过其他非 oracle baseline。

---

### 五、预实验一：HumanEval

HumanEval 实验采用离线代码生成矩阵。我们先构造 20 个代码生成策略，也就是 4 个 prompt family 乘以 5 个 temperature，然后通过 EvalPlus 得到每个策略在每道题上的 pass/fail 结果。主实验阶段不再调用大模型，而是在这个离线矩阵上模拟 cheap feedback、true reward、反馈漂移和不同审计调度策略。

结果上，high 是性能上界，final reward 大约是 0.915808；adaptive 的 final reward 是 0.915761，几乎追平 high。相比 cheap-only 和 fixed-low，adaptive 在风险升高时通过高预算校准补足了更新方向，因此最终 reward 更接近高质量反馈上界。

这说明，在代码生成这种可离线验证的场景中，adaptive audit 能够以更有选择性的方式接近 high-audit 稳定性。

---

### 六、预实验二：RLHF / SHP

第二组预实验是 RLHF 风格的 SHP 偏好反馈实验。这里我们构造了 20 个 preference checkpoint，并模拟 cheap judge 的长度偏置漂移。cheap judge 的优势是便宜高频，但它可能逐渐偏向某些表面特征，例如回答长度，而不是偏好质量本身。

实验重点是看 audited calibration 能不能校正这个偏置。

结果显示，adaptive 和 high 的 calibration 曲线更能跟上真实 drift，而 cheap_only 因为没有校准，无法真正修正偏置。reward 曲线上，adaptive 和 high 都能维持接近上界的表现；gap 曲线上，cheap_only 的稳定性更差，而 adaptive 可以把 gap 压得更低。

这说明双通道机制在偏好反馈漂移场景里也成立：cheap channel 可以提供密集信号，但必须依靠 audited feedback 来校准漂移。

---

### 七、主实验一：ScienceWorld

接下来是主实验，首先是 ScienceWorld。

ScienceWorld 是一个文本驱动的科学实验环境，任务包括生命周期判断、物态变化、测量、分类等。它的特点是任务链条长、步骤严格、容错率低，非常适合检验 Agent 是否能在任务流中积累经验、修正记忆并提升后续表现。

在我的实验里，每个 scheduler 都在同一批 ScienceWorld 任务流上运行，系统会逐任务执行：检索 memory、生成动作、评估 success/progress、根据调度策略决定是否 refinement，并记录 LLM calls、partial rows 和 memory quality。

ScienceWorld 主实验中，adaptive_ole 的平均进度是 0.9931，几乎等于 oracle_high 的 0.9932，同时 LLM 调用次数是 390，而 oracle_high 是 810。也就是说，在保持接近 oracle 任务进度的同时，adaptive 使用的调用数大约只有 oracle 的一半。

再看 partial rows，adaptive 只有 10/270，是所有方法中最低的。这说明它不仅平均进度高，而且更少出现“只完成一部分”的情况。

Reduced prompt 消融中，提示信息变弱后，所有方法都有下降，但 adaptive 仍然达到 0.8407，高于 cheap_only 的 0.7418 和 fixed_low 的 0.7613，也高于 exp3 的 0.8191。这说明 adaptive 的收益不只是来自 prompt hints，而是来自基于风险的记忆修正调度。

---

### 八、主实验二：ALFWorld / AIfWorld

第二个主实验是 ALFWorld，也就是 PPT 中写的 AIfWorld 部分。这个任务更接近日常家居环境，Agent 需要完成取放、清洗、加热、冷却、检查等多步任务。相比 ScienceWorld，ALFWorld 的语义多义性和常识依赖更强，因此对记忆复用和错误修正提出了更高要求。

主实验中，adaptive 的 success rate 是 0.8856，avg progress 是 0.8002。oracle_high 的 avg progress 是 0.8375，是最高上界；但 oracle_high 需要 1608 次 LLM 调用，而 adaptive 只用了 788 次，节省约 51% 调用。

和其他非 oracle baseline 比较，adaptive 明显更强：cheap_only 的 avg progress 是 0.7040，fixed_low 是 0.7172，exp3 是 0.6468。adaptive 相比 cheap_only 提升 0.0962，相比最强非 oracle baseline 也高 0.0829。

这说明在 ALFWorld 这种更复杂的长期任务流中，adaptive 虽然没有完全追平 oracle_high，但已经在成本可控的前提下，显著优于其他可比方法。

Reduced prompt ablation 中，adaptive 的 avg progress 是 0.7861，仍然保持竞争力，并且调用数约为 oracle_high 的一半。这进一步说明：高质量反馈不是越密越好，关键在于什么时候触发，以及如何把纠偏结果沉淀进 memory。

---

### 九、原型演示：完整双通道记忆更新闭环

除了批量实验，我还基于 ScienceWorld 做了一个离线可运行的演示脚本，放在项目的 `演示` 目录下。

这个 demo 的目标不是重新跑完整实验，而是展示一次完整的双通道学习与记忆更新过程。

它的流程是：首先读取一个 ScienceWorld 任务，检索当前 memory；然后 cheap channel 给出一个低成本动作候选；接着 risk gate 根据检索质量、cheap feedback 和 progress 判断是否需要 refinement；如果风险高，就调用 slow refinement，用高质量动作模板修正，并写入 refined memory；如果风险低，就直接接受 cheap memory。

这个 demo 默认运行 4 个任务，其中 2 次触发 slow refinement，2 次直接接受 cheap memory。最终 memory snapshot 中包含 2 条 accurate memory 和 2 条 cheap memory。这个结果可以直观展示：系统不是每一步都调用高成本反馈，而是在无记忆、部分进度或任务族不匹配时触发修正；当已有记忆可靠时，就直接复用。

这正对应本工作的核心主张：**让 Agent 学会在有限高质量反馈预算下，判断何时相信经验，何时修正经验。**

---

### 十、研究意义与总结

总结来说，这项工作有三点贡献。

第一，在问题层面，我把已有 Two-Channel / A-OMP 的 cheap-audit 思想迁移到了长期 Agent 自进化记忆这个新场景。研究重点从“如何优化后训练外循环”转向“如何调度记忆修正”。

第二，在系统层面，我实现了一个 A-OMP-Mem 风格的闭环系统，包括 memory store、retrieval、cheap execution、OLE certificate、refinement、memory update 和结果评估。

第三，在实验层面，我用 HumanEval 和 RLHF 验证了双通道审计机制本身的有效性，并在 ScienceWorld 和 ALFWorld 两个长期任务流中验证了 adaptive memory refinement 的成本—效果优势。

整体结论是：对于长期任务 Agent 来说，高质量反馈不应该被简单地每步调用，也不应该完全不用；它应该作为一种稀缺资源，在记忆风险升高时被集中使用。OLE certificate 提供了一个判断风险、触发 refinement 的机制，使系统能在降低 LLM 调用成本的同时，保持接近 oracle 的任务执行效果。

---

### 十一、局限与未来展望

当然，目前工作还有一些局限。

第一，adaptive 在 ALFWorld 上还没有完全追平 oracle_high，说明在更复杂的家居任务中，记忆质量估计和 refinement 触发仍有提升空间。

第二，当前 OLE certificate 主要依赖已有 memory quality 特征，未来可以进一步研究这些特征和真实任务成功率之间的关系，让 certificate 不只是调度信号，也成为长期记忆系统的诊断工具。

第三，目前实验仍然是在固定任务集上进行，未来可以扩展到更长时间跨度、更复杂分布变化的开放任务流，检验 Agent 的记忆是否能持续积累、迁移和自我修正。

最后，我认为这项工作的核心价值在于：它不是简单复现 A-OMP 理论，而是把双通道反馈思想落到了长期 Agent 记忆系统中，形成了一个可运行、可复现、可分析的 memory correction scheduler。

我的汇报到这里结束，谢谢各位老师，敬请批评指正。

---

## 备用删减建议

如果实际录制超过 11 分钟，可以优先删减：

1. HumanEval 的具体数值，只保留“adaptive 接近 high”。
2. RLHF/SHP 的 calibration、gap 曲线解释，只保留“cheap judge 漂移需要 audit 校准”。
3. ScienceWorld benchmark 的任务类型介绍。
4. 未来展望中的第二、第三点。

如果需要压缩到 8 分钟，建议保留：

1. 背景问题：记忆不是只要存，还要会修正。
2. 方法迁移：Two-Channel/A-OMP → memory correction scheduling。
3. OLE gate：何时触发 refinement。
4. ScienceWorld 和 ALFWorld 两页核心结果。
5. 最后三点贡献总结。

