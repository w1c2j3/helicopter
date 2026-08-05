# RWKV7 1.5B lm-eval 错例分析

本报告分析 2026-08-02 两批完整 sample artifacts，不重新解释或替换正式指标。能力
补充套件包含 76,401 条记录；CMMLU 包含 11,582 条记录。二元任务按原生 sample
metric 判断正误，WMT14 等连续生成任务仅报告质量异常，不把单参考低重合直接等同于
答错。

## 结论

| 能力 | 错误证据 | 主要判断 |
| --- | ---: | --- |
| LongBench 段落检索 | 191 / 200 错 | 200 条全部生成 `Paragraph 1`，分数 4.5% 完全来自标准答案恰好为第 1 段的 9 条；属于输出位置塌缩，不是分散的检索失误 |
| RACE 阅读理解 | 665 / 1,045 错 | 错题中 448 条（67.4%）选择了比正确答案更短的选项，且错误 scorer margin 中位数为 5.94；既有阅读/整合失败，也存在 raw continuation loglikelihood 的长度偏置 |
| LAMBADA 完形 | 1,738 / 5,153 错 | 标准末词不是模型的 greedy continuation；该协议不生成替代词，因此不能从现有 artifact 声称模型具体答了哪个词 |
| BLiMP 句法 | 12,297 / 67,000 错 | 总体错误率 18.35%，但弱项高度集中在长距离 `wh/that`、疑问句 NPI、存在句量词和重构，而非基础性数/性一致 |
| WMT14 英法翻译 | 连续指标 | 发现 10 条明显重复循环、54 条翻译元回答、126 条字符重合低于 0.1 的异常输出；这些是诊断计数，不替代 BLEU/chrF/TER |
| CMMLU | 6,251 / 11,582 错 | 总体错误率 53.97%；数学、化学、生物、古汉语、解剖和法律最弱，错误并非单一学科造成 |

LongBench 还存在一个必须先修协议再复测的风险：原始 dataset 的 `context` 和
`question` 已经带任务说明，而 lm-eval 0.4.12 YAML 又套了一遍相同模板。实际 prompt
中 `Here are 30 paragraphs`、`The following is an abstract` 和答案格式说明均重复两
次。三模型仍使用相同协议，所以当前横向对比保持一致，但绝对检索能力不能只归因于
模型。RWKV 与 Qwen3.5-0.8B Base 都是 4.5%；Qwen3.5-2B Base 为 8.5%，并开始输出
多个段落位置，说明模型尺寸能部分缓解、但没有消除该问题。

## 代表错例

### LongBench：位置塌缩

- `doc_id=0`
- 摘要描述 Sipowicz，标准答案 `Paragraph 15`
- 模型答案 `Paragraph 1`

这不是孤例：RWKV 的预测分布是 `{1: 200}`，而标准答案中只有 9 条是第 1 段。模型
满足了输出格式，却没有执行内容定位。下一轮应先去掉重复模板，再分别做目标位于
1-10、11-20、21-30 的分桶复测。

### RACE：读取到事实但没有完成组合

- `doc_id=524`
- 文章给出 Hiroshima 死亡 70,000-80,000，Nagasaki 死亡 35,000-40,000
- 问题要求两枚炸弹合计死亡人数
- 模型选择 `Between 70,000 and 80,000 people.`
- 标准答案 `Between 105,000 and 120,000 people.`
- 四个 continuation 分数为 `[-25.87, -17.37, -13.97, -21.96]`

模型直接复用了文中的单个局部数字，没有做两地相加；错误选项相对正确选项的分数
优势达到 11.90，不属于边界摇摆。另一个系统性信号是 39.3% 的全部题选择了字符数
最短的选项，错题中 67.4% 的预测比标准答案短。由于 RACE 原生主指标使用未归一化
continuation loglikelihood，下一轮应并列报告原生 `acc` 与经确认的长度归一化诊断，
但不能回写或替换本轮正式分数。

### WMT14：重复生成

- `doc_id=912`
- 输入要求翻译一条包含 Wanda Sykes、Bill Maher 和 Madison Square Garden 的新闻句
- 模型连续 50 余次重复 `Je suis un homme`，直到接近生成上限
- 标准答案是对应事件的完整法语翻译

共有 10 条输出满足“四词片段至少重复五次”的循环规则，其中 3 条产生完全相同的
919 字符输出。这是确定性的解码退化，不应只被平均 BLEU 稀释。另有 54 条输出包含
`The French phrase`、`translates to` 或 `in English` 等元回答模式，说明部分样本把翻译
请求误解成了解释翻译行为。

### BLiMP：长距离依赖弱于局部一致

最弱的五个叶子任务如下：

| 子任务 | accuracy | 错题数 |
| --- | ---: | ---: |
| `wh_vs_that_with_gap_long_distance` | 30.3% | 697 |
| `matrix_question_npi_licensor_present` | 35.4% | 646 |
| `existential_there_quantifiers_2` | 39.9% | 601 |
| `wh_vs_that_with_gap` | 41.7% | 583 |
| `principle_A_reconstruction` | 47.8% | 522 |

相对地，`principle_A_case_1` 为 100%，`anaphor_gender_agreement` 为 99.5%，
`anaphor_number_agreement` 为 99.2%。因此问题不是“英语语法整体差”，而是长距离
依赖、NPI licensing、量词与 reconstruction 明显薄弱。训练数据与后续评测应针对
这些结构，不应继续追加同分布的基础一致性样本。

### CMMLU：知识短板与选项先验并存

- `cmmlu_agronomy/doc_id=40`
- 问题：从植物学上看，水稻的穗是什么花序
- 模型选择 `B. 穗状花序`
- 标准答案 `D. 圆锥花序`
- B 相对 D 的 scorer margin 为 2.45

最弱学科为大学数学 25.7% accuracy、高中化学 25.8%、高中生物 30.8%、古汉语和
解剖学各 31.8%、专业法律 32.7%。四个标准选项在数据中接近平衡，但模型预测
`D` 3,761 次（32.5%）、`B` 3,172 次（27.4%），存在可见选项先验。同时 70.1%
的错题 scorer margin 不超过 1.0，说明大量错误处于低置信竞争状态，和 RACE 的大
margin 错误机制不同。

## 后续复测门槛

1. LongBench 先消除重复模板，保留同一 200 条、相同生成参数，并新增目标位置分桶；
   未完成前不把 4.5% 作为纯模型长上下文能力结论。
2. RACE 保留原生 `acc`，额外输出选项 token 数、raw/normalized score 与预测变化，
   用于区分阅读失败和长度偏置。
3. 生成任务把 repetition、meta-answer、empty、format-error 作为独立质量指标；平均
   BLEU 或 retrieval score 之外必须给出计数与样本。
4. 每轮报告固定附 `bad_cases.json` 和 `error_analysis.md`，所有结论回指
   `task_name + doc_id`；LAMBADA 等 likelihood task 不伪造模型自由回答。
