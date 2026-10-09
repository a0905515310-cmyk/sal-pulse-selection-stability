# 图表交付说明与审稿审查

本次未获得论文原稿、原始结果图片或现成绘图版式。因此，实际完成的是基于冻结数值证据的重新绘图，不能声称已逐幅复原或改正原图。图号与英文图注均为临时编号，最终应与论文结果段落的论证顺序对应。

## 当前证据可以支持什么

这组图的主线应是：**联合时序与脉宽选择在 HPRF 单一干扰下维持较高的正确选择概率，但持续正确、错误后恢复和非正确游程长度之间的取舍取决于干扰类型及对照方法。**

“TW 在所有场景、所有指标上优于单特征方法”与冻结结果直接冲突。TW 相对 T 在 F1、F4 的正确选择概率差值区间全部小于零；F2、F3 对应区间包含零。F1–F4 的正确状态持续概率差值区间全部小于零。TW 相对 W 在全部 18 个干扰条件中的正确状态持续概率差值区间全部大于零，但在 11 个条件中，错误后恢复概率差值区间全部小于零、观察到的非正确游程长度差值区间全部大于零。这些是区间几何位置与效应方向的描述，不是新增显著性检验。

## 建议的主文与补充材料分工

建议主文优先使用 `fig01_hprf` 与 `fig05_tw_minus_w`：前者呈现主要工作区间下的绝对表现，后者直接暴露持续正确与错误后恢复之间的取舍。若论文主要贡献是相对时序准则的改进，则将 `fig04_tw_minus_t` 同时放入主文。该选择须由论文的研究问题决定，不能依据某张图是否“好看”。

`fig02_f_only`、`fig03_composite`、另一张完整配对差值图、`fig06_width_strata` 与 `figS01_precision` 可放补充材料，但主文应明确引用 F-only 边界和 TW–W 取舍。图号暂定，不能在没有论文结果段落的情况下宣称已经确定最终图序。完整场景与方法在交付数据和图中保留，不删除不利条件。

## 统计口径与源数据

- 每个条件有 **n = 2,000 个独立仿真重复**，每个重复包含 **K = 200 个周期**。同一条件和重复中的五种方法使用同一物理世界；方法间对照是配对的。400,000 个周期不是 400,000 个独立重复。
- 点估计是相应分子计数总和除以分母计数总和，即 pooled-count ratio；不是先计算每个重复的比率再取平均。
- 误差条使用已冻结的 **B = 2,000 次整重复重采样、95% percentile Bootstrap 区间**，分位数采用线性插值。同一条件中的五种方法共享每次重采样的重复索引；差值区间由配对 Bootstrap 得到。
- 区间是逐项、pointwise 区间，没有进行多重比较校正；本次绘图不计算 p 值，不新增检验、对照或区间。不应将数十个区间拼接成“所有场景均成立”的同时推断。
- `P_cor` 是正确选择周期占比；`P_C_given_C` 是相邻周期中由正确状态继续保持正确的条件概率；`P_C_given_E` 是由错误选择状态转到正确状态的条件概率。这里的 E 不包括无候选事件 N，因此 `P_C_given_E` 不应改写为从所有非正确状态恢复的概率。
- `Mean_L_NC` 是**观察窗内非正确游程的平均已观察长度**，按已观察非正确周期总数除以非正确游程总数计算。非正确状态包括 E 和 N。第 200 周期仍未结束的游程保留，并仅计入截至该周期的已观察长度；它不是未受截断影响的平均恢复时间。
- H300 分层依据当周期 H 干扰脉冲的真脉宽与标称参考脉宽的绝对差，分为 0、10、≥20 ns；它不是含噪观测脉宽差。分层图只展示描述性 pooled-count 比率，不提供 Bootstrap 区间或 p 值。
- 无干扰数据只作为一致性检查。没有 E-origin 转移或非正确游程时，对应指标未定义，不能用 0 代替，也不能补造区间。

交付源数据位于 `source_data`：`individual.csv`、`paired.csv`、`precision.csv`、`width_strata.csv`、`ni_consistency.csv`。前三个概率指标在图中可用百分比表示，概率差值可用百分点表示；此单位变换不改变存储的原始数值。用于区分重叠方法的微小水平错位仅用于显示，不代表新增频率或延迟设置。

## 具体薄弱点与可检查的修改路径

1. **全面优越的论文定位站不住。** F-only 中 TW 对 T 的不利效应，以及 TW 对 W 的恢复代价，不能被平均正确率或宽量程掩盖。改法：正文将论断限定到干扰机制和指标；使用完整配对差值图呈现方向；结果段明确讨论 F1–F4，而不是只展示 HPRF 或三个有利代表点。
2. **条件恢复概率的估计精度并不一致。** H100、H200 的 TW `P_C_given_E` 分母分别只有 6,873、11,851 次 E-origin 转移；有正分母的重复分别是 1,920、1,995。总重复数相同不意味着每个条件指标的有效支持量相同。改法：保留宽区间，公开分母和有效支持重复数；不要用增加小数位掩盖不确定性，也不能在绘图阶段重新采样以得到更窄区间。
3. **非正确游程长度受固定观察窗影响。** 截至第 200 周期仍开放的游程已计入平均值，其长度仅反映已观察部分；在接近 200 周期的结果上尤其不能解释为实际最终恢复时长。改法：纵轴和正文统一使用 observed non-correct run length；若后续需要推断真正恢复时间，应另行设计更长观察窗或处理删失的分析，并与当前冻结结果严格区分。
4. **H300 分层只能作为描述性机制线索。** 单个 HPRF 条件、三个不等支持量分层，不能支撑一般化机制定律；分层汇总中的 C→E 计数没有匹配的 C-origin 分母，不能转成条件转移概率。改法：同时展示各层支持量，明确无推断区间；图注限定为 H300 描述性分析，不给显著性星号，不画伪造的 C→E 概率。
5. **逐项区间不能替代多重比较或等效性论证。** 区间包含零不等于方法等效，全部展示的 144 个配对结果也没有联合覆盖保证。改法：用效应量与区间描述结果，不写“无差异”“等效”或未定义的“显著优越”；若论文需要这些主张，须另行预设等效界限或校正策略，不能悄悄改动本次图中的冻结区间。
6. **固定仿真条件限制工程外推。** 当前图覆盖的是已设定的噪声与门宽条件，不能凭曲线外观推出对任意噪声、硬件误差或未测试干扰过程的鲁棒性。改法：方法章节列清噪声、门宽、脉宽编码和扰动来源的工程依据；把跨参数鲁棒性作为需要独立证据的问题，不将本次重绘当作该证据。

## 六项未达到预设精度目标

216 个预设精度检查中有 **6 个未达目标**。目标只适用于 T、TW 的单方法指标与 TW–T 对照，不适用于全部五种方法或 TW–W 对照。因此不得将 6/216 描述成“全部输出中仅 6 项有问题”，也不能推断 TW–W 已满足同样精度要求。

半宽定义为 `max(point − lower, upper − point)`。下表保留概率的原始 0–1 单位；概率目标 0.02 相当于 2 个百分点。游程长度单位为周期。显示值四舍五入，达标判断使用冻结原值。

| 条件 | 方法或对照 | 指标 | 实际最大单侧半宽 | 目标半宽 | 实际/目标 |
|---|---|---|---:|---:|---:|
| H100 | TW | P_C_given_E | 0.069962 | 0.020000 | 3.498 |
| H200 | TW | P_C_given_E | 0.049171 | 0.020000 | 2.459 |
| F4 | T | Mean_L_NC | 0.269794 | 0.250000 | 1.079 |
| H100 | TW–T | P_C_given_E | 0.071663 | 0.020000 | 3.583 |
| H200 | TW–T | P_C_given_E | 0.049563 | 0.020000 | 2.478 |
| HF500-3 | TW–T | Mean_L_NC | 0.250534 | 0.250000 | 1.002 |

HF500-3 的最后一项虽然仅略超目标，也不能因为显示精度或舍入而判作通过。精度目标不是方法优劣判据；未达目标与区间是否跨零是两个不同问题。

# Provisional English figure legends

The following legends are draft captions for the new figures. Panel lettering and final numbering should be reconciled with the manuscript before submission. Absolute probability panels use percentages; paired probability differences use percentage points. The observed run-length axes in Figs. 1–3 are logarithmic. Some intervals are narrower than the plotting symbols; exact endpoints are included in the source tables.

## fig01_hprf — Selection stability under HPRF-only interference

**Fig. 1 | Selection stability across HPRF-only interference rates.** Correct-selection probability, P_cor (a); correct-state persistence, P(C at k+1 | C at k) (b); recovery from an erroneous selection, P(C at k+1 | E at k) (c); and mean observed non-correct run length (d) are shown for FIRST, LAST, T, W and TW at 100, 200, 300, 400 and 500 kHz. Points are ratios of pooled counts from n = 2,000 independent simulation repeats per condition, with K = 200 cycles per repeat. Error bars are pointwise 95% percentile intervals from B = 2,000 whole-repeat bootstrap resamples; the same repeat indices were resampled for all methods within a condition. Connecting lines guide the eye between the simulated settings. Small horizontal offsets distinguish overlapping methods and do not denote additional rate settings. Non-correct runs include E and N states; terminal runs are retained using only their lengths observed through cycle 200. The intervals are not multiplicity-adjusted, and no p values are reported.

## fig02_f_only — Selection stability under F-only interference

**Fig. 2 | Selection stability across F-only interference delays.** Correct-selection probability (a), correct-state persistence (b), recovery from an erroneous selection (c) and mean observed non-correct run length (d) are shown for all five selection methods at F delays of +1, +2, +3 and +4 μs. Each condition contains n = 2,000 independent simulation repeats of K = 200 cycles. Points are pooled-count ratios, and error bars are pointwise 95% percentile intervals from B = 2,000 whole-repeat bootstrap resamples paired across methods. Connecting lines link the simulated settings; horizontal display offsets only separate overlapping methods. Recovery probability is conditioned on E, not on all non-correct states. Non-correct run lengths include terminal runs truncated by the 200-cycle observation window. These data should be read together with the paired contrasts in Figs. 4 and 5 to assess method-dependent trade-offs. No p values or multiplicity adjustments were added.

## fig03_composite — Selection stability under composite interference

**Fig. 3 | Selection stability under combined HPRF and F interference.** Columns correspond to HPRF rates of 200, 300 and 500 kHz; rows show correct-selection probability, correct-state persistence, recovery from an erroneous selection and mean observed non-correct run length, respectively. Each panel includes F delays of +1, +2 and +3 μs and all five selection methods, retaining all nine composite conditions. Points are pooled-count ratios from n = 2,000 independent repeats per condition and K = 200 cycles per repeat. Error bars are pointwise 95% percentile intervals from B = 2,000 whole-repeat bootstrap resamples, paired across methods within each condition. Lines join the discrete simulated delays; small display offsets distinguish overlapping methods. Non-correct run lengths retain the observed portions of runs still open at cycle 200 and should not be interpreted as uncensored recovery times. No p values or multiplicity-adjusted intervals are shown.

## fig04_tw_minus_t — Paired effects relative to timing-only selection

**Fig. 4 | Paired differences between joint and timing-only selection.** Each row in a–d is one of the 18 interference conditions, ordered by the frozen experimental design; F-only rows are shaded. Panels show TW minus T for correct-selection probability (a), correct-state persistence (b), recovery from an erroneous selection (c) and mean observed non-correct run length (d). Panels e and f enlarge the F-only contrasts from a and b, respectively, with unchanged estimates and interval endpoints, to resolve small negative effects and intervals crossing zero. Points are differences between pooled-count ratios. Horizontal intervals are pointwise 95% percentile intervals obtained by resampling n = 2,000 whole simulation repeats B = 2,000 times, using identical resampled repeat indices for both methods within each condition; each repeat contains 200 cycles. The vertical zero line denotes equal point-estimate values. Positive differences favour TW for the three probability metrics; negative differences favour TW for observed non-correct run length. Probability differences are expressed in percentage points, and run-length differences in cycles. All contrast signs remain TW minus T. Intervals crossing zero are not evidence of equivalence. No p values or multiplicity adjustment are used.

## fig05_tw_minus_w — Paired effects relative to width-only selection

**Fig. 5 | Paired differences between joint and width-only selection.** TW minus W is shown for correct-selection probability (a), correct-state persistence (b), recovery from an erroneous selection (c) and mean observed non-correct run length (d), retaining all 18 interference conditions in design order. Points are differences between ratios of pooled counts; intervals are pointwise 95% percentile intervals from B = 2,000 whole-repeat bootstrap resamples paired across methods. Each condition contains n = 2,000 independent repeats of 200 cycles. Positive differences favour TW for probability metrics, whereas negative differences favour TW for observed non-correct run length; the zero line marks equal metric values. Probability differences are in percentage points and run-length differences are in cycles. The paired results distinguish correct-state persistence from subsequent recovery, rather than combining these outcomes into a single score. Intervals are not multiplicity-adjusted, and no p values are reported.

## fig06_width_strata — Descriptive dependence on true width separation

**Fig. 6 | Descriptive correctness by true pulse-width separation at 300 kHz.** Correct-selection probability is shown for all five methods in three strata defined by the absolute difference between the true H-interference pulse width and the nominal reference pulse width of the current cycle: 0, 10 and ≥20 ns. Stratum support is shown alongside the correctness estimates. These are true-width strata, not bins of the noisy measured width difference. Probabilities are ratios of pooled correct-cycle counts to pooled cycle counts from n = 2,000 simulation repeats of 200 cycles. The strata contain 24,986, 47,051 and 327,963 cycles, respectively, with identical stratum membership and support across methods; these cycle counts are not independent sample sizes. The analysis is descriptive and specific to H300. No confidence intervals, p values or stratum-specific transition probabilities are estimated. The final ≥20 ns category groups multiple separations and is not a single width value.

## figS01_precision — Precision audit

**Supplementary Fig. 1 | Audit of the prespecified precision targets.** All 216 checks cover 18 interference conditions, four metrics and three inference entries: T, TW and TW minus T. Cell values represent the maximum one-sided 95% interval half-width, max(point estimate − lower bound, upper bound − point estimate), divided by its prespecified target. Targets are 0.02 for probabilities or probability differences and 0.25 cycles for mean observed non-correct run length or its difference. A ratio exceeding 1 indicates failure to meet the target; six checks exceed 1. The audit uses the frozen pointwise percentile intervals from B = 2,000 whole-repeat bootstrap resamples of n = 2,000 repeats per condition. It evaluates precision rather than the direction of a method effect or statistical significance. FIRST, LAST, W and TW minus W were outside this precision-target contract. Values are displayed with rounding, but threshold classifications use the unrounded values.

## 无干扰一致性数据的表注

**No-interference consistency check.** All five methods produce identical repeat-level results under the no-interference condition (n = 2,000 repeats, 200 cycles per repeat). Undefined recovery probabilities and non-correct run lengths arise from absent denominator events and are retained as undefined. This condition is a descriptive consistency check; no bootstrap inference is performed.
