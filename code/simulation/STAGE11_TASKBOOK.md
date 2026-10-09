# Stage 11：结果导出与论文证据数据冻结 Codex 正式任务书

> 日期：2026-08-26  
> 状态：**正式封口版**  
> 上一阶段：**Stage 10 PASS**  
> 本阶段定位：**整个 Python 数值研究的最后一个代码 Stage**  
> Stage 11 完成后立即 STOP；不得创建或执行 Stage 12。

---

## 0. 唯一执行目标

本阶段只完成：

**Stage 11：结果导出与论文证据数据冻结**

Stage 11 不是新的模拟阶段，也不是新的统计推断阶段。

允许做的事情只有：

1. 验证 Stage 10 PASS ZIP 与所有冻结源文件身份；
2. 从 Stage 10 已冻结的 individual / paired / precision summary 中做确定性字段补充与完整场景切片；
3. 从 Stage 9 H300 width strata 做**描述性 pooled-count 聚合**；
4. 从冻结 condition/method contracts 导出论文数据元信息；
5. 建立论文证据 artifact 注册表与 SHA 追溯链；
6. 自动打包完整 Stage 11 ZIP；
7. STOP。

严禁：

- 重新运行 Stage 9 Formal；
- 重新运行 Stage 10 Bootstrap；
- 重新读取 Bootstrap distribution 后重新计算 CI；
- 新增统计方法、CI、p-value 或 contrast；
- 删除不利 Condition；
- 只导出“好看”结果；
- 修改任何科学参数；
- 生成论文最终图片；
- 撰写论文正式数值结论；
- 进入 Stage 12。

---

## 1. 唯一 active baseline

文件：

`sal_stability_stage10_20260825.zip`

SHA256 必须严格为：

`fb3666b9a9cd74d2a52978df98b2e4f08b7ddcb11bff0134bbdaed6cc09fb389`

除此之外任何 Stage 10 ZIP、手工目录、RAR 或旧版本均不得作为 active baseline。

Stage 11 必须从该 ZIP 完整解压后的工程开始。

---

## 2. Stage 10 PASS 与冻结源文件身份

执行前必须逐一验证：

```text
artifacts/stage10/status.json
SHA256=91cb29d8c06aeabfed7b8dbfe3a46f26ee236533e12a045753b4cee24a57ecf6

artifacts/stage10/bootstrap_manifest.json
SHA256=69748b6f37e3caf5d3abb561042ca190c8f0068d2217b44b14cb770faa50b71d

artifacts/stage10/bootstrap_individual_summary.csv
SHA256=ad9bca843ff063d1a9cc26d134f8fc43b3cdeb9460f850e9e5afbda6c0d5f0b1

artifacts/stage10/bootstrap_paired_summary.csv
SHA256=2e864379ff8985fce28cfc7fd9d71f29ca5ec593fd7340dfeaf586684d7e58fe

artifacts/stage10/stage8_precision_target_audit.csv
SHA256=361e5b017cb078545c032a17f98104887ae46703a1a921d6cf8689e3ae9988e4

artifacts/stage10/bootstrap_distributions.npz
SHA256=370e11118c828c092a0c86037488025555578e2040364d776940887a8b2c8cad

artifacts/stage10/bootstrap_audit_indices.npz
SHA256=ff21b44d5d6bf638cbde0f881e9e0bca6a8778b4cf154b125d8511e2ec873b07

artifacts/stage10/stage10_science_contract.json
SHA256=fdaa84d99e3b9e8fbcebe091c7a173a90ba5a42cef054f48c79eca01179656fd

artifacts/stage10/ni_consistency_report.json
SHA256=3fcc11e13a40b7ae177643c71c7287878c33fb43a124d8d8604c96288f9d19bc

artifacts/stage9/formal_repeat_metrics.csv
SHA256=e3084733adf9fa3f2325b0407b4358aeae118df8f6a152a4c1f5984757f3f8f6

artifacts/stage9/formal_aggregated_metrics.csv
SHA256=803856c4d776b53b63e7e6f8be4425ce96ddb05c49aaca97c049134b04644865

artifacts/stage9/formal_h300_width_strata.csv
SHA256=4e70f33ba195070a75e35fc6b3d14b7d5d7851b4ca414fcac7e8f24046a108b4
```

并要求：

```text
Stage10 status = PASS
bootstrap_executed = true
stage11_executed = false
blocking_issue_count = 0
condition_dropped = false
parameter_changed = false
formal_rerun_executed = false
```

任何 SHA 或状态不一致：

**FAIL + STOP。**

---

## 3. Stage 11 科学合同文件

外部冻结合同：

`Stage11_Results_Export_and_Paper_Evidence_Freeze_Science_Contract_20260826.json`

canonical SHA256：

`9fc2141695a6e73697bbee2e7590eb54678be5ca094ae06e753a36a38f3282a1`

Stage 11 开始前必须按原字节写入：

`artifacts/stage11/stage11_science_contract.json`

并重新计算 SHA。

必须严格等于：

`9fc2141695a6e73697bbee2e7590eb54678be5ca094ae06e753a36a38f3282a1`

不得根据 Stage 10 的结果方向修改合同。

---

## 4. Stage 11 的数据读取边界

### 4.1 允许作为结果导出来源

只允许：

```text
artifacts/stage10/bootstrap_individual_summary.csv
artifacts/stage10/bootstrap_paired_summary.csv
artifacts/stage10/stage8_precision_target_audit.csv
artifacts/stage10/ni_consistency_report.json
artifacts/stage9/formal_aggregated_metrics.csv
artifacts/stage9/formal_h300_width_strata.csv
src/sal_stability_stage1/contracts.py
```

### 4.2 只允许验证 SHA，不允许用于重新统计

下列文件 Stage 11 可验证身份，但**不得重新从中计算新 CI 或新 Bootstrap 结果**：

```text
artifacts/stage10/bootstrap_distributions.npz
artifacts/stage10/bootstrap_audit_indices.npz
artifacts/stage10/bootstrap_manifest.json
artifacts/stage9/formal_repeat_metrics.csv
```

换言之：

Stage 10 的 PointEstimate / CI95Lower / CI95Upper 已经冻结。

Stage 11 的职责是导出，不是“再算一次看看”。

---

## 5. 冻结 Condition 元数据

Condition 必须直接来自 Stage 10 基线中的 `contracts.py`：

```text
0  NI       NI
1  H100     HPRF       100 kHz
2  H200     HPRF       200 kHz
3  H300     HPRF       300 kHz
4  H400     HPRF       400 kHz
5  H500     HPRF       500 kHz

6  F1       IDF        +1 us
7  F2       IDF        +2 us
8  F3       IDF        +3 us
9  F4       IDF        +4 us

10 HF200-1  COMPOSITE  200 kHz +1 us
11 HF200-2  COMPOSITE  200 kHz +2 us
12 HF200-3  COMPOSITE  200 kHz +3 us
13 HF300-1  COMPOSITE  300 kHz +1 us
14 HF300-2  COMPOSITE  300 kHz +2 us
15 HF300-3  COMPOSITE  300 kHz +3 us
16 HF500-1  COMPOSITE  500 kHz +1 us
17 HF500-2  COMPOSITE  500 kHz +2 us
18 HF500-3  COMPOSITE  500 kHz +3 us
```

正式导出字段：

```text
ConditionOrdinal
ConditionCode
Scene
HPRF_Hz
HPRF_kHz
FDelay_s
FDelay_us
```

缺失参数必须保留为空值/NaN，不得填 0 伪装成物理值。

---

## 6. 冻结 Method 与 Metric 元数据

Method：

```text
0 FIRST  首脉冲准则
1 LAST   末脉冲准则
2 T      最优时序准则
3 W      脉宽单特征准则
4 TW     双特征联合准则
```

Metric：

```text
P_cor         PreferredDirection=HIGHER
P_C_given_C   PreferredDirection=HIGHER
P_C_given_E   PreferredDirection=HIGHER
Mean_L_NC     PreferredDirection=LOWER
```

Contrast：

```text
0 TW-T  Comparator=T
1 TW-W  Comparator=W
```

不得新增任何 contrast。

---

## 7. paper_results_individual_master.csv

来源只能是：

`bootstrap_individual_summary.csv`

源表必须先验证恰好：

`18 × 5 × 4 = 360`

行，唯一键：

```text
ConditionOrdinal
MethodOrdinal
Metric
```

Stage 11 输出：

`artifacts/stage11/paper_results_individual_master.csv`

必须仍然恰好 360 行。

只允许在源列基础上补充：

```text
Scene
HPRF_Hz
HPRF_kHz
FDelay_s
FDelay_us
MethodNameZH
MetricPreferredDirection
SourceArtifact
SourceArtifactSHA256
```

源表中的以下内容必须原值保留：

- FormalSumA/B
- denominator support
- PointEstimate
- PointDefined
- undefined reason
- BootstrapB
- Defined/UndefinedBootstrapCount
- CI95Defined
- CI95Lower
- CI95Upper
- CI95Width
- MaxOneSidedHalfWidth
- Stage8 precision fields

禁止重新计算或覆盖 CI。

---

## 8. paper_results_paired_master.csv

来源只能是：

`bootstrap_paired_summary.csv`

源表必须恰好：

`18 × 2 × 4 = 144`

行。

唯一键：

```text
ConditionOrdinal
ContrastOrdinal
Metric
```

输出：

`artifacts/stage11/paper_results_paired_master.csv`

仍必须恰好 144 行。

允许补充：

```text
Scene
HPRF_Hz
HPRF_kHz
FDelay_s
FDelay_us
ComparatorMethodNameZH
MetricPreferredDirection
CIPositionRelativeToZero
SourceArtifact
SourceArtifactSHA256
```

`CIPositionRelativeToZero` 唯一定义：

```text
if CI95Defined == false:
    UNDEFINED
elif CI95Lower > 0:
    POSITIVE
elif CI95Upper < 0:
    NEGATIVE
else:
    INCLUDES_ZERO
```

禁止生成：

```text
SIGNIFICANT
NOT_SIGNIFICANT
SUPERIOR
INFERIOR
WIN
LOSE
```

禁止 p-value。

---

## 9. paper_precision_audit_master.csv 与限制项

完整复制并验证 Stage 10：

`stage8_precision_target_audit.csv`

216 行。

输出：

`paper_precision_audit_master.csv`

必须仍为 216 行，并增加 source identity 字段即可。

同时输出：

`paper_statistical_precision_limitations.csv`

它必须是上述 216 行中：

```text
MeetsStage8PrecisionTarget == false
```

的**精确全集**。

不得：

- 改 target；
- 删除 borderline row；
- 只保存 H100/H200；
- 根据论文需要选择性遗漏。

limitations 行数不作为方法优劣 PASS 条件，但必须与 deterministic filter 完全一致。

---

## 10. NI_consistency.csv

Stage 10 NI report 必须 PASS：

```text
AllCyclesCorrectPassed=true
FiveMethodRepeatMetricsIdentityPassed=true
InferentialBootstrapExecuted=false
```

然后从：

`formal_aggregated_metrics.csv`

中取 ConditionOrdinal=0 的完整 5 Method 行。

输出：

`artifacts/stage11/paper_data/NI_consistency.csv`

恰好 5 行。

至少包含：

```text
MethodOrdinal
MethodCode
MethodNameZH
R
K
Total_N_C
Total_N_E
Total_N_N
P_cor_value
P_cor_defined
P_C_given_C_value
P_C_given_C_defined
P_C_given_E_value
P_C_given_E_defined
P_C_given_E_undefined_reason
Mean_L_NC_value
Mean_L_NC_defined
Mean_L_NC_undefined_reason
```

不得给 NI 的 undefined 指标人为补 CI 或补 0。

---

## 11. 三类干扰场景必须完整导出

### 11.1 HPRF

ConditionOrdinal：

`1,2,3,4,5`

必须全部保留。

输出：

```text
paper_data/HPRF_individual.csv
```

行数：

`5 × 5 × 4 = 100`

以及：

```text
paper_data/HPRF_paired.csv
```

行数：

`5 × 2 × 4 = 40`

禁止只留 H100/H300/H500。

### 11.2 F-only

ConditionOrdinal：

`6,7,8,9`

输出：

```text
paper_data/F_only_individual.csv
```

80 行。

```text
paper_data/F_only_paired.csv
```

32 行。

F1—F4 必须全部存在。

### 11.3 Composite

ConditionOrdinal：

`10...18`

输出：

```text
paper_data/composite_individual.csv
```

180 行。

```text
paper_data/composite_paired.csv
```

72 行。

9 个复合 Condition 必须全部存在。

禁止只挑三个“代表点”。

---

## 12. H300 width strata 描述性聚合

唯一输入：

`artifacts/stage9/formal_h300_width_strata.csv`

输入必须：

30000 行。

仅允许 groupby：

```text
MethodOrdinal
MethodCode
WidthStratum
```

固定三层顺序：

```text
WIDTH_STRATUM_0NS
WIDTH_STRATUM_10NS
WIDTH_STRATUM_GE20NS
```

聚合：

```text
TotalCycleCount = sum(StratumCycleCount)
TotalCorrectCount = sum(StratumCorrectCount)
TotalErrorCount = sum(StratumErrorCount)
Total_C_to_E_Count = sum(Stratum_C_to_E_Count)
PositiveSupportRepeatCount = count of rows with StratumCycleCount > 0
```

必须满足：

```text
TotalCorrectCount + TotalErrorCount == TotalCycleCount
```

确定性描述指标：

```text
P_cor_stratum =
    TotalCorrectCount / TotalCycleCount

P_error_stratum =
    TotalErrorCount / TotalCycleCount
```

使用 pooled counts。

**禁止 mean of per-Repeat proportions。**

输出：

`paper_data/H300_width_strata_summary.csv`

必须：

`5 Methods × 3 strata = 15 rows`

不执行 Bootstrap。

不生成 CI。

不生成 p-value。

`Total_C_to_E_Count` 只保留计数；因为当前 strata 文件没有对应的 C-origin denominator，禁止擅自构造所谓 `P_C_to_E_stratum`。

---

## 13. 注册表导出

必须生成：

### condition_registry_export.csv

19 行。

### method_registry_export.csv

5 行。

### metric_registry_export.csv

4 行。

### contrast_registry_export.csv

2 行。

这些只是论文数据解释元信息，不得包含结果方向判断。

---

## 14. paper_evidence_registry.csv

Stage 11 所有论文证据 CSV 生成完毕后，建立：

`artifacts/stage11/paper_evidence_registry.csv`

每个证据 artifact 一行。

至少包含：

```text
ArtifactPath
ArtifactClass
RowCount
SHA256
ImmediateSourceArtifacts
InferenceClass
AllowedUse
SelectiveFilteringAllowed
```

InferenceClass 只能从以下值选择：

```text
STAGE10_FROZEN_INFERENCE
STAGE9_FROZEN_DESCRIPTIVE
IDENTITY_OR_METADATA
```

所有结果类 artifact：

```text
SelectiveFilteringAllowed=false
```

registry 不递归记录自己的 SHA。

最终 `stage11_manifest.json` 再记录 registry 的 SHA。

---

## 15. 禁止 Stage 11 生成论文图

Stage 11 只输出 graph-ready CSV。

不得调用：

- matplotlib
- seaborn
- plotly
- Excel chart
- SVG/PDF/PNG 绘图
- 任何论文图渲染

Stage 11 的最后产物是数据，不是图。

图表形式由 Stage 11 PASS 后单独审查。

---

## 16. 禁止 Stage 11 写论文结论

不得自动生成诸如：

```text
TW significantly outperforms T
TW is superior
the proposed method improves...
```

Stage 11 只能导出客观数值、CI 与结构标签。

`CIPositionRelativeToZero` 不是论文结论，只是区间几何位置。

科学解释由 Stage 11 独立验收后另行完成。

---

## 17. Stage 1—10 冻结树保护

Stage 10 ZIP 是唯一 current baseline。

Stage 11 必须新增：

`validate_stage1_through_stage10_tree_unchanged()`

直接以 Stage 10 ZIP archive member SHA 为权威。

只允许修改现有：

```text
README.md
pyproject.toml
```

只允许新增：

```text
STAGE11_TASKBOOK.md
scripts/build_stage11.py
src/sal_stability_stage1/results_export.py
src/sal_stability_stage1/stage11.py
tests/test_stage11_contract.py
tests/test_stage11_export.py
tests/test_stage11_storage.py
tests/test_stage11_execution.py
artifacts/stage11/**
```

其余任何 Stage 1—10 文件：

- 修改；
- 删除；
- 重命名；
- 越界新增；

均：

**FAIL + STOP。**

尤其禁止修改任何历史 test 使其变绿。

---

## 18. 历史 Stage boundary forward-compatibility

Stage 11 新文件加入后，旧阶段的封闭树测试天然可能拒绝后续 Stage additions。

允许识别的历史边界节点仅有：

```text
tests/test_stage8_decision.py::
test_stage1_through_stage7_byte_identity_and_frozen_science_gate
```

代码：

`LEGACY_STAGE_BOUNDARY_NOT_FORWARD_COMPATIBLE`

```text
tests/test_stage9_contract.py::
test_stage8_identity_and_decision_boundary_are_independently_verified
```

代码：

`LEGACY_STAGE9_BOUNDARY_NOT_FORWARD_COMPATIBLE`

```text
tests/test_stage10_contract.py::
test_stage9_pass_and_stage1_through_stage9_zip_tree_identity
```

代码：

`LEGACY_STAGE10_BOUNDARY_NOT_FORWARD_COMPATIBLE`

规则：

1. 不得修改/skip/xfail/delete 旧测试；
2. 如果完整历史 pytest 在所有原始外部 baseline ZIP 都存在时执行，失败节点集合只能等于上述三个；
3. 任何第四个 failure/error：FAIL；
4. 三个节点的差异只能来自任务书明确批准的后续 Stage additions；
5. Stage 11 自己的 Stage10-ZIP byte identity gate 必须 PASS；
6. 若某历史测试因其原始外部 baseline ZIP 缺失而无法运行，必须明确记录缺失 archive，不能伪称 full pytest 全绿；
7. Stage 11 PASS 的核心权威门禁仍是：Stage10 ZIP identity + source SHA + Stage11 专属测试 + deterministic export reconstitution。

---

## 19. Stage 11 专属测试

### test_stage11_contract.py

至少检查：

- Stage10 ZIP SHA exact；
- Stage10 status PASS；
- Stage11 science contract SHA exact；
- 所有 source SHA exact；
- FinalCodeStage=true；
- no bootstrap/no random/no p-value/no new contrast；
- 三类场景 condition universe exact；
- allowed file-change scope exact。

### test_stage11_export.py

至少检查：

- individual master 360 unique；
- paired master 144 unique；
- precision master 216 unique；
- precision limitations 为 false subset 精确全集；
- HPRF 100/40；
- F-only 80/32；
- composite 180/72；
- NI 5；
- H300 summary 15；
- source numerical fields 与 Stage10 summaries 完全一致；
- `CIPositionRelativeToZero` 规则 exact；
- 禁止 significance/superiority labels。

### test_stage11_storage.py

至少检查：

- evidence files 全部存在；
- row counts；
- SHA manifest；
- registry 与 artifacts 一致；
- Stage11 ZIP 成员完整；
- Stage12 artifact 不存在；
- PNG/PDF/SVG/xlsx figure artifact 不存在。

### test_stage11_execution.py

至少检查：

- `validate_stage1_through_stage10_tree_unchanged()` PASS；
- 无 RNG 调用；
- 无 Bootstrap 执行；
- 无 physics/kernel/formal execution；
- no plotting library execution path；
- no p-value；
- no new CI；
- repeated Stage11 export produces byte-identical CSV/JSON artifacts where deterministic serialization applies。

---

## 20. 确定性序列化规则

CSV：

- UTF-8
- `index=False`
- 固定列顺序
- 固定 row sort order
- `lineterminator="\n"`
- 浮点不得主动四舍五入成论文显示位数；保留源 CSV 可回读精度

JSON：

- UTF-8
- `sort_keys=True`
- `indent=2`
- 末尾单个换行

Stage 11 是“证据母数据”，不是排版输出。

论文保留几位小数以后单独裁决。

---

## 21. 推荐 Stage 11 artifact 结构

```text
artifacts/stage11/
    stage11_science_contract.json
    source_identity.json
    stage11_manifest.json
    status.json
    validation_report.md
    test_report.txt
    changed_files.txt

    registry/
        condition_registry_export.csv
        method_registry_export.csv
        metric_registry_export.csv
        contrast_registry_export.csv

    paper_results_individual_master.csv
    paper_results_paired_master.csv
    paper_precision_audit_master.csv
    paper_statistical_precision_limitations.csv
    paper_evidence_registry.csv

    paper_data/
        NI_consistency.csv
        HPRF_individual.csv
        HPRF_paired.csv
        F_only_individual.csv
        F_only_paired.csv
        composite_individual.csv
        composite_paired.csv
        H300_width_strata_summary.csv
```

---

## 22. Stage 11 PASS 硬门禁

必须全部满足：

1. Stage10 baseline ZIP SHA exact。
2. Stage10 status=PASS。
3. Stage10 blocking_issue_count=0。
4. 全部冻结 source SHA exact。
5. Stage11 science contract SHA exact。
6. Stage1—10 tree identity gate PASS。
7. 只发生任务书允许的 Stage11 additions / README/pyproject 修改。
8. Stage11 tests 全部 PASS。
9. individual master=360 rows unique。
10. paired master=144 rows unique。
11. precision audit master=216 rows unique。
12. precision limitations 是 false subset 精确全集。
13. HPRF individual=100。
14. HPRF paired=40。
15. F-only individual=80。
16. F-only paired=32。
17. composite individual=180。
18. composite paired=72。
19. NI=5。
20. H300 summary=15。
21. 三类干扰 Condition 无遗漏。
22. 所有 Stage10 PointEstimate/CI 源字段原值保留。
23. paired `CIPositionRelativeToZero` exact。
24. H300 pooled-count conservation PASS。
25. H300 无新 CI / p-value。
26. evidence registry SHA/rowcount/source lineage valid。
27. no new bootstrap。
28. no RNG。
29. no new CI。
30. no p-value。
31. no new contrast。
32. no condition dropping。
33. no parameter change。
34. no Formal rerun。
35. no physical simulation。
36. no paper figure rendering。
37. no result conclusion writing。
38. no result-driven selection。
39. no Stage12。
40. `StopAfterStage11=true`。

注意：

**结果方向不是 Stage 11 PASS 条件。**

CI 是否包含 0、TW 是否优于某方法、precision limitations 有多少行，均不能用于改变输入、删行或决定 Stage11 实现 PASS。

---

## 23. status.json 最少字段

```text
stage=11
stage_name=RESULTS_EXPORT_AND_PAPER_EVIDENCE_FREEZE
status=PASS/FAIL
final_code_stage=true

stage10_baseline_zip_sha256_verified
stage10_status_pass_verified
all_frozen_source_sha256_verified
stage11_science_contract_sha256_verified
stage1_through_stage10_tree_unchanged

individual_master_rows
paired_master_rows
precision_audit_master_rows
precision_limitation_rows
ni_rows
hprf_individual_rows
hprf_paired_rows
f_only_individual_rows
f_only_paired_rows
composite_individual_rows
composite_paired_rows
h300_width_strata_summary_rows

all_scenario_conditions_exported
source_inference_values_preserved
ci_position_labels_verified
h300_pooled_count_conservation_passed
evidence_registry_verified

new_bootstrap_executed=false
rng_executed=false
new_ci_generated=false
p_value_generated=false
new_contrast_added=false
condition_dropped=false
parameter_changed=false
formal_rerun_executed=false
physical_simulation_executed=false
paper_figure_generated=false
paper_result_conclusion_generated=false
result_driven_selection_used=false
stage12_executed=false

legacy_boundary_exceptions
blocking_issue_count
stop_after_stage11=true
```

---

## 24. 自动打包与 STOP

成功后自动生成：

`sal_stability_stage11_20260826.zip`

ZIP 必须包含：

- 完整 Stage 1—11 工程；
- `artifacts/stage11/` 全部证据数据；
- manifest/status/report；
- 不得存在 Stage 12 artifact。

生成 ZIP 后重新打开 ZIP，验证：

- 必需成员存在；
- row counts；
- artifact SHA；
- tree identity；
- forbidden artifact absence。

然后计算最终 ZIP SHA256。

自动 ZIP 成功后立即 STOP。

不得：

- 开始画论文图；
- 开始写第 4 章；
- 创建 Stage 12；
- 对数据重新筛选。

---

## 25. Codex 最终返回

只返回：

1. Stage 11 PASS/FAIL；
2. 自动生成的 `sal_stability_stage11_20260826.zip`；
3. ZIP SHA256；
4. source identity 摘要；
5. Stage11 tests / legacy boundary 摘要；
6. evidence artifact row-count 与 SHA 摘要；
7. 若 FAIL，明确 blocking issue。

Codex 自报 PASS 不作为最终裁决。

完整 Stage 11 ZIP 必须交由 ChatGPT 独立验收。

---

## 26. Stage 11 之后的唯一工作方向

Stage 11 是最后一个代码 Stage。

Stage 11 PASS 后，不再创建 Stage 12。

后续工作转换为：

```text
Stage11 frozen evidence data
→ 论文结果逻辑审查
→ 图表证据链设计
→ 第4章“结果与分析”
→ 反向校准第3章“数值模拟”
→ 参数工程/文献依据补强
→ 第5章“结论”
→ 摘要
→ 引言贡献闭环
```

任何后续论文调整都不得反向修改 Stage 9 Formal 或 Stage 10 Bootstrap 数据。
