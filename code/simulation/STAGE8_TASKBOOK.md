```text
# Stage 8：Formal R 自动裁决 Codex 正式执行任务书
# FORMAL_R_AUTOMATIC_DECISION
#
# 本任务只能执行 Stage 8。
# Stage 8 的唯一职责是：
# 读取 Stage 7 在 Pilot 前已冻结的 R 裁决合同和 Pilot 决策输入，
# 计算 144 个 individual precision cells 与 72 个 T–TW paired precision cells，
# 按冻结 ALL-rule 自动给出唯一 Formal R，并立即 STOP。
#
# 禁止重新运行 Pilot；禁止运行 FORMAL；禁止 Bootstrap；禁止生成论文正式结果。


===============================================================================
0. 唯一输入基线
===============================================================================

唯一代码基线：

sal_stability_stage7_20260824.zip

ZIP SHA256 必须为：

d6412152a199a820ef40c1485689846ee80d9227c51f9e3b6f5ff4dbac55e145

不一致：

FAIL
STOP

不得使用 Stage 6 或更早工程重新拼接。


===============================================================================
1. Stage 8 固定身份与职责
===============================================================================

StageNumber = 8

StageName =
FORMAL_R_AUTOMATIC_DECISION

Stage 8 只允许：

1. 验证 Stage 7 已 PASS；
2. 验证 Stage 1—7 冻结科学文件 SHA 未改变；
3. 验证 Stage 8 合同 SHA；
4. 验证 Stage 8 输入 SHA 与 7200 行完整性；
5. 计算 144 个 individual precision cells；
6. 计算 72 个 paired precision cells；
7. 执行冻结的 R=1000/2000 二值裁决；
8. 记录 ProjectedPrecisionWarning；
9. 输出 Stage 8 artifacts；
10. STOP。

Stage 8 禁止：

- 重新生成任何 G/H/F 物理事件；
- 重新运行 Pilot；
- 从 trajectory 重新统计 RepeatMetrics；
- 修改 Pilot 数据；
- 使用 FORMAL Namespace；
- 运行 R=1000；
- 运行 R=2000；
- Bootstrap；
- 置信区间；
- p-value；
- 根据 TW 是否优于 T 决定 R；
- 修改 Stage 8 合同；
- 删除 Condition；
- 修改参数；
- 输出论文正式结论。


===============================================================================
2. Stage 7 状态门禁
===============================================================================

必须读取：

artifacts/stage7/status.json

至少要求：

stage = 7
stage_name = PILOT_R200
status = PASS

pilot_executed = true
pilot_r = 200
pilot_namespace = PILOT

pilot_condition_count = 19
pilot_method_count = 5

pilot_repeat_metrics_row_count = 19000
pilot_aggregated_metrics_row_count = 95
pilot_h300_strata_row_count = 3000
stage8_decision_input_row_count = 7200

science_contract_changed = false
frozen_hashes_passed = true

r_decision_executed = false
formal_executed = false
bootstrap_executed = false
paper_result_generated = false

blocking_issue_count = 0

任一不满足：

FAIL
STOP


===============================================================================
3. Stage 8 唯一统计输入
===============================================================================

Stage 8 只允许读取以下两份文件作为统计裁决依据。

A.

artifacts/stage7/stage8_r_decision_contract.json

SHA256：

1991d8cec0a91438ccd95a5e96ed02cfd4d27285e9e2fa4eba047f9938e285e6


B.

artifacts/stage7/stage8_decision_input.csv

SHA256：

655ffd9f03a7a7824662c9a7a1a2281ed620c73019239236e386fb15eac0eb7b


任一 SHA 不一致：

FAIL
STOP

禁止 Stage 8 从 trajectory、Pilot chunks 或物理事件重新生成该输入。


===============================================================================
4. Stage 8 合同冻结常量
===============================================================================

合同必须满足：

ContractName =
STAGE8_FORMAL_R_AUTOMATIC_DECISION

ContractVersion = 1.0

LockedBeforePilot = true

Rpre = 200

IndependentUnit = Repeat

InterferenceConditionOrdinals = 1..18

MethodOrdinals = [2,4]

MethodCodes = [T,TW]

Metrics：

P_cor
P_C_given_C
P_C_given_E
Mean_L_NC

VarianceUCBMultiplier =
1.1890464657347386

Z975 =
1.96

ProbabilityHalfWidthTarget =
0.020

MeanLncHalfWidthTarget =
0.25

MinimumTotalDenominator =
100

MinimumPositiveDenominatorRepeats =
30

IndividualPrecisionCellCount =
144

PairedPrecisionCellCount =
72

TotalPrecisionCellCount =
216

RformalCandidates =
[1000,2000]

NoOutcomeDirectionRule = true
NoManualOverride = true
NoConditionDropping = true
NoParameterChange = true

FormalUsesSingleRForAllConditionsAndMethods = true

任一常量漂移：

FAIL
STOP


===============================================================================
5. Stage 8 输入完整性
===============================================================================

stage8_decision_input.csv 必须恰好：

18 Conditions × 2 Methods × 200 Repeats
=
7200 rows

唯一键：

ConditionOrdinal
MethodOrdinal
RepeatID

键全集必须严格为：

ConditionOrdinal = 1..18

MethodOrdinal ∈ {2,4}

RepeatID = 1..200

Method 身份：

2 -> T
4 -> TW

禁止包含：

NI
FIRST
LAST
W
RepeatID > 200
重复键
缺失键


===============================================================================
6. 四核心指标采用 ratio-of-sums
===============================================================================

对每一个：

Condition–Method–Metric

固定：

R0 = 200

统一估计量：

theta_hat =
sum(A_r) / sum(B_r)

禁止：

mean(A_r / B_r)


四项定义如下。


1. P_cor

A_r = N_C

B_r = 200


2. P_C_given_C

A_r = N_CC

B_r = N_Cdot


3. P_C_given_E

A_r = N_EC

B_r = N_Edot

特别禁止：

N_CE


4. Mean_L_NC

A_r = Sum_L_NC_obs

B_r = N_run_NC


===============================================================================
7. Repeat 是唯一独立统计单元
===============================================================================

禁止把：

200 cycles

视为 200 个独立 Monte Carlo 样本。

唯一独立统计单元：

Repeat。

对于每一个 individual cell：

B_bar =
(1/R0) * sum(B_r)

定义：

psi_r =
(A_r - theta_hat * B_r) / B_bar

然后：

s_psi_sq =
sum(
    (psi_r - mean(psi))^2
) / (R0 - 1)

不得用 cycle-level Bernoulli 独立假设替代。


===============================================================================
8. Pilot 方差保守放大
===============================================================================

固定：

VARIANCE_UCB_MULTIPLIER =
1.1890464657347386

计算：

s_psi_upper_sq =
VARIANCE_UCB_MULTIPLIER
*
s_psi_sq

不得根据 Pilot 输出修改该系数。


===============================================================================
9. 精度目标
===============================================================================

固定：

Z975 = 1.96

ProjectedHalfWidth(R) =

Z975
*
sqrt(
    s_variance_upper_sq / R
)


对于三个概率指标：

P_cor
P_C_given_C
P_C_given_E

TargetHalfWidth =
0.020


对于：

Mean_L_NC

TargetHalfWidth =
0.25 cycle


===============================================================================
10. individual Required R
===============================================================================

对每个 individual precision cell：

RequiredR = ceil(

    Z975^2
    *
    s_psi_upper_sq
    /
    TargetHalfWidth^2

)

必须计算：

18 × 2 × 4
=
144

个 individual precision cells。


===============================================================================
11. individual 输出字段
===============================================================================

生成：

artifacts/stage8/stage8_individual_precision.csv

必须恰好：

144 rows

每行至少保存：

ConditionOrdinal
ConditionCode

MethodOrdinal
MethodCode

Metric

NumeratorField
DenominatorField

TargetHalfWidth

Defined
UndefinedReason

SumA
SumB

PositiveDenominatorRepeatCount
SupportPass

ThetaHat
BBar
PsiMean

SPsiSq
SPsiUpperSq

RequiredR

ProjectedHalfWidthR1000
ProjectedHalfWidthR2000

R1000Pass
R2000TargetPass
ExceedsR2000


===============================================================================
12. denominator support gate
===============================================================================

P_cor：

B_r = 200

固定存在，
不需要非固定分母支持门禁。


对于：

P_C_given_C
P_C_given_E
Mean_L_NC

必须满足：

TotalDenominator =
sum(B_r)
>=
100

且：

PositiveDenominatorRepeatCount =
count(B_r > 0)
>=
30


如果：

sum(B_r) = 0

则：

Defined = false

不得填 0。

任何 undefined：

R=1000 eligibility 失败。

任何 denominator support fail：

R=1000 eligibility 失败。


===============================================================================
13. T–TW 严格配对
===============================================================================

同：

Condition
Metric
RepeatID

严格配对：

T

与：

TW


individual influence：

psi_T_r

psi_TW_r


定义 paired influence：

psi_delta_r =
psi_TW_r - psi_T_r


计算：

s_delta_sq =
sample variance(
    psi_delta_r
)

ddof = 1


保守方差：

s_delta_upper_sq =

VARIANCE_UCB_MULTIPLIER
*
s_delta_sq


paired RequiredR：

RequiredR_paired = ceil(

    Z975^2
    *
    s_delta_upper_sq
    /
    TargetHalfWidth^2

)


===============================================================================
14. delta_hat 禁止参与样本量裁决
===============================================================================

允许保存：

delta_hat =
theta_TW - theta_T

用于审计。

但是以下内容不得进入 R 判断：

delta_hat 的正负号；

delta_hat 是否大于 0；

TW 是否优于 T；

改善比例；

p-value；

所谓“结果是否理想”。


Formal R 只能使用：

估计精度。


===============================================================================
15. paired precision cells
===============================================================================

必须计算：

18 Conditions × 4 Metrics
=
72

个 paired precision cells。

输出：

artifacts/stage8/stage8_paired_precision.csv

必须恰好：

72 rows

每行至少保存：

ConditionOrdinal
ConditionCode

Metric

TargetHalfWidth

Defined
UndefinedReason

ThetaT
ThetaTW
DeltaHat

SPsiDeltaSq
SPsiDeltaUpperSq

RequiredR

ProjectedHalfWidthR1000
ProjectedHalfWidthR2000

R1000Pass
R2000TargetPass
ExceedsR2000


===============================================================================
16. R=1000 ALL-rule
===============================================================================

只有以下十项全部 PASS：

Rformal = 1000


A.

Stage7PilotDataComplete


B.

R0Equals200


C.

Conditions1Through18Complete


D.

MethodsTAndTWComplete


E.

RepeatIDs1Through200UniqueAndComplete


F.

AllFourMetricsDefined


G.

AllNonfixedTotalDenominatorsAtLeast100


H.

AllNonfixedPositiveDenominatorRepeatCountsAtLeast30


I.

All144IndividualRequiredRValuesAtMost1000


J.

All72PairedRequiredRValuesAtMost1000


只有：

A AND B AND C AND D AND E
AND F AND G AND H AND I AND J

全部成立：

Rformal = 1000


其他任一失败：

Rformal = 2000


禁止人工 override。


===============================================================================
17. ProjectedPrecisionWarning
===============================================================================

如果任意：

individual cell

或：

paired cell

满足：

RequiredR > 2000

则：

ProjectedPrecisionWarning = true


但是：

Rformal 仍固定使用合同候选上限：

2000


禁止因为 warning：

改成 R=3000；

改成 R=5000；

删除 Condition；

修改指标；

修改参数；

修改精度阈值；

修改 Stage 8 合同。


===============================================================================
18. Stage 8 独立验收 oracle
===============================================================================

以下 oracle 已在 Stage 8 实现前独立冻结。

正确实现必须得到：

IndividualPrecisionCellCount =
144

PairedPrecisionCellCount =
72

TotalPrecisionCellCount =
216


IndividualFailR1000Count =
11

PairedFailR1000Count =
9


IndividualExceedR2000Count =
4

PairedExceedR2000Count =
3


MaxIndividualRequiredR =
41967

MaxPairedRequiredR =
42840


R1000Eligible =
false


FormalR =
2000


ProjectedPrecisionWarning =
true


任何一项不一致：

FAIL
STOP

禁止修改 oracle 来让程序通过。


===============================================================================
19. 两个关键 oracle cell
===============================================================================

至少必须验证：

H100
TW
P_C_given_E

RequiredR =
41967


以及：

H100
T–TW paired
P_C_given_E

RequiredR =
42840


这两项只承担实现验收功能。

不得根据它们修改任何科学条件。


===============================================================================
20. RequiredR > 2000 的预警规则
===============================================================================

Stage 8 应将所有：

RequiredR > 2000

的 cell 写入：

artifacts/stage8/stage8_precision_warning.json


必须包含：

ProjectedPrecisionWarning

IndividualExceedR2000Count

PairedExceedR2000Count

IndividualCells

PairedCells


正确结果必须为：

IndividualExceedR2000Count =
4

PairedExceedR2000Count =
3


===============================================================================
21. Stage 8 正式决策文件
===============================================================================

生成：

artifacts/stage8/stage8_r_decision.json


至少包含：

StageNumber
StageName

DecisionContractSHA256
DecisionInputSHA256

Rpre
IndependentUnit

IndividualPrecisionCellCount
PairedPrecisionCellCount
TotalPrecisionCellCount

R1000Checks

R1000Eligible

Rformal

FormalRSource

IndividualFailR1000Count
PairedFailR1000Count

IndividualExceedR2000Count
PairedExceedR2000Count

MaxIndividualRequiredR
MaxPairedRequiredR

ProjectedPrecisionWarning

ManualOverrideApplied
OutcomeDirectionUsed
ConditionDropped
ParameterChanged

FormalUsesSingleRForAllConditionsAndMethods

FormalExecuted
BootstrapExecuted
PaperResultGenerated


正确值：

Rformal = 2000

ProjectedPrecisionWarning = true

ManualOverrideApplied = false

OutcomeDirectionUsed = false

ConditionDropped = false

ParameterChanged = false

FormalExecuted = false

BootstrapExecuted = false

PaperResultGenerated = false


===============================================================================
22. Stage 8 实现文件边界
===============================================================================

优先新增：

src/sal_stability_stage1/stage8.py

scripts/build_stage8.py

tests/test_stage8_decision.py


允许只修改：

pyproject.toml
的版本号与描述；

README.md
的 Stage 8 状态说明。


禁止修改 Stage 1—7 冻结科学/执行文件。

Stage 8 开始和结束时：

必须重新计算冻结 SHA。


如果旧文件 SHA 改变：

FAIL
STOP


===============================================================================
23. Stage 8 测试策略
===============================================================================

Stage 7 已经独立验收 PASS。

Stage 8 不修改 Stage 1—7 冻结文件。

因此 Stage 8 不需要通过重新运行整个多进程 Pilot/preflight
来重新证明 Stage 7。

旧实现完整性通过：

已验收 Stage 7 baseline

+

frozen SHA byte identity

保证。


Stage 8 新测试必须至少覆盖：

1. Stage 7 baseline ZIP SHA；

2. Stage 7 status PASS；

3. Stage 7 无 R decision/Formal/Bootstrap；

4. contract SHA；

5. contract 所有冻结常量；

6. NoOutcomeDirectionRule；

7. NoManualOverride；

8. NoConditionDropping；

9. NoParameterChange；

10. P_C_given_E 使用 N_EC；

11. decision input 恰好 7200；

12. Condition 1..18 完整；

13. T/TW 完整；

14. RepeatID 1..200 完整；

15. 四核心指标定义；

16. Repeat-level influence；

17. 144 individual cells；

18. 72 paired cells；

19. all metrics defined；

20. denominator support；

21. individual fail R1000 = 11；

22. paired fail R1000 = 9；

23. individual RequiredR>2000 = 4；

24. paired RequiredR>2000 = 3；

25. FormalR = 2000；

26. ProjectedPrecisionWarning = true；

27. MaxIndividualRequiredR = 41967；

28. MaxPairedRequiredR = 42840；

29. no outcome direction；

30. no manual override；

31. no Formal；

32. no Bootstrap。


任一新测试失败：

FAIL
STOP


===============================================================================
24. Stage 8 artifacts
===============================================================================

至少生成：

artifacts/stage8/stage8_individual_precision.csv

artifacts/stage8/stage8_paired_precision.csv

artifacts/stage8/stage8_r_decision.json

artifacts/stage8/stage8_precision_warning.json

artifacts/stage8/test_report.txt

artifacts/stage8/changed_files.txt

artifacts/stage8/validation_report.md

artifacts/stage8/status.json


===============================================================================
25. status.json PASS 条件
===============================================================================

PASS 时至少必须：

stage = 8

stage_name =
FORMAL_R_AUTOMATIC_DECISION

status = PASS


baseline_stage7_zip_sha256_verified =
true

stage7_status_passed =
true

science_contract_changed =
false

frozen_hashes_passed =
true


stage8_contract_sha256_verified =
true

stage8_decision_input_sha256_verified =
true

stage8_decision_input_row_count =
7200


individual_precision_cell_count =
144

paired_precision_cell_count =
72

total_precision_cell_count =
216


individual_fail_r1000_count =
11

paired_fail_r1000_count =
9


individual_exceed_r2000_count =
4

paired_exceed_r2000_count =
3


all_metrics_defined =
true

all_denominator_support_passed =
true


r1000_eligible =
false

r_decision_executed =
true

formal_r =
2000

projected_precision_warning =
true


manual_override_applied =
false

outcome_direction_used =
false

condition_dropped =
false

parameter_changed =
false


formal_executed =
false

bootstrap_executed =
false

paper_result_generated =
false


blocking_issue_count =
0

stop_after_stage8 =
true


===============================================================================
26. Stage 8 PASS 硬门禁
===============================================================================

以下全部满足才允许：

Stage 8 PASS


1. Stage 7 ZIP SHA 正确；

2. Stage 7 status PASS；

3. Stage 1—7 frozen SHA 完全不变；

4. contract SHA 正确；

5. decision input SHA 正确；

6. 7200 行输入完整；

7. 144 individual cells；

8. 72 paired cells；

9. 216 total cells；

10. 所有指标可定义；

11. denominator support 全 PASS；

12. individual fail R1000 = 11；

13. paired fail R1000 = 9；

14. individual RequiredR>2000 = 4；

15. paired RequiredR>2000 = 3；

16. MaxIndividualRequiredR = 41967；

17. MaxPairedRequiredR = 42840；

18. R1000Eligible=false；

19. Rformal=2000；

20. ProjectedPrecisionWarning=true；

21. 无人工 override；

22. 未使用 outcome direction；

23. 未删除 Condition；

24. 未改参数；

25. 未运行 Formal；

26. 未运行 Bootstrap；

27. 未生成论文正式结果；

28. Stage 8 tests PASS；

29. blocking_issue_count=0；

30. stop_after_stage8=true。


任一失败：

FAIL
STOP


===============================================================================
27. Stage 8 完成后立即停止
===============================================================================

Stage 8 完成后：

立即 STOP。


不得：

顺手运行 Stage 9。


Stage 9 必须等待：

Stage 8 完整工程 ZIP

经 ChatGPT 独立验收 PASS。


Stage 9 后续只能读取：

FormalR = 2000

不得再次自行决定 R。


===============================================================================
28. 最终 ZIP
===============================================================================

生成：

sal_stability_stage8_20260825.zip


ZIP 必须包含：

完整源码；

全部 tests；

Stage 1—8 artifacts；

Stage 8 taskbook；

README；

pyproject.toml。


禁止包含：

.venv

__pycache__

.pytest_cache

*.pyc

*.tmp

外部 Stage 6 baseline ZIP

外部 Stage 7 baseline ZIP

无关文件。


计算最终 ZIP SHA256。


===============================================================================
29. Codex 最终回复格式
===============================================================================

Stage 8 status:
PASS / FAIL

Baseline Stage 7 ZIP SHA256:
...

Frozen science hashes:
PASS / FAIL

Stage 8 contract SHA256:
...

Stage 8 decision input SHA256:
...

Decision input rows:
7200

Individual precision cells:
144

Paired precision cells:
72

Individual fail R1000:
11

Paired fail R1000:
9

Individual RequiredR > 2000:
4

Paired RequiredR > 2000:
3

Max individual RequiredR:
41967

Max paired RequiredR:
42840

R1000 eligible:
NO

Formal R:
2000

ProjectedPrecisionWarning:
TRUE

Outcome direction used:
NO

Manual override:
NO

Formal executed:
NO

Bootstrap executed:
NO

Science contract changed:
NO

Stage 8 tests:
PASS

Blocking issues:
0 或具体问题

Final ZIP:
sal_stability_stage8_20260825.zip

Final ZIP SHA256:
...

然后立即 STOP。
```
