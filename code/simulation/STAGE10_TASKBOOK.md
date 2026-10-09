# Stage 10：B=2000 整 Repeat 配对 Bootstrap 正式执行 Codex 任务书

> 状态：**正式封口版**  
> 日期：2026-08-25  
> 上一阶段：**Stage 9 PASS**  
> 本阶段职责：**只执行统计 Bootstrap，不重新运行任何物理模拟；完成后 STOP，不得进入 Stage 11。**

---

## 0. 唯一执行口令

只完成 **Stage 10：B=2000 整 Repeat 配对 Bootstrap**。

不得重新解释或修改 Stage 1—9 科学合同，不得重新执行 Formal，不得调整 R，不得删 Condition，不得新增 Formal 后对照，不得生成论文图或论文结论，不得执行 Stage 11。

Stage 10 PASS 只由输入身份、Bootstrap 统计合同、可复现性、数据完整性和输出完整性决定，**不得以 TW 是否优于任何方法作为 PASS 条件**。

---

## 1. 唯一输入基线与固定 SHA256

### 1.1 唯一 active baseline

文件：

`sal_stability_stage9_20260825.zip`

SHA256 必须严格等于：

`df1f4611d968e7911527592e32a7a24101c711e4306069f5dcaafebd745f2af5`

Stage 10 必须从该 ZIP 解压出的完整 Stage 9 PASS 工程开始。其他 Stage 9 ZIP、RAR、手工目录或旧版本一律禁止。

### 1.2 Stage 8 ZIP 仅作为历史 regression 依赖

为使历史 Stage 8/9 pytest 节点能够运行，允许在 Stage 10 工程同级放置：

`sal_stability_stage8_20260825.zip`

SHA256：

`a440cae9bbef5474bf30d8abc32078646bf76219910a6a37e31863065e233a3b`

注意：该 ZIP **不是 Stage 10 active baseline，也不是 Bootstrap 数据源**；唯一 active baseline 仍是 Stage 9 ZIP。它只能用于历史 regression test 依赖。

### 1.3 Stage 9 正式数据身份

`artifacts/stage9/formal_repeat_metrics.csv`

SHA256：

`e3084733adf9fa3f2325b0407b4358aeae118df8f6a152a4c1f5984757f3f8f6`

`artifacts/stage9/formal_aggregated_metrics.csv`

SHA256：

`803856c4d776b53b63e7e6f8be4425ce96ddb05c49aaca97c049134b04644865`

`artifacts/stage9/formal_h300_width_strata.csv`

SHA256：

`4e70f33ba195070a75e35fc6b3d14b7d5d7851b4ca414fcac7e8f24046a108b4`

`artifacts/stage9/stage10_bootstrap_contract_preformal.json`

SHA256：

`79fdae99716ea047d44171b332b2e2fc5bdeb311fa454b8092ba1f008c6b6b7a`

Stage 10 正式科学合同文件内容由本任务书冻结，canonical JSON SHA256：

`fdaa84d99e3b9e8fbcebe091c7a173a90ba5a42cef054f48c79eca01179656fd`

必须在执行前写入：

`artifacts/stage10/stage10_science_contract.json`

要求 UTF-8、`sort_keys=True`、`indent=2`、文件末尾单个换行；写入后 SHA 必须等于上述值。不得根据 Stage 9 Formal 结果修改合同。

---

## 2. Stage 10 冻结科学合同

### 2.1 Bootstrap 范围

固定：

- `B = 2000`
- `Namespace = BOOTSTRAP`
- 独立重采样单位：`Repeat`
- 每个 Condition Formal Repeat 总数：2000
- 每个 Bootstrap replicate 抽取 2000 个 RepeatID，**有放回**
- 只对 ConditionOrdinal `1...18` 做推断 Bootstrap
- NI（ConditionOrdinal=0）只做一致性核查，不做 inferential Bootstrap

五种方法：

1. FIRST
2. LAST
3. T
4. W
5. TW

四个核心指标：

1. `P_cor`
2. `P_C_given_C`
3. `P_C_given_E`
4. `Mean_L_NC`

Individual interval：五种 Method 全部生成。

配对差值只允许：

- Primary：`TW - T`
- Secondary ablation：`TW - W`

禁止新增 `TW-FIRST`、`TW-LAST`、`T-W` 或任何其他 Formal 后 contrast。

---

## 3. 重采样必须严格保持 Method 配对

对某个固定 Condition 和 BootstrapID=b，首先生成**唯一一组**长度 2000 的 RepeatID 序列：

`I_b = (r_1, ..., r_2000)`

随后 FIRST、LAST、T、W、TW 必须全部使用同一 `I_b`。

禁止为每个 Method 独立抽样。

因此 Method 绝对不得进入 Bootstrap RandomAddress。

例如同一 Bootstrap replicate 抽中 Repeat 12 两次，则该 Condition 下五种 Method 的 Repeat 12 数据都同步出现两次。

---

## 4. Bootstrap RNG 地址与无偏索引生成

必须继续使用已有：

- `RandomNamespace.BOOTSTRAP = 3`
- `VariableFamily.BOOTSTRAP_INDEX = 13`
- `SAL_STABILITY_MASTER_SEED_V1`
- Philox-4x32-10

对：

- `ConditionOrdinal = c`
- `BootstrapID = b`
- `DrawPosition = d`，其中 d=1...2000
- `Retry = q`，初始 q=0

固定地址：

```python
RandomAddress(
    namespace=RandomNamespace.BOOTSTRAP,
    condition_ordinal=c,
    repeat_id=b,
    variable_family=VariableFamily.BOOTSTRAP_INDEX,
    event_index=((d - 1) << 32) | q,
)
```

取：

```python
word = raw_words(address)[0]
```

固定：

```python
UINT32_SIZE = 2**32
R = 2000
ACCEPTANCE_LIMIT = (UINT32_SIZE // R) * R
# exact value:
ACCEPTANCE_LIMIT = 4294966000
```

若：

```python
word < 4294966000
```

则：

```python
RepeatID = (word % 2000) + 1
```

否则：

```python
q += 1
```

使用同一个 DrawPosition 的新 retry 地址重新生成，直到接受。

禁止简单使用 `floor(U*2000)+1`、`np.random.choice`、`default_rng`、Mersenne Twister 或 modulo without rejection。

此拒绝采样用于消除 2^32 不能被 2000 整除产生的有限状态映射偏差。

---

## 5. 向量化 Philox 允许，但必须证明与旧标量 RNG 完全等价

Stage 10 允许在新文件中实现 NumPy 向量化 Philox，以避免 7200 万个索引逐个执行 Python 标量调用。

但是：

- 不得修改 `rng.py`
- 不得改变 Philox 算法
- 不得改变 key 派生
- 不得改变 counter words
- 不得改变地址定义
- 向量化 `raw_words()[0]` 必须与冻结标量实现 bitwise 一致

必须包含 scalar-vs-vector hard tests。

固定 golden RepeatID：

```text
Condition=1, BootstrapID=1, DrawPosition 1..10
[1143, 1364, 408, 200, 1923, 1139, 1575, 944, 709, 1143]

Condition=1, BootstrapID=1000, DrawPosition 1..10
[35, 141, 1994, 838, 686, 1074, 1381, 1338, 405, 1308]

Condition=1, BootstrapID=2000, DrawPosition 1..10
[1302, 1139, 1444, 1613, 847, 1917, 1400, 1244, 1760, 1907]

Condition=18, BootstrapID=1, DrawPosition 1..10
[341, 245, 1697, 143, 943, 1662, 428, 1766, 403, 1615]

Condition=18, BootstrapID=2000, DrawPosition 1..10
[1623, 1043, 1810, 1183, 1335, 673, 1554, 1254, 1010, 785]
```

任何一项不一致：STOP。

---

## 6. Bootstrap audit 预注册

固定：

`BootstrapAuditIDs = {1, 1000, 2000}`

18 个干扰 Condition 全部参与。

必须保存：

`18 × 3 × 2000 = 108000`

个 1-based RepeatID 到：

`artifacts/stage10/bootstrap_audit_indices.npz`

建议 dtype=`uint16`。

必须在主 Bootstrap 完成后独立重新生成这些索引，并要求 primary 与 rerun 完全一致。

Audit 不用于选择“异常” Bootstrap replicate，不承担结果代表性，只用于随机索引可重构性审计。

---

## 7. 四个指标的 Bootstrap 计算公式

Bootstrap replicate b 的所有指标必须使用 **ratio of resampled sums**。

绝对禁止先计算 2000 个 Repeat 内部比例后再取均值。

### 7.1 P_cor

设重采样索引为 `r_j`：

`P_cor^(b) = sum_j N_C(r_j) / (2000 × 200)`

### 7.2 P_C_given_C

`P_C_given_C^(b) = sum_j N_CC(r_j) / sum_j N_Cdot(r_j)`

### 7.3 P_C_given_E

`P_C_given_E^(b) = sum_j N_EC(r_j) / sum_j N_Edot(r_j)`

注意分子必须是 `N_EC`，不得误用 `N_CE`。

### 7.4 Mean_L_NC

`Mean_L_NC^(b) = sum_j Sum_L_NC_obs(r_j) / sum_j N_run_NC(r_j)`

---

## 8. 零分母与 undefined 的唯一处理规则

若某个 individual bootstrap replicate 中：

- `sum N_Cdot = 0`
- 或 `sum N_Edot = 0`
- 或 `sum N_run_NC = 0`

则相应指标标记 `NaN/undefined`。

禁止：

- 填 0
- 填前一个值
- 重抽该 replicate
- 删除整个 Condition
- 删除该指标

必须统计：

- `BootstrapB`
- `DefinedBootstrapCount`
- `UndefinedBootstrapCount`
- `DefinedFraction`

Paired replicate：

`TW - comparator`

只有当**同一组 Repeat indices**下 TW 与 comparator 两个对应指标都 defined 时才 defined；否则 paired replicate 为 undefined。

若一个 inference cell 的 `DefinedBootstrapCount < 2`，CI 标记 undefined：

`INSUFFICIENT_DEFINED_BOOTSTRAP_REPLICATES`

该事实本身不自动导致 Stage 10 实现 FAIL；必须如实记录。

---

## 9. 95% CI 正式冻结

固定使用：

**percentile Bootstrap CI**

不得使用：

- Normal/Wald CI
- Studentized CI
- BCa
- Wilson
- Clopper–Pearson
- 任何 p-value 驱动区间

固定端点：

```python
lower = np.quantile(defined_values, 0.025, method="linear")
upper = np.quantile(defined_values, 0.975, method="linear")
```

点估计固定来自 Stage 9 全部 2000 Formal Repeat 的 ratio-of-sums。

**不得使用 bootstrap mean 替代正式点估计。**

Individual probability CI 必须位于 `[0,1]`。

Paired probability difference 必须位于 `[-1,1]`。

---

## 10. Paired contrast 方向固定

所有指标统一：

`Delta = TW - Comparator`

包括 `Mean_L_NC`。

因此：

- `TW - T`
- `TW - W`

对 `Mean_L_NC` 仍然保持原始减法方向。

禁止为了让“正数代表更优”而把 Mean_L_NC 改成 `T-TW` 或 `W-TW`。

Stage 10 不解释“正/负谁更好”；解释留到 Stage 11/论文结果分析。

---

## 11. Stage 8 精度合同的实际复核

Stage 8 原冻结目标：

概率指标：

`TargetHalfWidth = 0.020`

Mean_L_NC：

`TargetHalfWidth = 0.25`

Stage 10 对 percentile CI 定义实际最大单侧距离：

`ActualMaxOneSidedHalfWidth = max(PointEstimate - Lower, Upper - PointEstimate)`

只在 Stage 8 原 scope 上核查：

### Individual

- T
- TW

共：

`18 × 2 × 4 = 144`

### Paired

- TW-T

共：

`18 × 1 × 4 = 72`

因此：

`stage8_precision_target_audit.csv` 必须恰好 216 行。

FIRST/LAST/W individual 和 TW-W paired 不属于 Stage 8 R 决策保证范围。

TW-W 可以有 CI，但：

`Stage8PrecisionContractApplicable = false`

不得声称 R=2000 对 TW-W 保证 Stage 8 half-width target。

`ProjectedPrecisionWarning=true` 必须原样保留。

即使某些 actual CI 未达到 0.020 / 0.25：

**Stage 10 仍可 PASS。**

禁止因此：

- 把 R 改成 5000
- 重跑 Stage 9
- 删除 Condition
- 改精度目标
- 隐藏 CI

---

## 12. NI 的唯一处理

ConditionOrdinal=0：

不得执行 inferential bootstrap。

必须从 Stage 9 Formal 母表重新核查：

- 5 Methods × 2000 Repeat 完整
- `N_C=200`
- `N_E=0`
- `N_N=0`
- 五 Method RepeatMetrics 对同一 RepeatID 一致

形成：

`artifacts/stage10/ni_consistency_report.json`

NI 不进入 360 individual inference rows，也不进入 144 paired inference rows。

---

## 13. 执行方式冻结

固定：

```text
workers = 1
BootstrapB = 2000
BootstrapVectorizationBlockSize = 50
PersistentResumeUnit = Condition
```

`BlockSize=50` 只是内存/向量化计算分块，不是统计重采样单位。

无论内部用 25、50 或其他分块理论上结果都必须由 RandomAddress 唯一决定；正式执行固定 50，不再调参。

Persistent resume 单位固定为一个完整 Condition。

18 个 Condition 各自形成一个持久化目录。

如果某个 Condition manifest 或 distribution SHA 无效：

**整个 Condition Bootstrap 重算。**

禁止修补个别 replicate。

---

## 14. 推荐向量化计算方式

对每个 Condition：

1. 将 Stage 9 Formal RepeatMetrics 排成 `Method × Repeat` 数组。
2. 每次生成 50 个 BootstrapID 对应的 `50 × 2000` Repeat index matrix。
3. 五种 Method 共用此 index matrix。
4. 使用 NumPy gather + `sum(axis=...)` 计算 numerator/denominator sums。
5. 形成 5×4 individual replicate metrics。
6. 直接从同一 replicate 的 TW/T/W individual 值形成两组 paired differences。
7. 不需要构造或持久化 2000×2000 multinomial count matrix。

必须使用 int64 做统计计数求和。

---

## 15. Stage 10 持久化目录与文件

必须生成：

```text
artifacts/stage10/
    stage10_science_contract.json
    stage9_input_identity.json
    bootstrap_manifest.json
    status.json
    validation_report.md
    test_report.txt
    changed_files.txt

    ni_consistency_report.json

    bootstrap_individual_summary.csv
    bootstrap_paired_summary.csv
    stage8_precision_target_audit.csv

    bootstrap_distributions.npz
    bootstrap_audit_indices.npz

    bootstrap_conditions/
        condition_01_H100/
            distributions.npz
            manifest.json
        ...
        condition_18_HF500-3/
            distributions.npz
            manifest.json
```

恰好 18 个 condition distribution/manifest 对。

---

## 16. Distribution 固定形状

最终：

`bootstrap_distributions.npz`

必须至少包含：

```text
condition_ordinals : shape (18,)
method_ordinals    : shape (5,)
metric_codes       : shape (4,)
contrast_codes     : shape (2,)

individual_values  : float64 shape (18,5,4,2000)
individual_defined : bool    shape (18,5,4,2000)

paired_values      : float64 shape (18,2,4,2000)
paired_defined     : bool    shape (18,2,4,2000)
```

undefined value 位置必须为 NaN，且 defined mask=False。

禁止用 0 表示 undefined。

---

## 17. Individual summary 固定 360 行

`bootstrap_individual_summary.csv`

必须：

`18 × 5 × 4 = 360`

行。

至少包含：

- ConditionOrdinal
- ConditionCode
- MethodOrdinal
- MethodCode
- Metric
- FormalSumA
- FormalSumB
- FormalPositiveDenominatorRepeatCount
- PointEstimate
- PointDefined
- PointUndefinedReason
- BootstrapB
- DefinedBootstrapCount
- UndefinedBootstrapCount
- DefinedFraction
- CI95Defined
- CI95UndefinedReason
- CI95Lower
- CI95Upper
- CI95Width
- MaxOneSidedHalfWidth
- Stage8PrecisionContractApplicable
- Stage8TargetHalfWidth
- MeetsStage8PrecisionTarget

其中 Stage8PrecisionContractApplicable 只有 T/TW 为 true。

---

## 18. Paired summary 固定 144 行

`bootstrap_paired_summary.csv`

必须：

`18 × 2 × 4 = 144`

行。

至少包含：

- ConditionOrdinal
- ConditionCode
- ContrastOrdinal
- ContrastCode
- ComparatorMethod
- Metric
- PointEstimateTW
- PointEstimateComparator
- DeltaPoint
- BootstrapB
- DefinedBootstrapCount
- UndefinedBootstrapCount
- DefinedFraction
- CI95Defined
- CI95UndefinedReason
- CI95Lower
- CI95Upper
- CI95Width
- MaxOneSidedHalfWidth
- Stage8PrecisionContractApplicable
- Stage8TargetHalfWidth
- MeetsStage8PrecisionTarget

只有 `TW-T` 的 Stage8PrecisionContractApplicable=true。

`TW-W=false`。

---

## 19. 点估计交叉校验

Stage 10 必须重新从：

`formal_repeat_metrics.csv`

独立计算 95 个 Formal point estimates。

与：

`formal_aggregated_metrics.csv`

比较。

允许仅有正常 IEEE-754/CSV 十进制序列化的末位差异；不得出现科学数值差异。

Stage 10 individual summary 中 18×5×4 的 PointEstimate 必须来自此次独立重算。

Paired：

`DeltaPoint = PointEstimateTW - PointEstimateComparator`

必须直接成立。

---

## 20. Stage 1—9 文件冻结与 Stage 10 新文件范围

Stage 9 ZIP 是唯一 current baseline。

Stage 10 必须新增自己的 tree validator：

`validate_stage1_through_stage9_tree_unchanged()`

它直接将当前工程与 Stage 9 ZIP archive member hashes 比较。

只允许修改：

```text
README.md
pyproject.toml
```

只允许新增静态文件：

```text
STAGE10_TASKBOOK.md
scripts/build_stage10.py
src/sal_stability_stage1/bootstrap.py
src/sal_stability_stage1/bootstrap_storage.py
src/sal_stability_stage1/stage10.py
tests/test_stage10_contract.py
tests/test_stage10_rng.py
tests/test_stage10_metrics.py
tests/test_stage10_storage.py
tests/test_stage10_execution.py
```

以及：

`artifacts/stage10/**`

除此之外任何 Stage 1—9 文件新增/修改/删除：

**FAIL + STOP。**

尤其禁止修改：

- rng.py
- metrics.py
- stage8.py
- stage9.py
- formal_storage.py
- 所有旧 tests

---

## 21. 历史 pytest forward-compatibility 例外

Stage 10 增加新文件后，两个历史阶段封闭边界节点预期不具备 forward compatibility。

只允许以下两个节点作为 legacy boundary exception：

### Exception A

```text
tests/test_stage8_decision.py::test_stage1_through_stage7_byte_identity_and_frozen_science_gate
```

记录代码：

`LEGACY_STAGE_BOUNDARY_NOT_FORWARD_COMPATIBLE`

### Exception B

```text
tests/test_stage9_contract.py::test_stage8_identity_and_decision_boundary_are_independently_verified
```

记录代码：

`LEGACY_STAGE9_BOUNDARY_NOT_FORWARD_COMPATIBLE`

严格要求：

1. 先保存完整 pytest 原始输出。
2. full pytest 失败节点集合必须**严格等于上述两个节点**。
3. 不允许第三个失败或 error。
4. Exception A 的失败内容只能由已批准 Stage 9 + Stage 10 后续新增文件造成，不得有冻结文件修改/删除。
5. Exception B 的失败内容只能由任务书批准 Stage 10 新增文件造成。
6. Stage 10 自己的 Stage9-ZIP tree identity gate 必须 PASS。
7. Stage 8 ZIP SHA、Stage 9 ZIP SHA、Stage 1—9 frozen science/hash 必须独立 PASS。
8. 保留 raw failures；不得修改、skip、xfail 或删除旧测试。
9. 再使用 `--deselect` 精确排除上述两个节点运行 remaining suite；必须 0 failed、0 errors。

若历史节点失败原因不是严格的 forward-stage additions-only：

**不得套用例外，直接 FAIL。**

---

## 22. Stage 10 必须新增的测试

### test_stage10_contract.py

至少检查：

- Stage9 ZIP SHA
- Formal RepeatMetrics SHA
- preformal contract SHA
- Stage10 science contract SHA
- B=2000
- conditions 1..18
- NI no bootstrap
- contrasts 仅 TW-T/TW-W
- percentile CI
- no p-value
- no physical simulation
- Stage8 precision scope

### test_stage10_rng.py

至少检查：

- scalar/vector Philox raw word exact
- 本任务书五组 golden first-10 RepeatIDs
- Method 不进入地址
- address uniqueness
- rejection sampling mapping
- audit IDs 1/1000/2000
- audit primary/rerun exact

### test_stage10_metrics.py

至少检查：

- ratio-of-sums 而非 mean-of-ratios
- P_C_given_E 分子 N_EC
- zero denominator -> undefined/NaN
- paired undefined propagation
- TW-minus-comparator direction对 Mean_L_NC 不翻转
- quantile method=linear
- formal point estimate reconstitution

### test_stage10_storage.py

至少检查：

- 18 condition manifests
- per-condition distribution SHA
- resume valid condition skip
- invalid condition whole recompute
- combined NPZ shapes
- 360/144/216 row counts
- NaN 与 defined mask 一致

### test_stage10_execution.py

至少检查：

- 一个小 Bootstrap 子集的 serial deterministic replay
- vectorization block partition invariance
- same bootstrap indices for all methods
- NI never enters inferential engine
- no physics/event/kernel execution path called
- Stage11 not executed

---

## 23. Stage 10 正式执行顺序

严格执行：

```text
STEP 1  Verify Stage9 ZIP SHA and extract unique baseline
STEP 2  Verify Stage9 status PASS and all frozen input SHAs
STEP 3  Write/verify Stage10 science contract exact SHA
STEP 4  Verify Stage10 preformal contract unchanged
STEP 5  Verify Stage1-9 tree identity against Stage9 ZIP
STEP 6  Run full pytest and audit exactly two authorized legacy boundary exceptions
STEP 7  Run remaining pytest with only those two nodes deselected -> zero failures
STEP 8  Run Bootstrap RNG scalar/vector preflight + golden vectors
STEP 9  Recompute Stage9 Formal point estimates and NI consistency
STEP 10 Execute 18 interference Conditions, B=2000, condition-level resume
STEP 11 Reload all 18 condition distributions and validate identities
STEP 12 Independently regenerate audit Bootstrap IDs 1/1000/2000
STEP 13 Build combined bootstrap_distributions.npz
STEP 14 Build 360-row individual summary
STEP 15 Build 144-row paired summary
STEP 16 Build 216-row Stage8 precision target audit
STEP 17 Run post-Bootstrap regression and all hard gates
STEP 18 Write manifest/status/validation report
STEP 19 Package complete Stage10 ZIP
STEP 20 STOP
```

---

## 24. Stage 10 PASS 硬门禁

必须全部满足：

1. Stage9 ZIP SHA 正确。
2. Stage9 status=PASS。
3. Formal RepeatMetrics SHA 正确。
4. Stage10 preformal contract SHA 正确。
5. Stage10 science contract SHA 正确。
6. Stage1—9 current baseline 无越界修改/删除。
7. full pytest 仅允许两个已批准 legacy boundary failures。
8. remaining suite 0 failed / 0 errors。
9. Stage10 专属 tests 全 PASS。
10. `BootstrapNamespace=BOOTSTRAP`。
11. `B=2000`。
12. Condition 1..18 全部执行。
13. NI 未执行 inferential bootstrap。
14. 每 Bootstrap replicate 2000 draws with replacement。
15. 五 Method 同一 replicate 使用完全相同 Repeat indices。
16. Method 不进入 RandomAddress。
17. vectorized Philox 与 scalar exact。
18. 108000 audit indices primary/rerun exact。
19. metric estimator 为 ratio-of-resampled-sums。
20. undefined 未填 0、未重抽。
21. individual summary=360 行且唯一。
22. paired summary=144 行且唯一。
23. Stage8 precision audit=216 行且唯一。
24. individual distributions shape=(18,5,4,2000)。
25. paired distributions shape=(18,2,4,2000)。
26. probability individual defined values ∈[0,1]。
27. probability paired defined differences ∈[-1,1]。
28. Mean_L_NC individual defined values ≥0。
29. Formal point estimates可由 Formal母表重构。
30. DeltaPoint=TW point−Comparator point。
31. CI 为 percentile 2.5%/97.5%, method=linear。
32. Stage8 precision target未达到不会触发重跑/删 Condition。
33. `ProjectedPrecisionWarning=true` 保留。
34. `PValueGenerated=false`。
35. `ConditionDropped=false`。
36. `ParameterChanged=false`。
37. `PostFormalContrastAdded=false`。
38. `OutcomeDirectionUsedForPass=false`。
39. `PhysicalSimulationExecuted=false`。
40. `FormalRerunExecuted=false`。
41. `PaperFigureGenerated=false`。
42. `PaperResultConclusionGenerated=false`。
43. `Stage11Executed=false`。
44. `StopAfterStage10=true`。

任何实现/身份/完整性门禁失败：Stage10 FAIL + STOP。

**不得以 CI 是否跨 0、TW 是否更优、某个 half-width 是否满足 Stage8 target 作为 Stage10 PASS/FAIL 条件。**

---

## 25. status.json 必须至少包含

```text
stage = 10
stage_name = BOOTSTRAP_B2000_PAIRED_REPEAT
status = PASS/FAIL

stage9_baseline_zip_sha256_verified
formal_repeat_metrics_sha256_verified
preformal_contract_sha256_verified
stage10_science_contract_sha256_verified
stage1_through_stage9_tree_unchanged

bootstrap_executed
bootstrap_namespace
bootstrap_b
bootstrap_independent_unit
bootstrap_draws_per_replicate
bootstrap_condition_count

individual_summary_row_count
paired_summary_row_count
stage8_precision_audit_row_count

bootstrap_audit_ids
bootstrap_audit_index_count
bootstrap_audit_exact_reproducibility_passed

ratio_of_sums_verified
paired_indices_shared_verified
method_absent_from_random_address
vectorized_philox_scalar_exact

projected_precision_warning
p_value_generated
condition_dropped
parameter_changed
post_formal_contrast_added
outcome_direction_used_for_pass
physical_simulation_executed
formal_rerun_executed
paper_figure_generated
paper_result_conclusion_generated
stage11_executed

legacy_boundary_exceptions
blocking_issue_count
stop_after_stage10
```

---

## 26. 最终 ZIP

成功后必须自动生成：

`sal_stability_stage10_20260825.zip`

ZIP 必须包含完整 Stage 1—10 工程与 `artifacts/stage10/`。

打包前、打包后都必须验证：

- 必需文件存在
- 18 condition distributions/manifests 完整
- summary row counts 正确
- artifact SHA 与 manifest 一致
- Stage 11 产物不存在

自动计算并返回 Stage10 ZIP SHA256。

不得让用户手工再次 RAR/ZIP 覆盖自动生成包。

---

## 27. STOP 条件

Stage 10 自动 ZIP 成功后立即 STOP。

不得：

- 开始 Stage 11
- 画论文最终图
- 写第 4 章正式数值结论
- 根据 CI 方向修改参数/场景/方法
- 重新执行 Stage 9 Formal
- 追加新的 contrast

只返回 Stage 10 完整 ZIP、SHA256、PASS/FAIL 状态和必要执行摘要，交由 ChatGPT 独立验收。

---

## 28. Codex 最终执行原则

如果任务书与旧 Stage 文件发生冲突：

1. 不得自行修改 Stage 1—9 冻结文件。
2. 先判断是否属于本任务书明确批准的两个历史 forward-compatibility boundary exception。
3. 不是上述限定例外则 FAIL + STOP。
4. 不得为了得到理想统计结果而调整任何科学合同。
5. 结果方向永远不属于实现 PASS 门禁。
