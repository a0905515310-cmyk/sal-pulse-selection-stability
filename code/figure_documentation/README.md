# Python 论文结果图

本次从已冻结的数值表重新绘制了七张结果图。没有获得论文原稿或旧图，因此图号暂定；不能据此声称完成了与旧图逐幅对应的校正。

直接查看 [全部图片 PDF](all_figures.pdf) 或 [图片总览](figure_overview.png)。科学边界、六项精度问题和可用于论文的英文图注见 [图注与审稿审查](captions_and_review.md)。

| 图 | 内容 | PDF | SVG |
|---|---|---|---|
| 1 | HPRF 单独干扰：五种方法、四个指标 | [PDF](figures/fig01_hprf.pdf) | [SVG](figures/fig01_hprf.svg) |
| 2 | F 单独干扰：全部四个延迟 | [PDF](figures/fig02_f_only.pdf) | [SVG](figures/fig02_f_only.svg) |
| 3 | 复合干扰：全部九个工况 | [PDF](figures/fig03_composite.pdf) | [SVG](figures/fig03_composite.svg) |
| 4 | TW−T：18 个工况的配对差值与区间 | [PDF](figures/fig04_tw_minus_t.pdf) | [SVG](figures/fig04_tw_minus_t.svg) |
| 5 | TW−W：18 个工况的配对差值与区间 | [PDF](figures/fig05_tw_minus_w.pdf) | [SVG](figures/fig05_tw_minus_w.svg) |
| 6 | 300 kHz 下真脉宽分层与支持量 | [PDF](figures/fig06_width_strata.pdf) | [SVG](figures/fig06_width_strata.svg) |
| S1 | 全部 216 项预设精度检查 | [PDF](figures/figS01_precision.pdf) | [SVG](figures/figS01_precision.svg) |

每图同时提供 600 dpi TIFF 和 300 dpi PNG。最终宽度统一为 183 mm，正文文字 5.2–7 pt，面板字母 8 pt；概率图显示百分比，概率差值显示百分点。游程长度图采用对数纵轴，森林图保留有符号的线性差值。微小水平显示错位仅用于区分重叠方法，连接线使用真实工况位置。

## 必须保留的科学边界

- TW 并非所有工况、所有指标都优越，F-only 中存在相对 T 的不利结果；相对 W 存在持续正确与错误后恢复之间的取舍。
- n=2,000 指独立仿真重复，每次包含 200 周期；不能把周期数当独立样本量。图中区间为冻结的整重复 Bootstrap 95% percentile 区间，B=2,000，未经多重比较校正。
- 非正确游程指标是观察窗口内的已观察长度，包括终端截断片段；不能标为无删失的平均恢复时间。
- H300 分层仅描述性；6/216 仅描述预设精度检查范围（T、TW、TW−T），不是全部结果的总体质量比例。

## 无干扰一致性

五种方法的 P(cor) 与 P(C|C) 均为 1；P(C|E) 因没有 E-origin 事件而未定义，游程平均长度因没有非正确游程而未定义。这里不以 0 代替未定义，也不补造区间。完整记录见 `source_data/ni_consistency.csv`。

## 复现

需要 Python 3.12 或兼容依赖版本的环境。在本目录运行：

```bash
python -m pip install -r requirements.txt
python plot_results.py
```

脚本仅读取随包提供的 `source_data`，先核验 SHA-256、行数、键唯一性、点估计与原始计数比率，再绘图。统计点估计与区间端点不会重新计算或替换。源数据数值文本保持原样，仅移除不参与绘图的工作文件溯源列；原始表的哈希记录在数据清单内。

`qa/verification.json` 记录可编辑 SVG 文本、PDF 嵌入字体、最终尺寸、TIFF 分辨率与画布边界检查。`qa` 中保留实际 PDF 的 Python 渲染预览和灰度预览，以便检查；这些不是新增实验结果。

## 排版依据

按 [Nature 官方图件规格](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) 设置可编辑文字、标准字体、坐标单位和简洁底色（核对日期：2026-10-02）。Nature 风格排版不等于已满足具体目标期刊的所有投稿要求；图号、图注与正文对应关系仍需结合论文核对。
