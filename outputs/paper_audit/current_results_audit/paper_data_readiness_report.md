# Paper Results Audit: Current Results

生成目录：`outputs/paper_audit/current_results_audit/`

审计 commit：`2df7ea4f04bff4493aa6160713eb7cf180d6852c`

## 审计范围

本次只扫描当前 repo 中已经存在的结果文件，重点路径包括：

- `outputs/evaluation/`
- `outputs/evaluation/**/*.csv`
- `outputs/evaluation/**/*.json`
- `outputs/logs/`
- `paper_assets/` 中已有的结果/清单文件

未读取或复制 `.venv`，未复制 checkpoint，未删除、移动或覆盖任何原始结果。

## 发现结果概况

当前工作区没有发现 `outputs/evaluation/` 或 `outputs/logs/` 下的已生成评估结果文件。唯一被识别到的结果相关文件是 `paper_assets/final_showcase_v1/file_manifest.csv`，它本身也是一个 manifest，并说明当前环境没有可冻结的 scripted_showcase 输出图或 summary CSV。

因此，本次 audit 生成的是**当前结果缺失状态的清单**，不是论文数值结果汇总。

## 当前已有数据能支持哪些论文图

以当前 repo 中实际存在的文件为准，暂时不能直接支持论文中的定量结果图。原因是没有可用的：

- 低层 4×4 skill/specialization diagnostics CSV；
- high-level selector vs fixed/random baseline 对比 CSV；
- sequential benchmark summary；
- low-level SAC 或 high-level PPO 训练曲线日志；
- scripted_showcase 的已选 3D/top-view 图像与 candidate summary。

## 低层 4×4 结果是否足够做专长矩阵

当前没有找到低层 4×4 diagnostics 数据，因此不能直接生成专长矩阵。建议本地补齐并保存 `eval_lowlevel_diagnostics` 或等价 4×4 matrix CSV 后，再用 `lowlevel_4x4_summary.csv` 做论文矩阵图。

## fixed single policy 在复杂场景下是否仍然过高

当前没有找到 fixed_pi1/fixed_pi2/fixed_pi3/fixed_pi4、random、heuristic/oracle、highlevel PPO 的 baseline summary，因此无法从现有文件判断 fixed single policy 是否仍然过高。

## high-level selector 是否有明显优势

当前没有 high-level selector 与 fixed/random baseline 的同口径结果文件，因此不能审计 high-level selector 是否有统计上或数值上明显优势。

## 哪些指标最能凸显优势

一旦补齐结果，最推荐优先比较以下指标：

1. `success_rate`：主效果指标；
2. `out_of_bounds_rate`：突出安全/边界控制优势；
3. `avg_switch_count` 与 `option_usage_by_regime`：突出高层策略是否真的在切换；
4. `avg_episode_lowlevel_steps` / `timeout_rate`：突出长时域稳定性；
5. `capture_rate`：区分逃逸失败类型。

## 是否建议新增 hard/stress complex benchmark

建议新增，但应在基础 sequential benchmark 与 baseline 结果齐全之后再做。当前没有已落盘结果，优先级应是：

1. 先补齐正式 sequential / composite / lowlevel 4×4 的可复现实验 CSV；
2. 再新增 hard/stress complex benchmark，用于证明 hierarchical selector 在更困难场景下优于固定单策略和 random；
3. hard/stress benchmark 应单独标注，不要混入主 benchmark。

## 当前最推荐优先做的 3 张图

在补齐数据后，建议优先产出：

1. **Low-level 4×4 specialization heatmap**：场景 × policy 的 success_rate 矩阵；
2. **High-level selector vs fixed/random baseline bar chart**：比较 success_rate、out_of_bounds_rate、avg_steps；
3. **Scripted showcase qualitative figure**：同一 rollout 的 3D 主图 + top-view 俯视图，用于展示连续轨迹与多策略切换。

## 生成文件

- `result_file_inventory.csv`
- `lowlevel_4x4_summary.csv`
- `highlevel_baseline_summary.csv`
- `training_curve_inventory.csv`
- `paper_data_readiness_report.md`
