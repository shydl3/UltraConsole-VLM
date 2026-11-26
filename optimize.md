# UltraConsole-VLM 处理流水线优化指导书

> 版本：v1.1（2026-11-26）
> 定位：给后续 agent 的主施工指导书。本文维护性能路线 P0–P6 的现状、红线、优化路线与验收标准；`dev-guide.md` 只保留架构、设计契约、代码地图与开发约定，性能细节以本文为准。
> 范围：`Rules/src/ultra_rules/vlm`、`backend/app`、`frontend` 中与 VLM 分析流水线相关的部分。

---

## 0. 当前状态快照（2026-11-26）

- **性能 P0 ✅**：tracking + cues + integrity 合并为 `detect`，只解码一次；30min 真实任务 193.6s，9038 行 observations / cues / integrity 逐项一致。
- **性能 P1 ✅**：integrity 多区共享解码；30min cues+integrity 150s（旧分开约 4 分钟）。
- **性能 P2 ✅**：确定性证据默认 lazy，只写 manifest；后端首次访问按需生成预览。165 条结果 understanding 0.41s、0 预览；公共判定与旧结果一致。
- **性能 P3 🔶**：tracking I/O/缓存优化完成（batch insert、no-fsync 临时库、延迟索引、gzip level1）；tracking 100.8s → 89.3s（-11.4%），完整 detect 193.6s → 179.7s（-7.2%），9038 帧零漂移。TensorRT/ONNX/FP16/NVDEC 待独立评测。
- **下一步**：P4（队列、ETA、worker 分离）或 P3 剩余模型/解码评测。P5/P6 仍待做。
- **验证基线**：Python **223 passed / 1 skipped**；ruff 全绿；前端 46 用例、eslint、build 通过。
- **注意**：`optimize.md` 的 P0–P6 是性能编号，与 `dev-guide.md` 第 3 节的准确率 P0/P1/P2 无关。

---

## 1. 使用方式

1. 先读本文第 2、3 节，确认当前基线与不可改变的红线。
2. 从第 5 节按 P0 → P6 选择任务；一次只做一个 P 项，保持 diff 小、可回滚。
3. 每个任务必须同时交付：代码、测试、实测对比、风险说明。
4. 完成后更新第 10 节“进度看板”，不要只改代码不记录。
5. 不要为了“看起来更快”而改变判定语义；性能优化的前提是结果可复现、可对照。

---

## 2. 当前基线（实测）

### 2.1 真实长视频基线

样本：`Rules/data/vlm/runs/vlm-87b81cfb4aba4171a1c483992ef9dc7a`

- 视频：191.2 分钟，1080p，25fps
- 规则：1 条 `forced_passage` + 3 个完整性区
- 支撑候选：769 条；信号 episode：11 个；完整性结果：35 条
- cue 判定：4 `match` / 7 `no_match`

| 阶段 | 修复后实际耗时 | 说明 |
|---|---:|---|
| tracking | 约 11.4 分钟 | poseFps=5，682s |
| cues | 约 7 分钟 | 1 个信号区，全片扫描 |
| integrity | 约 21 分钟 | 3 个完整性区，当前每区各自全片解码 |
| understanding | 约 6.8 秒 | 11 个 cue episode，确定性判定 |
| 合计 | 约 39 分钟 | 约为视频时长的 **0.21×** |
| 不含 integrity | 约 18 分钟 | 约 **0.10×** |

> 批量任务 UI 上看到的 5–6 小时，很大一部分是单 GPU worker 串行排队，不是单任务计算时间。`startedAt` 在入队时写入，当前 UI 的“处理时长”包含排队等待。
>
> **P0 完成后**：30 分钟真实任务的完整 `detect`（tracking+cues+integrity）实测 **193.6 秒**，旧的分开阶段约 6 分钟；同 run 缓存重放 **1.5 秒**。191 分钟样本尚未重新跑完整 benchmark，上表是修复前的基线。

### 2.2 本轮已完成的修复（作为新基线）

| 修复 | 文件 | 效果 |
|---|---|---|
| 证据抽帧 seek 对齐 | `vlm/sampling.py` | 单样本约 5.5 分钟 → 约 0.7 秒 |
| 无目标 cue episode 不抽帧 | `vlm/qwen_worker.py` | `no_subject` 只留审计行 |
| cue-off 确定性候选不抽帧 | `vlm/qwen_worker.py` | 关门 `no_match` 不再解码 |
| 同采样率信号区共享解码 | `vlm/cues.py` | 默认 10fps 多信号区只解一遍 |
| 目标人查找建索引 | `vlm/decision.py` | 每个 episode 二分窗口查候选 |
| cues + integrity 共享解码 | `vlm/combined.py`、`runner.py` | 新增 `detect` 阶段；30min 真实任务 150s，输出与旧阶段完全一致 |
| tracking 并入共享解码（P0b） | `vlm/combined.py`、`vlm/observations.py` | 30min 完整 `detect` 193.6s；9038 行 observations、cues、integrity 完全一致；5min 任务 1500 行 observations + 165 candidates 完全一致 |
| lazy 证据预览（P2） | `vlm/sampling.py`、`vlm/qwen_worker.py`、`backend/app/vlm_api.py` | 默认只写 manifest；165 条结果 understanding 0.41s，0 预览；首次访问按需生成 |
| tracking I/O/缓存（P3） | `vlm/observations.py`、`vlm/combined.py` | batch insert + no-fsync 临时库 + 延迟索引 + gzip level1；tracking 100.8s→89.3s，detect 193.6s→179.7s，9038 帧零漂移 |
| 真实复算 | 87b8 任务 | understanding 63 分钟 → 6.8 秒，11 条结果逐条一致 |

### 2.3 当前主要瓶颈（按影响排序）

1. **全片解码已收敛为一次**：tracking + cues + integrity 现在共享 `detect` 阶段的同一次 `decoded()`。30 分钟真实任务 P0 时 193.6s，P3 I/O 后 **179.7s**；剩余成本主要是 pose 推理和 integrity 逐样本计算，而非重复解码。
2. **integrity 阶段（已缓解，仍有空间）**：多区共享解码后，30 分钟真实任务 cues+integrity 约 150 秒；`sampleFps` 与 `_refine_onset` 的进一步优化仍在 P1 内。
3. **证据预览已 lazy 化**：确定性结果只写 manifest，预览由后端首次访问时生成并缓存；VLM/dryRun 仍 eager。密集 episode 的 understanding 已不再随 JPEG 数线性增长。
4. **单 worker 串行**：`backend/app/vlm_api.py` 的 `MAX_CONCURRENT = 1`；批量任务按提交顺序排队，缺少优先级、ETA、排队位置。
5. **存储与可观测性**：`observations.sqlite` 4 小时约 271MB；run 目录、预览、raw 输出缺少统一保留策略；phase 耗时、episode 数、`no_subject` 比例等指标未进入 UI/API。

---

## 3. 红线（任何优化都不得违反）

1. **保留信号 episode 触发**：`forced_passage` 仍然“一个去重后的信号 episode 判一次”，不得退回“每个计数候选都判”。
2. **不得改 3 秒 episode 去重语义**：锚点是 episode 首次触发，不是滚动窗口；`cueDedupeSec` 默认 3.0。
3. **不得改通道配对**：入口线 + 可选计数区 + 信号区按 `pairIndex` 一一配对；同号入口线唯一。
4. **不得改退回判定**：入口线反向跨线为主，计数区 `zone.leave` 仅辅助；退回窗口从目标进入时刻起算 5 秒。
5. **确定性优先**：`forced_passage` 默认不调用 VLM；VLM 只用于 `uncertain`/显式开启的复核。
6. **不改 `poi.py` 的候选生成语义**；需要改候选时先写评测与回归。
7. **不改结果审计字段**：`resultId`、`cueEvidence`、`retreatSec`、`rawTriggerCount`、`window` 等保持可追溯。
8. **不引入未经验证的质量降级**：降低分辨率、降低 poseFps、跳过帧等必须用评测集证明不降准确率。
9. **默认保守**：新的并发/批处理能力用参数或环境变量开启，默认行为保持可对照。

---

## 4. 代码地图（优化时优先看这些位置）

| 路径 | 职责 |
|---|---|
| `Rules/src/ultra_rules/vlm/runner.py` | 阶段编排、worker lock、超时/取消 |
| `Rules/src/ultra_rules/vlm/media.py` | `decoded()` / `sampled()`，所有全片解码的入口 |
| `Rules/src/ultra_rules/vlm/observations.py` | tracking、观测缓存、支撑候选持久化 |
| `Rules/src/ultra_rules/vlm/cues.py` | 信号区检测、`detect_all()`、`calibrate()` |
| `Rules/src/ultra_rules/vlm/integrity.py` | 完整性检测，当前每区一次 `sampled()` |
| `Rules/src/ultra_rules/vlm/sampling.py` | 证据窗口、`source_images()`、contact sheet |
| `Rules/src/ultra_rules/vlm/decision.py` | 目标选择、退回、episode 合并、目标索引 |
| `Rules/src/ultra_rules/vlm/qwen_worker.py` | understanding 主循环、`_process_candidate` / `_process_cue_candidate` |
| `backend/app/vlm_api.py` | 上传、队列、`MAX_CONCURRENT`、run 状态 |
| `backend/app/vlm_jobs.py` | 启动/取消/状态、worker 存活判断 |
| `Rules/scripts/bench_tracking.py` | tracking 性能与一致性回归 |
| `Rules/tests/`、`backend/tests/` | 现有回归基线 |

---

## 5. 优化路线（P0 → P6）

### P0：单次解码、多消费者（最高收益）

> 状态：✅ **已完成**。P0a：cues + integrity 合并为 `detect` 阶段；P0b：tracking 也并入同一 `decoded()` 循环。cues+integrity 同时存在时，runner 只跑 `segment → detect → understanding`。

**目标**：同一视频段只解码一次，按各自采样率分发给 tracking / cues / integrity，消灭“每个阶段、每个区各解一遍”。

**现状**：

- `observations.track()` → `media.sampled(source, meta, poseFps)`
- `cues.detect_all()` → 每个不同 `cue.sampleFps` 一遍
- `integrity.detect_timeline()` → **每个完整性区一遍**
- `sampling.source_images()` → 每个证据窗口再 seek

**建议实现**：

1. 在 `media.py` 增加 `FrameBroker` / `SampledFanout`：
   - 只调用一次 `decoded()`；
   - 每个消费者维护自己的 `next_target`，用与 `sampled()` 完全一致的公式推进：`target = (floor(t*fps + 1e-6) + 1) / fps`；
   - 同一帧只做一次 `to_ndarray("bgr24")`，多个消费者共享。
2. ✅ **已做（P1）**：cues + integrity 在 `vlm/combined.py` 中共享一次解码，integrity 多区不再各自全片解码；缓存指纹复用 `cues.cache_key()` / `integrity.cache_key()`。
3. ✅ **已做（P0b）**：tracking 并入同一 `decoded()` 循环，`combined.analyze_run()` 负责：
   - observations 缓存命中且当前 run 无 `observations.sqlite`：`track()` 重放缓存（不再解码），再 `detect_run()` 做 cues+integrity；
   - observations 缓存命中且当前 run 已有完整 `observations.sqlite`：直接复用，不重建表；
   - 缓存未命中：在同一个循环里按 `SampleSchedule` 分发给 pose / cue / integrity；
   - integrity 的 persons 用最近 pose 结果（±1s 最近邻）替代 `_persons_near()` 回查 sqlite；
   - runner 阶段改为 `segment → detect → understanding`，不再单独跑 tracking。
4. ✅ 保留分阶段缓存与断点续跑；`cache_fingerprint()` / `tracker_config_for()` 由 tracking 与合并阶段共用，避免指纹漂移。

**验收**：

- ✅ `detect` 阶段在 cues+integrity 场景只解码一次（`Rules/tests/test_combined.py`）；
- ✅ cues `intervals`、integrity `intervals` 与旧阶段完全一致（30 分钟真实任务逐项相等）；
- ✅ tracking 并入后，30 分钟真实任务 **9038 行 observations 逐字段完全一致**；5 分钟真实任务 **1500 行 observations + 165 条 candidates 完全一致**；
- ✅ 30 分钟任务完整 `detect`（tracking+cues+integrity）**193.6 秒**，旧的分开阶段约 6 分钟；同 run 缓存重放 **1.5 秒**；
- ✅ 全量测试与 ruff 通过。

**风险**：阶段耦合、采样点漂移、内存背压、缓存失效逻辑、integrity 遮挡排除的 persons 时序。已用真实任务逐项对照；后续改动仍需保持等价性测试。

---

### P1：integrity 专项优化（先做，低风险）

> 状态：✅ **已完成第一版**。integrity 多区与 cues 共享一次解码；30 分钟真实任务 cues+integrity 约 150 秒，旧阶段约 4 分钟，输出完全一致。进一步的 `sampleFps` / `_refine_onset` 优化仍可选做。

**目标**：不改变完整性判定，先把最贵的阶段降下来。

**建议实现**：

1. ✅ 多区共享一次解码：`vlm/combined.py` 用 `SampleSchedule` 分发给多个 `_ZoneRuntime`，每帧只转换一次 BGR；cues 与 integrity 共用这次解码。
2. ⬜ 评估把默认 `sampleFps` 从 0.5 降到 0.2–0.3 是否影响确认区间；只作为可配置默认，不强制。
3. ⬜ `_refine_onset()` 批量处理：同一视频只打开一次容器，按 interval 顺序 seek/解码，避免每个 interval 重新 `av.open()`。
4. ✅ 没有完整性区时直接跳过；不要为“可能以后要用”保留空扫描。

**验收**：

- 多区测试视频的 `integrity.json` 与旧实现一致；
- 解码次数从“区数”降为“1 次/采样率组”；
- 191 分钟 3 区样本耗时明显下降（目标 < 10 分钟）；
- 有缓存版本号与回归测试。

---

### P2：证据预览延迟生成 / 按需生成

> 状态：✅ **已完成**。默认 `previewMode="lazy"`：确定性证据只写 sample manifest，不抽帧、不拼图；后端 artifact 端点首次访问时 `render_preview()` 生成 JPEG 并缓存。VLM / dryRun 仍 `render=True`。

**目标**：把证据成本从“每次分析必付”变成“人工复核才付”。

**现状**：

- `_build_deterministic_evidence()` 对 `match` / `no_match`（有目标）同步 `build_sample()`；
- 单样本约 0.7 秒，但信号密集时是 O(episode)；
- 已对 `no_subject`、cue-off 短路。

**建议实现**：

1. ✅ 增加 `previewMode`（`eager` / `lazy`，默认 `lazy`）：
   - `lazy`：`build_sample(render=False)` 只写 manifest（window、sourceFrameIndices、sampleTimesSeconds、observations、quality、poi、width/height）；
   - 后端 `/artifacts/previews/{sample}.jpg` 在文件缺失时按 manifest 调用 `render_preview()`，用全局锁串行生成并缓存；
   - 前端沿用现有 `previewPath`，文件按需生成；缺失时返回 404，结果与源视频仍可用。
2. ⬜ 对连续 episode 的窗口按时间排序，一次解码批量裁图；避免每个 sample 重新 `av.open()`（后续优化）。
3. ✅ 只对 `match` / `uncertain`（以及有目标的确定性结果）保留预览；`no_subject`、cue-off 审计行不生成证据。

**验收**：

- ✅ 结果 JSON 字段兼容；无预览的结果前端不报错（`previewPath` 保持不变，旧 eager 文件也可直接服务）；
- ✅ 懒加载端点有测试（首次生成、缓存命中、缺失 manifest 404）；
- ✅ 密集 episode 场景的 understanding 不再随 JPEG 数线性增长：165 条确定性结果的真实 run 复算 **0.41 秒**，生成 0 个预览、23 个 manifest（其余为 cue-off 短路）；
- ✅ lazy 与 eager 的 manifest 一致；单目标样本的预览字节一致。多人叠加时绘制顺序可能不同，视觉等价。

---

### P3：tracking 侧优化

> 状态：🔶 **I/O / 缓存侧已完成**（2026-11-26）。tracking 100.8s → **89.3s（-11.4%）**，完整 detect 193.6s → **179.7s（-7.2%）**，逐帧零漂移、观测逐字段一致。TensorRT / ONNX / FP16 / NVDEC 仍待独立环境评估。

**目标**：在不改变 track 语义的前提下降低 tracking 时间。

**本轮已完成**：

- `observations.open_observations_db()`：`journal_mode=MEMORY` + `synchronous=OFF`；observations.sqlite 是可重建的临时产物，gzip cache 才是完成标记。
- 延迟建索引：先批量插入，再 `CREATE INDEX`，避免每行维护索引。
- 批量写入：`executemany` 每 200 行 + commit，progress 每 50 行，减少 fsync/JSON 写盘。
- 单次序列化但保持语义：cache 写 `PoiTracker.update()` 之前的行，sqlite 写更新后的行（`trackSegmentId/identityAmbiguous` 不能丢）。
- gzip cache `compresslevel=1`：牺牲部分压缩率换写入/读取速度；旧 cache 仍可读。
- `combined._track_and_detect()` 同步使用上述路径。

**候选手段（未做）**：

1. `poseFps` 5 vs 10 的业务确认；固定 5fps 或按场景自适应。
2. 启用并验证 `ULTRA_VLM_POSE_BATCH` + `ULTRA_VLM_PIPELINE`；必须用 `bench_tracking.py` 做逐帧一致性回归（TF32 批量曾导致 ID 漂移，已修）。
3. 评估 TensorRT / ONNX / FP16；需要独立环境与精度回归。
4. 评估 NVDEC 硬解；当前分析链路主要是 PyAV 软解。
5. ROI 裁剪 / 运动区域优先是更大改动，可能影响 track 连续性，暂列观察项。

**验收**：

- ✅ `bench_tracking.py` 的 trackId/bbox/keypoints 与基线一致（9038 帧：0 mismatch，bbox/kp delta 0.0）；
- ✅ observations.sqlite 与旧实现逐字段一致（9038 行）；
- ✅ 实测加速比记录在案：tracking -11.4%，完整 detect -7.2%；
- ✅ 默认配置不变（batch/pipeline 仍由环境开关控制，poseFps 未改）；
- ⬜ TensorRT / ONNX / FP16 / NVDEC 的独立评测。

---

### P4：调度与吞吐

> 状态：🔶 **部分完成（2026-11-26）**：`startedRunningAt` 已作为未来 run 的时长起点，`durationSec` 不再包含排队时间，UI 也不再展示排队时长；`queuedAt` 仅留审计。runner 记录 `phaseSeconds` 并展示阶段耗时。**待做**：ETA/队列位置、优先级、worker 分离、多 GPU 分片。

**目标**：让多路摄像头/批量任务不再被单 worker 串行拖死。

**现状**：

- `backend/app/vlm_api.py`：`MAX_CONCURRENT = 1`；
- `_pump()` 按 `startedAt` 顺序启动；
- `startedAt` 仍是入队时钟；新字段 `startedRunningAt` 是实际开始，未来 run 的 `durationSec` 从它计算；`queuedAt` 仅审计，历史 run 回退原 `startedAt` 语义。

**建议实现**：

1. ✅ 分离 `queuedAt` / `startedRunningAt` / `finishedAt`；未来 run 的时长只从实际开始计算，排队不展示。⬜ 排队位置与 ETA 仍待做。
2. ⬜ 支持优先级（例如人工触发 > 批量补跑）和取消排队任务。
3. ⬜ 评估把 pose/cue/integrity worker 与 Qwen worker 分离：CPU/IO 阶段与 GPU 推理可重叠。
4. ⬜ 多 GPU 时按视频或 segment 分片；单 GPU 保持串行，避免显存争用。
5. ✅ `phaseSeconds` 已进入 status 并在复核窗口展示；⬜ 队列深度/运行中任务/平均阶段耗时 API 仍待做。

**验收**：

- ✅ 未来 run 的 UI 时长从 worker 实际开始计算；排队时间不计入、不展示；
- ✅ 不出现重复启动（提交锁 + active videoId 去重）；
- ⬜ 重启后队列可恢复（已有队列落盘，未针对新字段做回归）；
- ⬜ 有并发/优先级测试。

---

### P5：信号质量治理（也是性能治理）

**目标**：减少误报 episode 和无效审计，避免“信号噪声把成本放大”。

**建议实现**：

1. 用 `cues.calibrate()` 做每路摄像头阈值标定；把 `minOnRatio`、`minOnSec`、`minOffSec`、`cueDedupeSec` 纳入部署配置。
2. 对 `no_subject` episode 做聚合/限流（例如同一通道每 N 分钟最多一条审计），避免刷屏。
3. 长亮信号目前一个 episode 只判一次；如业务需要，增加 `repeatSec` 周期性复判（作为规则级可选，不改变默认）。
4. 增加每通道每小时 episode 上限与异常告警，防止阈值错误导致任务爆炸。
5. 在结果/日志中记录 `rawTriggerCount`、episode 数、`no_subject` 比例，便于回溯。

**验收**：

- 噪声视频不产生无界结果；
- 标定流程有文档和脚本；
- 默认语义不变，新增能力可关闭。

---

### P6：存储、可观测性与数据治理

**目标**：让生产运行可监控、可清理、可定位。

**建议实现**：

1. `observations.sqlite` 瘦身：只存必要字段、压缩 body、按 segment 分表或改用列式格式；先测查询/写入收益。
2. run 目录保留策略：按天数/大小清理 `samples/`、`previews/`、`raw/`；保留 `results.jsonl`、`summary.json` 的审计周期。
3. 指标：每阶段墙钟、解码次数、episode 数、`no_subject` 比例、缓存命中、GPU 利用率、队列深度。
4. 在 `progress.json`/API/UI 暴露关键指标；不要只写日志。
5. 失败路径：阶段超时、ffmpeg 错误、磁盘满、模型加载失败要有明确状态与告警。

**验收**：

- 能从 API 看到阶段耗时与队列状态；
- 清理策略有测试/干跑模式；
- 长跑任务不会因磁盘增长静默失败。

---

## 6. 建议实施顺序

| 里程碑 | 内容 | 目的 |
|---|---|---|
| M1（低风险快赢） | P1 integrity 共享解码；P2 懒预览；P6 最小指标 | 先把 191 分钟样本从 39 分钟压到 25 分钟以内 |
| M2（系统性） | ✅ P0 FrameBroker 单次解码 | 已把 tracking+cues+integrity 收敛到 1 次解码；30min 任务 193.6s |
| M3（吞吐） | P4 队列/ETA/worker 分离；P3 tracking 调优 | 多路/批量场景可运营 |
| M4（生产化） | P5 标定与治理；P6 存储/告警；评测集 | 从试点走向有限自动告警 |

每个里程碑结束后，用同一段真实视频复算并记录：总墙钟、各阶段耗时、判定结果是否一致。

---

## 7. 生产化判断与上线路径

### 7.1 判断

- **算法路线：合理，保留。** 信号 episode 触发 + 确定性优先 + 人工复核，是长视频低成本分析的正确方向。
- **当前工程状态：适合试点/辅助复核，不建议直接无人值守执法。** 主要差距是重复解码、单 worker 排队、预览与存储治理、缺少每路摄像头的质量评测。

### 7.2 上线路径

1. **影子模式**：生产环境跑，只出结果不告警；人工对比至少几天/每路摄像头。
2. **人工复核模式**：告警进入复核队列；此时完成 P1/P2/P6，控制成本与存储。
3. **有限自动告警**：只在评测达标、规则与阈值冻结的摄像头/通道开启；保留审计与回滚开关。

### 7.3 生产就绪检查表

- [ ] 每路摄像头有标注评测集（正/负样本、白天/夜间、遮挡、多人）
- [ ] 信号区召回率/误报率达标，阈值已标定
- [ ] forced_passage 的 precision/recall 已统计
- [ ] 单视频处理时间满足 SLA（例如 24h 录像 ≤ X 小时）
- [ ] 队列有 ETA、优先级、取消、重启恢复
- [ ] 预览/观测/原始输出的保留策略已生效
- [ ] 阶段耗时、episode 数、失败率有监控与告警
- [ ] 失败任务可重试且不会重复写结果
- [ ] 规则/阈值版本随任务快照冻结，可追溯
- [ ] 人工复核流程与责任边界明确

---

## 8. 验证命令

```bash
# 全量测试
PYTHONPATH=backend:Rules/src /home/raider/miniconda3/envs/ultraconsole/bin/python \
  -m pytest Rules/tests backend/tests

# 静态检查
/home/raider/miniconda3/envs/ultraconsole/bin/python -m ruff check \
  Rules/src Rules/tests backend/app backend/tests
```

### 8.1 真实任务复算（不改生产 run）

1. 复制一个真实 run 目录到 `/tmp`；
2. 用硬链接复用 `observations.sqlite`（不要复制 200MB+ 文件）；
3. 只保留要验证的规则（例如 `forced_passage`），删除 integrity 事件/区；
4. 直接调用 `understand(Path(tmp_dir))`；
5. 对比新旧 `results.jsonl` 的 `resultId / decision / retreatSec / rawTriggerCount`；
6. 清理 `/tmp` 临时目录。

参考命令（按实际 run 调整）：

```bash
TMP=$(mktemp -d /tmp/ultra-bench.XXXXXX)
RUN=Rules/data/vlm/runs/<run-id>
cp "$RUN/request.json" "$TMP/"
cp "$RUN/cues.json" "$TMP/"
cp "$RUN/candidates.jsonl" "$TMP/"
cp "$RUN/segments.json" "$TMP/"
ln "$RUN/observations.sqlite" "$TMP/observations.sqlite"
# 用 python 修改 request.json：去掉 integrityZones 与非目标事件
PYTHONPATH=Rules/src /home/raider/miniconda3/envs/ultraconsole/bin/python -c \
  "from pathlib import Path; from ultra_rules.vlm.qwen_worker import understand; understand(Path('$TMP'))"
```

---

## 9. Agent 工作规则

1. **先读后改**：读相关代码、测试，以及 `dev-guide.md` 第 2 节（设计契约/红线）与代码地图。
2. **一次一个 P 项**：不要在一次提交里同时改解码、调度和前端。
3. **先测基线**：改动前用同一段视频记录当前耗时与结果；改动后用相同输入复算。
4. **语义等价优先**：性能改动必须证明判定结果不变；不能证明时，先加开关并默认关闭。
5. **新增测试**：每个优化至少一个回归测试（等价性、边界、失败路径）。
6. **不动无关文件**：不要顺手重构；不要动 `poi.py` 的候选语义。
7. **不提交**：除非用户明确要求 commit；完成后报告 diff、测试和实测数字。
8. **记录进度**：更新第 10 节看板，写清楚“做了什么、实测多少、还差什么”。

---

## 10. 进度看板

| 编号 | 任务 | 状态 | 备注 |
|---|---|---|---|
| P0 | 单次解码多消费者 | ✅ **完成** | tracking+cues+integrity 合并为 `detect`；30min 任务 193.6s，observations/candidates/cues/integrity 全部一致 |
| P1 | integrity 多区共享/降采样 | ✅ **第一版完成** | 多区共享解码；30min 真实任务 150s（仅 cues+integrity），输出一致 |
| P2 | 证据预览 lazy | ✅ **完成** | 默认只写 manifest；API 首次访问按需生成；165 结果复算 0.41s、0 预览 |
| P3 | tracking 调优 | 🔶 **I/O/缓存完成** | tracking 100.8s→89.3s，detect 193.6s→179.7s，零漂移；TensorRT/ONNX/FP16/NVDEC 待评测 |
| P4 | 队列/ETA/worker 分离 | 🔶 **部分完成** | 时长从 startedRunningAt 计算、排队不展示；phaseSeconds 已展示；ETA/优先级/worker 分离待做 |
| P5 | 信号质量治理 | ⬜ 待做 | 标定、聚合、`repeatSec` |
| P6 | 存储与可观测性 | ⬜ 待做 | 指标、保留策略、失败告警 |

> 状态标记：⬜ 待做 / 🔶 进行中或部分完成 / ✅ 已完成。完成时补一行“实测：<数字>；测试：<命令/结果>”。

### 10.1 本轮实施记录（2026-11-26）

- **P0a / P1**：新增 `Rules/src/ultra_rules/vlm/combined.py` 与 `detect` 阶段；`runner.phases_for()` 在 cues+integrity 同时存在时只跑 `segment → tracking → detect → understanding`。
  - `media.SampleSchedule` 与 `sampled()` 的取帧规则一致（VFR 测试锁定）。
  - `cues.cache_key()` / `integrity.cache_key()` 让合并阶段与单独阶段共用缓存。
  - `integrity._ZoneRuntime` 抽出流式检测器，供单独阶段与合并阶段复用。
  - 真实 30 分钟任务：cues+integrity 150s（旧阶段约 4 分钟），`cues.json` / `integrity.json` 完全一致。
  - 测试：`Rules/tests/test_combined.py`、`test_vlm.py::test_sample_schedule_matches_sampled_on_vfr`、`test_runner_limits.py::test_phases_share_one_detect_pass_only_when_both_consumers_need_video`。
- **P0b（已完成）**：`combined.analyze_run()` + `_track_and_detect()`：
  - 同一个 `decoded()` 循环按 `SampleSchedule` 分发给 pose / cue / integrity；pose 保持原 `poseFps`、batch 与 pipeline 行为。
  - integrity persons 用最近 pose 结果（±1s 最近邻），与 `_persons_near()` 的 sqlite 查询语义一致；pending 样本在 pose batch flush 后评估。
  - observations 缓存命中时：当前 run 无 sqlite → `track()` 重放；已有完整 sqlite → 直接复用，避免重建表。
  - `observations.cache_fingerprint()` / `tracker_config_for()` 被 tracking 与合并阶段共用，缓存键不漂移。
  - 真实 30 分钟任务：完整 `detect` 193.6s（旧分开阶段约 6 分钟）；9038 行 observations 逐字段一致；cues/integrity 完全一致。
  - 真实 5 分钟任务：1500 行 observations + 165 条 candidates 完全一致；cues/integrity 完全一致。
  - 同 run 二次 `detect` 缓存重放 1.5s。
  - 测试：`test_combined.py::test_analyze_run_joins_tracking_cues_and_integrity`（含同 run 重试与跨 run 缓存命中）。
- **P2（已完成）**：lazy 证据预览：
  - `contracts.AnalysisParameters.previewMode` 默认 `lazy`；`build_sample(render=None)` 按它决定是否抽帧/拼图。
  - 确定性证据只写 `samples/{id}.json`；VLM/dryRun 显式 `render=True`。
  - `backend/app/vlm_api.py` 的 artifact 端点发现 `previews/*.jpg` 缺失时，按 manifest 调 `render_preview()`，全局锁串行生成并缓存。
  - 真实 5 分钟、165 条确定性结果复算：understanding **0.41 秒**，生成 **0 个预览 / 23 个 manifest**（其余为 cue-off 短路）；165 条公共判定与旧结果完全一致。
  - lazy/eager manifest 一致；单目标预览字节一致，多人叠加时绘制顺序可能不同但视觉等价。
  - 测试：`test_lazy_preview.py`、`backend/tests/test_vlm_api.py::test_artifact_endpoint_renders_missing_preview_on_demand`。
- **P3（I/O/缓存部分完成）**：
  - `observations.open_observations_db()`：`journal_mode=MEMORY` + `synchronous=OFF`；临时库可重建，cache 才是完成标记。
  - 延迟 `CREATE INDEX`；`executemany` 每 200 行 + commit；progress 每 50 行。
  - cache 写 tracker.update 前行，sqlite 写更新后行；修复了“单次序列化导致 trackSegmentId 丢失”的回归。
  - gzip cache `compresslevel=1`；旧 cache 仍可读。
  - 实测（30min、9038 帧、batch16+pipeline+poseFps5）：tracking 100.8s → **89.3s**（-11.4%）；完整 detect 193.6s → **179.7s**（-7.2%）；observations 逐字段一致，bench diff 0 mismatch、bbox/kp delta 0.0。
  - 测试：`Rules/tests/test_tracking_io.py`。
  - 未做：TensorRT / ONNX / FP16 / NVDEC 独立评估。
- **P4（部分完成）**：
  - `status.json` 新增 `queuedAt`（仅审计）/ `startedRunningAt`；未来 run 的 `durationSec` 从 `startedRunningAt` 计算，排队时间不计入也不展示。
  - 仍在排队的新 run `durationSec` 为空并显示 `—`；历史 run 无 `startedRunningAt` 时回退到原 `startedAt` 语义（不迁移）。
  - runner 记录 `phaseSeconds`（segment/tracking/detect/cues/integrity/understanding），复核窗口展示阶段耗时。
  - `media._frame_timestamp`：片尾缺失 PTS 的帧按“上一帧 + 一帧间隔”回退；真实 PTS 回退/解码错误在 `truncate_on_error=True` 时只处理最后可用前缀，run 置 `partial` 并记录 `processedDurationSec`/原因，UI 显示警告。
  - 未做：ETA/排队位置、优先级、worker 分离、队列深度 API。

---

## 11. 已知修复与回归测试（避免回退）

| 修复 | 回归测试 |
|---|---|
| seek 落后目标帧时重试，不回退全片解码 | `Rules/tests/test_vlm.py::test_source_images_retries_a_seek_that_lands_after_the_first_frame` |
| 同采样率信号区共享一次解码 | `Rules/tests/test_cues.py::test_detect_all_uses_one_decode_pass_per_sample_rate` |
| cues + integrity 共享一次解码且缓存可复用 | `Rules/tests/test_combined.py::test_combined_detects_cues_and_integrity_with_one_decode` |
| tracking 并入共享解码、observations/candidates 一致、同 run 重试安全 | `Rules/tests/test_combined.py::test_analyze_run_joins_tracking_cues_and_integrity` |
| 共享解码的取帧规则与 `sampled()` 一致 | `Rules/tests/test_vlm.py::test_sample_schedule_matches_sampled_on_vfr` |
| runner 在 cues+integrity 时选择 `detect` 阶段 | `Rules/tests/test_runner_limits.py::test_phases_share_one_detect_pass_only_when_both_consumers_need_video` |
| lazy manifest 不抽帧、按需预览与 eager 一致 | `Rules/tests/test_lazy_preview.py::test_lazy_sample_writes_manifest_only_then_renders_same_preview` |
| 确定性证据 lazy 不触发源解码 | `Rules/tests/test_lazy_preview.py::test_deterministic_evidence_lazy_skips_source_decode` |
| artifact 端点按需生成缺失预览 | `backend/tests/test_vlm_api.py::test_artifact_endpoint_renders_missing_preview_on_demand` |
| 批量 tracking 写库仍保留 trackSegmentId/identityAmbiguous | `Rules/tests/test_tracking_io.py::test_track_batch_sqlite_rows_keep_post_tracker_fields` |
| `no_subject` 不抽帧 | `Rules/tests/test_cue_trigger.py::test_no_subject_cue_episode_skips_evidence_decoding` |
| cue-off 确定性候选不抽帧 | `Rules/tests/test_vlm_compact.py::test_worker_all_off_candidate_skips_evidence_decoding` |
| 目标索引与二分查找 | `Rules/tests/test_decision.py::test_select_cue_target_prefers_the_entry_line_then_auxiliary_zone` |

---

## 12. 附：本文不做什么

- 不重新引入“逐计数候选分析”；
- 不把 forced_passage 改成 VLM 默认复核；
- 不改变 3 秒 episode 去重、通道配对、5 秒退回窗口；
- 不为了提速降低视频分辨率/poseFps 而不做评测；
- 不在本文重复 `dev-guide.md` 的架构、代码地图与用户操作说明；性能细节以本文为准。
