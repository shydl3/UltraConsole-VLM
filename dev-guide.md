# UltraConsole 开发指南（Dev Guide）

> 面向开发者 / 后续 AI 协作：系统架构、稳定设计契约、代码地图、开发测试与当前风险。
> 项目介绍见 [README](README.md)；部署与启停见 [deploy.md](deploy.md)；用户操作见 [使用手册](docs/USAGE.md)；性能路线与实测见 [optimize.md](optimize.md)；历史变更见 [CHANGELOG](CHANGELOG.md)。

---

## 1. 系统架构与数据流

### 1.1 一次分析任务的阶段

```text
segment
  └─ 规划分段 / 探测视频元数据

detect（cues + integrity 同时存在）
  └─ 单次 media.decoded()，SampleSchedule 分发给：
       · tracking  → observations.sqlite + candidates.jsonl
       · cues      → cues.json
       · integrity → integrity.json
  └─ 仅 cues 或仅 integrity 时：tracking → cues / integrity 分开执行

understanding
  └─ 逐规则类别判定 → results.jsonl
```

- `detect` 由 `Rules/src/ultra_rules/vlm/combined.py` 实现；tracking、cue、integrity 共享同一次解码，避免长视频重复解码。
- `understanding` 由 `qwen_worker.understand()` 驱动；确定性结论不调用模型，`uncertain` 才进入 Qwen 复核。
- 阶段状态写入 `status.json` / `progress.json`；阶段失败、取消、超时由 `runner.py` 管理。
- 观测缓存以 gzip + 指纹保存；改规则通常命中缓存，只重算候选/判定。

### 1.2 候选事件如何产生

- **POI 候选**（`poi_event`）：`PoiTracker` 对每个 `(trackId, trackSegmentId, POI)` 维护状态机，在跨线 / 进区 / 出区 / 首次在区内时触发。
- **信号候选**（`forced_passage`）：不逐人产生。`_cue_pairs()` 读取 `cues.json`，按 **3 秒 episode 去重**（以首次触发为锚）生成候选，锚定到同编号入口计数线。
- **支撑候选**：信号规则仍需要 `line.cross / zone.enter / zone.leave / initial_inside` 来选目标和判退回；这些事件以 `eventDefinitionIds=[] + supportRuleIds=[...]` 写入 `candidates.jsonl`，不产生独立判定。
- **完整性**：`integrity.json` 中的变化区间直接进入完整性判定，不依赖跟踪候选。

> `poi.py` 的候选生成语义是红线，除非业务模型变化并有评测集，否则不要改。

### 1.3 判定路径

- **`poi_event`**：候选 → 若绑定信号区，先走 `decision.py` 确定性门禁；确定时直接 `match/no_match`，否则交给 Qwen。
- **`forced_passage`**：信号 episode → 选最近正向跨入口线的目标 → 从目标进入时刻起 5 秒内找反向跨线/离开计数区 → `match`（未退回）/ `no_match`（退回）。默认纯确定性，不调用 VLM；无人通过输出 `no_subject` 审计行，且不生成证据。
- **`integrity`**：像素级基线比对 → 变化区间 → CV 判定，不依赖大模型。
- **VLM 后处理**：模型逐条件输出 `yes / no / unknown + evidenceTimeSec`，`results.py` 按合取真值表算最终 decision；证据时间越窗/缺失会降级为 `unknown`。模型自由文本结论仅作对照。

### 1.4 前端主流程

- 服务器目录浏览/导入视频；浏览器不能直放时先显示首帧并允许绘制。
- 在首帧/预览上绘制计数线、计数区、信号区、完整性区；支持矩形/多边形。
- 定义事件规则并绑定区域；支持单文件与同机位文件夹批量。
- 查看结果、逐条件证据、回放定位、人工复核；记录与视频统一在管理页查看。
- 绘制区域时按住 `Ctrl` 显示 3× 放大镜（`utils/draw.ts` 的 `drawRegionMagnifier`）。

### 1.5 性能现状

| 场景 | 实测 |
|---|---:|
| 30min 1080p、poseFps=5、1 信号区 + 3 完整性区，完整 `detect` | **179.7s** |
| 同任务 tracking 阶段 | **89.3s** |
| 5min、165 条确定性结果 understanding | **0.41s**（lazy 预览，0 张图） |
| 同 run 全缓存重放 | **1.5s** |

> 性能路线、P0–P6、基线与红线统一维护在 [optimize.md](optimize.md)，本文不重复展开。

---

## 2. 稳定设计契约（红线）

以下决策是当前系统正确性的基础，除非业务模型变化并有评测集，否则不要推翻：

1. **报警信号是安检门门框的竖直红色 LED 灯带**；顶部绿色指示灯按规则忽略。
2. **红灯亮灭用经典 CV 检测**（红色像素占比 + 持续时间），不让大模型判断。
3. **确定性优先 + VLM 兜底**：能用轨迹/红灯/退回算清楚的直接出结果；只有 `uncertain` 才调模型。
4. **退回语义**：POI 路径窗口为 `[entry_t, onset+5]`；信号触发路径为 `[entry_t, entry_t+5]`，从目标进入时刻起算，避免灯亮后晚进入观察窗过短。
5. **规则类别稳定**：`poi_event / forced_passage / integrity` 决定必需区域、触发方式与判定路径；名称可改，类别标识不复用。
6. **信号触发**：入口计数线（必填、必须有进入方向）+ 信号区按 `pairIndex` 配对；计数区可选辅助。同一信号区首次触发后 **3 秒内去重**，一个 episode 只判一次。
7. **入口方向**：只作用于最新画的一条计数线；反向跨越视为退回。
8. **确定性证据默认 lazy**：只写 sample manifest；预览由后端首次访问生成。VLM/dryRun 仍 eager。
9. **`no_subject` 只留审计记录**，不抽帧、不生成证据；cue-off 的确定性 `no_match` 同样不抽帧。
10. **单次解码**：tracking/cues/integrity 共享 `detect` 的解码流；改采样/缓存逻辑必须做逐帧等价性回归。
11. **不要退回逐计数候选分析**：信号 episode 触发是长视频成本控制的核心，不得为了“简单”改回逐个候选。
12. **分析前强制校验**：只画完整性区时只能启用完整性规则；强行通过缺入口线/信号区/配对/方向时前端行内提示 + 后端 422。

---

## 3. 准确率模型与改进路线

### 3.1 已知失败模式

1. **证据不可见**：抽帧分辨率过低时红灯只有 1–2 像素，模型看不清。
2. **规则语义不对齐**：窗口不以信号亮起为基准、触发类型混入 `initial_inside` / `zone.leave` 时，模型会把“进入”误判成“经过”。
3. **模型/提示词偏差**：小模型倾向复述规则文字；“未看见不等于没发生”会把不确定推向 `match`。
4. **后处理不拦截**：早期版本直接采信模型结论，质量警告不否决；现在改为逐条件合取 + 证据校验降级。

### 3.2 路线

| 阶段 | 状态 | 内容 |
|---|---|---|
| **P0 确定性判定** | ✅ 已完成 | 红灯 CV、入口线退回、后处理 veto、触发收紧、阈值校准工具 |
| **P1 证据 / 提示词 / 模型** | 🔶 部分完成 | 结构化逐条件输出 ✅、默认否认提示词 ✅；高分辨率 ROI 抽帧、4B/8B 对比、self-consistency、置信度校准待做 |
| **P2 评测 / 生产化** | ⬜ 待做 | 人工标注集、precision/recall/误报率、人工复核闭环、监控与告警 |

> 强行通过/信号 episode/通道配对的详细设计见 [signal-triggered-rules-plan.md](docs/signal-triggered-rules-plan.md)。

---

## 4. 关键代码地图

### 4.1 后端 / 算法

| 路径 | 职责 |
|---|---|
| `Rules/src/ultra_rules/vlm/contracts.py` | Poi / CueDefinition / EventDefinition / RunRequest / AnalysisParameters；规则类别、pairIndex、cueDedupeSec、previewMode |
| `Rules/src/ultra_rules/vlm/models.py` | 项目内 Qwen3-VL 权重发现与校验 |
| `Rules/src/ultra_rules/vlm/observations.py` | YOLO-Pose + ByteTrack → observations.sqlite；缓存指纹；支撑候选；`open_observations_db()` 写路径优化 |
| `Rules/src/ultra_rules/vlm/combined.py` | `analyze_run` / `_track_and_detect` / `detect_run`：tracking + cues + integrity 单次解码 |
| `Rules/src/ultra_rules/vlm/media.py` | `probe / decoded / sampled / SampleSchedule`：解码与多消费者采样 |
| `Rules/src/ultra_rules/vlm/cues.py` | 红灯 CV、`detect_all()`、`calibrate()` |
| `Rules/src/ultra_rules/vlm/integrity.py` | 完整性 CV；`_ZoneRuntime` 流式检测器 |
| `Rules/src/ultra_rules/vlm/decision.py` | 目标选择、退回、轨迹连续性、`build_target_index()`；信号 episode 合并 |
| `Rules/src/ultra_rules/vlm/sampling.py` | 证据窗口、选帧、manifest、`build_sample(render=...)`、`render_preview()` |
| `Rules/src/ultra_rules/vlm/qwen_worker.py` | `understand()` 编排；信号路径 `_cue_pairs()` / `_process_cue_candidate()` |
| `Rules/src/ultra_rules/vlm/results.py` | 逐条件解析、合取真值表、证据校验降级、结果聚合 |
| `Rules/src/ultra_rules/vlm/prompting.py` | Qwen 提示词与版本 |
| `Rules/src/ultra_rules/vlm/runner.py` | 阶段编排、worker lock、超时/取消、状态写入 |
| `Rules/src/ultra_rules/pose_adapter.py` | YOLO-Pose 单帧/批量推理适配 |
| `backend/app/vlm_api.py` | VLM HTTP、任务队列、artifact 懒预览、文件夹导入 |
| `backend/app/vlm_jobs.py` | 启动/状态/取消、owner 存活、reap |
| `backend/app/event_rules.py` | 规则库 CRUD、类别迁移、配对校验、resolve |
| `backend/app/video_api.py` / `video_transcode.py` | 视频浏览/预览；remux / NVDEC→NVENC |
| `backend/app/main.py` | FastAPI 入口、lifespan 自动选设备/加载模型、实时推理 |

### 4.2 前端

| 路径 | 职责 |
|---|---|
| `ui/viewer/tabs/VideoTab.tsx` | 视频选择、预览、首帧绘制、VLM 面板、批量入口 |
| `ui/viewer/widgets/EventEditOverlay.tsx` | 计数线/区/信号区/完整性区绘制；Ctrl 放大镜 |
| `ui/viewer/widgets/RoiOverlay.tsx` | 跟踪 ROI 绘制；Ctrl 放大镜 |
| `ui/vlm/VlmPanel.tsx` | 语义分析控制、参数、任务记录、结果列表 |
| `ui/vlm/EventRulesEditor.tsx` | 规则类别、配对通道表、校验提示 |
| `ui/review/ResultCard.tsx` / `ReviewVideo.tsx` | 结果卡片、证据帧、回放定位 |
| `utils/draw.ts` | Canvas 绘制、`fitContain`、`drawRegionMagnifier` |
| `utils/vlm.ts` | 结果/进度/参数类型与前端契约 |

---

## 5. 开发、测试与调试

### 5.1 核心命令

```bash
# 后端 + Rules
PYTHONPATH=backend:Rules/src python -m pytest Rules/tests backend/tests
python -m ruff check Rules/src Rules/tests backend/app backend/tests

# 前端
cd frontend
npm test -- --run
npm run lint
npm run build
```

性能基线 / 一致性回归：

```bash
python Rules/scripts/bench_tracking.py run --input <视频> --name base --batch 16 --pipeline --pose-fps 5 --fresh
python Rules/scripts/bench_tracking.py diff --a <基线run> --b <新run>
```

部署、启动/停止、常见问题见 [deploy.md](deploy.md)；完整性能基线命令见 [optimize.md](optimize.md)。

### 5.2 调试入口

- 任务目录：`Rules/data/vlm/runs/vlm-<id>/`
- 状态/进度：`status.json`、`progress.json`
- 阶段日志：`segment.log`、`detect.log`、`tracking.log`、`cues.log`、`integrity.log`、`understanding.log`
- 关键产物：`observations.sqlite`、`candidates.jsonl`、`cues.json`、`integrity.json`、`results.jsonl`、`samples/`、`previews/`
- 前端开发：`cd frontend && npm run dev`；生产构建：`npm run build`

### 5.3 改动约定

- 涉及判定语义的改动必须补等价性/回归测试；性能改动必须证明 `observations / candidates / cues / integrity` 不漂移。
- 不要在一次改动里同时重写解码、规则和前端。
- 新增长期架构/契约写进本文件；性能路线写进 `optimize.md`；历史变更写进 `CHANGELOG.md`；用户操作写进 `docs/USAGE.md`。
- 不要把部署步骤、性能基线和“本轮变更清单”重新塞回本文件。

---

## 6. 当前有效风险

| 风险 | 说明 |
|---|---|
| 真实红灯端到端验收 | 尚未用含真实红灯的录像跑完整 forced_passage 链路；需验证 episode 数、目标选择、退回与证据窗口 |
| 准确率评测集缺失 | 当前无法量化 precision/recall/误报率；P1/P2 调整前应先建最小标注集 |
| 4B/8B、证据增强未落地 | 高分辨率 ROI 抽帧、self-consistency、置信度校准仍待做 |
| 191min 完整 benchmark 未复跑 | 30min 数据已对齐；超长录像端到端、GPU 占空比/显存峰值仍待测 |
| 单 GPU 串行队列 | `MAX_CONCURRENT=1`，批量任务排队；多路/生产吞吐需 P4 |
| 阈值依赖相机 | 红灯/完整性阈值需按相机标定；噪声可能放大 episode 数 |
| 预览依赖源文件 | lazy 预览首次访问需要源视频与 observations；源被删除时 404，结果本身仍可读 |
| 生产定位 | 当前适合试点/人工复核；无人值守告警需评测集与运维体系 |

性能相关的风险、基线与后续 P3/P4 计划见 [optimize.md](optimize.md)。

---

## 7. 维护与交接

- **文档分工**：README 介绍项目；deploy 负责部署/启停；USAGE 负责用户操作；本文件负责架构/契约/代码地图；optimize 负责性能；CHANGELOG 负责历史。
- **红线优先**：第 2 节的决策优先于“看起来更简单”的实现；改红线必须先改评测与业务假设。
- **交接顺序**：先读 README → deploy（跑起来）→ USAGE（用一遍）→ 本文件（架构/契约）→ optimize（性能）。
- **提交前**：至少跑对应测试；涉及前端跑 build/lint；涉及性能跑 `bench_tracking diff`；涉及规则语义跑 `Rules/tests` + `backend/tests`。

---

[返回 README](README.md) · [部署指南](deploy.md) · [使用手册](docs/USAGE.md) · [性能路线](optimize.md) · [CHANGELOG](CHANGELOG.md)
