# UltraConsole VLM 语义分析 — 开发指南（Dev Guide）

> 本文档面向两类读者：
> - **非技术读者（业务 / 管理 / 其他部门）**：请先读「第 1 节 执行摘要」和「第 2 节 项目目标」，了解我们在做什么、做到哪了、下一步为什么。
> - **开发者 / 后续 AI 协作**：从「第 3 节」开始是技术细节、代码地图、路线图与验收标准，可直接据此继续开发。

---

## 1. 执行摘要（给非技术读者）

本项目是一个**对视频做“事件语义分析”的工具**：用户上传一段监控视频，在画面上框出“计数线 / 计数区 / 信号区”，定义一条“事件规则”，系统自动：

1. 用视觉模型（YOLO 姿态）检测并跟踪画面里的人；
2. 根据“谁在何时进入/离开区域、是否跨线”生成**候选事件时间点**；
3. 用视觉语言大模型（Qwen3-VL）判断每个候选点“事件是否真的发生”；
4. 输出带时间点、原因和证据截图的结果，供人工复核。

**当前状态一句话**：候选事件的时间点已经可靠；**唯一的主要短板是“单个事件判断”的准确率还不够高**（存在较多误报 / 漏报），而我们的目标是把准确率提升到**可上线生产环境部署**的水平。

**我们已经做了两件关键事，直接针对误报：**

- **把“门框红灯是否亮起”这类可明确测量的信号，改成用传统图像算法（CV）直接检测**，不再让大模型“看”一个在缩略图里只有一两个像素的红灯——这消除了最大的一类误报（模型把规则文字复述成“红灯亮起”）。
- **把判定改成“确定性优先 + 大模型兜底”**：能用轨迹和红灯状态算清楚的，直接算；算不清楚的才交给大模型复核。

**本轮又做完两件事：**

- **把大模型回答改成“逐条件、带证据时间”的结构化输出**：模型对每个判定条件给“是 / 否 / 不确定 + 证据画面秒数”，由代码按“所有条件都满足才算发生”的合取逻辑**计算**最终结论，不再直接采信模型的一段话；提示词同时改为**“默认否认”**（看不清就不算发生）并**删掉衣着外貌等无关描述**。GPU 实测：描述更短更聚焦、逐条件证据可点击回放。
- **控制台启动即就绪**：后端启动自动选 GPU（CUDA）并加载姿态模型 `yolo11m-pose.pt`，前端无需再手动选设备 / 模型（下拉与按钮仍保留可手动切换）。

**本轮另修复一处监控视频兼容问题**：HEVC/H.265 等浏览器解不出画面的监控录像，过去预览黑屏、点分析报 `Missing JPEG payload`；现在前端检测到“有时长无画面”会自动转入显卡兼容预览（NVDEC 硬解 → H.264），本轮已用一段 25 分钟真实 HEVC 录像实测跑通转码链路（见 4.5）。

**长视频吞吐 / 显卡利用率（现状与下一步）**：目前处理几小时的录像时，两条分析链路都是“逐帧串行”——取画面、显卡推理、事件计算轮流等待，显卡大部分时间空闲。本轮已完成第一期提效施工（tracking 解码/推理流水线 + 批量姿态推理，默认关闭、可开关），并在真实 25 分钟监控录像上实测：**语义无损的出厂配置提速约 1.19×**；本机属推理瓶颈而非解码瓶颈，更大空间在“基础分析离线批处理”与“understanding 批处理”（见 6.4.5）。所有提效改动以“不改变任何判定结论”为前提，默认保守串行。

**接下来要做的（按优先级）：**

- **P1（剩余）**：让大模型“看得更清楚”（高分辨率裁剪、围绕红灯亮起时刻抽帧）、并评估更大更强的模型（4B/8B、self-consistency）。
- **P2**：建立**人工标注的评测集和量化指标**（准确率、召回率、误报率），用数据证明“是否达到上线标准”，并接入人工复核闭环。
- **PT（性能与吞吐专项，与 P1/P2 正交）**：先测基线，再做不改变判定语义的流水线/批处理提效；understanding 批处理排在 P1 证据方案冻结之后，避免返工（见 6.4）。

详细方案见第 6 节。这份文档的目标是：让任何一位开发者（或 AI）接手后，不需要重新摸索，就能按路线图继续推进。

---

## 2. 项目定位与目标

- **定位**：UltraConsole 的二次开发版本。系统由两个子系统组成：
  1. **经典 CV 实时视频控制台**（保留自上游）：实时 YOLO 检测、计数线/区、系统监控、告警引擎、事件日志等（见 `ControlPanel` / `MonitorPanel` / `VideoTab` 的基础分析 / `ws/infer`）。
  2. **VLM 语义事件分析**（本项目核心扩展）：上传视频 → 画 POI/信号区 → 事件规则 → 候选事件 → 大模型判定。
- **已移除**：仅 **Webhook 推送**能力（前端 `WebhookCard`/`useWebhookDispatcher`、后端 `webhook.py` 与 `/api/webhook*`、`/api/notify`）。记录/视频管理已改为**单页内切换**（不再弹窗）。
- **目标**：把 VLM 的“事件级判定”准确率提升到**生产环境可部署**水平，即误报、漏报低到业务可接受，且结果可审计、可回放。

> 上游说明：本项目是作者开源项目的衍生作品，保留 LICENSE 与署名；开发在 `llm` 分支，`origin` 指向自有仓库，`upstream` 指向原作者仓库。

---

## 3. 系统如何工作（架构概览）

### 3.1 一次分析任务的完整链路

```
前端(上传视频 + 画线/区/信号区 + 定义规则)
        │  POST /api/vlm/runs/from-rules
        ▼
后端 event_rules.resolve()  →  RunRequest(request.json)
        │
        ▼
runner.py 依次执行三个子进程阶段：
  ① tracking    姿态跟踪：YOLO-Pose + ByteTrack → observations.sqlite
  ② cues        信号检测：CV 红灯检测 → cues.json（可选，仅当规则绑定了信号区）
  ③ understanding 判定：确定性判定 或 VLM 复核 → results.jsonl
        │
        ▼
前端轮询 /runs 与 /results，展示结果 + 证据截图
```

> 完整阶段序列为 `segment → tracking → cues（可选）→ integrity（可选）→ understanding`（`runner.py`）；上图省略了 segment 与 integrity 两个阶段。各阶段的串行点与 GPU 空闲来源见 3.4。

### 3.2 候选事件如何生成（时间点）

- 候选生成在 `Rules/src/ultra_rules/vlm/poi.py` 的 `PoiTracker`：对每个 `(跟踪ID, 跟踪段, POI)` 维护状态机，在跨线 / 进区 / 出区 / 首次在区内时产生**触发（trigger）**。
- `Rules/src/ultra_rules/vlm/observations.py` 再按“启用的规则 + 触发类型 + 绑定的 POI”过滤，写入 `candidates.jsonl`。
- **这部分（候选时间点）用户已确认可靠，不要再改动其生成逻辑。**

### 3.3 判定如何产生（结果）

- `Rules/src/ultra_rules/vlm/qwen_worker.py` 的 `understand()` 对每条候选：
  - 若规则绑定了**信号区（cueIds）** → 先走 `decision.py` 的**确定性判定**；
  - 确定性给出 `match` / `no_match` → 直接出结果，不调用大模型；
  - 确定性无法判定（`uncertain`）→ 交给 VLM 复核（结构化逐条件输出 + 默认否认提示词，见 6.2）。
- `Rules/src/ultra_rules/vlm/results.py` 负责解析模型回答与聚合（当前版本 `model-decision-v5`）：解析模型逐条件输出 `conditions`（`index / status(yes|no|unknown) / evidenceTimeSec`），由 `decision_from_conditions` 按**合取真值表**计算 decision（任一 `no`→`no_match`；全 `yes`→`match`；否则 `uncertain`），模型自由文本结论降级为 `modelDecision` 仅供对照；证据时间做校验（越窗 / 约等于 `triggerSec` / 缺失 → 该条件降级 `unknown` 并记警告）；`conditions` 缺失或数量不符则回退采信 `modelDecision`（保持 v4 行为）。质量项（可见度 / 帧数 / 目标过小）仅作 advisory 警告，不否决。

### 3.4 两条链路的串行点与 GPU 空闲来源（性能视角）

> 本节是**现状诊断（事实，带代码位置）**；对应的提效提案全部在 6.4 节并标注 ⬜ 待做。量化数字一律待实测基线（6.4.1），此处只做定性描述。

**链路 A：视频基础分析（前端驱动，YOLO + 计数线/区）**

```
VideoTab.analyze()：每采样帧一次完整往返，同一时刻仅 1 帧在途
  ① 浏览器 seek（从最近关键帧重新解码）
  ② drawImage → canvas.toDataURL（同步 JPEG 编码 + base64）
  ③ WS /ws/infer 发送单帧
  ④ main.py ws_infer：reader → 单个 infer_worker → sender
  ⑤ infer_runtime.infer_image：_predict_lock 全局串行（与实时控制台共用）
  ⑥ WS 回传 pred → events.ts processFrame（线/区事件）→ 回到 ①
```

- 单帧在途约束：`frontend/src/utils/videoAnalysis.ts` 的 `VideoInferenceSession.infer`（上一帧 `pending` 未回即拒绝发下一帧）。
- 后端 `ws_infer` 的 `latest_frame` 单槽 + 满队列丢弃是**为直播设计**（新帧覆盖旧帧）；离线分析靠前端逐帧等待避免丢帧，代价是吞吐。
- GPU 空闲来源：① seek 重复解码、② 同步编码、③⑥ 往返与前端事件/渲染；GPU 推理只占单帧耗时的一部分（占比待实测）。
- 事件规则计算（`processFrame`）是 O(框数×规则数) 的纯几何，单帧成本可忽略——**链路 A 的瓶颈不是规则多，而是取帧方式与单帧乒乓**。

**链路 B：VLM 语义分析（后端离线 worker）**

```
runner.run()：持 worker.lock，子进程阶段严格串行
  segment → tracking → cues(可选) → integrity(可选) → understanding
  （vlm_api._pump：MAX_CONCURRENT=1，全局同时仅一个 run）

tracking（observations.track，单进程单线程生成器）：
  media.sampled 全帧软解 → 采样帧 to_ndarray(bgr24)（CPU）
    → PoseAdapter.infer 单帧 YOLO.track（GPU）
    → JSON 序列化 + sqlite/gzip 写入 + PoiTracker.update（CPU）
  ——解码时 GPU 闲，推理时 CPU 闲，逐帧交替

cues / integrity：各自再次调用 media.sampled，把整个文件重新解码一遍

understanding（qwen_worker.understand）：
  逐候选 × 逐绑定事件 × 逐窗口（sampling.windows：16s 分段、重叠 2s）
    build_sample（CPU：查 sqlite + seek 抽帧 + 画标注 + 拼图）
    → Qwen.infer batch=1 生成（GPU）→ 解析/落盘（CPU）
```

- 串行点：`Rules/src/ultra_rules/vlm/runner.py` 阶段 for 循环；`vlm/observations.py` 的 `generate()`；`vlm/qwen_worker.py` 的 `_process_candidate()` 窗口 for 循环；`backend/app/vlm_api.py` 的 `MAX_CONCURRENT=1` + `worker.lock`。
- GPU 空闲来源：解码/转换/抽帧/画图/写盘等 CPU 段与 GPU 段交替；batch=1 生成填不满 SM；tracking→understanding 之间 GPU 完全空闲（Qwen 懒加载一次）。

**多事件规则如何共享计算（成本随规则数的增长关系）**

| 计算 | 是否随事件规则数增长 | 说明 |
|---|---|---|
| tracking 解码 + 姿态推理 | 否 | 全片只跑一次，帧数由 `poseFps` 决定；观测缓存指纹（`observations.py` fingerprint）**不含规则/POI**，改规则重跑命中缓存 |
| 候选生成 `PoiTracker.update` | 否（随 POI 数与人流） | 缓存命中后仍逐帧重算候选，纯 CPU，成本低 |
| cues / integrity CV 扫描 | 否（随信号区/完整性区数量） | 与事件规则数无关 |
| 候选→规则匹配过滤 | 线性但极廉价 | `observations.track` 内 `applicable` 列表推导 |
| understanding VLM 调用 | **线性（主成本）** | 调用数 ≈ Σ候选(绑定启用事件数 × 窗口数) × (1−确定性短路率)；规则越多，每候选复核越多 |

> 结论：加规则不会重跑跟踪与抽帧；真正随规则数线性放大的是 VLM 复核调用次数。因此“多规则长视频”的提效重点是 tracking 流水线化 + understanding 批处理/短路，而不是候选生成（候选生成逻辑按 4.1/7.3 红线不动）。

---

## 4. 当前开发进度（已完成）

| 模块 | 内容 | 状态 |
|---|---|---|
| VLM 语义分析管线 | 上传视频 → 画 POI → 规则 → 候选 → 判定 → 结果 | ✅ 已完成 |
| P0 确定性判定 | 红灯 CV 检测 + 入口线退回判定 + 后处理 veto + 触发收紧 | ✅ 已完成 |
| P0 边界：红灯阈值校准 | `cli calibrate` 测量工具 + 前端“信号区阈值”面板 | ✅ 已完成 |
| P0 边界：入口方向 | 计数线 `entryDirection`，翻转只作用于最新一条线 | ✅ 已完成 |
| 收敛调整 | 仅移除 Webhook；保留实时控制台与基础分析；记录管理改为单页内切换 | ✅ 已完成 |
| Schema 迁移容错 | `event-rules.json` 旧记录缺新字段时用 `.get` 读取 | ✅ 已完成 |
| P1② 结构化条件输出 | 模型逐条件 `status + evidenceTimeSec`，`results.py` 合取真值表算 decision（`model-decision-v5`）；前端逐条件证据展示 | ✅ 已完成 |
| P1③ 默认否认提示词 | 去衣着/外貌、默认否认、3 个 few-shot、输出含 conditions（`target-event-described-json-v6`） | ✅ 已完成 |
| 控制台启动自选设备/模型 | 后端 lifespan 自动选 CUDA（不可用回退 CPU）+ 加载 `yolo11m-pose.pt`；`GET /api/state` 回显；前端下拉/按钮仍可手动切换 | ✅ 已完成 |
| 库记录命名 + 处理时长 | 分析记录展示名 = 原视频名 + 分钟级时间戳（`library.json` displayName）；计时从「开始分析」起算，库列表/详情显示时长 | ✅ 已完成 |
| 监控 HEVC 兼容预览触发 | 浏览器“有时长无画面”（HEVC 无硬解）时自动转入 GPU 兼容预览；分析入口/帧循环空帧防护（见 4.5） | ✅ 已完成（本轮） |
| PT 性能与吞吐专项 | PT0-1/2 批量姿态 + 解码/推理重叠已实现并实测（默认关，语义无损 1.19×）；基线/回归脚本落地；PT0-3/4/5、PT1、PT2 待做（见 6.4） | 🔶 部分完成 |
| P1（其余：证据 / 模型） | 高分辨率 ROI 抽帧、4B/8B、self-consistency、置信度校准 | ⬜ 待做 |
| P2（评测 / 生产化） | 标注集、指标、人工复核、监控 | ⬜ 待做 |

### 4.1 P0 已落地的关键决策（请勿推翻，除非业务变化）

1. **报警信号 = 安检门门框的竖直红色 LED 灯带**；顶部绿色指示灯按规则忽略。
2. **红灯亮灭用经典 CV（红色像素占比 + 持续时间）检测**，不用大模型判断。
3. **判定策略 = 确定性优先 + VLM 仅复核不确定样本**。
4. **“退回” = 红线亮起后 5 秒内，反向跨回入口计数线**；触发只保留进入类（`line.cross`）。
5. **红灯阈值在前端“信号区阈值”高级设置里可编辑**，随任务快照冻结，每相机各存一套；校准工具输出建议值。
6. **入口方向翻转只作用于最新画的一条计数线**。
7. **仅移除 Webhook**；实时控制台（经典 CV 检测/计数/监控）与基础分析保留。
8. **启动即就绪，无需手动选设备/模型**：后端启动时（FastAPI `lifespan` → `_autostart`）自动选 CUDA（不可用回退 CPU）并加载 `yolo11m-pose.pt`；前端 `ControlPanel` 挂载读 `GET /api/state` 回显当前引擎与模型，Model Hub 下拉与 CPU/CUDA 按钮仍可手动切换。VLM 分析侧 `poseDevice` 默认 `cuda`、姿态模型固定 `yolo11m-pose.pt`。可用 `ULTRA_AUTOSTART / ULTRA_DEFAULT_MODEL / ULTRA_DEVICE` 配置；自动选择失败只记日志、降级为 CPU/无模型，不阻塞启动。
9. **库记录展示名与处理时长语义**：VLM 记录展示名 = `原视频名_YYYYMMDD_HHMM`（分钟级，如 `监控.mp4_20261105_1730`），由 `prepare()` 写 `library.json` 的 `displayName`，内部 `run_id` 不再作展示主标识；处理时长 `startedAt` 在 `prepare()`（用户点「开始分析」那一刻）写入 `status.json`，`runner` 不得重置，`durationSec = finishedAt(或当前时间) - startedAt`，运行中任务显示「进行中」。

### 4.2 本轮变更清单（P1②③ 结构化条件输出 + 控制台启动自选）

**VLM 结构化条件输出与默认否认提示词**
- `contracts.py`：`ConditionCheck` 新增 `evidenceTimeSec`；`ModelAnswer.conditions` 转为主输出。
- `results.py`：`POSTPROCESS_VERSION` v4→**v5**；`parse_answer(trigger)` 解析 conditions + 证据校验降级；新增 `decision_from_conditions` 合取真值表；`apply_answer` 映射 `frameIndex`；`verified_decision` 精简为 advisory。
- `prompting.py`：`PROMPT_VERSION` →**v6**；去衣着/外貌禁令、默认否认、字段顺序 `videoDescription/conditions/decision/confidence/reason`、3 个 few-shot。
- `qwen_worker.py`：`parse_answer/apply_answer` 接线传 `trigger` + 采样清单；output 带 `eventCriteria` 快照。
- 前端 `utils/vlm.ts` + `ui/vlm/VlmPanel.tsx` + i18n：逐条件证据列表（状态标记 + 点击跳转 `evidenceTimeSec`）。
- 测试：`test_vlm.py` 更新、新增 `test_prompting.py`、合取真值表与证据降级覆盖。

**控制台启动自动选 CUDA + yolo11m-pose**
- `config.py`：`Settings` 新增 `autostart/default_model/device`（`ULTRA_AUTOSTART/ULTRA_DEFAULT_MODEL/ULTRA_DEVICE`）。
- `main.py`：FastAPI `lifespan`→`_autostart`（选 cuda/回退 cpu + 加载默认模型，失败降级不阻塞）；新增 `GET /api/state`。
- `state.py`：初始 `engine.device` 改用 `settings.device`。
- 前端 `contracts.ts`（`ConsoleBootstrap`）+ `client.ts`（`apiGetState`）+ `ControlPanel.tsx`（挂载同步 engine/model，下拉与按钮保留）。
- 测试：`conftest.py` autouse 关闭 autostart；新增 `test_autostart.py`（/api/state、优先 cuda、回退 cpu）。
- 验证：后端 pytest 53 通过 + 1 skip、ruff 通过、前端 build/lint/test 30 通过；重启后 `GET /api/state` 实测回显 `cuda` + `yolo11m-pose.pt`。

### 4.3 本轮变更清单（库记录命名 + 处理时长）

- **展示名 = 原视频名 + 分钟级时间戳**：`cli.py` 的 `prepare()` 创建任务时写 `run_dir/library.json` 的 `displayName = <fileName>_<YYYYMMDD_HHMM>`（如 `监控.mp4_20261105_1730`）；`library_api._entry()` 优先取 `displayName`，内部 `run_id`（`vlm-<uuid>`）仅作目录/URL/去重，不再作展示主标识。旧任务无 `library.json` 时仍显示原文件名。
- **计时器与处理时长**：`prepare()` 写 `status.json` 的 `startedAt`（用户点「开始分析」那一刻），`runner.run()` 改为沿用该值不重置；`library_api._entry()` 计算 `durationSec = finishedAt(或当前时间) - startedAt` 并随每条记录返回。
- 前端：`utils/library.ts` 的 `LibraryItem` 增加 `durationSec` 与 `formatDuration()`（时/分/秒、中英双语）；`LibraryPage` 新增「处理时长」列、`LibraryDetail` 显示时长，运行中任务标注「进行中」。
- 测试：新增 `test_display_name_and_duration`（displayName 时间戳 + durationSec 精确断言）。验证：后端 pytest 137 passed + 1 skip、ruff 通过、前端 build/lint/test 30 通过。

### 4.4 路径自包含（权重与解释器）

- **VLM 权重根目录**：`models.py` 的 `MODEL_ROOT` 原为 `parents[5]`（实际解析到 `/home/user`，即项目外），现改为项目内 `models/`，并支持 `VLM_MODEL_ROOT` 覆盖（外部挂载到其他路径时使用）；2B/4B 权重已复制到 `models/Qwen3-VL-2B-Instruct` / `models/Qwen3-VL-4B-Instruct`，`VLM_MODEL_PATH` 仅作为单模型路径的覆盖入口。
- **VLM 解释器**：`cli.py` 的 `qwenPython` 默认改为 `sys.executable`，不再写死开发机 conda 路径；Qwen 跑在独立环境时用 `VLM_PYTHON` 覆盖（本机放 `backend/.env`，外部环境由环境变量注入）。
- `.gitignore` 忽略 `models/*`（保留 `.gitkeep`）；README 权重/解释器说明与 `backend/.env.example` 已同步。
- 验证：`/api/vlm/parameters` 返回项目内路径，2B/4B 均 `available`；后端 + Rules 测试通过。

### 4.5 本轮变更清单（监控 HEVC 兼容预览触发修复）

- **现象**：加载 HEVC/H.265 监控录像（本轮实测样本：1920×1080 Main / 25fps / AAC / 25 分钟）预览黑屏，分析报 `video.analysis.failed: Error: Missing JPEG payload`。
- **根因**：浏览器无 HEVC 硬解时仍能读出时长（`loadeddata` 触发、`videoWidth=0`），**不触发 `error` 事件**，仅挂在 `error` 上的兼容预览回退永不执行；0×0 画布 `toDataURL` 得 `"data:,"`，空 `imageJpegBase64` 被后端 `ws_infer` 拒收记 WARN，前端会话把 WARN 当致命错误断开。
- **修复（仅前端 `ui/viewer/tabs/VideoTab.tsx`）**：`onLoadedData` 就绪条件加 `videoWidth/videoHeight>0`，“有时长无画面”主动调 `onDecodeError()` 转 GPU 兼容预览；`analyze()` 入口同检查；帧循环空 JPEG 抛本地化错误。后端无改动。
- **验证**：tsc / eslint / vitest（43 用例）通过；用该真实文件跑通 `backend/app/video_transcode.py`（NVDEC hevc_cuvid → h264_nvenc，输出 H.264 1080p、时长完整、抽帧非黑，**本轮实测**）。兼容预览按 SHA-256 缓存（`video_api.py`），同文件二次加载免转码。
- **与 PT 专项的关系**：该转码链路已验证的 cuvid/NVDEC 解码模式，是 PT0-2（tracking 硬解化）可复用的现成参考；无 NVIDIA 硬解的机器上此回退不可用（现有设计禁用 CPU 回退，见第 8 节）。

### 4.6 本轮变更清单（PT0 第一期：tracking 流水线 + 批量化，默认关闭）

- **PT0-2 流水线**（`Rules/src/ultra_rules/vlm/observations.py` 的 `_FramePipeline`）：解码线程（PyAV 软解 + bgr24 转换）→ 有界队列（maxsize=max(8, batch×2)，背压封顶内存）→ 推理消费者；FIFO 保帧序，生产端异常在消费端重抛，close() 让生产线程退出。开关 `ULTRA_VLM_PIPELINE=0/1`（默认 0）。**与串行比特一致**（本轮实测 15000 帧 0.0 漂移）。
- **PT0-1 批量姿态**（`Rules/src/ultra_rules/pose_adapter.py` 的 `infer_batch / _bytetrack`）：一次 batch predict 前向 + 按帧序独立 BYTETracker 关联，精确复刻 ultralytics `on_predict_start / on_predict_postprocess_end`（persist、非 stream 单 tracker、frame_rate=30、永不 reset）；`batch_size=1` 保持原单帧 `model.track` 路径不变。开关 `ULTRA_VLM_POSE_BATCH`（默认 1，上限 64）。
- **红线发现与修复**：TF32 下批量前向因 batch 形状不同产生 rounding 微差，长视频被 Kalman/关联放大成 trackId 切换（本轮实测 15000 帧中 3623 帧漂移；候选携带 `trackSegmentId`、退回配对依赖它，属语义问题）。**batch>1 现在内部强制 FP32 + cuDNN 确定性**（禁 TF32/benchmark），漂移收敛为 0（实测 bbox 差 ≤0.0005px、关键点 ≤0.0022px，trackId/人数/候选逐条一致）。
- **缓存指纹 bump**：新增 `poseBatch / poseInfer / posePrecision` 三键，**存量观测缓存一次性失效**（重算）；pipeline 不入指纹（结果不变，开关间共享缓存）；批量与串行缓存隔离。
- **测量/回归脚本** `Rules/scripts/bench_tracking.py`（run/diff；`--fresh` 强制重算、`--fp32` 同精度基线、`--limit-sec` 无损截段）；diff 门禁：trackId/人数/候选完全一致 + bbox≤1px 硬门禁，关键点 advisory（5px 仅警告）。
- **测试**：新增 `Rules/tests/test_pose_pipeline.py` 10 用例（帧序/背压/分块/异常传播/缓存隔离/FP32 标志/GPU 冒烟）；全量 184 收集、1 skip 全绿；改动文件 ruff 绿。
- **本轮实测墙钟**（RTX 3080 Laptop，25min 1080p25 HEVC，poseFps=10，15000 帧，tracking 阶段）：串行 201.1s → 出厂配置（batch8+pipeline+FP32）169.1s = **1.19×**；单开 pipeline 190.1s（1.06×）。本机推理瓶颈而非解码瓶颈，PT0 空间有限，更大空间见 PT1-4 / PT1-1（6.4.5）。
- `backend/.env.example` 增加两个开关的注释说明。

---

## 5. 核心问题：事件判定准确率

### 5.1 问题表现（一次实测）

- 某 20 分钟视频产出 **70 条候选 → 70 条全部判 `match`**，且置信度只有 0.85 / 0.95 两档。
- 其中 21 条是 `initial_inside`（首次出现已在区内，本就不满足“进入”）、25 条是 `zone.leave`（语义方向相反），仍全部判 `match`。
- 抽检全分辨率帧：门框**没有竖直红线、顶部是绿灯**，但模型输出反复写“门框有明显竖直红线亮起”——说明是**模型复述规则文字，而非真正观察画面**。
- 模型还把候选的触发时间当成红灯起始时间（违反提示词里的明确要求）。

### 5.2 根因分层

1. **证据不可见（最关键）**
   - 抽帧被缩到 `576×320`（`maxPixels=196608`，线性缩小约 0.3 倍），一条 5px 的红灯只剩约 1.5px，模型根本看不清。
   - 采样 `2 FPS / 最多 32 帧`，短闪的红灯可能整段漏掉。
2. **规则语义不可直接判定**
   - 原窗口以“进入区”触发点为基准，而不是以“红灯亮起”时刻为基准；红灯延迟亮起时，5 秒窗口对不齐。
   - `initial_inside` / `zone.leave` 等触发类型与“进入 + 未退回”的事件语义不匹配。
3. **模型与提示词偏差**
   - 2B 模型容量不足，倾向于把 `criteria` 复述成“已发生”。
   - 提示词含“未看见不等于没发生”，把小模型推向“看不清也判 match”。
   - 置信度是模型自报、未校准，只有两个值，不能当概率用。
4. **后处理不拦截**
   - `model-decision-v4` 保留模型结论；质量警告只提示不否决；分段聚合是“任一段 match 则整体 match”。
5. **没有评测集 / 指标**，无法量化“改好了没有”。

> P0 已解决第 2、4 层的大部分（红灯 CV 检测 + 确定性判定 + veto + 触发收紧）。**第 1、3、5 层是 P1 / P2 的核心。**

---

## 6. 改进路线图

### 6.1 P0 —— 确定性判定（✅ 已完成）

**做了什么**

- 新增 `CueDefinition`（信号区几何 + 红色阈值参数）与 `RunRequest.cues`、`EventDefinition.cueIds`。
- 新增 `Rules/src/ultra_rules/vlm/cues.py`：在绘制 ROI 内做红色掩码 + 滞回，输出红灯亮/灭区间；含独立缓存；`cli calibrate` 输出红占比时间序列、直方图、Otsu 建议阈值、峰值帧 RGB 统计。
- 新增 `Rules/src/ultra_rules/vlm/decision.py`：确定性判定（红灯状态 / 是否退回 / 轨迹是否连续），并做“进入—退回”配对抑制，避免一次进出被重复计数。
- `understand()` 改为确定性优先，Qwen 模型改为**懒加载**（纯确定性结果不加载模型）。
- 前端新增“画信号区”工具、“信号区阈值”面板、入口方向翻转；规则编辑器支持绑定信号区。

**如何校准红灯阈值（运维步骤）**

```bash
PYTHONPATH=Rules/src python -m ultra_rules.vlm.cli calibrate \
  --input <视频文件> --cue '{"id":"red","points":[[x1,y1],[x2,y2],...]}'
```

把输出的 `suggestedMinOnRatio` 等建议值填进前端「信号区阈值」面板。**注意**：校准需要一段“红灯确实亮起”的片段；若只有“灭”的样本，只能确定“灭”的基线上限，无法验证召回。

### 6.2 P1 —— 证据、提示词与模型（🔶 部分完成：②③ 已落地并 GPU 实测，①④⑤ 待做）

**目标**：让模型“看得见、说得出证据、判得稳”。

1. **高分辨率 ROI 抽帧**
   - 输入改为多路：全帧（上下文）+ 门框 ROI 原分辨率/2x + 目标裁剪。
   - 把 `maxPixels` 提高到 786432（或按 ROI 单独给预算）；红灯线宽 ≥3px、目标高度 ≥120px。
   - 围绕“红灯亮起时刻”以 4–6 FPS 抽 5–6 秒，`maxFrames` 提到 48–64（而不是固定 2 FPS 覆盖 8 秒）。
   - 相关代码：`Rules/src/ultra_rules/vlm/sampling.py` 的 `windows / choose_rows / build_sample`。

2. **结构化条件输出，代码做最终决策**（✅ 已完成）
   - 模型对每个 `criteria` 输出 `{index, status: yes/no/unknown, evidenceTimeSec}`；`results.py` 的 `decision_from_conditions` 按合取真值表算 decision（任一 no→no_match；全 yes→match；否则 uncertain），模型自由文本结论降级为 `modelDecision` 对照，两者不一致记 `condition_decision_mismatch`。
   - 证据校验：`evidenceTimeSec` 越窗 / 约等于 `triggerSec`（|差|<0.15s）/ 为 null 时把该条件降级 `unknown` 并记警告；`conditions` 缺失或数量不符则整体回退采信 `modelDecision`（`conditions_unavailable`），保证旧模型/坏格式不崩。
   - `apply_answer` 把 `evidenceTimeSec` 映射到最近采样帧 `frameIndex`（`sampleTimesSeconds`+`sourceFrameIndices`）供前端回放；`qwen_worker` output 带 `eventCriteria` 快照供前端按 index 映射条件文本。
   - **未做**：条件的 ROI/bbox 输出（2B 模型不稳定，依赖 P1①）。相关代码：`prompting.py`、`results.py`、`contracts.py`（`ModelAnswer/ConditionCheck`）、`qwen_worker.py`、前端 `utils/vlm.ts` + `ui/vlm/VlmPanel.tsx`。

3. **提示词改为“默认否认”**（✅ 已完成，`PROMPT_VERSION=target-event-described-json-v6`）
   - 删除“未看见不等于没发生”，改为默认否认：只有清晰看到条件所需现象才认定发生；固定 ROI 线索在采样完整覆盖下“未出现”作为 no 证据，仅遮挡/窗口外才 uncertain；红灯亮起时刻以画面为准，不得用 `triggerSec` 代替。
   - 禁止描述衣着/外貌/体型/发色等与判定无关属性（目标身份由粉红框标识）；`videoDescription` 收紧到 2–4 句、`reason` 逐项交代各必要条件的可见依据。
   - 输出字段顺序 `videoDescription, conditions, decision, confidence, reason`；加 3 个 few-shot（match / no_match / uncertain）。
   - **GPU 实测**：输出不再出现衣着词、`videoDescription` 字数降约 3–4 倍且关键动作/时序保留、conditions↔criteria 映射与合取决策生效、`evidence_time_equals_trigger` 降级在真实数据触发。

4. **模型升级与集成**
   - 至少用 4B 做对比（16GB 显存可跑 4B BF16，此前峰值约 9GB）；8B 需量化/卸载。
   - 对同一候选做 2–3 次不同帧子集/提示的 self-consistency，或加一个 critic 二次核验。
   - **不要指望只换大模型解决“红灯只有 1.5px”的问题——先修输入证据。**

5. **置信度校准 + 拒答阈值**
   - 在标注集上校准置信度（ECE / Brier），设“低置信度转人工”的阈值。

### 6.3 P2 —— 评测与生产化（⬜ 待做）

**目标**：用数据证明“达到上线标准”，并可持续运营。

1. **建立标注评测集**
   - 从历史任务里抽 300–500 条候选，按触发类型、遮挡、灯光状态分层；做双重标注。
   - 每条标注两个层面：**事件级 label** + **逐条件 label**（进入 / 红灯亮 / 5 秒内未退回）。
2. **指标（不要只看 accuracy，类别不均衡会失真）**
   - 事件级：precision / recall / F1 / PR-AUC、每小时误报数、abstain（不确定）率。
   - 条件级：尤其“红灯是否亮”的单条件准确率。
   - 置信度校准：ECE / Brier。
3. **上线验收门槛（建议值，需业务确认）**
   - 真实事件 recall ≥ 95%；precision ≥ 90%，或误报 ≤ 1 次 / 2 小时；abstain ≤ 10%，且不确定样本全部进人工复核；每条结果可回放到证据帧。
4. **人工复核闭环 + 监控**
   - 复核队列、纠错写回、按相机/机位监控红灯检测漂移；规则 / 提示词 / 模型版本化。

### 6.4 PT —— 性能与吞吐专项（⬜ 待做 / 提案；与 P0/P1/P2 正交，不占用既有编号）

> **定位**：P0/P1/P2 解决“判得准”，PT 解决“判得快”。内部优先级 PT0（低成本先行）→ PT1（中等改动）→ PT2（高成本谨慎）。
> **红线（继承 4.1 / 7.3）**：一切提效以“不改变判定语义与准确率”为前提；不改 `poi.py` 候选生成；红灯仍 CV 检测、确定性优先 + VLM 仅复核、退回语义不变；默认保守串行，并发/批大小只能以环境变量/参数开启；不预设加速比，量化一律以基线实测为准。

#### 6.4.1 瓶颈量化模型与基线采集（PT0 第一步，不改代码）

**变量与默认值**

| 符号 | 含义 | 现默认值 | 来源 |
|---|---|---|---|
| D | 视频时长（小时） | — | `request.json` 的 `video.durationSec` |
| f_p | 姿态采样率 `poseFps` | 10 | `contracts.AnalysisParameters` |
| f_s / F | VLM 抽帧率 `sampleFps` / 上限 `maxFrames` | 2 / 32 | 同上 |
| C | 候选数 / 小时 | 待实测（量级参考：§5.1 一次实测 20min→70 条，外推约 210/h，**非正式基线**） | `candidates.jsonl` |
| E / W | 每候选绑定启用事件数 / 每事件窗口数 | W 由 `preSeconds+postSeconds` 按 16s 分段、重叠 2s 决定 | `sampling.windows` |
| p | 确定性短路率（不调 VLM 的比例） | 待实测 | `results.jsonl` 的 `inferenceStatus=deterministic` 占比 |
| t_vlm / t_sample | 单窗口 VLM 生成 / 抽帧耗时 | 待实测 | `attempts[].inferenceSeconds` |
| t_pose / t_dec | 单帧姿态推理 / 解码分摊耗时 | 待实测 | 基线测量 |

**估算公式（现状串行；用于定位瓶颈，不是实测结论）**

```
N_pose = D×3600×f_p                      # 姿态推理帧数
N_dec  ≈ D×3600×fps_source               # 需解码的总帧数（帧间编码须全解）
T_tracking ≈ N_pose×(t_dec分摊 + t_pose + t_io)
N_vlm ≈ C×D×E×W×(1−p)                    # VLM 调用次数（规则数在此线性放大）
T_understand ≈ N_vlm×(t_sample + t_vlm)  # batch=1
T_端到端 ≈ T_segment + T_tracking + T_cues + T_integrity + T_understand
链路A：T_A ≈ D×3600×f_a×(t_seek+t_encode+t_rtt+t_infer+t_event)
       GPU 占空比 ≈ t_infer / 单帧总耗时
流水线/批处理后：T_tracking ≈ max(N_dec×t_dec, N_pose×t_pose/batch有效加速) + 队列填充
```

**基线表模板（⬜ 待实测：跑一次 ≥1h 真实视频填写，2B/4B 分别测）**

| 阶段 | 单位耗时 | GPU 利用率 | 显存峰值 | 备注 |
|---|---|---|---|---|
| segment | 待实测 | 待实测 | 待实测 | |
| tracking·解码 | 待实测（s/千帧） | 待实测 | — | 软解 / NVDEC 分别测；本机解码非瓶颈（6.4.5） |
| tracking·姿态 | 待实测（ms/帧，纯推理） | 待实测 | 待实测 | 全阶段分摊 13.4ms/帧（串行）/ 11.3ms/帧（出厂配置），本轮实测（含解码/写盘） |
| cues / integrity | 待实测 | — | — | 含重复解码成本 |
| understanding·抽帧 | 待实测（s/窗口） | — | — | `build_sample` |
| understanding·生成 | 待实测（s/窗口） | 待实测 | 待实测 | |
| 链路A·单帧往返 | 待实测（ms/帧） | 待实测 | 待实测 | seek/编码/推理分段计时 |

> 本轮已部分填写（6.4.5）：tracking 全阶段墙钟 串行 201.1s / 仅 pipeline 190.1s / batch8+pipeline+FP32 169.1s（25min 视频、15000 帧）；其余单元格待实测。

**采集方法（全部用现有产物，无需新埋点即可起步；命令见 7.2）**

- 阶段墙钟：`status.json` 的 `startedAt/finishedAt/updatedAt` + `run_dir/{phase}.log`；`progress.json` 的 `trackingSeconds / actualPoseFps / sampledFrames / trackingFraction`（`observations.py` 已写）。
- VLM 单窗口耗时与显存：`results.jsonl` 各 `segments[].attempts[]` 的 `inferenceSeconds / inputTokens / outputTokens / peakAllocatedGiB / peakReservedGiB`（`qwen_worker` 已写）；模型加载 `runtime.json.loadSeconds`。
- 短路率 p：`results.jsonl` 按 `inferenceStatus`（deterministic / succeeded / dry_run）计数；`summary.json` 的 `resultCount/failedRequests/decisions`。
- 候选密度 C：`candidates.jsonl` 行数 ÷ `video.durationSec`。
- GPU 利用率/显存时间线：任务运行期间 `nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 1`。
- 链路 A 分段计时：后端 `pred.telemetry`（preprocessMs/inferenceMs/postprocessMs）已有；前端 seek/编码/往返需临时 `performance.now` 埋点（⬜ 提案，测量后移除或留作调试开关）。
- 缓存命中：`observations.ref.json.cacheHit`。

#### 6.4.2 分层方案（每项 ⬜ 待做 / 提案；收益为方向性判断，倍数待实测）

**PT0 低成本先行**

| 编号 | 方案 | 涉及文件 | 前置依赖 | 收益 / 风险 |
|---|---|---|---|---|
| PT0-1 | ✅ 已实现（默认关；实测见 6.4.5）批量姿态推理：batch predict 前向 + 按帧序独立 BYTETracker 关联；**batch>1 强制 FP32 确定性**（TF32 批量在长视频上有 ID 漂移，已修复） | `Rules/src/ultra_rules/pose_adapter.py`、`vlm/observations.py` | 基线 + 逐条一致性回归；观测缓存指纹 bump（已做） | 高（GPU 占空比↑）/ 中（须验证 trackId 逐帧一致） |
| PT0-2 | ✅ 已实现（默认关；比特一致）解码/推理流水线重叠：解码线程 → 有界帧队列 → 推理消费；HEVC 硬解（复用 `video_transcode.transcode_gpu` 的 cuvid/NVDEC 模式）仍待做 | `vlm/observations.py`、`vlm/media.py` | 可与 PT0-1 独立 | 高 / 低（队列上限控内存；NVDEC 需 NVIDIA） |
| PT0-3 | ⬜ 本轮缓做：抽帧去重复用。**缓做原因**：`cue.sampleFps` / `zone.sampleFps` 各自独立且有状态采样调度（`media.sampled`），integrity 还依赖**已完成**的 observations.sqlite 做遮挡排除；朴素共享解码会改变采样点与 CV 结果（红线）。需多消费者采样器 + 阶段编排重构，升格为 PT1 级改动 | `vlm/runner.py` 阶段编排、`vlm/cues.py`、`vlm/integrity.py`、`vlm/media.py` | 多消费者采样器设计评审 | 中（长视频显著）/ 中 |
| PT0-4 | 确定性短路率提升：在既有“确定性优先”框架内统计 p、识别可安全扩大的可测量信号（不得放宽否决、不得改退回语义） | `vlm/decision.py`、`vlm/qwen_worker.py`（门禁统计） | **P2 标注集**验证不降准确率 | 随规则数↑ / 高（语义红线，无评测集不动） |
| PT0-5 | 结果级缓存复用：按已有 `requestFingerprint` 跨 run 复用 understanding 结果（现状仅 run 内断点续跑） | `vlm/qwen_worker.py`、`vlm/storage.py` | 指纹含模型/提示词版本，变更即失效 | 中（改规则重跑场景）/ 低 |

> 事实注记：观测缓存指纹不含规则/POI（`observations.py`），**改规则重跑 tracking 已命中缓存**，仅重算候选与判定——这是现有设计，不是待做项。

**PT1 中等改动**

| 编号 | 方案 | 涉及文件 | 前置依赖 | 收益 / 风险 |
|---|---|---|---|---|
| PT1-1 | understanding 预取/批生成：生产者-消费者重叠 `build_sample`(CPU) 与 generate(GPU)；进一步 batch generate（padding + 逐序列 `JsonAnswerStop`） | `vlm/qwen_worker.py`（`Qwen.infer / understand / _process_candidate`） | **P1① 证据方案冻结**（maxPixels/maxFrames 会变）+ 显存门禁 | 高（多规则主成本）/ 中（显存峰值随 batch×帧数×分辨率；batch 数值微差须过一致性回归） |
| PT1-2 | tracking 写盘/候选聚合移出推理线程（sqlite/gzip/PoiTracker 消费者线程） | `vlm/observations.py` | PT0-2 | 中 / 低 |
| PT1-3 | 按时间分片并行跟踪：分片独立 ByteTrack，边界需 ≥ `maxGapSeconds` 的重叠预热 + 跨界轨迹重关联 | `vlm/observations.py`、`vlm/runner.py` | 评测集验证候选不重不漏 | **默认不做**：ID 切换/轨迹断裂会波及 `poi.py` 候选可靠性（红线）；单 GPU 下分片主要收益是崩溃恢复而非吞吐 |
| PT1-4 | 链路 A 改后端离线批处理 job：顺序解码（消灭逐帧 seek）+ batch 推理 + 按序事件聚合；复用 vlm_jobs 队列/进度/取消与 cuvid 解码；前端改“提交 job→轮询→回放叠加存储预测”；直播 `/ws/infer` 保留不动 | 新增 `backend/app` 分析 job 模块、`video_api.py`、前端 `VideoTab.tsx` / `utils/videoAnalysis.ts` | PT0 基线；新 API 契约先评审（7.3-6） | 最大（数量级）/ 改动面最大 |

**PT2 高成本 / 谨慎**

| 编号 | 方案 | 涉及文件 | 前置依赖 | 收益 / 风险 |
|---|---|---|---|---|
| PT2-1 | 跨 run 并发：`vlm_api.MAX_CONCURRENT`（现常量 1）改环境变量 + 显存门禁；`worker.lock` 按阶段/资源细化 | `backend/app/vlm_api.py`、`vlm_jobs.py`、`vlm/runner.py` | 显存基线（4B BF16 此前峰值约 9GB，见 6.2④ 既有记录；10GB 级卡 4B+YOLO 并发不可行，2B 余量待实测） | 中 / 高（OOM 会杀任务） |
| PT2-2 | 与实时控制台 GPU 隔离：离线 job 与 `/ws/infer` 争用同一 `_predict_lock`/同卡；提案独立进程/stream + 显存预算 + 控制台抢占（离线批处理可暂停让路） | `backend/app/infer_runtime.py`、`main.py`、新 job 模块 | PT1-4；控制台帧延迟 P95 基线 | 中 / 高 |
| PT2-3 | 跨分片事件去重合并（仅当 PT1-3 落地）：在候选输出之后的合并层按 (POI, 时间窗, 外观) 去重，不改 `poi.py` 生成逻辑 | 新合并层 | PT1-3 + 评测集 | 默认不做 |

**单 GPU 现实约束（所有 PT 项的共同前提）**

- **显存**：Qwen bf16 权重 + KV/激活 + 帧张量；峰值 ≈ f(batch × maxFrames × 分辨率)。P1① 提高分辨率/帧数后 PT1-1 批处理空间被压缩——**先冻结证据方案再定批大小**。
- **SM 争用**：同卡并发 kernel 不保证线性加速；NVDEC/NVENC 是独立硬件单元，解码硬解化是“真并行”，与推理争用最小。
- **与实时控制台共用 GPU**：现状 `_predict_lock` 全局串行是事实上的隔离（离线与直播互斥）；提效后需显式抢占策略，验收以控制台帧延迟 P95 不劣化为判据。
- **“并行不一定线性加速”判据**：并发总吞吐 < 单独吞吐之和的 ~90%，或显存峰值超预算，或控制台延迟劣化，或一致性回归不过 → 回退串行。阈值待基线后校准。

#### 6.4.3 落地路线与验收（每阶段默认保守串行、可开关）

| 阶段 | 目标 | 改动清单 | 验收指标 | 回滚开关（提案命名，默认值保守） |
|---|---|---|---|---|
| PT0 | tracking 墙钟下降、GPU 占空比上升（目标值待基线后定） | PT0-1/2/3/5 + 基线表 | 每小时视频端到端墙钟；`actualPoseFps`（帧/秒）；nvidia-smi GPU 利用率；显存峰值；**trackId/人数/候选逐条一致 + bbox≤1px**（基线选择见下文） | `ULTRA_VLM_POSE_BATCH=1`、`ULTRA_VLM_PIPELINE=0`、`ULTRA_VLM_SHARED_DECODE=0` |
| PT1 | understanding 吞吐↑、链路 A 离线化 | PT1-1/2/4 | N_vlm 调用数与短路率 p；单窗口耗时；端到端墙钟；**results.jsonl 逐条 decision/conditions.status 与串行基线一致**（evidenceTimeSec 容差 ±1 采样帧） | `ULTRA_VLM_PREFETCH=0`、`ULTRA_VLM_GEN_BATCH=1` |
| PT2 | 多任务/控制台共存 | PT2-1/2 | 并发总吞吐 vs 串行和；控制台帧延迟 P95；显存峰值不超预算 | `ULTRA_VLM_MAX_CONCURRENT=1`、`ULTRA_GPU_VRAM_BUDGET_GIB` |

**一致性回归方法（脚本已落地：`Rules/scripts/bench_tracking.py diff`）**：固定视频 + 规则快照，先跑基线 run（保留 `candidates.jsonl` / observations / `results.jsonl`），开启开关重跑后按帧/按 resultId 逐条 diff。门禁：**trackId / 人数 / 候选必须完全一致**；bbox 硬门禁 ≤1px；关键点 advisory（≤5px，超限仅警告）。**基线选择（本轮实测教训）**：pipeline 模式结果比特一致，可直接对生产串行基线；batch 模式强制 FP32，**必须对同精度串行基线**（`bench_tracking.py run --fp32 --fresh`），不得对 TF32 生产基线——TF32 批量本轮实测有 3623/15000 帧 ID 漂移，FP32 批量漂移 ≤0.0005px。understanding 层回归仍按 `resultId` 比 decision/conditions.status（greedy `do_sample=False`，同输入应同输出）；数值微差导致翻转 → 判不通过、回退。

#### 6.4.4 与 P1 / P2 的依赖与排序

- **PT0（tracking 侧）与证据方案正交，可先行**；PT0-4（短路扩展）与 PT1-3（分片跟踪）语义敏感，**必须等 P2 评测集**。
- **PT1-1（understanding 批处理）排在 P1① 证据冻结、P1④ 模型选型之后**：maxPixels/maxFrames/模型变化都会改变显存与耗时结构，先并行化再改证据会返工。
- P1④ self-consistency 会把 N_vlm ×2–3、4B 会抬高 t_vlm 与显存——届时 PT1-1 批处理收益更大，但批大小须按新显存基线重定。
- **建议总排序**：PT0 基线测量（不改代码）→ P2 最小标注集 → PT0 工程优化 ∥ P1① 证据 → P1④ 模型对比 → PT1-1 批处理跟进 → PT1-4 链路 A 离线化（可与 PT0 并行立项，契约先评审）→ PT2 按需。
- **本轮实测后的排序修正**：PT0 出厂配置在本机仅 1.19×（推理瓶颈，6.4.5），不是长视频的主收益项；**PT1-4（基础分析离线批处理，消灭逐帧 seek）与 PT1-1（understanding 批处理，多规则线性成本）优先级应提前**；PT0 开关保持默认关，经评测集回归后再议默认开启。

#### 6.4.5 本轮实测结果（PT0 第一期；其余项仍 ⬜ 待做）

**条件**：RTX 3080 Laptop（10GB）/ 25 分钟 1080p25 HEVC 监控录像 / poseFps=10 / 15000 采样帧 / tracking 阶段墙钟（`progress.json.trackingSeconds`）；测量脚本 `Rules/scripts/bench_tracking.py`，严格单链串行无 GPU 争用。

| 配置 | 墙钟 | vs 生产串行 | 一致性（对相应基线） |
|---|---|---|---|
| 串行（默认 TF32，生产基线） | 201.1s | 1.00× | 基线 |
| 仅 pipeline=1 | 190.1s | 1.06× | ✅ 比特一致（0.0 漂移） |
| batch8+pipeline（TF32，已废弃的行为） | 156.9s | (1.28×) | ❌ 3623 帧 trackId 漂移 |
| 串行 FP32（同精度基线） | 237.2s | 0.85× | FP32 基线 |
| batch8 FP32 | 187.6s | 1.07× | ✅ 0 ID 漂移，bbox≤0.0005px |
| **batch8+pipeline FP32（出厂配置）** | **169.1s** | **1.19×** | ✅ 0 ID 漂移，bbox≤0.0005px、kp≤0.0022px、候选 14=14 |

**结论与发现**

1. **pipeline 比特一致**，可放心开启；本机仅 1.06×（解码不是瓶颈），在解码瓶颈机器/软解 HEVC 上收益会更大。
2. **batch 必须配 FP32**（已内部强制）：TF32 批量微差被跟踪器放大为长视频 ID 切换（语义红线）；FP32 后漂移收敛为 0。公平加速（vs 同精度串行）1.40×，vs 生产 TF32 串行 1.19×（FP32 本身慢约 18%）。
3. **本机 tracking 是推理瓶颈**：单开 pipeline 仅 1.06× → PT0 不是长视频主收益项；主收益在 PT1-4 / PT1-1。
4. **GPU 利用率 / 显存峰值本轮未采集**（待实测，采集方法见 6.4.1）；120s 截段归因：串行 17.55s / 仅 pipeline 16.89s / 仅 batch8 15.46s（TF32）。

---

## 7. 开发者指引

### 7.1 关键代码地图

**后端（Python，`Rules/` 是纯算法库，`backend/` 是 FastAPI 服务）**

| 路径 | 职责 |
|---|---|
| `Rules/src/ultra_rules/vlm/contracts.py` | 全部数据模型（Poi / CueDefinition / EventDefinition / RunRequest 等） |
| `Rules/src/ultra_rules/vlm/models.py` | 本地 Qwen3-VL 权重发现与校验（默认根目录为项目内 `models/`，不读项目外路径） |
| `Rules/src/ultra_rules/vlm/observations.py` | 跟踪阶段：YOLO-Pose + ByteTrack → observations.sqlite；观测缓存指纹不含规则/POI（改规则重跑命中缓存） |
| `Rules/src/ultra_rules/vlm/runner.py` | 阶段协调子进程：`worker.lock` 互斥、阶段串行/超时/取消、status 写入 |
| `Rules/src/ultra_rules/vlm/media.py` | 统一解码与时间基：`probe / decoded`（全帧软解）/ `sampled`（按 fps 采样）；tracking/cues/integrity 各自调用（重复解码点，见 3.4） |
| `Rules/src/ultra_rules/vlm/integrity.py` | 完整性 CV 检测：抽样与基线比对 → 变化区间 |
| `Rules/src/ultra_rules/vlm/storage.py` | JSON/JSONL 读写、指纹 `digest`、`progress.json`、缓存工具 |
| `Rules/src/ultra_rules/pose_adapter.py` | YOLO-Pose 单帧推理适配（每文件独立 tracker；PT0-1 批量化改造点） |
| `Rules/src/ultra_rules/vlm/poi.py` | 候选生成：PoiTracker 状态机 |
| `Rules/src/ultra_rules/vlm/cues.py` | 红灯 CV 检测 + `calibrate()` 校准工具 |
| `Rules/src/ultra_rules/vlm/decision.py` | 确定性判定（红灯 / 退回 / 轨迹连续性） |
| `Rules/src/ultra_rules/vlm/sampling.py` | 抽帧窗口、选帧、标注预览图 |
| `Rules/src/ultra_rules/vlm/prompting.py` | VLM 提示词 |
| `Rules/src/ultra_rules/vlm/qwen_worker.py` | Qwen 模型封装 + `understand()` 编排 |
| `Rules/src/ultra_rules/vlm/results.py` | 解析 / 裁决（`decision_from_conditions` 合取真值表 + 证据校验降级）/ 聚合 |
| `backend/app/vlm_api.py` | VLM HTTP 接口 + 任务调度器（`_pump`，`MAX_CONCURRENT=1` 单 worker 槽，队列落盘可重启恢复） |
| `backend/app/vlm_jobs.py` | 任务启动/状态/取消：`launch/status/cancel`、owner 进程存活判定、reap 线程 |
| `backend/app/event_rules.py` | 规则库 CRUD + ManagedRun + resolve |
| `backend/app/library_api.py` / `video_api.py` | 记录管理（展示名/处理时长 `durationSec`）/ 视频预览转码（NVDEC+NVENC、SHA-256 缓存、`ULTRA_VIDEO_MAX_CONCURRENT`、禁用 CPU 回退） |
| `backend/app/video_transcode.py` | 隔离 GPU 转码 worker：cuvid 硬解 → `h264_nvenc`（PT0-2 硬解化可复用此模式） |
| `backend/app/main.py` | FastAPI 入口：VLM/视频/库路由 + 实时推理接口（`/api/models` `/api/state` `/api/params` `/api/system/stats` `/ws/infer`）+ 启动 `lifespan` 自动选设备/加载默认模型 |
| `backend/app/infer_runtime.py` / `devices.py` / `state.py` / `config.py` / `system_stats.py` / `logging_store.py` / `schemas.py` | 经典 CV 实时推理运行时（模型加载、设备、参数、启动自选配置、系统状态、日志、Schema）；`_predict_lock` 全局推理串行、每流独立 tracker 上下文（PT2-2 隔离改造点） |

**前端（`frontend/src/`）**

| 路径 | 职责 |
|---|---|
| `ui/viewer/tabs/VideoTab.tsx` | 视频选择 + 预览 + 基础检测分析 + 画线/区/信号区 + 挂载 VlmPanel |
| `ui/viewer/widgets/EventEditOverlay.tsx` | 画 POI（含入口方向箭头） |
| `ui/control/ControlPanel.tsx` | 实时控制台左栏（引擎/NMS/类别过滤/模型；挂载读 `/api/state` 回显启动自选的设备与模型） |
| `ui/monitor/MonitorPanel.tsx` | 实时控制台右栏（系统状态/配置/计数事件/告警/日志） |
| `ui/vlm/VlmPanel.tsx` | 语义分析控制 + 结果展示（含逐条件证据 + 点击跳转 `evidenceTimeSec`） |
| `ui/vlm/EventRulesEditor.tsx` | 规则编辑 + POI/信号区绑定 |
| `ui/vlm/CueSettings.tsx` | 信号区阈值编辑 |
| `ui/vlm/AnalysisParameters.tsx` | 分析参数（模型 / 姿态设备 / 抽帧等） |
| `ui/library/LibraryPage.tsx` | 记录与视频管理（单页内切换） |
| `store/useConsoleStore.ts` | 全局状态（引擎/参数/模型/预测/日志/告警/ROI/计数/事件/`eventsConfig`） |
| `store/useUiStore.ts` | 控制台 / 库 视图切换 |
| `utils/vlm.ts` / `eventRules.ts` / `vlmParameters.ts` | VLM 请求构造与类型 |
| `utils/videoAnalysis.ts` / `events.ts` / `roi.ts` / `useEventTracker.ts` / `useSystemStats.ts` / `draw.ts` / `configIO.ts` | 实时视频分析（`VideoInferenceSession` 单帧在途）、事件追踪、ROI、画图、配置导入导出 |
| `utils/videoPreview.ts` | 兼容预览：上传 → 轮询转码进度 → 取播放 URL → 释放（HEVC 回退入口，见 4.5） |

### 7.2 运行与验证命令

```bash
# 后端 + 规则算法测试
cd <项目根目录>
PYTHONPATH=backend:Rules/src python -m pytest backend/tests Rules/tests

# 新 WSL 一键初始化（Miniconda/Node/依赖/权重/启动）
bash deploy.sh

# WSL 单环境依赖（一个 requirements 同时装 torch/ultralytics/transformers）
pip install -r backend/requirements-wsl.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 然后把 backend/.env 的 VLM_PYTHON 指向同一解释器

# 后端 lint
python -m ruff check backend/app Rules/src

# 前端类型检查 / 测试 / lint / 构建
cd frontend && npm run build && npm test && npm run lint

# 启动（后端 8001 / 前端默认端口，示例 5100）
PATH=<conda>/envs/ultraconsole/bin:$PATH ./start.sh 5100

# 模型权重位置（项目内，默认加载；不需要设置 VLM_MODEL_PATH）
# models/Qwen3-VL-2B-Instruct/   ← 默认模型
# models/Qwen3-VL-4B-Instruct/
# Qwen 独立解释器写在本机 backend/.env 的 VLM_PYTHON（不入库）

# 红灯阈值校准
PYTHONPATH=Rules/src python -m ultra_rules.vlm.cli calibrate \
  --input <视频> --cue '<信号区JSON>'

# ---- 性能基线采集与一致性回归（PT0，见 6.4.1 / 6.4.3；<run_dir> 为一次真实任务目录）----
# GPU 利用率/显存时间线（任务运行期间执行）
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 1 > gpu-trace.csv

# 阶段吞吐与墙钟（现有产物，无需埋点）
cat <run_dir>/progress.json          # trackingSeconds / actualPoseFps / sampledFrames
cat <run_dir>/runtime.json           # Qwen loadSeconds / dtype / gpu
cat <run_dir>/observations.ref.json  # cacheHit

# VLM 单窗口耗时/显存分布 + 确定性短路率
python - <<'PY'
import json, pathlib, statistics
run = pathlib.Path("<run_dir>")
rows = [json.loads(l) for l in (run/"results.jsonl").read_text().splitlines() if l.strip()]
att = [a for r in rows for s in r.get("segments", []) for a in s.get("attempts", [])]
secs = [a["inferenceSeconds"] for a in att]
vram = [a["peakAllocatedGiB"] for a in att]
det = sum(1 for r in rows if r.get("inferenceStatus") == "deterministic")
print("windows:", len(secs),
      "median_s:", statistics.median(secs) if secs else None,
      "p95_s:", sorted(secs)[int(len(secs)*0.95)] if secs else None)
print("vram_peak_gib:", max(vram) if vram else None,
      "deterministic_ratio:", det/len(rows) if rows else None)
PY

# 候选密度（条/小时）
python - <<'PY'
import json, pathlib
run = pathlib.Path("<run_dir>")
n = sum(1 for l in (run/"candidates.jsonl").open() if l.strip())
d = json.loads((run/"request.json").read_text())["video"]["durationSec"]
print("candidates_per_hour:", n / (d/3600))
PY

# 一致性回归：两次 run 的 results.jsonl 按 resultId 逐条比对 decision/conditions
python - <<'PY'
import json, pathlib
def load(p):
    return {r["resultId"]: r for r in
            (json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip())
            if r.get("resultId")}
a, b = load("<baseline_run>/results.jsonl"), load("<candidate_run>/results.jsonl")
diff = [k for k in a.keys() & b.keys()
        if a[k]["decision"] != b[k]["decision"]
        or [c.get("status") for c in a[k].get("conditions", [])]
        != [c.get("status") for c in b[k].get("conditions", [])]]
print("mismatch:", len(diff), diff[:10],
      "only_a:", len(a.keys()-b.keys()), "only_b:", len(b.keys()-a.keys()))
PY

# PT0 tracking 基线测量与逐帧一致性回归（见 6.4.1 / 6.4.3 / 6.4.5）
python Rules/scripts/bench_tracking.py run --input <视频> --name serial
python Rules/scripts/bench_tracking.py run --input <视频> --name fast --batch 8 --pipeline
python Rules/scripts/bench_tracking.py diff --a tmp/bench-tracking/serial --b tmp/bench-tracking/fast
# batch 内部强制 FP32：其回归基线必须是同精度串行（不得用 TF32 生产基线）
python Rules/scripts/bench_tracking.py run --input <视频> --name serial-fp32 --fp32 --fresh
# 归因/快速迭代：--fresh 强制重算（独立缓存），--limit-sec 无损截段
python Rules/scripts/bench_tracking.py run --input <视频> --name clip --limit-sec 120 --pipeline --fresh
```

### 7.3 给后续 AI / 开发者的约定

1. **候选时间点已可靠**：不要改动 `poi.py` 的候选生成逻辑。
2. **红灯语义已定**：门框竖直红色 LED 灯带；顶部绿灯忽略；用 CV 检测，不是让 VLM 看。
3. **判定策略已定**：确定性优先，VLM 只复核不确定样本。
4. **退回语义已定**：入口线反向跨回，5 秒窗口从红灯亮起时刻算起。
5. **Schema 兼容**：给 `EventDefinition` 等新增字段时，`event_rules.py` 的 `resolve()` 里读取必须用 `.get()` 容错（历史 `event-rules.json` 可能缺新键）。
6. **先解释方案再动工**：涉及功能设计 / UI 交互 / 阈值语义等有多种路径的改动，先给方案让用户拍板。
7. **验收要能量化**：任何“准确率提升”都要有标注集 + precision/recall/F1 支撑，不能只凭样例感觉。
8. **资源路径必须自包含**：权重、配置、示例数据的默认路径一律落在项目内；解释器与外部挂载只能通过 `VLM_PYTHON` / `VLM_MODEL_PATH` / `ULTRA_MODELS_DIR` 等环境变量覆盖，代码里不得写死家目录或机器专属绝对路径。
9. **性能优化红线（PT 专项）**：提效不得改变判定语义与准确率；不改 `poi.py` 候选生成；默认保守串行，批大小/并发度/流水线只能以环境变量开启（提案命名见 6.4.3）；每项必须跑“与串行基线逐条一致性回归”（6.4.3）；不预设加速比，量化一律以 6.4.1 基线实测为准，未实测写“待实测”。

---

## 8. 已知边界 / 风险 / 待办清单

- [ ] **红灯阈值仍是默认值**：需用真实“亮/灭”片段跑 `calibrate` 校准，尤其 `minOnRatio / minR / minRG / minRB`。
- [ ] **入口方向**：默认双向 + 退回配对；单向闸门若仍出现“退回被当新进入”的边界，用前端“进入方向”翻转控件设定方向。
- [ ] **红灯起始时刻 vs 触发窗口**：红灯延迟亮起超过 `preSeconds/postSeconds` 时，5 秒窗口会对不齐（P1 以 onset 定窗口后可解决）。
- [ ] **置信度不可信**：当前是模型自报、未校准，不能作为概率或阈值。
- [x] **CUDA 端到端**：本机有 GPU，已验证控制台启动自动选 cuda + 加载 `yolo11m-pose.pt`（`GET /api/state` 回显），并跑通一次真实 VLM 分析（`model-decision-v5`）。
- [ ] **P1② 证据降级 vs 缺失即否定**：`missing_evidence_time` 会把“基于未出现”的 no（`evidenceTimeSec=null`，如“红灯从未亮起”）降级为 unknown，使 `no_match`→`uncertain`，与目标“未出现作为 no 证据”存在张力；待决定是否放宽（`status=no` 且 null 不降级，仅 `yes` 且 null 才降级）。
- [ ] **规则 criteria 混入指令型条目**：如“忽略安检门顶部灯光”不是可检验条件，合取（全 yes→match）下模型无法标 yes 会阻断 match；需在规则库层面把“指令”与“可检验条件”分开。
- [ ] **采样 manifest 长度不一致**：个别窗口 `sampleTimesSeconds` 与 `sourceFrameIndices` 长度不等时 `frameIndex` 映射被跳过（前端 seek 用 `evidenceTimeSec` 不受影响）；待排查 `sampling.py` 或加防御性对齐。
- [ ] **评测集缺失**：P2 前无法量化准确率，属于当前最高优先级的“基础设施”缺口。
- [x] **监控 HEVC 兼容预览触发（本轮修复）**：黑屏 + `Missing JPEG payload` 根因与修复见 4.5；GPU 转码链路本轮实测跑通，预览按 SHA-256 缓存。
- [ ] **长视频吞吐无基线**：链路 A 逐帧 seek 乒乓、链路 B 阶段串行 + batch=1，GPU 利用率低（定性诊断，见 3.4）；倍数/占空比/显存峰值全部待实测（6.4.1 基线表）。
- [ ] **单 GPU 显存争用**：4B BF16 此前峰值约 9GB（6.2④ 既有记录）+ 姿态模型 + 实时控制台同卡；PT2 并发前必须显存门禁与抢占策略。
- [ ] **分片并行跟踪的边界风险**：轨迹断裂/ID 切换可能破坏候选可靠性红线（PT1-3），默认不做；跨分片去重（PT2-3）同。
- [ ] **HEVC 兼容预览依赖 NVIDIA 硬解**：`video_api.transcode` 明确禁用 CPU 回退，无 NVDEC 的机器上长 HEVC 视频无法生成兼容预览（现有设计；PT0-2 硬解化同样受此约束）。
- [x] **PT0 第一期落地（默认关）**：pipeline 比特一致；batch 强制 FP32 消除 ID 漂移；出厂配置 1.19×（本轮实测，见 4.6 / 6.4.5）；存量观测缓存因指纹 bump 一次性失效。**本机已按用户确认在 `backend/.env` 启用 batch8+pipeline**（机器本地配置，不入库）；代码默认仍为关，其他机器/生产默认开启仍待 P2 评测集回归。
- [ ] **批量数值漂移教训**：TF32 批量前向的微差会被跟踪器放大为长视频 trackId 切换（本轮实测 3623/15000 帧）；后续任何批量化（understanding 等）必须先跑长视频一致性回归，必要时配确定性精度。
- [ ] **本机 tracking 推理瓶颈**：PT0 空间有限（1.19×）；长视频主收益在 PT1-4 / PT1-1；GPU 利用率/显存峰值待实测。

---

## 9. 下一步建议（按顺序）

1. **P2 先行打地基**：先做最小标注集 + 指标脚本（哪怕只标 100 条），否则 P1 的每次调整都无法证明好坏。
2. **P1 的“证据”优先于“换大模型”**：先做高分辨率 ROI 抽帧 + 结构化条件输出，再做 4B/8B 对比。
3. **红灯阈值校准**：找一个含真实红灯亮起的片段，跑 `calibrate` 落地阈值。
4. **PT0 基线测量先行（不改代码）**：按 6.4.1 用一次 ≥1h 真实任务填基线表（命令见 7.2），之后所有提效 PR 以该表为对照。
5. **PT0 工程优化可与 P1① 并行**：批量姿态 + 解码重叠 + 抽帧复用与判定语义正交，带开关与一致性回归即可推进；PT0-4 短路扩展、PT1-3 分片跟踪必须等 P2 评测集。**本轮实测修正**：PT0 出厂配置在本机仅 1.19×（推理瓶颈），PT1-4（基础分析离线批处理）与 PT1-1（understanding 批处理）优先级应提前；PT0 开关保持默认关，经评测集回归后再议默认开启。
6. **understanding 批处理（PT1-1）排在证据/模型冻结之后**：避免“先并行化再改证据”返工；self-consistency/4B 落地后按新显存基线重定批大小（见 6.4.4）。
