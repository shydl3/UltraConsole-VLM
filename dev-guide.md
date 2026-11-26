# UltraConsole VLM 语义分析 — 开发指南（Dev Guide）

> 本文档面向两类读者：
> - **非技术读者（业务 / 管理 / 其他部门）**：请先读「第 1 节 执行摘要」和「第 2 节 项目目标」，了解我们在做什么、做到哪了、下一步为什么。
> - **开发者 / 后续 AI 协作**：从「第 3 节」开始是技术细节、代码地图、路线图与验收标准，可直接据此继续开发。

---

## 1. 执行摘要（给非技术读者）

本项目是一个**对视频做“事件语义分析”的工具**：用户上传单段监控视频，或一次导入同一机位的整个文件夹批量处理；在画面上框出“计数线 / 计数区 / 信号区”，定义“事件规则”后，系统自动：

1. 用视觉模型（YOLO 姿态）检测并跟踪画面里的人；
2. 根据“谁在何时进入/离开区域、是否跨线”生成**候选事件时间点**（强行通过类则改为信号区亮起触发）；
3. 按规则类别判定：通行/计数事件用视觉语言大模型（Qwen3-VL）；强行通过由信号区亮起触发、**确定性判定是否退回**；完整性用像素级 CV；
4. 输出带时间点、原因和证据截图的结果，供人工复核。

**当前状态一句话**：候选事件的时间点已经可靠；**唯一的主要短板是“单个事件判断”的准确率还不够高**（存在较多误报 / 漏报），而我们的目标是把准确率提升到**可上线生产环境部署**的水平。

**我们已经做了两件关键事，直接针对误报：**

- **把“门框红灯是否亮起”这类可明确测量的信号，改成用传统图像算法（CV）直接检测**，不再让大模型“看”一个在缩略图里只有一两个像素的红灯——这消除了最大的一类误报（模型把规则文字复述成“红灯亮起”）。
- **把判定改成“确定性优先 + 大模型兜底”**：能用轨迹和红灯状态算清楚的，直接算；算不清楚的才交给大模型复核。

**本轮又做完两件事：**

- **把大模型回答改成“逐条件、带证据时间”的结构化输出**：模型对每个判定条件给“是 / 否 / 不确定 + 证据画面秒数”，由代码按“所有条件都满足才算发生”的合取逻辑**计算**最终结论，不再直接采信模型的一段话；提示词同时改为**“默认否认”**（看不清就不算发生）并**删掉衣着外貌等无关描述**。GPU 实测：描述更短更聚焦、逐条件证据可点击回放。
- **控制台启动即就绪**：后端启动自动选 GPU（CUDA）并加载姿态模型 `yolo11m-pose.pt`，前端无需再手动选设备 / 模型（下拉与按钮仍保留可手动切换）。

**本轮另修复一处监控视频兼容问题**：HEVC/H.265 等浏览器解不出画面的监控录像，过去预览黑屏、点分析报 `Missing JPEG payload`；现在不再自动转码，而是先显示**首帧**供绘制区域，用户按需点「生成预览」（NVDEC 硬解 → H.264；H.264 源则免 GPU 直接 remux）。本轮已用一段 25 分钟真实 HEVC 录像实测跑通转码链路（见 4.5）；首帧/手动预览见 4.9。

**本轮新增“信号区触发规则”**：强行通过类规则不再由计数区候选逐人触发，而是 **信号区亮起才分析**（同一信号区从首次触发起 3 秒内去重）；入口计数线 + 信号区按编号配对，计数区可选辅助；“是否退回”从目标进入时刻起算 5 秒，入口线反向跨线为主判据。计数区 1000 人次、信号只亮 10 次时只判 10 次。规则类别（通行 / 强行通过 / 完整性）同时用于分析前强制校验（见 4.8）。

**本轮完善视频输入与批量处理**：浏览器不能直接播放的源**不再自动转码**，页面先显示首帧供绘制区域，用户按需点「生成预览」；语义分析面板新增「批量处理」，可一次导入一个文件夹内的全部视频，同一机位共用一套区域与规则，任务按队列连续执行（见 4.9）。

**长视频吞吐 / 显卡利用率（现状与下一步）**：本轮已完成性能路线 P0/P1/P2（`optimize.md`，与第 1 节的准确率 P0/P1/P2 无关）：tracking + cues + integrity 合并为只解码一次的 `detect` 阶段；integrity 多区共享解码；确定性证据默认 lazy 预览（只写 manifest，首次查看时生成 JPEG）。真实 30 分钟任务完整 `detect` 实测 **179.7 秒**（P3 I/O 优化后；旧分开阶段约 6 分钟），5 分钟任务 understanding 复算 **0.41 秒**、0 个预览。P3 tracking I/O/缓存优化已完成：tracking 100.8→89.3 秒，逐帧零漂移；TensorRT/NVDEC 仍待评测。tracking 的 batch/pipeline 仍默认保守，实测 1.19×（6.4.5）；下一步是 P4 队列/ETA（见 6.4.7）。所有提效改动以“不改变任何判定结论”为前提。

**接下来要做的（按优先级）：**

- **P1（剩余）**：让大模型“看得更清楚”（高分辨率裁剪、围绕红灯亮起时刻抽帧）、并评估更大更强的模型（4B/8B、self-consistency）。
- **P2**：建立**人工标注的评测集和量化指标**（准确率、召回率、误报率），用数据证明“是否达到上线标准”，并接入人工复核闭环。
- **PT（性能与吞吐专项，与 P1/P2 正交）**：先测基线，再做不改变判定语义的流水线/批处理提效；understanding 批处理排在 P1 证据方案冻结之后，避免返工（见 6.4）。

详细方案见第 6 节。这份文档的目标是：让任何一位开发者（或 AI）接手后，不需要重新摸索，就能按路线图继续推进。

---

## 2. 项目定位与目标

- **定位**：UltraConsole 的二次开发版本。系统由两个子系统组成：
  1. **经典 CV 实时视频控制台**（保留自上游）：实时 YOLO 检测、计数线/区、系统监控、告警引擎、事件日志等（见 `ControlPanel` / `MonitorPanel` / `VideoTab` 的基础分析 / `ws/infer`）。
  2. **VLM 语义事件分析**（本项目核心扩展）：上传视频 → 画 POI/信号区 → 事件规则 → 候选事件 → 按规则类别走确定性判定或大模型判定（强行通过 = 信号触发 + 确定性退回判定）；支持单文件与文件夹批量，同机位共用区域/规则。
- **已移除**：仅 **Webhook 推送**能力（前端 `WebhookCard`/`useWebhookDispatcher`、后端 `webhook.py` 与 `/api/webhook*`、`/api/notify`）。记录/视频管理已改为**单页内切换**（不再弹窗）。
- **目标**：把 VLM 的“事件级判定”准确率提升到**生产环境可部署**水平，即误报、漏报低到业务可接受，且结果可审计、可回放。

> 上游说明：本项目是作者开源项目的衍生作品，保留 LICENSE 与署名；开发在 `llm` 分支，`origin` 指向自有仓库，`upstream` 指向原作者仓库。

---

## 3. 系统如何工作（架构概览）

### 3.1 一次分析任务的完整链路

```
前端(从服务器选择/导入视频 + 画线/区/信号区 + 定义规则)
        │  POST /api/vlm/runs/from-rules
        ▼
后端 event_rules.resolve()  →  RunRequest(request.json)
        │
        ▼
runner.py 阶段序列（子进程）：
  segment
  detect（cues + integrity 同时存在）—— 一次解码、多消费者：
        · tracking  → observations.sqlite + candidates.jsonl
        · cues      → cues.json
        · integrity → integrity.json
    （仅 cues 或仅 integrity 时：tracking → cues / integrity 分开执行）
  understanding 判定：
        · poi_event 规则：POI 候选 → 确定性门禁 / VLM → results.jsonl
        · forced_passage 规则：信号 episode + 通道配对 → 目标/退回 → results.jsonl
        · integrity 规则：完整性变化区间 → CV 判定 → results.jsonl
        │
        ▼
前端轮询 /runs 与 /results，展示结果 + 证据截图（预览可按需生成）
```

> 完整阶段序列为 `segment → detect（或 tracking → cues/integrity）→ understanding`（`runner.py`）。`detect` 由 `vlm/combined.py` 实现，`media.SampleSchedule` 让 pose / cue / integrity 各自按原采样率消费同一解码流；各阶段串行点见 3.4，信号触发规则见 4.8，lazy 预览见 6.4.7。

### 3.2 候选事件如何生成（时间点）

- **POI 触发候选**（`poi_event` 规则）：`Rules/src/ultra_rules/vlm/poi.py` 的 `PoiTracker` 对每个 `(跟踪ID, 跟踪段, POI)` 维护状态机，在跨线 / 进区 / 出区 / 首次在区内时产生**触发（trigger）**；`observations.py` 再按“启用的规则 + 触发类型 + 绑定的 POI”过滤，写入 `candidates.jsonl`。
- **信号触发候选**（`forced_passage` 规则）：不再逐人产生候选。`cues.json` 已有每个信号区的亮灭区间，`qwen_worker._cue_pairs()` 按 **3 秒 episode 去重**（以首次触发为锚）后，每个 episode 生成一条候选，锚定到同编号的入口计数线；一个 episode 只判一次。
- **支撑候选**：信号规则需要 `line.cross / zone.enter / zone.leave / initial_inside` 来选目标与判退回。`observations.candidate_owners()` 会把这些事件以 `eventDefinitionIds=[] + supportRuleIds=[...]` 写入 `candidates.jsonl`，只作为信号路径的输入，不产生 POI 判定。
- **这部分（候选时间点）用户已确认可靠，不要再改动 `poi.py` 的生成逻辑。**

### 3.3 判定如何产生（结果）

- `Rules/src/ultra_rules/vlm/qwen_worker.py` 的 `understand()` 按规则类别分流：
  - **`poi_event`（通行/计数）**：对每个 POI 候选，若规则绑定了**信号区（cueIds）** → 先走 `decision.py` 的确定性判定；确定性给出 `match` / `no_match` → 直接出结果，不调用大模型；无法判定（`uncertain`）→ 交给 VLM 复核（结构化逐条件输出 + 默认否认提示词，见 6.2）。
  - **`forced_passage`（强行通过）**：由 `_cue_pairs()` 从 `cues.json` 生成信号 episode 候选，`_process_cue_candidate()` 做纯确定性判定：选最近正向跨入口线的目标 → 从进入时刻起 5 秒内找反向跨线/离开计数区 → `match`（未退回）/ `no_match`（退回）。默认不调用 VLM；无人通过输出 `no_match + no_subject` 审计记录，**且不抽帧/不生成证据**。有目标的确定性结果默认只写 sample manifest（`previewMode=lazy`），预览在首次查看时由后端生成；VLM/dryRun 仍 eager。详见 4.8、6.4.6 与 6.4.7。
  - **`integrity`（完整性）**：按完整性区间走 CV 判定（`_process_integrity`），不依赖跟踪候选。
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
  segment → detect（cues+integrity）或 tracking → cues/integrity → understanding
  （vlm_api._pump：MAX_CONCURRENT=1，全局同时仅一个 run）

detect（vlm/combined.py: analyze_run）：
  缓存未命中时 media.decoded 全帧软解一次（CPU）
    SampleSchedule 分发给 pose / cue / integrity：
      pose（poseFps）→ 批量 YOLO-Pose（GPU）→ observations.sqlite/candidates
      cue（cue.sampleFps）→ 红灯 ROI ratio → cues.json
      integrity（zone.sampleFps）→ 复用最近 pose persons 做遮挡排除 → integrity.json
  ——解码只做一次；解码与 GPU 推理可通过 ULTRA_VLM_PIPELINE 重叠
  缓存命中时：track() 重放 observations（不推理） + detect_run() 只做 cues/integrity

仅 cues 或仅 integrity：tracking 仍单独跑，紧随其后的 CV 阶段再解码一次

understanding（qwen_worker.understand）：
  逐候选 × 逐绑定事件 × 逐窗口（sampling.windows：16s 分段、重叠 2s）
    确定性结果：build_sample(render=False) 只查 sqlite + 写 manifest（不抽帧/不拼图）
    VLM/dryRun：build_sample(render=True) 抽帧 + 画标注 + 拼图 → Qwen.infer → 解析/落盘
    预览：确定性结果的 previews/*.jpg 由后端 artifact 端点首次访问时 render_preview() 生成
```

- 串行点：`Rules/src/ultra_rules/vlm/runner.py` 阶段 for 循环；`vlm/combined.py` 的 `analyze_run()` / `_track_and_detect()`；`vlm/qwen_worker.py` 的 `_process_candidate()` 窗口 for 循环；`backend/app/vlm_api.py` 的 `MAX_CONCURRENT=1` + `worker.lock`。
- GPU 空闲来源：解码/转换/抽帧/画图/写盘等 CPU 段与 GPU 段交替；batch=1 生成填不满 SM；detect→understanding 之间仍有模型加载空档（Qwen 懒加载一次）。

**多事件规则如何共享计算（成本随规则数的增长关系）**

| 计算 | 是否随事件规则数增长 | 说明 |
|---|---|---|
| tracking 解码 + 姿态推理 | 否 | 全片只跑一次，帧数由 `poseFps` 决定；观测缓存指纹（`observations.py` fingerprint）**不含规则/POI**，改规则重跑命中缓存 |
| 候选生成 `PoiTracker.update` | 否（随 POI 数与人流） | 缓存命中后仍逐帧重算候选，纯 CPU，成本低 |
| cues / integrity CV 扫描 | 否（随信号区/完整性区数量） | 与事件规则数无关 |
| 候选→规则匹配过滤 | 线性但极廉价 | `observations.track` 内 `applicable` 列表推导 |
| understanding VLM 调用 | **线性（主成本）** | 调用数 ≈ Σ候选(绑定启用事件数 × 窗口数) × (1−确定性短路率)；规则越多，每候选复核越多 |

> 结论：加规则不会重跑跟踪与抽帧；真正随规则数线性放大的是 VLM 复核调用次数。因此“多规则长视频”的提效重点是 tracking 流水线化 + understanding 批处理/短路，而不是候选生成（候选生成逻辑按 4.1/7.3 红线不动）。

### 3.5 视频输入、首帧绘制与文件夹批量（前端）

- **单文件**：服务器浏览/导入按引用登记 → 浏览器尝试直接播放；若容器/编码不可直放，显示后端抽取的**首帧**，允许在首帧上直接绘制区域；点「生成预览」后才走 remux/GPU 转码。语义分析不依赖浏览器预览（后端直接读源文件），预览只用于本页播放/姿态分析。
- **文件夹批量**：语义分析面板的「批量处理」导入当前文件夹内全部允许后缀的视频（非递归、SHA-256 复用、失败项单独返回）；自动把第一个视频加载到中栏用于绘制。同一批共用 POI/信号区/完整性区、规则与参数；后端队列逐条执行，单条失败不中断。
- **绘制坐标**归一化，首帧与后续预览/源视频共用同一套区域。

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
| 监控兼容预览（HEVC） | 浏览器“有时长无画面”（HEVC 无硬解）时按需生成 GPU 兼容预览（NVDEC 硬解 → H.264）；分析入口/帧循环空帧防护（见 4.5） | ✅ 已完成 |
| 信号区触发规则（强行通过） | 规则类别 + 通道配对（入口线+可选计数区+信号区）+ 3 秒 episode 去重 + 从进入时刻起 5 秒退回判定 + 分析前强制校验（见 4.8） | ✅ 已完成（本轮） |
| 首帧绘制 + 手动预览 | 不可直放源显示首帧并可绘制区域；「生成预览」显式触发 remux/GPU 转码；语义分析不依赖浏览器预览（见 4.9） | ✅ 已完成（本轮） |
| 文件夹批量处理 | 「批量处理」一次导入当前文件夹全部视频（非递归、SHA-256 复用、失败项单独返回）；同机位共用区域/规则，队列连续执行（见 4.9） | ✅ 已完成（本轮） |
| 复核窗口排版 + 处理时长 | 修复右侧网格行/通用按钮样式导致的筛选栏与结果重叠；头部显示处理时长；窄屏整页滚动（见 4.3） | ✅ 已完成（本轮） |
| PT 性能与吞吐专项 | 性能 P0/P1/P2 已完成；P3 I/O/缓存完成（tracking -11.4%）；PT0-1/2 批量姿态/重叠默认关（1.19×）；PT0-4/5、P3 模型/解码评测、PT1、PT2 待做（见 6.4.7） | 🔶 部分完成 |
| P1（其余：证据 / 模型） | 高分辨率 ROI 抽帧、4B/8B、self-consistency、置信度校准 | ⬜ 待做 |
| P2（评测 / 生产化） | 标注集、指标、人工复核、监控 | ⬜ 待做 |

### 4.1 P0 已落地的关键决策（请勿推翻，除非业务变化）

1. **报警信号 = 安检门门框的竖直红色 LED 灯带**；顶部绿色指示灯按规则忽略。
2. **红灯亮灭用经典 CV（红色像素占比 + 持续时间）检测**，不用大模型判断。
3. **判定策略 = 确定性优先 + VLM 仅复核不确定样本**。
4. **“退回” = 反向跨回入口计数线**。两条路径的窗口起点不同：POI 触发路径沿用 `[entry_t, onset+5]`；**信号触发（强行通过）路径为 `[entry_t, entry_t+5]`**（从目标进入时刻起算，避免灯亮后晚进入只剩很短观察窗）。触发只保留进入类（`line.cross`）。
5. **红灯阈值在前端“信号区阈值”高级设置里可编辑**，随任务快照冻结，每相机各存一套；校准工具输出建议值。
6. **入口方向翻转只作用于最新画的一条计数线**。
7. **仅移除 Webhook**；实时控制台（经典 CV 检测/计数/监控）与基础分析保留。
8. **启动即就绪，无需手动选设备/模型**：后端启动时（FastAPI `lifespan` → `_autostart`）自动选 CUDA（不可用回退 CPU）并加载 `yolo11m-pose.pt`；前端 `ControlPanel` 挂载读 `GET /api/state` 回显当前引擎与模型，Model Hub 下拉与 CPU/CUDA 按钮仍可手动切换。VLM 分析侧 `poseDevice` 默认 `cuda`、姿态模型固定 `yolo11m-pose.pt`。可用 `ULTRA_AUTOSTART / ULTRA_DEFAULT_MODEL / ULTRA_DEVICE` 配置；自动选择失败只记日志、降级为 CPU/无模型，不阻塞启动。
9. **库记录展示名与处理时长语义**：VLM 记录展示名 = `原视频名_YYYYMMDD_HHMM`（分钟级，如 `监控.mp4_20261105_1730`），由 `prepare()` 写 `library.json` 的 `displayName`，内部 `run_id` 不再作展示主标识；处理时长 `startedAt` 在 `prepare()`（用户点「开始分析」那一刻）写入 `status.json`，`runner` 不得重置，`durationSec = finishedAt(或当前时间) - startedAt`，运行中任务显示「进行中」。
10. **规则类别是稳定契约**：`category` 固定为 `poi_event=1 / forced_passage=2 / integrity=3`，类别决定必需区域、触发方式与判定路径；规则名称可改，类别代码不重命名、不复用。旧规则库一次性迁移（见 4.8），分析记录不迁移。
11. **强行通过 = 信号触发 + 通道配对**：入口计数线（必填，必须设置进入方向）+ 信号区（必填）按 `pairIndex` 配对；同号计数区为可选辅助。信号从首次触发起 **3 秒内去重**，一个 episode 只判一次；退回以入口线反向跨线为主判据，计数区 `zone.leave` 仅在目标来自计数区时辅助。默认纯确定性判定，不调用 VLM；`no_subject` 保留审计记录。
12. **分析前强制校验**：按“已绘制区域 × 已启用规则类别”校验。只绘制完整性区时只能启用完整性规则；强行通过缺入口线/信号区/配对/进入方向时直接在开始前拦截（前端行内提示 + 后端 422 兜底）。
13. **预览默认不自动生成**：浏览器不能直放时先显示首帧并可绘制区域；只有用户点「生成预览」才执行 remux/GPU 转码。语义批量分析直接读源文件，不依赖浏览器预览；预览只影响本页播放与姿态分析。首帧与预览共用归一化坐标。
14. **批量处理按文件夹、同机位**：文件夹导入仅当前层、非递归、同步执行，按 SHA-256 复用；同批视频共用一套区域/规则/参数，任务按后端队列连续执行，单条失败不中断，改区域/规则只影响之后提交的任务。用户负责保证同一批视频的画面角度完全一致。

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
- **复核窗口（记录查看窗口）**也显示处理时长：`GET /api/vlm/runs/{id}` 新增 `durationSec`（与库列表同一 `startedAt/finishedAt` 时钟），`ReviewDialog` 头部显示，运行中的任务随 2.5s 轮询递增。
- 测试：新增 `test_display_name_and_duration`（displayName 时间戳 + durationSec 精确断言）。验证：后端 pytest 137 passed + 1 skip、ruff 通过、前端 build/lint/test 30 通过。

### 4.4 路径自包含（权重与解释器）

- **VLM 权重根目录**：`models.py` 的 `MODEL_ROOT` 原为 `parents[5]`（实际解析到 `/home/user`，即项目外），现改为项目内 `models/`，并支持 `VLM_MODEL_ROOT` 覆盖（外部挂载到其他路径时使用）；2B/4B 权重已复制到 `models/Qwen3-VL-2B-Instruct` / `models/Qwen3-VL-4B-Instruct`，`VLM_MODEL_PATH` 仅作为单模型路径的覆盖入口。
- **VLM 解释器**：`cli.py` 的 `qwenPython` 默认改为 `sys.executable`，不再写死开发机 conda 路径；Qwen 跑在独立环境时用 `VLM_PYTHON` 覆盖（本机放 `backend/.env`，外部环境由环境变量注入）。
- `.gitignore` 忽略 `models/*`（保留 `.gitkeep`）；README 权重/解释器说明与 `backend/.env.example` 已同步。
- 验证：`/api/vlm/parameters` 返回项目内路径，2B/4B 均 `available`；后端 + Rules 测试通过。

### 4.5 本轮变更清单（监控 HEVC 兼容预览触发修复）

- **现象**：加载 HEVC/H.265 监控录像（本轮实测样本：1920×1080 Main / 25fps / AAC / 25 分钟）预览黑屏，分析报 `video.analysis.failed: Error: Missing JPEG payload`。
- **根因**：浏览器无 HEVC 硬解时仍能读出时长（`loadeddata` 触发、`videoWidth=0`），**不触发 `error` 事件**，仅挂在 `error` 上的兼容预览回退永不执行；0×0 画布 `toDataURL` 得 `"data:,"`，空 `imageJpegBase64` 被后端 `ws_infer` 拒收记 WARN，前端会话把 WARN 当致命错误断开。
- **修复（仅前端 `ui/viewer/tabs/VideoTab.tsx`）**：`onLoadedData` 就绪条件加 `videoWidth/videoHeight>0`，“有时长无画面”主动调 `onDecodeError()`；`analyze()` 入口同检查；帧循环空 JPEG 抛本地化错误。后端无改动。（4.9 起 `onDecodeError()` 不再自动转码，改为显示首帧 + 手动「生成预览」，此链路保留为显式操作。）
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

### 4.7 本轮变更清单（监控视频格式支持：H.264 免转码预览 + .ts 等传输流）

- **现场格式确认**：C5-4 目录里的 `*.mp4` 实际是 DVR 导出的 **MPEG-PS**（文件头 `IMKH`），视频 H.264 Baseline 1920×1080@25，音频 G.711 A-law。浏览器既不解该容器也不解 G.711 音频，因此“直接让播放器解码”不可行（PS 无浏览器 demuxer，ffmpeg.wasm 软解又慢）；正确路径是**服务端尽量只换容器、不重编码**。
- **免转码快路径**（`backend/app/video_transcode.py` 新增 `remux_copy`，CLI `--mode auto`）：源视频已是浏览器可解的 H.264 8-bit 4:2:0 时，用 `add_stream_from_template` 把码流原样拷进 MP4（包时间戳整体归零；TS 包不带时长，用相邻 DTS 补齐，避免末帧被 MP4 丢弃）。**不占 GPU、不降画质**。实测 30 分钟 537MB PS 文件 **0.5–0.6 秒**生成预览（原 NVDEC+NVENC 需数分钟）；输出 `avc1`、时长 1824.04s 与源一致、45600 包无丢失。
- **HEVC 等仍需硬转**：非 H.264（或 High10/4:2:2/4:4:4）继续走 `h264_nvenc`；复核播放代理（`playback.py`，1s GOP 以便快速 seek）保持 `--mode transcode` 不变。CPU 回退仍禁用。
- **`.ts` 等传输流**：`vlm_api.ALLOWED_SUFFIXES` 增加 `.ts/.mts/.m2ts/.mpg/.mpeg`；`/videos/{id}/source` 使用显式 MIME（`.ts → video/mp2t`，避免 `mimetypes` 把 `.ts` 判成文本）。probe / 分析 / 预览均由 PyAV 原生支持 MPEG-TS。
- **缓存上限**：`ULTRA_VIDEO_PREVIEW_CACHE_BYTES`（默认 64 GiB，0 = 不限）按最旧优先淘汰，正在使用的预览不删——免转码预览接近源文件大小，防止缓存无限增长。
- **验证**：pytest 全量绿、改动文件 ruff 绿；临时实例端到端：导入 537MB PS → 预览 **1.0s** 就绪；真实 5 分钟 HEVC `.ts` → 预览 39.7s（NVDEC→NVENC）。Chrome headless 实测两种预览时长/尺寸正确、可播放、可 seek 到 1500s 并 `drawImage` 抽帧（分析帧循环同路径）。
- **前端**：兼容预览链路不变（4.9 起改为「生成预览」显式触发）；i18n 提示改为“H.264 免转码秒级完成，HEVC 长录像可能需要数分钟”。

### 4.8 信号区触发规则（强行通过 / 规则类别 / 通道配对）

- **需求**：对“强行通过安检门且未退回”这类规则，不再由计数区候选逐人触发，而是**信号区亮起才分析**；同一信号区从首次触发起 3 秒内算同一次事件；计数区 1000 人次、信号只亮 10 次 → 只判 10 次。
- **规则类别**：规则模板新增稳定 `category`（1 `poi_event` 通行/计数、2 `forced_passage` 强行通过、3 `integrity` 完整性）；类别决定必需区域、触发方式、判定路径与分析前校验。旧规则一次性迁移：含 `integrity.change` → 完整性，其余 → 通行事件。
- **通道配对**：入口计数线（必填，需设置进入方向）+ 信号区（必填）按 `pairIndex` 一一对应；同编号计数区为**可选辅助**。入口线/计数区/信号区各自独立编号，`L1 + Z1(可选) + C1` 构成通道 1。
- **退回判定**：信号 episode 起点为 `triggerSec`，在 `[起点−pre, 区间末+post]` 内取最近的正向跨线目标；退回窗口**从目标进入时刻起算 5 秒**，入口线反向跨线为主判据，计数区 `zone.leave` 仅作辅助。无人通过仍输出 `no_match + no_subject` 审计记录，不调用大模型。
- **校验**：强行通过必须有完整的入口线↔信号区配对且入口线有方向；只绘制完整性区时，启用非完整性规则会在开始分析前返回 422（“当前仅绘制了完整性区，只能启用完整性规则”）。前端规则编辑器同步显示行内错误并禁用开始按钮。
- **代码落点**：`contracts.py`（category / cue.on / pairIndex / cueDedupeSec）、`observations.py`（支撑候选：信号规则仍需持久化跨线/进出事件）、`decision.py`（episodes / 目标选择 / 退回）、`qwen_worker.py`（`_cue_pairs` + `_process_cue_candidate`）、`backend/app/event_rules.py`（类别迁移 + 强制配对校验）、前端 `EventEditOverlay`（编号徽标）、`EventRulesEditor`（类别选择 + 配对表）、`ResultCard`/`ReviewVideo`（触发信号区与退回证据）。
- **验证**：后端 + Rules 全量通过（当轮 208；当前全量 **210 passed / 1 skipped**，新增 `Rules/tests/test_cue_trigger.py`、`backend/tests/test_event_rules.py` 用例），ruff 全绿；前端 **46 用例**、eslint、build 通过。实机验证：规则库迁移后 `forced_passage_unopposed` 为 `category=forced_passage / triggers=["cue.on"] / pre=8 / cueDedupeSec=3 / v3`；仅完整性区 + 非完整性规则的实机 POST 返回预期的 422 中文提示；前端 headless 冒烟可见类别、配对表与行内校验。
- **性能（2026-11-26）**：tracking + cues + integrity 已合并为 `detect` 阶段，只解码一次；30 分钟真实任务 P0 时 193.6s，P3 I/O 后 **179.7s**（旧分开阶段约 6 分钟），observations/candidates/cues/integrity 逐项一致。确定性证据默认 lazy，只写 sample manifest；165 条结果 understanding 复算 0.41s、0 预览。详见 6.4.7。
- **未做**：尚未用含真实红灯的录像跑完整分析（手头样本无红灯），worker 逻辑由合成 `cues.json + candidates.jsonl` 的 `understand()` 集成测试覆盖。
- 详细设计与验收标准见 `docs/signal-triggered-rules-plan.md`。

### 4.9 本轮变更清单（首帧绘制 + 文件夹批量处理）

- **首帧绘制替代自动预览**：浏览器解不出源视频时（MPEG-PS / HEVC / TS 等），不再自动启动转码；`VideoTab` 显示后端抽取的**首帧图**（`GET /api/vlm/videos/{id}/thumbnail`，首次请求解码并缓存为 `thumbnail.jpg`），线/区/信号区/完整性格子工具与 ROI 在首帧上直接可用（`stageReady = videoLoaded || (useStill && stillReady)`）。用户点击「生成预览」后才执行原来的 remux/GPU 转码；预览就绪前也能进入语义批量分析（后端直接读源文件），预览只影响本页播放与姿态分析。绘制坐标是归一化的，首帧与预览共用同一套区域。
- **文件夹批量导入**：`POST /api/vlm/videos/import-folder` 一次登记当前文件夹内全部允许后缀的视频（非递归；逐文件校验、硬链接/复用、失败项单独返回，不中断其余文件）。前端语义分析面板新增「批量处理」按钮，进入文件夹选择模式；导入成功后自动把**第一个视频**加载到中栏用于绘制（若当前视频已在批次内则保持不变），其余文件进入分析对象列表。
- **批量连续执行**：沿用既有任务队列（`MAX_CONCURRENT=1`、状态落盘、每 2 秒 `_pump` 调度）。同一批视频共用当前 POI/信号区/完整性区、规则与参数；单条失败只标记该任务，队列继续下一条，关闭页面也不影响已排队任务。
- **前端**：`ServerVideoDialog` 新增文件夹模式（当前目录视频计数、部分失败列表、截断提示）；`VlmPanel` 新增 `onSelectVideo` 与批次模式；`apiVlmImportFolder` / `apiVlmThumbnail`；中英文文案。
- **验证**：后端 **210 passed / 1 skipped**、ruff 全绿；前端 46 用例、eslint、build 通过。实机 headless 验证：文件夹 6 个 `.ts` 一次导入 → 自动加载首帧（1920×1080）→ 在首帧上画计数线（VlmPanel 显示 `1 POI`）→ 批次列表 6 项、开始按钮显示「Start analysis (6 videos)」（规则未配齐时正确禁用）。单视频导入路径与既有逻辑一致。
- **边界**：文件夹导入为同步请求，大文件夹需要持续读取/哈希（哈希用于 SHA-256 复用与预览缓存）；仅当前层视频，子文件夹不遍历；内容完全相同的文件按 SHA-256 复用同一登记，返回名称为首次登记的文件名。

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
>
> **信号触发规则（4.8）从触发源上进一步消除第 2 层的一类问题**：强行通过只分析“信号区亮起”的 episode，进入/退回锚定在配对入口线与信号时刻，不再对大量无关通行做判定（1000 人次、10 次亮灯 → 10 次判断）；但它**不解决**证据不可见与模型偏差（第 1、3、5 层），后者仍按 P1/P2 推进。

---

## 6. 改进路线图

### 6.1 P0 —— 确定性判定（✅ 已完成）

**做了什么**

- 新增 `CueDefinition`（信号区几何 + 红色阈值参数）与 `RunRequest.cues`、`EventDefinition.cueIds`。
- 新增 `Rules/src/ultra_rules/vlm/cues.py`：在绘制 ROI 内做红色掩码 + 滞回，输出红灯亮/灭区间；含独立缓存；`cli calibrate` 输出红占比时间序列、直方图、Otsu 建议阈值、峰值帧 RGB 统计。
- 新增 `Rules/src/ultra_rules/vlm/decision.py`：确定性判定（红灯状态 / 是否退回 / 轨迹是否连续），并做“进入—退回”配对抑制，避免一次进出被重复计数。
- `understand()` 改为确定性优先，Qwen 模型改为**懒加载**（纯确定性结果不加载模型）。
- 前端新增“画信号区”工具、“信号区阈值”面板、入口方向翻转；规则编辑器支持绑定信号区。
- **本轮补充（信号触发）**：`forced_passage` 类别改为“信号 episode 触发 + 通道配对”，不再逐人分析计数区候选；`decision.py` 新增 `merge_cue_episodes / select_cue_target / retreat_event`，`qwen_worker` 新增 `_cue_pairs / _process_cue_candidate`；规则类别与分析前校验见 4.8。

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

### 6.4 PT —— 性能与吞吐专项（P0–P6 性能路线；与第 1 节准确率 P0/P1/P2 无关）

> **定位**：第 1 节的准确率 P0/P1/P2 解决“判得准”；本节 PT 解决“判得快”，性能路线编号 P0–P6 见 `optimize.md`（P0/P1/P2 已完成，P3–P6 待做）。内部优先级 PT0（低成本先行）→ PT1（中等改动）→ PT2（高成本谨慎）。
> **红线（继承 4.1 / 7.3）**：一切提效以“不改变判定语义与准确率”为前提；不改 `poi.py` 候选生成；**信号触发路径不得改 3 秒 episode 去重、通道配对与退回窗口**；红灯仍 CV 检测、确定性优先 + VLM 仅复核；默认保守串行，并发/批大小只能以环境变量/参数开启；不预设加速比，量化一律以基线实测为准。

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
| cues / integrity | 待实测 | — | — | `detect` 与 tracking 共享一次解码；cues-only/integrity-only 仍两阶段 |
| understanding·抽帧 | 确定性 lazy 0；VLM 约 0.7s/窗口 | — | — | `build_sample(render=...)`；lazy 预览首次访问生成 |
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
| PT0-3 | ✅ **已完成（2026-11-26）**：`detect` 阶段用 `media.SampleSchedule` 让 tracking / cues / integrity 从同一次 `decoded()` 消费，异构 `poseFps`/`cue.sampleFps`/`zone.sampleFps` 均保留原采样点；integrity 用最近 pose persons 做遮挡排除（等价于 `_persons_near`）。30min 真实任务 observations 9038 行、candidates、cues、integrity 全部逐项一致（6.4.7）。剩余可选：cues-only/integrity-only 路径仍是两阶段；`_refine_onset` 可批量 seek | `vlm/combined.py`、`vlm/media.py`、`vlm/runner.py` | 已完成等价性回归 | 高（长视频显著）/ 中 |
| PT0-4 | 确定性短路率提升：在既有“确定性优先”框架内统计 p、识别可安全扩大的可测量信号（不得放宽否决、不得改退回语义） | `vlm/decision.py`、`vlm/qwen_worker.py`（门禁统计） | **P2 标注集**验证不降准确率 | 随规则数↑ / 高（语义红线，无评测集不动） |
| PT0-5 | 结果级缓存复用：按已有 `requestFingerprint` 跨 run 复用 understanding 结果（现状仅 run 内断点续跑） | `vlm/qwen_worker.py`、`vlm/storage.py` | 指纹含模型/提示词版本，变更即失效 | 中（改规则重跑场景）/ 低 |
| PT0-6 | ✅ **已完成（2026-11-26）** lazy 证据预览：`previewMode=lazy` 时确定性结果只写 manifest；后端 artifact 端点首次访问生成 JPEG 并缓存；VLM/dryRun 仍 eager。165 条结果 understanding 0.41s、0 预览（6.4.7） | `vlm/sampling.py`、`vlm/contracts.py`、`vlm/qwen_worker.py`、`backend/app/vlm_api.py` | 无 | 高（密集信号/大结果集）/ 低（前端契约不变） |
| PT0-7 | ✅ **已完成（2026-11-26）** tracking I/O/缓存：临时库 no-fsync、延迟索引、`executemany` 批量插入、gzip level1；tracking 100.8→89.3s（-11.4%），detect 193.6→179.7s（-7.2%），9038 帧零漂移（6.4.7） | `vlm/observations.py`、`vlm/combined.py` | 无 | 中 / 低（缓存可重建；旧 cache 兼容） |

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
- **2026-11-26 更新**：性能 P0/P1/P2 已完成（单次解码 `detect`、integrity 多区共享、lazy 预览），P3 I/O/缓存也已完成（tracking -11.4%，见 6.4.7）；下一步 P3 剩余模型/解码评测或 P4（队列、ETA、worker 分离）。PT1-4/PT1-1 与准确率评测集仍按原排序。

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

#### 6.4.6 本轮修复：证据抽帧 seek 对齐与空 episode 短路（2026-11-26）

**现象**：191 分钟录像的 understanding 阶段耗时约 63 分钟；11 个 cue episode 的证据样本每个约 5.5 分钟，远超信号检测（cues 阶段约 7 分钟）。

**根因**：`sampling.source_images()` 以首个选中帧的 PTS 做 `container.seek(..., backward=True)`，但容器/开放 GOP 会落到目标帧之后的 0.1–1.2 秒；首个选中帧缺失后 `len(images) != len(rows)`，代码回退到 `decoded()` 从第 0 帧重新解码到目标帧。episode 越靠后，单个样本越慢，整体接近 O(episode × 视频位置)。

**修复**：

1. `source_images()` 改为“先向前回退 2.5s，再按需 8s / 30s 重试，取最近时间戳帧”，并用 `MAX_FRAME_DRIFT_SEC=0.5` 兜底；不再因缺 1 帧回退到全片解码。无 PTS 时可用 `mediaTimeSec + timeOriginSec` 走同一时间轴。
2. `_process_cue_candidate()` 对 `no_subject` episode 跳过证据生成（只保留审计结果）；`poi_event` 候选在确定性 cue-off 时同样跳过抽帧，只保留 `no_match` 审计行。
3. `understand()` 一次构建目标索引（`build_target_index`），`select_cue_target()` 用二分窗口查候选，避免每个 episode 全量扫描。
4. `cues.detect_all()` 同采样率的信号区共享一次解码（BGR 只转换一次）。

**实测**：同一 191 分钟、11 个 cue episode 的真实任务，understanding 从约 63 分钟降到 **6.8 秒**；11 条结果与修复前逐条 `resultId/decision/retreatSec/rawTriggerCount` 完全一致。单样本抽帧从约 5.5 分钟降到约 0.7 秒。

**回归**：`Rules/tests` + `backend/tests` 全量通过（含新增 seek 重试、单次解码、空目标不抽帧、预构建索引用例）；ruff 全绿。

> 6.4.6 是 P0a/P1 的第一步；随后 P0b 已把 tracking 也并入同一次解码，并完成 lazy 预览，见 6.4.7。

#### 6.4.7 本轮性能 P0/P1/P2：单次解码、多区共享与 lazy 预览（2026-11-26）

> 命名说明：本节 P0/P1/P2 指 `optimize.md` 的性能路线；第 1 节的准确率优先级仍称 P0/P1/P2，二者独立。

**性能 P0/P1：tracking + cues + integrity 一次解码**

- `Rules/src/ultra_rules/vlm/combined.py` 新增 `analyze_run()` / `_track_and_detect()` / `detect_run()`：
  - `detect` 阶段用 `media.decoded()` 解码一次，`media.SampleSchedule` 按 `poseFps` / `cue.sampleFps` / `zone.sampleFps` 分发给三个消费者；
  - tracking 保持原 batch/pipeline 行为（`_FramePipeline(convert=False)`，同一帧只按需转换一次 BGR）；
  - integrity 的遮挡 persons 用最近 pose 结果（±1s 最近邻），等价于原 `_persons_near()` 的 sqlite 查询；
  - `observations.cache_fingerprint()` / `tracker_config_for()` 被 tracking 与合并阶段共用，缓存键不漂移；
  - runner 在 cues+integrity 同时存在时只跑 `segment → detect → understanding`；仅 cues 或仅 integrity 仍 `tracking → cues/integrity`。
- 缓存命中：当前 run 无 `observations.sqlite` 时 `track()` 重放；已有完整 sqlite 时直接复用；缓存未命中但残留部分 sqlite 时先清理再重算。

**实测（真实任务）**

| 任务 | 完整 detect | 一致性 |
|---|---:|---|
| 30min（1 信号区 + 3 完整性区） | 193.6s → **179.7s**（P3 I/O 后；旧分开阶段约 6 分钟） | 9038 行 observations 逐字段一致；cues/integrity 完全一致 |
| 5min（5 信号区 + 165 候选） | **32.9s** | 1500 行 observations + 165 条 candidates 完全一致；cues/integrity 一致 |
| 同 run 二次 detect（全缓存） | **1.5s** | 不再解码、不再加载 pose 模型 |

**性能 P2：lazy 证据预览**

- `contracts.AnalysisParameters.previewMode` 新增，默认 `lazy`。
- `sampling.build_sample(..., render=None)` 按 `previewMode` 决定：
  - 确定性证据：`render=False`，只查 observations + 写 `samples/{id}.json`（含 window / sourceFrameIndices / sampleTimesSeconds / observations / quality / poi / width / height），**不调用 `source_images()`、不拼图**；
  - VLM / dryRun：显式 `render=True`，仍需帧做推理/预览。
- 后端 `GET /api/vlm/runs/{run_id}/artifacts/previews/{sample}.jpg` 在文件缺失时按 manifest 调 `sampling.render_preview()`，全局锁串行生成并缓存；manifest 缺失返回 404，结果与源视频不受影响。
- 兼容：旧 eager 预览直接服务；前端继续用 `previewPath`，无契约变化。

**性能 P2 实测**

- 5min、165 条确定性候选复算：understanding **0.41s**，生成 **0 个预览 / 23 个 manifest**（其余为 cue-off 短路）；165 条公共候选判定与旧结果完全一致。
- lazy 与 eager 的 manifest 完全一致；单目标样本预览字节一致。多人叠加时绘制顺序可能不同（lazy 先画旁人再画目标），视觉等价。

**性能 P3：tracking I/O / 缓存（2026-11-26）**

- `observations.open_observations_db()`：临时 observations.sqlite 使用 `journal_mode=MEMORY` + `synchronous=OFF`；gzip cache 才是完成标记，部分库可在重试时删除。
- 延迟 `CREATE INDEX`，先 `executemany` 批量插入（每 200 行 commit），progress 每 50 行；减少 fsync、索引维护和进度写盘。
- cache 写 `PoiTracker.update()` 之前的行，sqlite 写更新后的行（保留 `trackSegmentId/identityAmbiguous`；不能为省一次序列化而丢失字段）。
- gzip cache `compresslevel=1`，牺牲压缩率换写入/读取速度；旧 cache 仍可读。
- 实测（30min、9038 帧、batch16+pipeline+poseFps5）：tracking **100.8s → 89.3s（-11.4%）**；完整 detect **193.6s → 179.7s（-7.2%）**；observations 9038 行逐字段一致，bench diff 0 mismatch、bbox/kp delta 0.0。
- 未做：TensorRT / ONNX / FP16 / NVDEC 独立环境评估；poseFps 默认值未改。
- 测试：`Rules/tests/test_tracking_io.py::test_track_batch_sqlite_rows_keep_post_tracker_fields`。

**回归**

- 新增：`Rules/tests/test_combined.py`、`Rules/tests/test_lazy_preview.py`、`backend/tests/test_vlm_api.py::test_artifact_endpoint_renders_missing_preview_on_demand`，以及 `SampleSchedule` VFR 等价、runner 阶段选择、`no_subject`/cue-off 短路用例。
- 当前全量：**223 passed / 1 skipped**；ruff 全绿；前端 **46 用例**、eslint、build 通过。

**红线保持**：信号 episode 触发、3 秒去重、通道配对、从进入时刻起 5 秒退回、确定性优先均未改变；`poi.py` 候选生成未改。

**下一步**：P3 剩余的 TensorRT/ONNX/FP16/NVDEC 独立评测，或 P4 队列/ETA/worker 分离；详见 `optimize.md`。

---

## 7. 开发者指引

### 7.1 关键代码地图

**后端（Python，`Rules/` 是纯算法库，`backend/` 是 FastAPI 服务）**

| 路径 | 职责 |
|---|---|
| `Rules/src/ultra_rules/vlm/contracts.py` | 全部数据模型（Poi / CueDefinition / EventDefinition / RunRequest 等）；规则类别 `category`、`cue.on`、通道编号 `pairIndex`、信号去重 `cueDedupeSec`、预览模式 `previewMode`（默认 lazy） |
| `Rules/src/ultra_rules/vlm/models.py` | 本地 Qwen3-VL 权重发现与校验（默认根目录为项目内 `models/`，不读项目外路径） |
| `Rules/src/ultra_rules/vlm/observations.py` | 跟踪阶段：YOLO-Pose + ByteTrack → observations.sqlite；观测缓存指纹不含规则/POI（改规则重跑命中缓存）；`candidate_owners()` 区分直接判定候选与信号规则的支撑候选；`cache_fingerprint()` / `tracker_config_for()` 与合并阶段共用；`_FramePipeline(convert=False)` 支持同帧多处消费；`open_observations_db()` + 批量插入/延迟索引/gzip level1 优化写路径 |
| `Rules/src/ultra_rules/vlm/runner.py` | 阶段协调子进程：`worker.lock` 互斥、阶段串行/超时/取消、status 写入；cues+integrity 时走 `segment → detect → understanding` |
| `Rules/src/ultra_rules/vlm/media.py` | 统一解码与时间基：`probe / decoded`（全帧软解）/ `sampled`（按 fps 采样）/ `SampleSchedule`（一次解码多消费者，取帧规则与 `sampled` 一致） |
| `Rules/src/ultra_rules/vlm/combined.py` | `analyze_run` / `_track_and_detect` / `detect_run`：tracking + cues + integrity 单次解码；缓存命中重放；lazy 输出 cues/integrity artifacts |
| `Rules/src/ultra_rules/vlm/integrity.py` | 完整性 CV 检测：抽样与基线比对 → 变化区间 |
| `Rules/src/ultra_rules/vlm/storage.py` | JSON/JSONL 读写、指纹 `digest`、`progress.json`、缓存工具 |
| `Rules/src/ultra_rules/pose_adapter.py` | YOLO-Pose 单帧推理适配（每文件独立 tracker；PT0-1 批量化改造点） |
| `Rules/src/ultra_rules/vlm/poi.py` | 候选生成：PoiTracker 状态机 |
| `Rules/src/ultra_rules/vlm/cues.py` | 红灯 CV 检测 + `calibrate()` 校准工具 |
| `Rules/src/ultra_rules/vlm/decision.py` | 确定性判定（红灯 / 退回 / 轨迹连续性）；信号 episode 合并 `merge_cue_episodes`、目标选择 `select_cue_target`、退回 `retreat_event` |
| `Rules/src/ultra_rules/vlm/sampling.py` | 抽帧窗口、选帧、sample manifest、`build_sample(render=...)`（lazy 只写 manifest）、`render_preview()`（按需生成 contact sheet） |
| `Rules/src/ultra_rules/vlm/prompting.py` | VLM 提示词 |
| `Rules/src/ultra_rules/vlm/qwen_worker.py` | Qwen 模型封装 + `understand()` 编排；信号触发路径 `_cue_pairs()` / `_process_cue_candidate()`（确定性，不默认调模型） |
| `Rules/src/ultra_rules/vlm/results.py` | 解析 / 裁决（`decision_from_conditions` 合取真值表 + 证据校验降级）/ 聚合 |
| `backend/app/vlm_api.py` | VLM HTTP 接口 + 任务调度器（`_pump`，`MAX_CONCURRENT=1` 单 worker 槽，队列落盘可重启恢复）；视频首帧 `GET /videos/{id}/thumbnail`；文件夹批量导入 `POST /videos/import-folder`；`/artifacts/previews/*.jpg` 缺失时按 manifest 懒生成并缓存 |
| `backend/app/vlm_jobs.py` | 任务启动/状态/取消：`launch/status/cancel`、owner 进程存活判定、reap 线程 |
| `backend/app/event_rules.py` | 规则库 CRUD + ManagedRun + resolve；规则类别迁移、强行通过通道配对校验、分析前区域/类别校验 |
| `backend/app/library_api.py` / `video_api.py` | 记录管理（展示名/处理时长 `durationSec`）/ 视频预览（H.264 免转码重封装优先、其余 NVDEC+NVENC、SHA-256 缓存 + `ULTRA_VIDEO_PREVIEW_CACHE_BYTES` 上限、`ULTRA_VIDEO_MAX_CONCURRENT`、禁用 CPU 回退） |
| `backend/app/video_transcode.py` | 隔离预览 worker：`remux_copy`（H.264 包拷贝进 MP4，无 GPU）与 `transcode_gpu`（cuvid 硬解 → `h264_nvenc`，PT0-2 硬解化可复用此模式） |
| `backend/app/main.py` | FastAPI 入口：VLM/视频/库路由 + 实时推理接口（`/api/models` `/api/state` `/api/params` `/api/system/stats` `/ws/infer`）+ 启动 `lifespan` 自动选设备/加载默认模型 |
| `backend/app/infer_runtime.py` / `devices.py` / `state.py` / `config.py` / `system_stats.py` / `logging_store.py` / `schemas.py` | 经典 CV 实时推理运行时（模型加载、设备、参数、启动自选配置、系统状态、日志、Schema）；`_predict_lock` 全局推理串行、每流独立 tracker 上下文（PT2-2 隔离改造点） |

**前端（`frontend/src/`）**

| 路径 | 职责 |
|---|---|
| `ui/viewer/tabs/VideoTab.tsx` | 视频选择 + 预览 + 基础检测分析 + 画线/区/信号区 + 挂载 VlmPanel；不可直放源显示首帧、手动画区域与「生成预览」按钮 |
| `ui/viewer/widgets/EventEditOverlay.tsx` | 画 POI（含入口方向箭头）；按类型分配 `pairIndex`，画布显示 `#1` 等通道角标 |
| `ui/control/ControlPanel.tsx` | 实时控制台左栏（引擎/NMS/类别过滤/模型；挂载读 `/api/state` 回显启动自选的设备与模型） |
| `ui/monitor/MonitorPanel.tsx` | 实时控制台右栏（系统状态/配置/计数事件/告警/日志） |
| `ui/vlm/VlmPanel.tsx` | 语义分析控制 + 结果展示（逐条件证据 + 点击跳转 `evidenceTimeSec`）；「批量处理」文件夹导入、批次列表与批量启动 |
| `ui/vlm/EventRulesEditor.tsx` | 规则编辑：类别选择、强行通过“配对通道”表、分析前行内校验；其他类别沿用 POI/信号区绑定 |
| `ui/vlm/CueSettings.tsx` | 信号区阈值编辑 |
| `ui/vlm/AnalysisParameters.tsx` | 分析参数（模型 / 姿态设备 / 抽帧等） |
| `ui/library/LibraryPage.tsx` | 记录与视频管理（单页内切换） |
| `store/useConsoleStore.ts` | 全局状态（引擎/参数/模型/预测/日志/告警/ROI/计数/事件/`eventsConfig`） |
| `store/useUiStore.ts` | 控制台 / 库 视图切换 |
| `utils/vlm.ts` / `eventRules.ts` / `vlmParameters.ts` | VLM 请求构造与类型；`eventRules.ts` 含类别定义、`pairIndex` 配对校验与导入导出 |
| `api/client.ts` | `apiVlmImportFolder`（文件夹批量登记）、`apiVlmThumbnail`（首帧图）、媒体 Range URL 与导入/浏览接口 |
| `utils/videoAnalysis.ts` / `events.ts` / `roi.ts` / `useEventTracker.ts` / `useSystemStats.ts` / `draw.ts` / `configIO.ts` | 实时视频分析（`VideoInferenceSession` 单帧在途）、事件追踪、ROI、画图、配置导入导出 |
| `utils/videoPreview.ts` | 兼容预览：从已导入视频发起转码（`/from-video`）→ 轮询进度 → 取播放 URL → 释放（HEVC 回退入口，见 4.5） |
| `ui/viewer/widgets/ServerVideoDialog.tsx` | 服务器视频浏览弹窗：`GET /api/vlm/browse` 逐级浏览 `VLM_BROWSE_ROOT`；单/多选走 `POST /api/vlm/videos/import`；文件夹模式一键 `POST /api/vlm/videos/import-folder`（当前层全部视频，显示计数/失败项/截断） |
| `ui/review/ReviewDialog.tsx` / `ReviewDialog.module.css` / `useReviewData.ts` | 记录查看窗口：头部状态/处理时长、视频+时间轴、覆盖/筛选/结果/分页；响应式单列整页滚动；播放代理 |
| `ui/review/ResultCard.tsx` / `ReviewVideo.tsx` / `utils/review.ts` | 复核展示：触发信号区/配对通道/退回证据；时间轴 marker 在无 `trackId` 时显示 `C1 → L1` |

**设计文档**

| 路径 | 职责 |
|---|---|
| `docs/signal-triggered-rules-plan.md` | 信号区触发规则（规则类别 / 通道配对 / 3 秒 episode / 退回 / 分析前校验）的设计、实现状态与验收标准 |

### 7.2 运行与验证命令

```bash
# 后端 + 规则算法测试
cd <项目根目录>
PYTHONPATH=backend:Rules/src python -m pytest backend/tests Rules/tests

# 信号触发规则聚焦回归（3 秒去重 / 通道隔离 / 退回 / 配对校验）
PYTHONPATH=backend:Rules/src python -m pytest \
  Rules/tests/test_cue_trigger.py Rules/tests/test_decision.py backend/tests/test_event_rules.py

# 新 WSL 一键初始化（Miniconda/Node/依赖/权重/启动）
bash deploy.sh

# WSL 单环境依赖（一个 requirements 同时装 torch/ultralytics/transformers）
pip install -r backend/requirements-wsl.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 然后把 backend/.env 的 VLM_PYTHON 指向同一解释器

# 后端 lint
python -m ruff check backend/app backend/tests Rules/src Rules/tests

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

1. **候选时间点已可靠**：不要改动 `poi.py` 的候选生成逻辑。信号触发是**另一条候选来源**（`cues.json` episode → `_cue_pairs`），不要把它塞回 `poi.py`。
2. **红灯语义已定**：门框竖直红色 LED 灯带；顶部绿灯忽略；用 CV 检测，不是让 VLM 看。
3. **判定策略已定**：确定性优先，VLM 只复核不确定样本；强行通过默认纯确定性（信号 + 进入 + 未退回），不调 VLM。
4. **退回语义已定**：入口线反向跨回；POI 触发路径沿用 `[entry_t, onset+5]`，信号触发路径为 `[entry_t, entry_t+5]`；计数区 `zone.leave` 仅在目标来自计数区时辅助。
5. **Schema 兼容**：给 `EventDefinition` 等新增字段时，`event_rules.py` 的 `resolve()` 里读取必须用 `.get()` / 默认值容错（历史 `event-rules.json` 可能缺新键）；规则库迁移必须幂等。
6. **先解释方案再动工**：涉及功能设计 / UI 交互 / 阈值语义等有多种路径的改动，先给方案让用户拍板。
7. **验收要能量化**：任何“准确率提升”都要有标注集 + precision/recall/F1 支撑，不能只凭样例感觉。
8. **资源路径必须自包含**：权重、配置、示例数据的默认路径一律落在项目内；解释器与外部挂载只能通过 `VLM_PYTHON` / `VLM_MODEL_PATH` / `ULTRA_MODELS_DIR` 等环境变量覆盖，代码里不得写死家目录或机器专属绝对路径。
9. **性能优化红线（PT 专项）**：提效不得改变判定语义与准确率；不改 `poi.py` 候选生成；默认保守串行，批大小/并发度/流水线只能以环境变量开启（提案命名见 6.4.3）；每项必须跑“与串行基线逐条一致性回归”（6.4.3）；不预设加速比，量化一律以 6.4.1 基线实测为准，未实测写“待实测”。
10. **类别/编号/术语是稳定契约**：`category`（poi_event / forced_passage / integrity）、`pairIndex`（通道号）、`episode`（信号事件）、`cue.on`（仅限强行通过）不要重命名或复用；新增类别必须同时更新前端类别选择、后端迁移与 `EventRulesEditor` 配对/校验。
11. **信号触发测试必跑**：改动 `decision.py` / `qwen_worker.py` / `observations.py` / `event_rules.py` 时至少跑 `Rules/tests/test_cue_trigger.py`、`Rules/tests/test_decision.py`、`backend/tests/test_event_rules.py`，覆盖 3 秒去重、通道隔离、退回/未退回、`no_subject` 与配对校验。
12. **首帧只是绘制锚点，不是分析源**：不能直放时先显示首帧；不要恢复“自动生成预览”。语义批量分析直接读源文件，预览是显式操作；绘图坐标归一化，首帧/预览共用。绘图工具的可用条件用 `stageReady`（首帧加载完成或视频可解码），不要用 `videoLoaded` 一刀切。
13. **文件夹批量按同机位假设**：导入仅当前层、非递归、同步、SHA-256 复用；同批共用一套区域/规则/参数，任务队列连续执行；不要为单条失败中止整批。用户负责保证同批画面角度完全一致，代码不逐视频校正区域。

---

## 8. 已知边界 / 风险 / 待办清单

- [ ] **红灯阈值仍是默认值**：需用真实“亮/灭”片段跑 `calibrate` 校准，尤其 `minOnRatio / minR / minRG / minRB`。
- [x] **入口方向**：强行通过规则**强制**要求入口计数线设置进入方向（未设置会在分析前拦截）；其余规则仍有默认双向 + 前端“进入方向”翻转控件。
- [ ] **红灯起始时刻 vs 触发窗口**：强行通过路径已改为以信号 episode 起点锚定、`preSeconds=8`；旧 POI 触发路径在红灯延迟亮起超过 `preSeconds/postSeconds` 时仍可能对不齐（P1 以 onset 定窗口后可解决）。
- [ ] **强行通过必须画入口计数线**：纯计数区 + 信号区不会开启该规则；分析前会提示补画入口线。目前也没有“跨号重配”UI，配对严格按 `pairIndex`（`L1+Z1(可选)+C1`）。
- [ ] **信号触发默认不调 VLM**：当前 `_process_cue_candidate()` 为纯确定性，没有 `vlmReview` 分支；若业务需要对少数信号触发做语义复核，需要后续补提示词与采样路径。
- [ ] **旧规则库迁移边界**：只自动识别完整性规则，并把内置 `forced_passage_unopposed` 转为强行通过；用户自建的“强行通过”规则需在编辑器手动切换类别。旧分析记录不迁移、不重算（按测试数据处理）。
- [ ] **真实红灯视频端到端验收**：尚未用含红灯亮起的真实录像跑通强行通过规则，需验证 episode 数、目标选择、退回判定与证据窗口。
- [ ] **postSeconds 证据覆盖**：内置规则当前 `postSeconds=5`，晚进入目标的退回证据可能落在证据窗口之外；可在规则编辑页调大（判定本身不受窗口限制）。
- [x] **信号触发规则已实现**：规则类别、通道配对、3 秒 episode 去重、从进入时刻起算退回、`no_subject` 审计、前后端分析前校验均已落地（见 4.8 / `docs/signal-triggered-rules-plan.md`）。
- [x] **首帧绘制 + 手动预览已实现**：不可直放源先显示首帧，绘图工具在首帧上可用；「生成预览」显式触发转码；语义分析不依赖浏览器预览（见 4.9）。
- [ ] **首帧绘制的场景一致性**：绘制的依据只有第一帧；若后续时段光照、遮挡或机位与首帧差异大，用户需自行确认区域仍适用（同机位前提下通常成立）。
- [x] **文件夹批量处理已实现**：「批量处理」导入当前文件夹全部视频（非递归、SHA-256 复用、失败项单独返回），自动加载第一个用于绘制，同批共用区域/规则，队列连续执行（见 4.9）。
- [ ] **文件夹导入是同步请求**：大文件夹需要持续读取/哈希（无逐文件进度），导入期间对话框显示 busy；超过 500 个视频会截断；内容完全相同的文件按 SHA-256 复用并返回首次登记名。大规模导入若耗时不可接受，后续改成后台 job + 进度轮询。
- [ ] **批量任务快照**：区域/规则在点击「开始语义分析」时冻结进每个任务；已排队任务不受后续修改影响。用户应在启动前完成绘制与规则配置。
- [ ] **真实 155 文件目录验收**：尚未用 C5-4 这类完整目录跑“批量导入 → 连续分析”全链路，需确认导入耗时、队列稳定性、磁盘（硬链接/预览缓存）与单条失败行为。
- [ ] **置信度不可信**：当前是模型自报、未校准，不能作为概率或阈值。
- [x] **CUDA 端到端**：本机有 GPU，已验证控制台启动自动选 cuda + 加载 `yolo11m-pose.pt`（`GET /api/state` 回显），并跑通一次真实 VLM 分析（`model-decision-v5`）。
- [ ] **P1② 证据降级 vs 缺失即否定**：`missing_evidence_time` 会把“基于未出现”的 no（`evidenceTimeSec=null`，如“红灯从未亮起”）降级为 unknown，使 `no_match`→`uncertain`，与目标“未出现作为 no 证据”存在张力；待决定是否放宽（`status=no` 且 null 不降级，仅 `yes` 且 null 才降级）。
- [ ] **规则 criteria 混入指令型条目**：如“忽略安检门顶部灯光”不是可检验条件，合取（全 yes→match）下模型无法标 yes 会阻断 match；需在规则库层面把“指令”与“可检验条件”分开。
- [ ] **采样 manifest 长度不一致**：个别窗口 `sampleTimesSeconds` 与 `sourceFrameIndices` 长度不等时 `frameIndex` 映射被跳过（前端 seek 用 `evidenceTimeSec` 不受影响）；待排查 `sampling.py` 或加防御性对齐。
- [ ] **评测集缺失**：P2 前无法量化准确率，属于当前最高优先级的“基础设施”缺口。
- [x] **监控 HEVC 兼容预览触发（本轮修复）**：黑屏 + `Missing JPEG payload` 根因与修复见 4.5；GPU 转码链路本轮实测跑通，预览按 SHA-256 缓存。
- [x] **长视频吞吐基线（部分）**：链路 B 已完成单次解码 + lazy 预览 + tracking I/O；30min 真实任务完整 `detect` 179.7s（P0 时 193.6s），5min 任务 understanding 0.41s（6.4.7）。191min 样本完整端到端、GPU 占空比/显存峰值仍待补测（6.4.1 基线表）。
- [x] **性能 P0/P1/P2 + P3 I/O 已落地（2026-11-26）**：tracking+cues+integrity 单次解码、integrity 多区共享、lazy 证据预览、tracking I/O/缓存优化；observations/candidates/cues/integrity 逐项一致（6.4.7）。下一步 P3 模型/解码评测与 P4。
- [ ] **单 GPU 显存争用**：4B BF16 此前峰值约 9GB（6.2④ 既有记录）+ 姿态模型 + 实时控制台同卡；PT2 并发前必须显存门禁与抢占策略。
- [ ] **分片并行跟踪的边界风险**：轨迹断裂/ID 切换可能破坏候选可靠性红线（PT1-3），默认不做；跨分片去重（PT2-3）同。
- [ ] **HEVC 兼容预览依赖 NVIDIA 硬解**：`video_api.transcode` 明确禁用 CPU 回退，无 NVDEC 的机器上长 HEVC 视频无法生成兼容预览（现有设计；PT0-2 硬解化同样受此约束）。
- [x] **PT0 第一期落地（默认关）**：pipeline 比特一致；batch 强制 FP32 消除 ID 漂移；出厂配置 1.19×（本轮实测，见 4.6 / 6.4.5）；存量观测缓存因指纹 bump 一次性失效。**本机已按用户确认在 `backend/.env` 启用 batch8+pipeline**（机器本地配置，不入库）；代码默认仍为关，其他机器/生产默认开启仍待 P2 评测集回归。
- [ ] **批量数值漂移教训**：TF32 批量前向的微差会被跟踪器放大为长视频 trackId 切换（本轮实测 3623/15000 帧）；后续任何批量化（understanding 等）必须先跑长视频一致性回归，必要时配确定性精度。
- [x] **tracking 推理瓶颈 + I/O 优化（部分）**：PT0 批处理/重叠空间有限（1.19×）；P3 I/O/缓存已把 tracking 100.8→89.3s（-11.4%）、detect 193.6→179.7s；TensorRT/NVDEC 与 GPU 利用率/显存峰值仍待实测。

---

## 9. 下一步建议（按顺序）

0. **性能路线 P0/P1/P2 已完成，P3 I/O/缓存已完成**（`optimize.md`，非准确率 P0/P1/P2）：单次解码 `detect`、integrity 多区共享、lazy 预览、tracking I/O 优化；真实任务已逐项对齐。下一步 P3 剩余模型/解码评测（TensorRT/ONNX/FP16/NVDEC）或 P4（队列、ETA、worker 分离），不要重复做单次解码、lazy 预览或 tracking I/O。
1. **P2 先行打地基**：先做最小标注集 + 指标脚本（哪怕只标 100 条），否则 P1 的每次调整都无法证明好坏。
2. **P1 的“证据”优先于“换大模型”**：先做高分辨率 ROI 抽帧 + 结构化条件输出，再做 4B/8B 对比。
3. **红灯阈值校准**：找一个含真实红灯亮起的片段，跑 `calibrate` 落地阈值。
4. **PT0 基线测量先行（不改代码）**：按 6.4.1 用一次 ≥1h 真实任务填基线表（命令见 7.2），之后所有提效 PR 以该表为对照。
5. **PT0 工程优化可与 P1① 并行**：批量姿态 + 解码重叠 + 抽帧复用与判定语义正交，带开关与一致性回归即可推进；PT0-4 短路扩展、PT1-3 分片跟踪必须等 P2 评测集。**本轮实测修正**：PT0 出厂配置在本机仅 1.19×（推理瓶颈），PT1-4（基础分析离线批处理）与 PT1-1（understanding 批处理）优先级应提前；PT0 开关保持默认关，经评测集回归后再议默认开启。
6. **understanding 批处理（PT1-1）排在证据/模型冻结之后**：避免“先并行化再改证据”返工；self-consistency/4B 落地后按新显存基线重定批大小（见 6.4.4）。
7. **强行通过规则真实录像验收**：用含真实红灯亮起的片段，按 `L1↔C1` 画好通道，核对 3 秒去重后的 episode 数、目标选择、退回/未退回判定与证据窗口；再决定是否给少数触发加 VLM 复核。
8. **评测集覆盖信号触发样本**：P2 标注集需包含“信号亮 + 进入 + 退回/未退回/无人”各若干例，单独统计强行通过规则的 precision/recall，避免只评通行事件。
9. **真实 155 文件目录全链路验收**：用 C5-4 目录走一遍「批量处理 → 导入 → 首帧绘制 → 选规则 → 队列连续分析」，记录导入耗时、队列稳定性、磁盘占用与失败项处理；这是下一步最直接的实战验证。
10. **大文件夹导入后台化（按验收结果决定）**：若 155 文件同步导入过慢或阻塞页面，改为后台 job + 逐文件进度轮询，并把 500 上限改为可配置；当前实现保持同步是为了简单可控。
