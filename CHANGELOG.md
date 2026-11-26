# Changelog

> 从 `dev-guide.md` 拆出的历史开发记录，按原章节保留。新的变更请按日期追加到顶部，并保持 dev-guide 只描述稳定架构与开发约定。

---

## 2026-11-26

### 处理时长从实际开始计算 + 阶段耗时

- run 状态新增 `queuedAt`（入队，仅审计）与 `startedRunningAt`（worker 实际开始）；未来 run 的 `durationSec` 从 `startedRunningAt` 计算，**排队时间不再计入，也不再展示**。
- 仍在排队的新 run `durationSec` 为空，UI 显示 `—`；历史 run 没有拆分时钟时继续使用原 `startedAt` 语义（不做迁移）。
- 复核窗口与分析记录列表只显示“处理时长 + 视频时长”；排队时长不作为 UI 指标。
- runner 记录 `phaseSeconds`（segment/tracking/detect/cues/integrity/understanding）；复核窗口展示阶段耗时。
- `decoded(truncate_on_error=True)`：遇到真实 PTS 回退或解码错误时，停止在最后一个可用帧，`progress` / `summary` 记录 `truncated`、`truncatedAtSec`、`truncationReason`、`processedDurationSec`；复核窗口显示“仅处理前 X / 共 Y”的警告，run 状态置为 `partial`。
- 单帧缺失 PTS 不截断，按“上一帧 + 一帧间隔”修复并累计 `timestampFallbackFrames`；UI 提示“N 帧缺失时间戳已修复”。`_3` 属于这一类，因此完整处理并提示，而不是中途失败。

### 分析记录视频时长

- 创建 run 时把已探测的视频时长写入 `status.json` 的 `videoDurationSec`；历史 run 由 `vlm_jobs.status` 从 `request.json` 回填。
- 复核窗口标题栏新增“视频时长”，预览尚未生成/加载时也能先看到；时间轴在 metadata 加载前用该值作为 fallback。
- 分析记录列表在“处理时长”下补充“视频时长”。

### 语义分析重复提交修复

- “开始语义分析”增加同步提交锁，快速双击不会重复进入提交函数。
- 任务队列把 active `videoId` 回传给面板；同一视频已在排队/运行时禁用开始按钮并提示“该视频已在队列中”，避免轮询刷新间隙重复入队。
- 提交完成后保留 2.5 秒 settle 窗口，覆盖 POST 返回但队列轮询尚未刷新的空档。

### 复核窗口事件时间格式

- 分析记录中的事件触发时间、条件证据时间、信号区起止/退回/进入时间、完整性起止/持续时间和证据窗口，统一显示为“x小时x分x秒”（不足 1 小时自动省略小时/分钟），不再显示原始秒数。
- 时间轴 marker 的 tooltip / aria-label 同步使用同一格式；播放时钟仍保留 `HH:MM:SS.mmm` 便于精确定位。

---

## 归档记录

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
- `.gitignore` 忽略 `models/*`（保留 `.gitkeep`）；deploy.md 权重/解释器说明与 `backend/.env.example` 已同步。
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
- **性能（2026-11-26）**：tracking + cues + integrity 已合并为 `detect` 阶段，只解码一次；30 分钟真实任务 P0 时 193.6s，P3 I/O 后 **179.7s**（旧分开阶段约 6 分钟），observations/candidates/cues/integrity 逐项一致。确定性证据默认 lazy，只写 sample manifest；165 条结果 understanding 复算 0.41s、0 预览。详见 [optimize.md](optimize.md)。
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

---

[返回 README](README.md) · [开发指南](dev-guide.md) · [性能路线](optimize.md)
