# 信号区触发事件分析 · 开发方案

> 版本：设计稿 v1.0（2026-11-23）
> 状态：**已按本方案实现**（规则类别 / 通道配对 / 信号 episode 去重 / 入口线+辅助区 / 确定性退回判定 / 前端配对表）。
> 验证：`backend/tests` + `Rules/tests` 全量通过（含新增 `test_cue_trigger.py` 与 event-rules 用例）；前端 46 用例、eslint、build 通过。
> 范围：VLM 语义分析链路（`Rules/src/ultra_rules/vlm` + `backend/app` + `frontend/src`）
> 关联：`dev-guide.md` 第 3 / 4.7 / 4.8 / 6 节

---

## 1. 需求定义（本轮确认）

以规则“该人员强行通过安检门且未退回”为样板：

1. **不判断“有无阻拦”**。核心判定只有一条：**人员是否退回**。
2. **触发源反转**：
   - 该规则不再由计数区（POI）产生候选；
   - 改为扫描整段视频内**信号区（cue）的亮起区间**，**信号区每亮起一次，才做一次判断**；
   - 例：计数区累计 1000 人次通过，但信号区只亮 10 次 → 只产生 10 次判断；其余时段视为正常，不分析。
3. **信号区去重**：同一信号区**从第一次触发起的 3 秒内**视为同一次事件，不重复产生结果。闪烁、短暂熄灭后再亮都合并为一个“信号事件（episode）”。
4. **默认一一配对（入口线为通道基准）**：每个通道 = 入口计数线 + 信号区（按编号对应）；计数区为**可选辅助**，同编号自动挂到该通道。编号 1 的信号区亮起，只分析编号 1 的通道；编号 2 同理。
5. **退回判定**：
   - 主判据（必填的入口计数线）：与进入方向相反的 `line.cross`；
   - 辅助判据（可选计数区）：目标来自计数区时，窗口内 `zone.leave` 也算退回。
6. **同一信号区间默认只有一人**；偶尔多人时取窗口内**距触发时刻最近**的一人，不按人拆成多条。
7. **规则类别（编号）**：规则名称可自定义，但规则背后有稳定的类别编号/代码，用于决定：
   - 该规则需要哪些区域（计数区 / 计数线 / 信号区 / 完整性区）；
   - 触发方式与判定逻辑；
   - 分析开始前的强制校验项。
8. **分析前强制校验**：按“已绘制区域 × 已启用规则”做组合校验。例如只绘制了完整性区时，必须只勾选完整性规则，否则在开始分析时给出明确提示。
9. **`preSeconds` 加长**：强行通过类别默认使用更长的回看窗口（建议 8 秒），覆盖“报警灯在动作之后才亮”的情况。
10. **无人员通过**：信号亮起但配对计数 POI 窗口内无人 → 仍输出一条 `no_match + no_subject` 审计记录，不调用大模型。
11. **存量数据**：已有分析记录按测试数据处理，不做兼容、不重算；新逻辑只对未来任务生效（规则模板仍需类别迁移，因为它们会被未来任务使用）。

---

## 2. 现状与差距（为什么必须改）

| 问题 | 现有实现 | 影响 |
|---|---|---|
| 候选由计数区触发 | `observations.py:191-194` 按 POI 生成候选并归属规则 | 1000 次通行就产生 1000 个候选，无法做到“信号亮几次判几次” |
| 信号区是规则级并集 | `decision.py:62-77` 把所有 `event.cueIds` 的亮灯区间合并；`decision.py:122-130` 窗口内任一信号区亮即 `on` | 多信号场景无法区分“是哪个信号区触发的”，A 门候选可能因 B 门灯亮被判满足 |
| 证据无归属 | `decision.py:130` 只记录 `{cueIds 全集, state, onsetSec}`；前端 `ResultCard.tsx:71` 只展示 state/onset | 事后无法追查触发信号区，也就是上一轮确认的“缺口” |
| 采样/提示词强依赖 POI + 目标人 | `sampling.py:95` 按 `candidate["poiId"]` 取 POI 画黄框；`sampling.py:101` 按 `trackSegmentId` 标 TARGET；`prompting.py` 要求“只判断粉红框 TARGET” | 纯信号候选（无 POI、无轨迹）不能直接接入，需要新的候选/目标定位策略 |

**结论**：必须在“候选生成”层反转，而不是在 `evaluate()` 内打补丁。信号区时间线已经存在（`cues.py` 的 `detect_all` 全时长扫描），反转**不增加视频解码成本**，只改变“谁产生候选、候选归属哪条规则、判定用哪对区域”。

---

## 3. 规则类别（category）

### 3.1 类别定义

在规则模板中新增稳定字段 `category`（不随名称变化）：

```python
RuleCategory = Literal["poi_event", "forced_passage", "integrity"]
```

| 类别号 | category | 中文名（可翻译） | 必需区域 | 触发方式 | 判定路径 |
|---|---|---|---|---|---|
| 1 | `poi_event` | 通行 / 计数 / 通用语义 | 计数线或计数区（至少一个） | 现有 POI 触发类型（`line.cross` / `zone.enter` / `zone.leave` / `initial_inside`） | 现有链路：候选 → 确定性门禁（如绑定信号区）→ VLM |
| 2 | `forced_passage` | 强行通过（信号联动） | **入口计数线 + 信号区，按编号 1:1 配对；计数区可选辅助** | 信号区亮起（`cue.on`，3 秒去重） | 信号亮（触发即成立）+ 目标正向跨线/进入 + `retreatSeconds` 内未退回 → `match`；反向跨线（或计数区 `zone.leave`）→ `no_match` |
| 3 | `integrity` | 完整性 | 完整性区（至少一个） | `integrity.change` | 现有 CV 检测路径（`qwen_worker._process_integrity`） |

> “类别号”是稳定编号（1/2/3），只用于后台关联与展示，不随规则名称变化；`category` 字符串是代码契约，二者一一对应。

### 3.2 类别决定的行为

| 用途 | 规则 |
|---|---|
| 触发类型 | `forced_passage` → 强制 `triggerKinds = ["cue.on"]`；`integrity` → 强制 `["integrity.change"]`；`poi_event` → 沿用用户勾选 |
| 必需区域 + 绑定 UI | 见上表；`forced_passage` 的绑定 UI 用“配对表”（见 4.3），不再分别勾 POI/信号区 |
| 分析前校验 | 见第 7 节 |
| 候选生成 | `forced_passage` → 第 5 节的 `_cue_pairs`；其余沿用现有路径 |
| 判定函数 | `forced_passage` → `qwen_worker._process_cue_candidate`（配合 `decision.select_cue_target` / `retreat_event`，第 6 节）；`poi_event` → 现有 `decision.evaluate`；`integrity` → 现有完整性路径 |

### 3.3 数据与迁移

- `EventDefinition` 增加 `category` 字段，默认 `"poi_event"`（保证存量规则行为不变）。
- 规则库 `Rules/data/vlm/event-rules.json` 的一次性迁移（在 `backend/app/event_rules.py:_load` 内，幂等）：
  - `id == "preset_integrity"` 或 `triggerKinds` 含 `integrity.change` → `integrity`；
  - 其他存量规则 → `poi_event`。
  - 自定义的“强行通过”规则无法可靠自动识别，编辑器需提示“该规则类别为通行事件，如需信号联动请改为强行通过”。
- 内置默认规则 `Rules/configs/vlm_events.json` 的 `forced_passage_unopposed` 增加 `"category": "forced_passage"`，并按本轮需求把描述/conditions 从“无人阻拦”改为“未退回”。
- `PRESET_INTEGRITY_RULE`（`event_rules.py:37-49`）增加 `category: "integrity"`。

**类别代码是稳定的公共契约**：一旦发布，不重命名、不复用；名称只做展示与 i18n。

---

## 4. 区域编号与默认配对

### 4.1 `pairIndex`

- 前端 `CountingZone` / `SignalRegion` / `CountingLine`（`frontend/src/api/contracts.ts:45-50`）增加 `pairIndex: number`。
- 后端 `Poi`（无论 line / zone）/ `CueDefinition`（`contracts.py`）增加 `pairIndex: int | None = None`（可空，兼容 CLI 与旧请求）。
- **编号分配（按类型独立编号）**：入口线、计数区、信号区各自取“本类型当前未使用的最小正整数”。因此 `L1`、`Z1`、`C1` 天然构成通道 1；删除区域不重排、不复用已删除编号；新区域继续取最小空闲号。
- **唯一性校验**：同一类型内 `pairIndex` 必须唯一；同一编号允许“一条入口线 + 一个辅助计数区 + 一个信号区”共存。后端 `resolve` 再校验一次，重复编号 → 422。
- 编号与区域名称解耦：`L1` / `Z1` / `C1` 名字只是默认名，可改名；配对永远按 `pairIndex`。
- 编号只存活于本次分析（区域配置本来也不跨会话持久化，`useConsoleStore.ts:66`），但会随 `request.json` 冻结进任务快照，便于复核。

界面表现（`EventEditOverlay.tsx`）：
- 计数 POI、信号区在画布角标显示编号与配对关系（如 `Z1 ↔ C1`、`L2 ↔ C2`）；
- 绘制计数 POI 时若已有同号信号区，直接显示配对提示；反之亦然。

### 4.2 配对规则（forced_passage）

- 规则绑定：
  - `poiIds`：入口计数线（必填）+ 可选辅助计数区；
  - `cueIds`：参与配对的信号区（必填）；
  - 两者为空时沿用“全部适用”语义：全部入口线 + 同号辅助区 / 全部信号区。
- **通道** = `{入口线 L_i, 可选计数区 Z_i, 信号区 C_i}`；配对按 `pairIndex`。
- **完整性校验**：
  - 入口线的 `pairIndex` 集合必须与 `cueIds` 的 `pairIndex` 集合**完全相等**；否则 422，并报告缺失编号；
  - 每条入口线必须设置 `entryDirection`（否则无法定义“正向进入/反向退回”）；
  - 同一编号出现两条入口线或两个信号区 → 422；
  - 显式绑定的辅助计数区若无同号入口线 → 422；未显式绑定的多余计数区忽略。
- 只配对、不猜测：不做“最近距离”自动配对，编号不同就是不同通道。

示例：通道1 = `L1 + Z1(可选) ↔ C1`，通道2 = `L2 ↔ C2`；`C1` 亮起只分析通道 1，绝不触发通道 2。

> 注意区分：`zone` 是计数区（无方向），`line` 是计数线（有方向）。本事件要求画入口线，是因为“退回”只能靠反向跨线可靠判定；计数区只作辅助证据。

### 4.3 规则编辑器的配对 UI

`forced_passage` 类别下，`EventRulesEditor` 的区域列改为**配对表**：

```
☑ 编号1：计数区 Z1  ↔  信号区 C1
☑ 编号2：计数线 L2  ↔  信号区 C2
⚠ 编号3：计数区 Z3  ↔  （缺少信号区）
```

- 勾选/取消整对；
- 未配对行不可启用该规则；
- 空白绑定（全部）时展示全部已绘制配对；
- 显式“跨号重配”UI 不纳入本期（默认按编号；如后续需要再加映射表）。

---

## 5. 触发与候选模型

### 5.1 信号时间线（已有，无需新增解码）

`cues.py` 的 `detect_timeline` 已按 `cue.sampleFps` 扫描全视频，经 `_hysteresis` 合并为 `intervals: [[on, off], ...]`，写入 `cues.json`。本方案直接消费这些区间。

### 5.2 候选生成：`_cue_pairs`

在 understanding 阶段（`qwen_worker.run`）新增，参照完整性路径 `_integrity_pairs`（`qwen_worker.py:230-255`）：

```python
def merge_episodes(intervals, dedupe_seconds):
    """从第一次触发起的 dedupe_seconds 内视为同一事件。"""
    episodes = []
    for start, end in intervals:
        if episodes and start - episodes[-1]["interval"][0] <= dedupe_seconds:
            episodes[-1]["interval"][1] = max(episodes[-1]["interval"][1], end)
            episodes[-1]["rawTriggerCount"] += 1
        else:
            episodes.append({"interval": [start, end], "rawTriggerCount": 1})
    return episodes


def _cue_pairs(request, events, timelines):
    """每条 forced_passage 规则 × 每个通道（入口线/可选区/信号区）× 每个信号事件 → 一个候选。"""
    lines = {p.id: p for p in request.pois if p.kind == "line" and p.pairIndex is not None}
    zones = {p.id: p for p in request.pois if p.kind == "zone" and p.pairIndex is not None}
    cues = {c.id: c for c in request.cues if c.pairIndex is not None}
    pairs = []
    for event in events.values():
        if event.category != "forced_passage":
            continue
        line_by_index = {lines[i].pairIndex: lines[i] for i in event.poiIds if i in lines}
        zone_by_index = {zones[i].pairIndex: zones[i] for i in event.poiIds if i in zones}
        cue_by_index = {cues[i].pairIndex: cues[i] for i in event.cueIds if i in cues}
        for index, line in line_by_index.items():
            cue = cue_by_index.get(index)
            if cue is None:          # resolve() 已拒绝不完整配对；这里只防御
                continue
            raw = timelines.get(cue.id, {}).get("intervals", [])
            for episode_index, episode in enumerate(merge_episodes(raw, event.cueDedupeSec)):
                pairs.append((event, line, zone_by_index.get(index), cue, episode_index,
                              episode["interval"], episode["rawTriggerCount"]))
    return pairs
```

候选结构（不写入 `candidates.jsonl`，在 understanding 内存生成）：

```json
{
  "candidateId": "cue-<eventId>-<cueId>-<episodeIndex>",
  "triggerKind": "cue.on",
  "cueId": "cue_a1b2c3",
  "cueLabel": "C1",
  "pairIndex": 1,
  "poiId": "line_x9y8z7",
  "poiLabel": "L1",
  "poiKind": "line",
  "cueInterval": [12.4, 13.1],
  "rawTriggerCount": 2,
  "triggerSec": 12.4,
  "trackId": 12,
  "trackSegmentId": "12-3",
  "direction": null
}
```

- `cueInterval` 是去重合并后的信号事件区间，证据窗口覆盖合并后的整段；
- `rawTriggerCount` 记录被合并的原始亮起次数，便于审计（例如“本次事件由 2 次闪烁合并”）。

### 5.3 与现有 POI 候选隔离（防重复）

`forced_passage` 的 `triggerKinds` 固定为 `["cue.on"]`，而 `observations.py:191-194` 的规则归属判断是 `candidate["triggerKind"] in e.triggerKinds`，因此 POI 候选**天然不会**进入该规则；不需要额外排除逻辑，也不会出现同一事件两份结果。

### 5.4 支撑候选的持久化（关键实现点）

现有 `observations.py:191-194` **只把“至少被一条规则直接使用”的候选写入 `candidates.jsonl`**。如果本次只启用了 `forced_passage` 规则（其触发类型是 `cue.on`），则 `zone.enter` / `zone.leave` 候选不会命中任何 `triggerKinds`，**不会落盘**；而 cue 路径恰恰需要这些轨迹事件来选择 TARGET、判断退回。

因此必须扩展候选持久化策略（二选一，推荐 A）：

- **A. 支撑候选**：在 `observations.py` 的适用性判断旁增加“支撑”判断：
  ```python
  support_kinds = ("line.cross", "zone.enter", "zone.leave", "initial_inside")
  support = [e for e in request.events
             if e.enabled and e.category == "forced_passage"
             and candidate["triggerKind"] in support_kinds
             and (not e.poiIds or candidate["poiId"] in e.poiIds)]
  if applicable or support:
      candidate["eventDefinitionIds"] = applicable        # 可为空：不直接产生 POI 判定
      candidate["supportRuleIds"] = [e.id for e in support]
      append_json(run_dir / "candidates.jsonl", candidate)
  ```
  - 计数线参与配对时，支撑行就是 `line.cross`（含正反两个方向），用于“进入目标”和“反向退回”判定；
  - 计数区参与配对时，支撑行是 `zone.enter/zone.leave/initial_inside`。
  - 这些行只作为 cue 路径的输入；`eventDefinitionIds` 为空时，现有 `_process_candidate` 循环天然不产生判定，`total` 统计不受影响；
  - `candidate_groups()`、`retreat_candidate_ids()` 会把它们纳入，退回检测可直接复用。
- B. 在 understanding 阶段直接从 `observations.sqlite` 重新推导进入/离开事件（不落候选）。实现更重、容易与 tracking 语义漂移，不推荐。

补充约束：

- 支撑行只针对 `forced_passage` 规则绑定的计数 POI，不会把无关区域的全部事件都落盘；
- 每行只是元数据（时间、track、POI、方向），不含帧；1000 人次的量级仅几百 KB；
- `candidates.jsonl` 的消费者（entry 过滤、retreat、进度）都不假设 `eventDefinitionIds` 非空，需补一条单元测试锁定该行为；
- understanding 遍历 POI 候选时对 `eventDefinitionIds` 为空的行直接跳过（只作为 cue 路径的支撑数据），避免无意义调用。

### 5.5 粒度与去重

- **一个信号事件（episode）= 一次判断**（episode 内的多个人不拆成多条，目标选择见 6.1）。
- **去重锚点**：以 episode 的**第一次触发时刻**为锚，`dedupeSeconds = 3.0` 内的后续亮起都并入同一 episode；超过 3 秒的亮起开启新 episode。
  - 例：`[10.0,10.2] [11.0,11.2] [12.5,13.0]` → 一个 episode `[10.0,13.0]`（3 次原始触发，1 条结果）；
  - 例：`[10.0,10.2] [13.1,13.3]` → 两个 episode（间隔 3.1s）；
  - 合并后的 episode 区间用于证据窗口和目标搜索，`rawTriggerCount` 记录原始触发次数。
- **去重作用域**：按 `(规则, 信号区)` 独立去重。不同信号区（C1/C2）互不抑制；不同规则绑定同一信号区时，各自产生结果（同一触发可被多条规则使用）。
- 信号闪烁毛刺已由 `minOnSec` / `minOffSec` 滞回合并；3 秒去重是第二道合并，二者叠加。
- 信号长时间常亮 = 一个 episode = 一次判断；如需“长亮期间按固定间隔重复判断”，后续加 `repeatSec`（本期不做）。
- **结果 ID**：`resultId = f"cue-{cueId}-{episodeIndex}-{event.id}"`，与现有 `state["completed"]` 配合，实现断点续跑不重复。
- **全程无信号亮起**：不产生任何 forced_passage 结果；`summary.json` 记 `reason = "no_candidates"`（若整个任务只有该规则）。

### 5.6 进度统计

`qwen_worker.run` 的 `totalRequests` 增加 cue 配对数：

```
total = Σ POI 候选事件数 + integrity pairs + cue pairs
```

`progress` 上报逻辑不变。

---

## 6. 目标人选择与退回判定

### 6.1 目标人选择（每个信号事件选一个 TARGET）

在配对通道的候选池中查找：

1. 时间窗 `W = [episode.start - preSeconds, episode.end + postSeconds]`；强行通过类别默认 `preSeconds = 8`（覆盖“报警灯在动作之后才亮”），`postSeconds = 6`；
2. **入口计数线（主）**：取 `triggerKind == "line.cross"`、`poiId == 配对入口线` 且方向为 `entryDirection` 的候选；
3. **可选计数区（辅）**：没有入口线候选时，取同号计数区的 `zone.enter`；再没有则 `initial_inside`；
4. 取距 `episode.start` 最近的一条（同时刻优先取更早的）；
5. 再没有 → 无目标；
6. **无目标**：仍输出一条审计记录，`decision = "no_match"`、`qualityWarnings = ["no_subject"]`、`reason = "信号区「C1」亮起，但配对通道「L1」窗口内无人通过"`；不调用大模型。

说明：
- 一个信号事件只选一个 TARGET（用户确认：同一信号区间基本只有一人；偶尔多人取最近者），其余在场人员作为旁框上下文（沿用 `otherPeople`）；
- 该 TARGET 的 `trackSegmentId` 用于后续退回判断与采样标注；
- 目标搜索用去重合并后的完整 episode 窗口，而不是只看首次触发点，避免“第二次闪烁时才通过”的人被漏掉。

### 6.2 退回判定

- 退回窗口：**从目标进入时刻 `entry_t` 起算** `retreatSeconds`，即 `[entry_t, entry_t + retreatSeconds]`，默认 `retreatSeconds = 5.0`（沿用 `decision.py:15` 的 `RETREAT_SECONDS`）。
  - 不用 `[onset, onset + 5]`：若人员是在灯亮后 3 秒才进入，onset 起算只剩 2 秒观察窗；从进入时刻起算更符合“进入后是否退回”。
- 退回条件（同一 `trackSegmentId` + 同一配对通道）：
  - **入口计数线（主）**：窗口内出现与 `entryDirection` 相反的 `line.cross`；
  - **可选计数区（辅）**：目标来自计数区（没有可用的入口线候选）时，窗口内 `zone.leave` 也算退回。
- 判定：
  - 观察到退回 → `decision = "no_match"`，`reason = "信号亮起后目标在 X.Xs 退回（反向跨线/离开计数区），不算强行通过"`；
  - 未观察到退回 → `decision = "match"`，`reason = "信号区「C1」亮起，通道「L1」内目标进入后 X 秒内未退回"`；
  - 跟踪中断 → 维持现有语义：`match` + `qualityWarnings=["tracking_gap"]`（遮挡不等于退回）；
  - `identityAmbiguous` → `uncertain`（若开启 VLM 复核则走复核，否则保留 `uncertain` 并标记）。

**模型选择说明**：入口线是必填通道基准，反向跨线是唯一无歧义的退回证据；计数区只为“人员已在区域内/入口线未捕获”等边界情况提供辅助，不覆盖入口线的结论。

实现注意：
- `decision.retreat_candidate_ids()`（`decision.py:19-38`）会把“5 秒内反向跨越”的候选从全局 entry 候选中剔除；cue 路径做退回检测时必须使用**未过滤**的原始候选池，否则看不到退回事件。
- 复用 `decision.retreat_event()`（由原 `_retreated` 抽出）与 `decision.select_cue_target()`；现有 `evaluate()` 语义不变。

### 6.3 确定性优先，VLM 可选

- 默认：`forced_passage` **只用确定性判定**（信号 + 进入 + 未退回），但每个结果仍生成证据窗口/抽帧供人工复核（复用 `_build_deterministic_evidence`）。
- 可选：规则级 `vlmReview`（新增字段，默认 `false`）为 `uncertain` / `no_subject` / 目标不明确的情况增加一次 VLM 复核；提示词在现有 `prompting.py` 的 `cue_evidence` 基础上增加：
  - 触发信号区名称/编号、亮起区间；
  - 配对计数区名称；
  - 退回判定结果与时刻；
  - 事件描述（此时应已改为“强行通过且未退回”）。
- 这样“1000 次通行 → 10 次信号触发 → 10 次判断”的成本可控；需要更强语义复核时再打开 VLM。

---

## 7. 分析前强制校验（核心）

### 7.1 能力集合

从前端 `eventsConfig`（即本次请求的 `pois` / `cues` / `integrityZones`）计算：

```
has_lines     = 任意 kind == "line"
has_zones     = 任意 kind == "zone"
has_counting  = has_lines or has_zones
has_cues      = 任意信号区
has_integrity = 任意完整性区
```

### 7.2 校验规则

对**每条已启用**规则，按 `category` 校验：

| category | 必须满足 | 失败提示（示例） |
|---|---|---|
| `poi_event` | `has_counting` | “规则《{name}》（通行事件）需要至少一个计数线/计数区” |
| `forced_passage` | `has_lines and has_cues`（计数区可选），入口线与信号区按 `pairIndex` 完整配对、同号入口线唯一、入口线已设置方向 | “规则《{name}》（强行通过）需要入口计数线 + 信号区，且编号一一对应；缺少：入口线 L2 ↔ 信号区 C2” |
| `integrity` | `has_integrity` | “规则《{name}》（完整性）需要至少一个完整性区” |

**组合场景提示**（对应本轮需求举例）：

- 只画了完整性区、却启用了非完整性规则：
  > “当前仅绘制了完整性区，只能启用完整性规则。请停用《…》或补充绘制计数区/信号区。”
- 只画了计数区、没有信号区、启用了强行通过规则：
  > “强行通过规则需要计数区与信号区按编号一一对应，请补充绘制信号区（如 C1）。”

### 7.3 校验位置

- **后端权威**：`backend/app/event_rules.py:resolve()` 在构建 `RunRequest` 前执行 7.2，失败返回 422，消息带规则 id/name 与缺失区域；新增校验放在现有 `poi_ok / integrity_ok / cue_ids` 校验附近（`event_rules.py:167-190`）。
- **前端镜像**：`frontend/src/utils/eventRules.ts:bindingError()` 增加 category 分支；`EventRulesEditor.tsx:39` 的 `ready` 纳入配对完整性；错误文本展示在对应规则行与面板顶部。
- 开始按钮在 `ready=false` 时禁用；若用户仍从 API 提交，后端 422 兜底。

### 7.4 类别与触发类型一致性

- `forced_passage`：后端强制 `triggerKinds=["cue.on"]`，编辑器隐藏触发类型多选；
- `integrity`：强制 `["integrity.change"]`；
- `poi_event`：保留现有触发类型编辑；
- 同一条规则禁止 `cue.on` 与 POI 触发类型混用（422）。

---

## 8. 数据模型与契约变更清单

### 8.1 后端 / Rules

| 文件 | 变更 |
|---|---|
| `Rules/src/ultra_rules/vlm/contracts.py` | `Trigger` 增加 `"cue.on"`；`EventDefinition` 增加 `category`、`cueDedupeSec`（默认 3.0）；`Poi` / `CueDefinition` 增加 `pairIndex`；model validator：`forced_passage` 必须含 `cue.on` 且不含 POI 触发类型 |
| `Rules/src/ultra_rules/vlm/observations.py` | 候选持久化增加“支撑候选”（5.4）：`forced_passage` 绑定区域的 `line.cross / zone.enter / zone.leave / initial_inside` 即使不直接触发规则也写入 `candidates.jsonl` |
| `Rules/src/ultra_rules/vlm/qwen_worker.py` | 新增 `_cue_pairs`、`_process_cue_candidate`；`run()` 汇总 cue pairs、进度、快照；结果标记 `triggerKind="cue.on"` |
| `Rules/src/ultra_rules/vlm/decision.py` | 抽出 `retreat_event` / `select_cue_target` / `merge_cue_episodes`；导出 `tracked_through`；保留现有 `evaluate` 不变 |
| `Rules/src/ultra_rules/vlm/cues.py` | 无功能改动；可在 `cues.json` 中冗余记录 `pairIndex`/label 便于排障（可选） |
| `Rules/src/ultra_rules/vlm/sampling.py` | `build_sample` 支持 cue 候选：仍按 `poiId`（配对计数区）画黄框；可选把信号区画成橙色虚线框；无 TARGET 的分支不进入 |
| `Rules/src/ultra_rules/vlm/prompting.py` | 仅可选 VLM 复核使用：注入触发信号区/区间/退回证据 |
| `backend/app/event_rules.py` | 规则库 `category` 持久化 + 迁移；preset integrity 加类别；`resolve()` 类别校验/配对校验/触发类型归一 |
| `backend/app/vlm_api.py` | `results` 的 `markers` 增加 `cueId`、`cueLabel`、`pairIndex`（`vlm_api.py:563-566`） |
| `Rules/configs/vlm_events.json` | 默认规则加 `category: "forced_passage"`；描述/条件改为“未退回” |
| `backend/tests/`、`Rules/tests/` | 见第 11 节 |

### 8.2 前端

| 文件 | 变更 |
|---|---|
| `frontend/src/api/contracts.ts` | `CountingZone` / `SignalRegion` 增加 `pairIndex`；`EventRule` 增加 `category` |
| `frontend/src/store/useConsoleStore.ts` | 类型透传（`EventsConfig`） |
| `frontend/src/ui/viewer/widgets/EventEditOverlay.tsx` | 绘制时按类型分配 `pairIndex`（最小空闲号）；画布编号角标 `#1`，`L1/Z1/C1` 同号成通道 |
| `frontend/src/utils/vlm.ts` | `vlmPois` / `vlmCues` 带上 `pairIndex`；规则行类型带 `category`（本地校验/展示用，提交仍走规则库版本） |
| `frontend/src/utils/eventRules.ts` | `newRule`/`presetRule`/`parseRuleImport` 增加 `category`；`bindingError` 按类别校验配对/区域 |
| `frontend/src/ui/vlm/EventRulesEditor.tsx` | 类别选择器；`forced_passage` 的配对表；校验提示（`prepare()` 仍只提交选择，类别由后端按规则库版本解析） |
| `frontend/src/ui/vlm/VlmPanel.tsx` | 开始前的组合校验错误展示（后端 422 也在此展示） |
| `frontend/src/ui/review/ResultCard.tsx` | 展示触发信号区/配对计数区/退回结论 |
| `frontend/src/utils/review.ts`、`ui/review/ReviewVideo.tsx` | marker 标签：`trackId` 为空时用信号区名称/编号，而非 `#null` |
| `frontend/src/i18n/locales/{zh,en}.json` | 类别名、`cue.on`、校验文案、证据标签 |
| 前端测试 | 见第 11 节 |

### 8.3 结果字段（`cueEvidence` 扩展，向后兼容）

```json
{
  "cueIds": ["cue_a1b2c3"],
  "triggerCueId": "cue_a1b2c3",
  "cueLabel": "C1",
  "pairIndex": 1,
  "poiId": "zone_x9y8z7",
  "poiLabel": "Z1",
  "poiKind": "zone",
  "state": "on",
  "onsetSec": 12.4,
  "interval": [12.4, 13.1],
  "rawTriggerCount": 2,
  "retreated": false,
  "retreatSec": null
}
```

- 保留 `cueIds/state/onsetSec`；
- 新增字段供结果卡片、复核窗口、审计导出使用。

---

## 9. 处理流程总览

```
segment ──► tracking ──► cues(全时长，已有) ──► integrity(可选)
                              │
                              ▼
                     understanding
                       ├─ POI 候选（poi_event 规则，原有路径）
                       ├─ integrity pairs（原有路径）
                       └─ cue pairs（新增）
                            for 每条 forced_passage 规则
                              for 每个通道 (L_i + Z_i(可选) ↔ C_i)
                                merge_episodes(C_i 的亮起区间, 3s)
                                for 每个 episode
                                  选 TARGET = 窗口内最近正向跨 L_i 的目标
                                  retreated = 窗口内反向跨 L_i（或 Z_i 的 zone.leave）
                                  输出 1 条结果（cue.on）
```

---

## 10. 兼容性与边界

- **存量规则**：默认 `poi_event`，行为不变；`cueIds` 并集语义仅保留给 `poi_event`（本期不改）。规则模板仍需一次性写入 `category`，因为未来任务会读取它。
- **存量任务/结果**：已有分析记录按测试数据处理——不迁移、不重算、不为旧结果做展示兼容；新逻辑只对**新提交的任务**生效。
- **旧请求**：不带 `pairIndex` 的 `forced_passage` 运行会被校验拒绝（422），前端新版会始终发送。
- **区域删除/重绘**：编号不复用；出现缺号时校验直接报缺配对，不做自动猜测。
- **多规则共享同一配对**：允许（同一 `L1↔C1` 可被多条强行通过规则使用），每条规则各自出结果。
- **同一信号事件多人**：只取距触发最近者做 TARGET，其余作为旁框上下文；不按人拆结果（与“1000→10”一致）。
- **去重作用域**：`(规则, 信号区)` 独立，3 秒窗口以 episode 首次触发为锚；不同信号区/不同规则互不抑制。
- **信号常亮跨多小时**：一个 episode 一次判断；如需周期复判后续加 `repeatSec`。
- **信号不亮但有人通过**：不产生任何结果（符合“红灯没亮代表一切正常”的语义）。

---

## 11. 测试与验收

### 11.1 后端单元/集成（`Rules/tests` + `backend/tests`）

1. **类别迁移**：旧规则无 `category` → `poi_event`；含 `integrity.change` → `integrity`；预设 → `integrity`；迁移幂等。
2. **配对生成**：`L1(+Z1 可选)↔C1`、`L2↔C2` 产生正确候选；缺号/多号/同号两条入口线或两个信号区 → 422 且消息包含缺失编号。
3. **一对一隔离**：C1 的 3 个 episode 只生成通道 1（L1，可带 Z1）的 3 条候选；C2 同理；C1 不出现 L2/Z2。
4. **3 秒去重**：`[10,10.2][11,11.2][12.5,13]` → 1 个 episode；`[10,10.2][13.1,13.3]` → 2 个；合并后 `rawTriggerCount` 正确；不同信号区/不同规则不互相抑制。
5. **触发次数**：构造 1000 个 POI 候选 + 10 个信号事件 → `forced_passage` 只产生 10 条结果；`poi_event` 规则行为不变。
6. **退回判定**：计数区未退回 → match、5s 内 `zone.leave` → no_match；计数线反向跨线 → no_match、未反向 → match；跟踪中断 → match + `tracking_gap`；无目标 → no_match + `no_subject`；退回窗口从目标进入时刻起算。
7. **断点续跑**：`resultId` 稳定；重复运行不重复追加。
8. **校验**：仅完整性区 + 非完整性规则 → 422；强行通过缺信号区/缺配对 → 422；前端 `ready=false`。
9. **触发隔离**：`cue.on` 与 POI 触发混用 → 422。
10. **证据字段**：结果含 `triggerCueId/cueLabel/pairIndex/poiId/poiKind/rawTriggerCount/retreated`。

### 11.2 前端

- 绘制编号分配（入口线/计数区/信号区各自独立编号、删除后不复用、最小编号优先）；
- 配对表勾选与未配对提示；
- 类别选择后触发类型/绑定 UI 的切换；
- `bindingError` 文案与 `ready`；
- 结果卡片和复核 marker 标签（无 trackId 时显示 C1/Z1）。

### 11.3 验收标准（Definition of Done）

- 只画完整性区时，启用强行通过规则会在开始分析前被拦截，提示明确；
- 画 `L1(+Z1)↔C1`、`L2↔C2`：C1 亮 10 次、C2 亮 5 次（均按 3 秒去重后），且计数线各有大量通行 → 强行通过规则正好产生 15 条结果（10 条通道1、5 条通道2），每条都带触发信号区与配对入口线；
- C1 的触发绝不出现在通道 2（L2/Z2）的结果里；
- 同一信号区在 3 秒内的连续亮起只产生 1 条结果，结果记录合并次数；
- 退回/未退回判定（入口线反向跨线，计数区 leave 仅辅助）与真实视频人工复核一致；
- 现有 `poi_event` / `integrity` 运行的判定结果不发生变化（旧记录本身不做兼容）。

---

## 12. 施工顺序（建议）

| 阶段 | 内容 | 可独立验证 |
|---|---|---|
| P1 | contracts：`category` / `cue.on` / `pairIndex`；规则库迁移；前后端类型 | 存量测试全绿 |
| P2 | 编号分配 + 配对 UI + 前后端校验（先不改变判定） | 完整性区场景拦截提示 |
| P3 | `_cue_pairs` + `_process_cue_candidate` + 结果/证据/进度 | 合成数据 1000→10 |
| P4 | 结果卡片/复核 marker/i18n；可选 VLM 复核 | 真实录像端到端 |
| P5 | dev-guide 更新 + 回归 | 全量测试 |

---

## 13. 本方案默认值与待确认项

**已确认（按本轮指令，直接施工）**

- 信号去重：同一信号区从第一次触发起的 **3 秒**内算同一次事件；固定锚点，不滚动顺延；一个 episode = 一条结果。
- 退回判定：**入口计数线反向跨线为主**，计数区 `zone.leave` 仅作辅助；**退回窗口从目标进入时刻起算 5 秒**。
- **入口计数线必填 + 信号区必填 + 计数区可选辅助**；同号组成一个通道。
- 同一信号事件多人时取最近者，不按人拆结果。
- `preSeconds = 8`，并在**规则编辑页**提供可调项（随任务快照冻结）。
- 无人通过：保留 `no_match + no_subject` 审计记录。
- 存量分析记录不兼容、不重算，只对未来的新任务生效；规则模板仍做一次性 `category` 迁移。
- 默认确定性判定（不调 VLM），VLM 复核默认关闭。

**施工默认值**

- `retreatSeconds = 5.0`（沿用现有常量），退回窗口从目标进入时刻起算；
- `cueDedupeSec = 3.0`（规则级可改）；
- `postSeconds = 6`（强行通过类别默认）；
- TARGET = 窗口内距首次触发最近的入口线候选（无入口线候选时用同号计数区）；
- 入口线与信号区按 `pairIndex` 一一配对，同号入口线唯一；计数区可选、同号挂接。

**仍需确认**

1. 是否需要显式跨号重配 UI（本期默认不做，按编号固定配对）。
