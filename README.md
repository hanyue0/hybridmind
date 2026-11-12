# HybridMind · 论文精读问答与证据锚点

> 2026 第九届传智杯 · AI 大模型创新应用挑战赛（个人赛 · B 组）参赛作品
> 参赛者：张云迪（商洛学院）

**一句话**：上传一篇英文论文，用中文提问；系统先让本地 1.5B 小模型作答，再用「答案能不能被原文背书」来校验——站得住就留在本地（零 API 费用、数据不出机），站不住才升级云端；每一条结论都挂着可点击的锚点，点了就跳到原文那一页、高亮那一行。

## 交付物

| 交付物 | 位置 | 说明 |
| --- | --- | --- |
| 在线演示 | <https://hanyue0-hybridmind.ms.show/paper> | ModelScope 创空间，2 vCPU / 8 GB |
| 演示视频 | `docs/HybridMind_演示视频_最终版.mp4` | 5 分 17 秒，1920×1080，含字幕 |
| 技术文档 | [`docs/HybridMind_技术文档.pdf`](docs/HybridMind_技术文档.pdf) | 25 页 |
| 作品介绍 | [`docs/HybridMind_作品介绍.pdf`](docs/HybridMind_作品介绍.pdf) | 11 页，对应报名表七项 |
| 代码仓库 | 本仓库 | 48 项单元测试（旧 22 + 新 26） |

## 实测结果

3 篇公开论文（63 页、328 个片段）、115 道中文题（44 数字定位 / 57 事实抽取 / 14 不可答），同一判分口径下：

| 策略 | 准确率 | 证据召回 | 引用有效 | 误拒率 | 云端占比 | 千次成本 | 平均延迟 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 全本地 1.5B | 73.0% | 69.3% | 100% | 17.8% | 0% | ¥0.00 | 15.2 s |
| 全云端 DeepSeek | **89.6%** | 91.1% | 100% | 7.9% | 100% | ¥1.66 | 1.06 s |
| **证据门控路由** | 87.8% | 90.1% | 100% | **4.0%** | **32.2%** | **¥0.51** | 8.43 s |

**准确率比全云端低 1.8 个百分点（2 道题，95% CI [−5.9, +4.0] 覆盖 0），云端调用少 68%，成本降 69%，误拒率降一半、锚定率最高。**

配套图：[`fig_paper_strategies.png`](docs/figures/fig_paper_strategies.png)、[`fig_paper_gate.png`](docs/figures/fig_paper_gate.png)、[`fig_paper_cost.png`](docs/figures/fig_paper_cost.png)。

### 一个重要否定结果

把上一版「预测本地会不会答错」的学习型路由原样搬到论文场景，5 折交叉验证 **AUC 只有 0.66**，阈值化后 89% 请求都要上云——路由等于失效。原因不是实现错误，而是错误形态变了：算术题里本地错在「多步推理保持不住」，与题目写法强相关；论文题里它错在「在上下文里挑错了那个数」，与题目写法无关。

因此本版把路由信号从**预测难度**换成**验证证据**：

```
本地先答 → 锚点对齐 → 一致性校验 → 站得住就保留，站不住才升级云端
```

门控的判别力：校验通过时本地答对率 **87.0%**，未通过时只有 **12.5%**（两组相差 74.5 个百分点）。

## 核心设计

```
PDF ──▶ ① 解析层  pymupdf 抽块，保留 页码 + 行级 bbox + 章节
          │
          ▼
      ② 检索层  BM25（英文词干 + 中文 bigram）+ 离线双语索引 → top-5 + 4 维特征
          │
          ▼
      ③ 路由层  先本地作答 ──▶ 锚点校验 ──通过──▶ 返回（0 次云端调用）
          │                        │
          │                        └─不通过─▶ 云端重答 → 重新校验
          ▼
      ④ 锚点层  约束生成 + 确定性对齐（数字/术语/译文）→ 三条一致性校验
          │
          ▼
      ⑤ 界面    答案 + 可点击引用角标 + 页图高亮 + 路由决策与账本
```

三个刻意的设计选择：

1. **离线双语索引**：中文问英文论文，BM25 词面重合度为 0。建库期一次性把每块译文写进索引（328 块约 ¥0.34），运行期零额外调用；锚点仍回指英文原文。
2. **不靠模型守规矩**：实测本地 1.5B 经常不写 `[S编号]`。锚点层用数字、英文术语、译文重合度三类与语言无关的信号做兜底对齐，模型标不标出处都能产出可高亮的锚点。
3. **判分不请大模型当裁判**：数值题比数词、短答题比别名、拒答题看是否说「原文未提及」。换判分口径时用 `scripts/rescore_rag.py` 在已有结果上离线重判，不重跑模型。

## 快速开始

```bash
# 1. 依赖
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
ollama pull qwen2.5:1.5b

# 2. 云端（可选；不配则纯本地运行）
copy .env.example .env        # 填 CLOUD_API_KEY

# 3. 语料与索引（约 ¥0.4：逐块中文翻译，一次性）
python scripts/fetch_papers.py
python scripts/build_paper_index.py

# 4. 启动
.venv\Scripts\python -m uvicorn app.web.main:app --port 8000
# 论文精读：http://127.0.0.1:8000/paper
# 旧版问答：http://127.0.0.1:8000/
```

没有独立显卡也能跑：本作品在 i5-13420H / 16 GB 单通道 / 无独显的笔记本上完成开发与全部评测，本地推理走 CPU（约 3–4 token/s）。

## 评测与复现

```bash
python -m app.eval.run_eval_rag --mode local    # 全本地
python -m app.eval.run_eval_rag --mode cloud    # 全云端
python -m app.eval.run_eval_rag --mode gated    # 证据门控（本作品）

python scripts/rescore_rag.py        # 换判分口径后离线重判
python scripts/make_figures_rag.py   # 三张图
python -m unittest discover -s tests -t .   # 48 项测试
```

旧的算术问答实验仍可复现：`python -m app.eval.run_eval --mode local|cloud|auto`，数据在 `data/eval_set.jsonl`。

## 目录结构

```
app/
  config.py              全部可调参数（阈值、价格表、后端选择）
  pipeline.py            编排：决策 → 推理 → 降级 → 记账
  rag/
    parser.py            PDF → 片段（页码、行级 bbox、章节）
    retriever.py         BM25 + 中文 bigram + 离线双语索引
    prompt.py            受约束 prompt
    anchor.py            确定性对齐 + 三条一致性校验
    qa.py                端到端：检索 → 门控 → 生成 → 锚点（含流式版本）
  router/                特征、策略、学习型分类器（14/18 维）
  eval/                  判分口径、论文评测、证据指标
  web/                   FastAPI + 零依赖前端（index.html 旧 / paper.html 新）
scripts/
  fetch_papers.py        下载 3 篇语料（PDF 不进 git，见 data/papers/README.md）
  build_paper_index.py   解析 + 逐块翻译，建离线双语索引
  gen_qa_draft.py        生成题目草案并逐字校验
  select_qa.py           筛选、去重、人工剔除
  rescore_rag.py         换判分口径后离线重判
  make_figures_rag.py    论文场景三张图
  record_demo_v2.py      CDP 录屏（演示视频的真实交互片段）
  make_voiceover.py      机器配音兜底（edge-tts）
  assemble_v2.py         按配音时长合成视频
tests/                   旧 22 项 + 新 26 项
docs/                    技术文档、作品介绍、图、AI 协作记录、视频脚本
data/papers/             语料与索引（PDF 与译文不进 git）
```

## 配置项

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LOCAL_MODEL` | `qwen2.5:1.5b` | 本地模型 |
| `CLOUD_MODEL` | `deepseek-chat` | 任何 OpenAI 兼容模型 |
| `ROUTER_MODE` | `rule` | `rule` / `learned`（论文场景用门控，不依赖它） |
| `THETA_LOCAL` | `0.45` | 规则路由阈值（论文场景另用模型内置阈值） |
| `ROUTER_MODEL_PATH_PAPER` | `data/router_model_paper.json` | 论文场景的 18 维模型 |

## 已知局限

- 样本 115 题、3 篇论文，准确率差 1.8 个百分点只相当于 2 道题，置信区间较宽；
- 表格跨行、跨页时分块会失效（101 道可答题里有 4 道检索未命中，其中 3 道是表格、1 道是同一事实多处出现，直接表现为拒答）；
- 数字可溯只能排除「凭空编造」，不能排除「抄错列」（7 条失败案例属于此类）；
- 本地推理约 15.2 s/题（CPU），门控让最终上云的题也要先等这一次；通过率低于 30% 时应直接上云。

## 许可

仅用于本届传智杯参赛与评审。第三方组件（pymupdf、FastAPI、Qwen2.5、llama.cpp）遵循各自许可证；演示语料为 3 篇 arXiv 论文，版权归原作者，依据 arXiv non-exclusive license 获取，仅用于非商业演示。
