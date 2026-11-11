# 演示语料说明

本目录存放论文精读的演示语料。**PDF 与解析缓存都不进 git**——三篇论文均为
arXiv 的 non-exclusive license，作者只授权 arXiv 分发，未授权第三方再分发。
仓库是公开的，因此这里只提交清单与下载脚本。

```bash
python scripts/fetch_papers.py     # 下载 3 篇 PDF
python scripts/build_paper_index.py  # 解析 + 生成中文译文索引（需要云端密钥，约 ¥0.4）
```

## 语料清单

| 文件 | 论文 | arXiv | 选它的理由 |
| --- | --- | --- | --- |
| `rag_2005.11401.pdf` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | [2005.11401](https://arxiv.org/abs/2005.11401) | 本系统技术路线的来源；含 21M 文档索引、EM/F1 等可核对数字 |
| `lora_2106.09685.pdf` | LoRA: Low-Rank Adaptation of Large Language Models | [2106.09685](https://arxiv.org/abs/2106.09685) | 参数效率方向，表格密集，适合数字定位类问题 |
| `lost_in_middle_2307.03172.pdf` | Lost in the Middle: How Language Models Use Long Contexts | [2307.03172](https://arxiv.org/abs/2307.03172) | 直接论证「长上下文 ≠ 读得准」，是本作品「检索 + 锚点」的动机 |

## 生成物（均可重建）

| 文件 | 内容 |
| --- | --- |
| `*.chunks.json` | 解析缓存：每块的页码、bbox、行级坐标 |
| `*.zh.json` | 逐块中文译文（建库期一次性生成，运行期零调用） |

## 版权

三篇论文版权归原作者所有，依据 arXiv non-exclusive license 从 arXiv 获取，
仅用于本作品的非商业演示与评测。若权利人要求，删除对应文件即可，系统会自动
在论文列表里隐藏该篇。
