# 公网部署指南

> 目标：交付物要求的「可公网访问的在线演示地址」。
> 现状：**已部署到 ModelScope 创空间**（Docker SDK，免费 CPU 2 vCPU / 8 GB）。
> 空间：https://modelscope.cn/studios/hanyue0/hybridmind

## 零、实际部署记录（踩过的坑，按发生顺序）

| # | 现象 | 原因 | 解法 |
| --- | --- | --- | --- |
| 1 | `git push` 报成功，但空间仍显示「未部署任何内容」 | 创空间对 Git 推送的识别有延迟/条件限制 | 改用网页「拖拽上传」文件夹 |
| 2 | 构建失败：`ResolutionImpossible` | `gradio 6.30` 要求 `starlette>=1.0.1`，而 `fastapi 0.115` 要求 `starlette<0.42` | 从依赖里删掉 gradio（Docker 模式跑的是 FastAPI，用不到它） |
| 3 | 端口探测不通 | Docker 模式端口**固定 7860**，不可配置 | `start.sh` 与健康检查统一改 7860 |
| 4 | 本地构不了镜像 | 本机未装 WSL2，Docker 守护进程起不来 | 直接在平台侧构建，本机不需要 Docker |

依赖修正后，构建耗时 **8 分钟**（14:37 → 14:45），
`llama-cpp-python` 取到预编译轮子，未触发源码编译。

## 一、为什么不能直接用现在的形态

开发时的本地推理走 Ollama 守护进程（`127.0.0.1:11434`），多数免费托管平台：

- 只允许**一个**容器，不保证能常驻第二个服务；
- 免费档内存 512 MB – 16 GB 不等，1.5B 模型需要约 1.2 GB；
- 构建机与运行机都不能假设有 GPU。

因此部署形态改为：**应用 + 本地模型同容器**，本地推理换成 llama.cpp 的
OpenAI 兼容服务（`llama_cpp.server`），应用侧复用云端那套 Provider，
不需要为本地单写一套客户端。这条代码路径已在本机验证：

```
LOCAL_BACKEND=openai
LOCAL_BASE_URL=http://127.0.0.1:11434/v1     # 本机用 Ollama 的 /v1 冒充
LOCAL_SERVED_MODEL=qwen2.5:1.5b
```

实测四条请求全部正常走通（`1+1` 走本地 p=0.99、`87+8` 走本地 p=0.97 等），
说明「本地后端 = OpenAI 兼容服务」这条路是通的。

## 二、已完成

| 产物 | 说明 |
| --- | --- |
| `app/providers/openai_provider.py` 的 `stream()` | 流式输出，云端与本地共用一个客户端 |
| `app/providers/__init__.py` 的 `build_local_provider()` | 按 `LOCAL_BACKEND` 选择 ollama / openai / llamacpp |
| `deploy/Dockerfile` | 单容器：应用 + llama-cpp server + 1 GB 模型 |
| `deploy/start.sh` | 容器内两进程编排与就绪等待 |

## 三、卡点

**本机 Docker 守护进程未启动**，无法构建镜像验证。盲写一个没跑过的 Dockerfile
直接交付是不负责任的——首次构建大概率会在 wheel 或模型下载上出问题。

## 四、三条路径（按推荐度排序）

### 路径 A：ModelScope 创空间（推荐，国内可访问、免费、16 GB 内存）

1. 注册 modelscope.cn（手机号即可）。
2. 「创空间」→ 新建 → SDK 选 **Docker**，硬件选 **CPU 免费实例**。
3. 把本仓库推到创空间的 Git 远端（页面会给地址）。
4. 在仓库根放一份 `Dockerfile`（把 `deploy/Dockerfile` 改名复制过去）。
5. 等待构建，得到 `https://modelscope.cn/studios/<用户名>/<空间名>` 的公网地址。

> 优势：评委在国内能直接打开，不需要梯子。这是我首选推荐的方案。

### 路径 B：Hugging Face Spaces（国际通用，但国内访问慢）

1. 注册 huggingface.co。
2. New Space → SDK **Docker** → CPU basic（16 GB 免费）。
3. 推送代码与 `Dockerfile`。
4. 得到 `https://<用户名>-<空间名>.hf.space`。

> 风险：国内访问可能超时，不适合作为唯一演示地址。

### 路径 C：临时公网隧道（无账号，即时可用，但不稳定）

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

得到形如 `https://xxx.trycloudflare.com` 的地址，本机关机即失效。
只适合提交前给老师/同学看一眼，**不适合写进省赛提交物**。

## 五、部署后需要验证的四件事

1. `GET /api/health` 返回 `cloud_ready: true`（云端密钥已注入）。
2. 首页能打开，输入框可输入。
3. 问「计算 87 + 8 的值。」→ 右侧显示**本地推理 LOCAL**，费用 ¥0。
4. 问一道两步应用题 → 右侧显示**云端推理 CLOUD**，费用非零。

## 六、安全注意

- 云端密钥通过环境变量注入（`CLOUD_API_KEY`），**绝不写进镜像层**。
- 公网实例会被爬虫扫到，密钥泄露会导致账单损失；建议单独申请一个限额密钥专供演示。
- 若平台支持，给 `/api/chat` 加一层简单的速率限制。
