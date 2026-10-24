const state = {
  mode: "auto",
  busy: false,
};

const $ = (id) => document.getElementById(id);
const thread = $("thread");
const input = $("input");
const sendBtn = $("send");

/* ---------- 元信息 ---------- */

async function loadHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    document.querySelector('[data-role="local"]').textContent = `本地 · ${data.local_model}`;
    document.querySelector('[data-role="cloud"]').textContent =
      `云端 · ${data.cloud_model}${data.cloud_ready ? "" : "（未配置）"}`;
    document.querySelector('[data-role="theta"]').textContent = `阈值 ${data.theta_local}`;
    $("threshold-mark").style.left = `${data.theta_local * 100}%`;
    setConn(true, "已连接");
  } catch (err) {
    setConn(false, "连接失败");
  }
}

function setConn(ok, text) {
  const el = $("conn");
  el.textContent = text;
  el.className = ok ? "ok" : "bad";
}

async function refreshStats() {
  try {
    const resp = await fetch("/api/stats");
    const s = await resp.json();
    $("stat-requests").textContent = s.requests;
    $("stat-local").textContent = `${Math.round(s.local_ratio * 100)}%`;
    $("stat-cost").textContent = `¥${s.cost_cny.toFixed(6)}`;
    $("stat-escalation").textContent = s.escalations;
  } catch (err) {
    /* 统计失败不影响主流程 */
  }
}

/* ---------- 渲染 ---------- */

function clearEmpty() {
  const empty = $("empty");
  if (empty) empty.remove();
}

function addMessage(role, text) {
  clearEmpty();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  thread.appendChild(wrap);
  thread.scrollTop = thread.scrollHeight;
  return { wrap, bubble };
}

function renderTrace(trace, usage) {
  const badge = $("route-badge");
  const label = trace.route === "local" ? "本地推理 LOCAL" : "云端推理 CLOUD";
  badge.textContent = trace.escalated ? `${label} · 已升级` : label;
  badge.dataset.route = trace.route;

  $("difficulty-value").textContent = trace.difficulty.toFixed(2);
  $("difficulty-fill").style.width = `${Math.min(trace.difficulty, 1) * 100}%`;
  $("threshold-mark").style.left = `${trace.threshold * 100}%`;

  const list = $("reasons");
  list.innerHTML = "";
  trace.reasons.forEach((reason) => {
    const li = document.createElement("li");
    li.textContent = reason;
    list.appendChild(li);
  });

  if (usage) {
    const metrics = [
      ["延迟", `${trace.latency_ms.toFixed(0)} ms`],
      ["费用", `¥${usage.cost_cny.toFixed(6)}`],
      ["输入 tokens", usage.prompt_tokens],
      ["输出 tokens", usage.completion_tokens],
    ];
    if (trace.confidence !== null && trace.confidence !== undefined) {
      metrics.push(["本地自评置信度", trace.confidence.toFixed(2)]);
    }
    metrics.push(["难度分 / 阈值", `${trace.difficulty.toFixed(2)} / ${trace.threshold.toFixed(2)}`]);
    $("metrics").innerHTML = metrics
      .map(([k, v]) => `<div class="metric"><span>${k}</span><b>${v}</b></div>`)
      .join("");
  }
}

/* ---------- 流式发送 ---------- */

async function send() {
  const text = input.value.trim();
  if (!text || state.busy) return;

  state.busy = true;
  sendBtn.disabled = true;
  input.value = "";
  addMessage("user", text);

  const { wrap, bubble } = addMessage("assistant", "思考中…");
  bubble.classList.add("streaming", "placeholder");
  const status = document.createElement("div");
  status.className = "meta";
  status.textContent = "正在判断难度…";
  wrap.appendChild(status);
  $("trace-hint").textContent = "推理中…";

  let buffer = "";
  let answer = "";

  try {
    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: text }], mode: state.mode }),
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });

      const chunks = pending.split("\n\n");
      pending = chunks.pop();

      for (const chunk of chunks) {
        const line = chunk.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const event = JSON.parse(line.slice(6));

        if (event.type === "trace") {
          status.textContent = "推理中…";
          renderTrace(
            { ...event, escalated: false, confidence: null, latency_ms: 0 },
            null
          );
        } else if (event.type === "delta") {
          if (bubble.classList.contains("placeholder")) {
            bubble.textContent = "";
            bubble.classList.remove("placeholder");
          }
          answer += event.text;
          bubble.textContent = answer;
          bubble.classList.remove("streaming");
          thread.scrollTop = thread.scrollHeight;
        } else if (event.type === "reset") {
          answer = "";
          bubble.textContent = "";
          bubble.classList.remove("placeholder");
          bubble.classList.add("streaming");
        } else if (event.type === "status") {
          status.textContent = event.text;
        } else if (event.type === "escalate") {
          status.textContent = `⚠ ${event.text}`;
        } else if (event.type === "reasons") {
          renderTrace(
            { route: "cloud", difficulty: 0, threshold: 0, reasons: event.reasons, escalated: false, confidence: null, latency_ms: 0 },
            null
          );
        } else if (event.type === "done") {
          answer = event.answer;
          bubble.textContent = answer;
          bubble.classList.remove("streaming");
          status.textContent = "";
          $("trace-hint").textContent = event.model;
          renderTrace(event.trace, event.usage);
          refreshStats();
        } else if (event.type === "error") {
          throw new Error(event.detail);
        }
      }
    }
    buffer = answer;
  } catch (err) {
    bubble.textContent = answer || `请求失败：${err.message}`;
    status.textContent = "请检查 Ollama 是否运行、.env 中云端密钥是否配置";
    $("trace-hint").textContent = "出错";
  } finally {
    state.busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

document.querySelectorAll(".mode").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.mode = btn.dataset.mode;
  });
});

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  send();
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    send();
  }
});

loadHealth();
refreshStats();
input.focus();

/* 支持 ?q=问题&mode=auto 直接发起一次演示请求：
   录制视频时可复现同一镜次，也方便截图与自动化冒烟测试。 */
const params = new URLSearchParams(location.search);
const preset = params.get("q");
const presetMode = params.get("mode");
if (presetMode) {
  const btn = document.querySelector(`.mode[data-mode="${presetMode}"]`);
  if (btn) btn.click();
}
if (preset) {
  input.value = preset;
  setTimeout(send, 400);
}
