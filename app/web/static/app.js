const state = {
  mode: "auto",
  busy: false,
  history: [],
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
    document.querySelector('[data-role="cloud"]').textContent = `云端 · ${data.cloud_model}${data.cloud_ready ? "" : "（未配置）"}`;
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

function typewriter(node, text, speed = 12) {
  return new Promise((resolve) => {
    let i = 0;
    const step = () => {
      if (i >= text.length) return resolve();
      node.textContent = text.slice(0, ++i);
      thread.scrollTop = thread.scrollHeight;
      setTimeout(step, speed);
    };
    step();
  });
}

function renderTrace(trace, usage, model) {
  const badge = $("route-badge");
  const label = trace.route === "local" ? "本地推理 LOCAL" : "云端推理 CLOUD";
  badge.textContent = trace.escalated ? `${label} · 已升级` : label;
  badge.dataset.route = trace.route;
  $("trace-hint").textContent = model;

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

/* ---------- 交互 ---------- */

async function send() {
  const text = input.value.trim();
  if (!text || state.busy) return;

  state.busy = true;
  sendBtn.disabled = true;
  input.value = "";
  addMessage("user", text);
  state.history.push({ role: "user", content: text });

  const pending = addMessage("assistant", "…");
  $("trace-hint").textContent = "推理中…";

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "user", content: text }],
        mode: state.mode,
      }),
    });

    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    await typewriter(pending.bubble, data.answer);
    renderTrace(data.trace, data.usage, data.model);
    state.history.push({ role: "assistant", content: data.answer });
    refreshStats();
  } catch (err) {
    pending.bubble.textContent = `请求失败：${err.message}`;
    $("trace-hint").textContent = "出错";
    if (pending.wrap.querySelector(".meta")) return;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = "请检查 Ollama 是否运行、.env 中云端密钥是否配置";
    pending.wrap.appendChild(meta);
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
