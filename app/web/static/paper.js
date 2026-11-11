/* 论文精读前端：原文预览 + 锚点高亮 + 路由决策。
 *
 * 三个约束驱动的写法：
 * 1. 零依赖、单文件——录屏时构建链越短越不容易翻车；
 * 2. 高亮用「服务端渲染页图 + 绝对定位框」，不引 pdf.js；
 * 3. 流式用 fetch + 手写 SSE 解析（EventSource 不支持 POST）。
 */

const state = {
  paper: null,
  page: 1,
  pages: 0,
  sizes: [],
  mode: "auto",
  busy: false,
  stats: { requests: 0, cloud: 0, cost: 0 },
};

const $ = (id) => document.getElementById(id);

/* ---------------- 论文与页面 ---------------- */

async function loadPapers() {
  const res = await fetch("/api/papers");
  const data = await res.json();
  const select = $("paper-select");
  select.innerHTML = "";
  (data.papers || []).forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = `${prettyName(p.name)}${p.pages ? ` · ${p.pages} 页` : ""}`;
    select.appendChild(opt);
  });
  if (select.options.length) {
    state.paper = select.value;
    await loadMeta();
  }
}

function prettyName(name) {
  if (name.startsWith("rag_2005")) return "RAG · 检索增强生成";
  if (name.startsWith("lora_2106")) return "LoRA · 低秩适配";
  if (name.startsWith("lost_in_middle")) return "Lost in the Middle · 长上下文";
  return name.replace(/_\d+\.\d+$/, "");
}

async function loadMeta() {
  const res = await fetch(`/api/paper/${encodeURIComponent(state.paper)}/meta`);
  const meta = await res.json();
  state.pages = meta.pages;
  state.sizes = meta.page_sizes || [];
  state.page = 1;
  $("page-indicator").textContent = `1 / ${state.pages}`;
  renderPage(1);
}

function renderPage(n) {
  if (!state.paper || n < 1 || n > state.pages) return;
  state.page = n;
  const img = $("page-img");
  const url = `/api/paper/${encodeURIComponent(state.paper)}/page/${n}.png`;
  img.src = url;
  img.onload = () => {
    fitPage();
    clearHighlights();
    drawHighlights(lastAnchors.filter((a) => a.page === state.page));
  };
  $("page-indicator").textContent = `${n} / ${state.pages}`;
}

function fitPage() {
  const img = $("page-img");
  const holder = $("page-holder");
  const viewer = $("viewer");
  const maxW = viewer.clientWidth - 34;
  const maxH = viewer.clientHeight - 34;
  const ratio = img.naturalWidth / img.naturalHeight;
  let w = Math.min(maxW, maxH * ratio, img.naturalWidth);
  img.style.width = `${w}px`;
  img.style.height = "auto";
  holder.style.width = `${w}px`;
}

window.addEventListener("resize", () => {
  fitPage();
  clearHighlights();
  drawHighlights(lastAnchors.filter((a) => a.page === state.page));
});

/* ---------------- 高亮 ---------------- */

let lastAnchors = [];

function pageScale() {
  const size = state.sizes[state.page - 1] || [612, 792];
  const img = $("page-img");
  return img.clientWidth > 0 ? img.clientWidth / size[0] : 1;
}

function clearHighlights() {
  $("highlights").innerHTML = "";
}

function drawHighlights(anchors) {
  clearHighlights();
  if (!anchors.length) return;
  const scale = pageScale();
  const box = $("highlights");
  anchors.forEach((a) => {
    const [x0, y0, x1, y1] = a.bbox || [0, 0, 0, 0];
    const el = document.createElement("div");
    el.className = "hl";
    el.style.left = `${x0 * scale}px`;
    el.style.top = `${y0 * scale}px`;
    el.style.width = `${Math.max(6, (x1 - x0) * scale)}px`;
    el.style.height = `${Math.max(4, (y1 - y0) * scale)}px`;
    el.dataset.chunk = a.chunk_id;
    box.appendChild(el);
  });
}

function focusAnchor(chunkId) {
  const a = lastAnchors.find((x) => x.chunk_id === chunkId);
  if (!a) return;
  if (a.page !== state.page) renderPage(a.page);
  setTimeout(() => {
    drawHighlights([a]);
    const el = document.querySelector(`.hl[data-chunk="${chunkId}"]`);
    if (el) {
      el.classList.add("pulse");
      el.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }, 160);
}

/* ---------------- 问答 ---------------- */

function modeLabel(route) {
  return route === "cloud" ? "云端 DeepSeek" : "本地 1.5B";
}

function setRoute(trace) {
  const bar = $("route-bar");
  bar.dataset.route = trace.route;
  $("route-name").textContent = modeLabel(trace.route);
  const pct = trace.score_kind === "p_local_correct"
    ? `本地答对概率 ${(trace.difficulty * 100).toFixed(0)}% / 阈值 ${(trace.threshold * 100).toFixed(0)}%`
    : `难度 ${trace.difficulty.toFixed(2)} / 阈值 ${trace.threshold.toFixed(2)}`;
  $("route-meta").textContent = `${pct} · ${Math.round(trace.latency_ms)} ms`;
}

function setFeatures(features) {
  const box = $("features");
  box.innerHTML = "";
  const rows = [
    ["检索置信", features.top1_score?.toFixed(2)],
    ["证据分散", `${features.n_pages_hit} 页`],
    ["词覆盖", `${Math.round((features.coverage || 0) * 100)}%`],
  ];
  rows.forEach(([label, value]) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `${label} <b>${value}</b>`;
    box.appendChild(chip);
  });
}

function appendUser(text) {
  const wrap = document.createElement("div");
  wrap.className = "msg user";
  wrap.innerHTML = `<div class="bubble"></div>`;
  wrap.querySelector(".bubble").textContent = text;
  $("thread").appendChild(wrap);
  scrollThread();
}

function appendBot() {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.innerHTML = `<div class="bubble"><span class="caret"></span></div><div class="meta"></div>`;
  $("thread").appendChild(wrap);
  scrollThread();
  return wrap;
}

function renderAnswer(bubble, text) {
  // 把 [S3] 变成可点击的引用角标
  const html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/\[(S\d+(?:\s*[,，]\s*S\d+)*)\]/g, (_, ids) =>
      ids
        .split(/[,，]/)
        .map((id) => `<span class="cite" data-chunk="${id.trim()}">${id.trim()}</span>`)
        .join("")
    );
  bubble.innerHTML = html || '<span class="caret"></span>';
  bubble.querySelectorAll(".cite").forEach((el) => {
    el.addEventListener("click", () => focusAnchor(el.dataset.chunk));
  });
}

function scrollThread() {
  const t = $("thread");
  t.scrollTop = t.scrollHeight;
}

function setEvidence(anchors, report) {
  lastAnchors = anchors || [];
  const list = $("anchors");
  list.innerHTML = "";
  if (!lastAnchors.length) {
    $("evidence-hint").textContent = report && report.refused
      ? "模型判断原文未提及，未给出锚点"
      : "本轮没有可对齐的锚点";
    return;
  }
  const badChunks = new Set([
    ...(report?.invalid_refs || []),
  ]);
  lastAnchors.forEach((a) => {
    const li = document.createElement("li");
    const ok = !badChunks.has(a.chunk_id) && report?.number_ok !== false;
    if (!ok) li.classList.add("bad");
    li.innerHTML = `
      <span class="sid">${a.chunk_id}</span>
      <span class="quote"></span>
      <span class="pg">第 ${a.page} 页</span>`;
    li.querySelector(".quote").textContent = a.quote;
    if (a.method === "aligned") {
      const tag = document.createElement("span");
      tag.className = "warn";
      tag.textContent = "对齐";
      li.insertBefore(tag, li.querySelector(".pg"));
    }
    li.addEventListener("click", () => focusAnchor(a.chunk_id));
    list.appendChild(li);
  });

  const notes = [];
  if (report?.invalid_refs?.length) notes.push(`无效引用 ${report.invalid_refs.join(",")}`);
  if (report?.missing_numbers?.length) notes.push(`数字不可溯 ${report.missing_numbers.join(",")}`);
  if (report?.missing_terms?.length) notes.push(`术语不可溯 ${report.missing_terms.join(",")}`);
  $("evidence-hint").textContent = notes.length ? notes.join(" · ") : "全部锚点已通过一致性校验";
  drawHighlights(lastAnchors.filter((a) => a.page === state.page));
}

let currentAnswer = "";

async function ask(question) {
  if (state.busy || !state.paper) return;
  state.busy = true;
  $("send").disabled = true;
  $("conn").textContent = "推理中…";
  currentAnswer = "";
  const empty = $("empty");
  if (empty) empty.remove();
  appendUser(question);
  const bot = appendBot();
  const bubble = bot.querySelector(".bubble");
  const meta = bot.querySelector(".meta");

  try {
    const res = await fetch("/api/paper/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        paper: state.paper,
        question,
        mode: state.mode,
        k: 5,
      }),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        let evt;
        try {
          evt = JSON.parse(line.slice(5).trim());
        } catch {
          continue;
        }
        handleEvent(evt, bubble, meta);
      }
    }
  } catch (err) {
    meta.textContent = `请求失败：${err.message}`;
  } finally {
    state.busy = false;
    $("send").disabled = false;
    $("conn").textContent = "就绪";
    scrollThread();
  }
}

function handleEvent(evt, bubble, meta) {
  switch (evt.type) {
    case "retrieval":
      setFeatures(evt.features || {});
      break;
    case "trace":
      setRoute(evt);
      break;
    case "status":
      meta.textContent = evt.text;
      break;
    case "delta":
      currentAnswer += evt.text;
      renderAnswer(bubble, currentAnswer);
      scrollThread();
      break;
    case "reset":
      currentAnswer = "";
      renderAnswer(bubble, "");
      break;
    case "escalate":
      meta.textContent = evt.text;
      break;
    case "done": {
      currentAnswer = evt.answer || currentAnswer;
      renderAnswer(bubble, currentAnswer);
      setRoute(evt.trace);
      const u = evt.usage || {};
      meta.textContent =
        `${modeLabel(evt.trace.route)} · ${Math.round(evt.trace.latency_ms)} ms · ¥${(u.cost_cny || 0).toFixed(6)}` +
        ` · ${u.prompt_tokens || 0}+${u.completion_tokens || 0} tokens`;
      setEvidence(evt.anchors || [], evt.report || {});
      updateStats(evt.trace, u);
      if (window.__autoJump && (evt.anchors || []).length) {
        setTimeout(() => focusAnchor(evt.anchors[0].chunk_id), 350);
      }
      break;
    }
    case "error":
      meta.textContent = `出错：${evt.detail}`;
      break;
    default:
      break;
  }
}

function updateStats(trace, usage) {
  state.stats.requests += 1;
  if (trace.route === "cloud") state.stats.cloud += 1;
  state.stats.cost += usage.cost_cny || 0;
  $("stat-requests").textContent = state.stats.requests;
  $("stat-cloud").textContent = `${Math.round((state.stats.cloud / state.stats.requests) * 100)}%`;
  $("stat-cost").textContent = `¥${state.stats.cost.toFixed(6)}`;
}

/* ---------------- 交互绑定 ---------------- */

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("input");
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  ask(q);
});

$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});

document.querySelectorAll(".mode").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.mode = btn.dataset.mode;
  });
});

$("paper-select").addEventListener("change", async (e) => {
  state.paper = e.target.value;
  lastAnchors = [];
  $("anchors").innerHTML = "";
  $("thread").innerHTML = "";
  $("evidence-hint").textContent = "回答后显示，点击可跳转到原文";
  await loadMeta();
});

$("prev-page").addEventListener("click", () => renderPage(state.page - 1));
$("next-page").addEventListener("click", () => renderPage(state.page + 1));

document.addEventListener("keydown", (e) => {
  if (e.key === "PageDown") renderPage(state.page + 1);
  if (e.key === "PageUp") renderPage(state.page - 1);
});

loadPapers()
  .then(() => {
    // 演示/截图用：/paper?paper=xxx&ask=问题 会自动选论文并提问
    const params = new URLSearchParams(location.search);
    const wantPaper = params.get("paper");
    const about = params.get("ask");
    window.__autoJump = params.get("jump") === "1";
    if (wantPaper) {
      const select = $("paper-select");
      if ([...select.options].some((o) => o.value === wantPaper)) {
        select.value = wantPaper;
        state.paper = wantPaper;
        return loadMeta().then(() => {
          if (about) ask(about);
        });
      }
    }
    if (about) ask(about);
  })
  .catch((err) => {
    $("conn").textContent = `初始化失败：${err.message}`;
  });
