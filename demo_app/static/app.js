// RiskLens offline demo: render the attribution pack and drive the Ask panel. No external dependencies.
const $ = (id) => document.getElementById(id);
const state = { pack: null, tab: "Summary", answers: null };
// Snapshot mode (GitHub Pages): read files written by scripts/export_static_demo.py instead of calling the server.
const STATIC = window.RISKLENS_STATIC === true;
const esc = (s) => String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
const packFile = (q, ext) => `data/pack_${q.get("framework")}_${q.get("previous_date")}_${q.get("current_date")}.${ext}`;

const isZero = (v) => Math.abs(v) < 0.5;  // rounding remnants such as -0.000001 display as zero
const FMT = {
  money: (v) => (isZero(v) ? "-" : (v < 0 ? "(" : "") + "$" + Math.round(Math.abs(v)).toLocaleString("en-US") + (v < 0 ? ")" : "")),
  pct: (v) => (v === 0 ? "-" : (v < 0 ? "(" : "") + (Math.abs(v) * 100).toFixed(1) + "%" + (v < 0 ? ")" : "")),
  pct2: (v) => (v * 100).toFixed(2) + "%",
  pp: (v) => (v === 0 ? "-" : (v < 0 ? "(" : "") + Math.abs(v).toFixed(1) + (v < 0 ? ")" : "")),
};
const short = (v) => (Math.abs(v) >= 1e6 ? "$" + (v / 1e6).toFixed(2) + "M" : FMT.money(v));

function fmt(c) {
  if (c.v === null || c.v === undefined) return "";
  if (c.f && typeof c.v === "number") return FMT[c.f](c.v);
  return String(c.v);
}

function td(c, sourceTools) {
  const el = document.createElement("td");
  el.textContent = fmt(c);
  if (c.k !== "text") el.classList.add(c.k);
  if (c.f) el.classList.add("num");
  if (c.x) el.title = (c.k === "link" ? "Link: " : "Formula: ") + c.x;
  if (c.src) {
    el.title = "Tool output: " + (sourceTools[c.src] || "") + " · " + c.src;
    el.addEventListener("click", () => openSource(c.src));
  }
  return el;
}

function table(t, sourceTools) {
  const wrap = document.createElement("div");
  wrap.className = "grid";
  if (t.caption) { const h = document.createElement("h4"); h.textContent = t.caption; wrap.appendChild(h); }
  const tb = document.createElement("table");
  const head = tb.createTHead().insertRow();
  t.columns.forEach((c) => { const th = document.createElement("th"); th.textContent = c; head.appendChild(th); });
  const body = tb.createTBody();
  const add = (rows, cls) => (rows || []).forEach((r) => {
    const tr = body.insertRow();
    if (cls) tr.className = cls;
    r.forEach((c) => tr.appendChild(td(c, sourceTools)));
  });
  add(t.rows); add(t.totals, "total"); add(t.tail);
  wrap.appendChild(tb);
  return wrap;
}

function waterfall(chart) {
  // Floating bars from the previous allowance to the current one; one scale for bars and labels.
  const steps = [{ label: "Start", base: 0, value: chart.start, kind: "end" }];
  let level = chart.start;
  chart.steps.forEach((s) => { steps.push({ label: s.label, base: level, value: s.impact, kind: s.impact >= 0 ? "up" : "down" }); level += s.impact; });
  steps.push({ label: "End", base: 0, value: chart.end, kind: "end" });
  const tops = steps.map((s) => Math.max(s.base, s.base + s.value));
  const lows = steps.map((s) => Math.min(s.base, s.base + s.value));
  const hi = Math.max(...tops), lo = Math.min(...lows.filter((v, i) => steps[i].kind !== "end"), chart.start, chart.end) * 0.96;
  const W = 1000, H = 300, padL = 10, padB = 62, padT = 24, bw = (W - padL * 2) / steps.length;
  const y = (v) => padT + (hi - v) / (hi - lo) * (H - padT - padB);
  const color = { up: "var(--pos)", down: "var(--neg)", end: "var(--accent)" };
  const wrap = (label, width) => {  // greedy word wrap to at most 3 lines that fit the bar slot
    const lines = [""];
    label.split(/\s+/).forEach((w) => {
      const cur = lines[lines.length - 1];
      if (cur && (cur + " " + w).length > width) lines.push(w); else lines[lines.length - 1] = cur ? cur + " " + w : w;
    });
    return lines.length > 3 ? [...lines.slice(0, 2), lines.slice(2).join(" ").slice(0, width - 1) + "…"] : lines;
  };
  let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Allowance waterfall">`;
  svg += `<line class="axis" x1="0" x2="${W}" y1="${H - padB}" y2="${H - padB}"/>`;
  svg += `<text x="${padL}" y="12" text-anchor="start" class="note">Axis starts at ${short(lo)}, not zero</text>`;
  steps.forEach((s, i) => {
    const top = s.kind === "end" ? s.value : Math.max(s.base, s.base + s.value);
    const bottom = s.kind === "end" ? lo : Math.min(s.base, s.base + s.value);
    const x = padL + i * bw + bw * 0.15, w = bw * 0.7;
    svg += `<rect x="${x}" y="${y(top)}" width="${w}" height="${Math.max(1, y(bottom) - y(top))}" fill="${color[s.kind]}" rx="2"/>`;
    const val = s.kind === "end" ? short(s.value) : (s.value >= 0 ? "+" : "−") + short(Math.abs(s.value));
    svg += `<text x="${x + w / 2}" y="${y(top) - 6}" text-anchor="middle">${val}</text>`;
    wrap(s.label, Math.floor(bw / 7.2)).forEach((ln, j) => {
      svg += `<text x="${x + w / 2}" y="${H - padB + 16 + j * 14}" text-anchor="middle">${esc(ln)}</text>`;
    });
  });
  return svg + "</svg>";
}

const RUNS_TAB = "Model runs";
const SCEN_TAB = "Scenarios";      // every evaluated model configuration side by side

function renderTabs() {
  const tabs = $("tabs");
  tabs.innerHTML = "";
  [...state.pack.sheets.map((s) => s.name), SCEN_TAB, RUNS_TAB].forEach((name) => {
    const b = document.createElement("button");
    b.type = "button"; b.role = "tab"; b.textContent = name;
    b.setAttribute("aria-selected", String(name === state.tab));
    b.addEventListener("click", () => { state.tab = name; renderTabs(); renderCurrent(); });
    tabs.appendChild(b);
  });
}

function simpleTable(columns, rows) {
  const wrap = document.createElement("div");
  wrap.className = "grid";
  const tb = document.createElement("table");
  const head = tb.createTHead().insertRow();
  columns.forEach(([label]) => { const th = document.createElement("th"); th.textContent = label; head.appendChild(th); });
  const body = tb.createTBody();
  rows.forEach((r) => {
    const tr = body.insertRow();
    columns.forEach(([, get, num]) => {
      const c = tr.insertCell();
      const v = get(r);
      c.textContent = v === null || v === undefined ? "–" : String(v);
      if (num) c.className = "num";
    });
  });
  wrap.appendChild(tb);
  return wrap;
}

function renderCurrent() {
  document.querySelector(".legend").hidden = state.tab === RUNS_TAB || state.tab === SCEN_TAB;   // colour key is for pack sheets
  if (state.tab === RUNS_TAB) return renderRuns();
  if (state.tab === SCEN_TAB) return renderScenarios();
  return renderSheet();
}

// ---------------------------------------------------------------- scenario scoreboard
const pctText = (v) => (v === null || v === undefined ? "–" : v.toFixed(3));

function qualityCell(td, v) {
  // value plus a bar scaled 0-1; colour bands mark fully right / partly right / weak
  td.className = "num qcell";
  if (v === null || v === undefined) { td.textContent = "–"; return; }
  const band = v >= 0.999 ? "full" : v >= 0.8 ? "mid" : "low";
  td.classList.add(band);
  const bar = document.createElement("span"); bar.className = "qbar"; bar.style.width = `${Math.round(v * 100)}%`;
  const txt = document.createElement("span"); txt.className = "qval"; txt.textContent = pctText(v);
  td.append(bar, txt);
}

function scoreTable(columns, rows) {
  // columns: [label, getter, kind] where kind is "q" (quality bar), "num" or text
  const wrap = document.createElement("div"); wrap.className = "grid";
  const tb = document.createElement("table"); tb.className = "scoreboard";
  const head = tb.createTHead().insertRow();
  columns.forEach(([label]) => { const th = document.createElement("th"); th.textContent = label; head.appendChild(th); });
  const body = tb.createTBody();
  rows.forEach((r) => {
    const tr = body.insertRow();
    columns.forEach(([, get, kind]) => {
      const td = tr.insertCell(); const v = get(r);
      if (kind === "q") qualityCell(td, v);
      else { td.textContent = v === null || v === undefined ? "–" : String(v); if (kind === "num") td.className = "num"; }
    });
  });
  wrap.appendChild(tb);
  return wrap;
}

function fullNames(tableWrap, rows) {
  // per-question tables use short column names; the full configuration name shows on hover
  tableWrap.querySelectorAll("thead th").forEach((th, i) => { if (i > 0) th.title = rows[i - 1].label; });
}

function section(el, title, text) {
  const h = document.createElement("h4"); h.textContent = title; el.appendChild(h);
  if (text) { const p = document.createElement("p"); p.className = "muted"; p.textContent = text; el.appendChild(p); }
}

async function renderScenarios() {
  const el = $("sheet");
  el.innerHTML = '<p class="empty">Loading scenarios…</p>';
  let data;
  try {
    const r = await fetch(STATIC ? "data/scenarios.json" : "api/scenarios");
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch (e) { el.innerHTML = ""; const p = document.createElement("p"); p.className = "error"; p.textContent = `Could not load the scoreboard: ${e.message}`; el.appendChild(p); return; }
  el.innerHTML = "";
  const h = document.createElement("h3"); h.textContent = "Scenario comparison: every model configuration we evaluated"; el.appendChild(h);
  const note = document.createElement("p"); note.className = "muted"; note.textContent = data.note; el.appendChild(note);

  const base = data.base;
  section(el, "Base story: 15 evaluation-bank questions",
    "Same questions, tools, prompts and verifier for every row. Time is per question where it was measured (the frontier reference was hand-driven; NeMo Gym does not record wall time).");
  el.appendChild(scoreTable([
    ["Configuration", (r) => r.label], ["Agents", (r) => r.agents], ["Mean quality", (r) => r.mean_quality, "q"],
    ["Fully right", (r) => `${r.fully_right} / ${r.answered}`, "num"],
    ["Median s / question", (r) => r.median_s, "num"], ["Source", (r) => r.source],
  ], base.rows));
  section(el, "Base story by question", "Quality per question; hover a question ID for its text.");
  const qcols = [["Question", (q) => q.task_id]].concat(base.rows.map((r) => [r.short || r.label,
    (q) => (r.per_question[q.task_id] || {}).quality, "q"]));
  const qtable = scoreTable(qcols, base.questions);
  qtable.querySelectorAll("tbody tr").forEach((tr, i) => { tr.cells[0].title = base.questions[i].question; });
  fullNames(qtable, base.rows);
  el.appendChild(qtable);

  const ho = data.heldout;
  if (ho.rows.length) {
    section(el, `Held-out story (${ho.story}): base vs LoRA vs local 4B`,
      "This story was never in training and has different correct answers, so compare these rows with each other, not with the base story. Thinking off is the fast mode the LoRA was trained for.");
    el.appendChild(scoreTable([
      ["Configuration", (r) => r.label], ["Thinking", (r) => r.thinking], ["Mean quality", (r) => r.mean_quality, "q"],
      ["Answers with no tool call", (r) => `${r.no_tool_answers} / ${r.answers}`, "num"],
      ["Output tokens / answer", (r) => r.output_tokens.toLocaleString("en-US"), "num"],
      ["Attempts / question", (r) => r.attempts_per_question, "num"],
    ], ho.rows));
    section(el, "Held-out story by question", "Mean quality per question across attempts.");
    const htable = scoreTable([["Question", (q) => q.task_id]].concat(ho.rows.map((r) => [r.short || r.label, (q) => r.per_question[q.task_id], "q"])),
      ho.questions);
    fullNames(htable, ho.rows);
    el.appendChild(htable);
    if (ho.adapter) {
      const p = document.createElement("p"); p.className = "muted"; p.append("LoRA adapter: ");
      const a = document.createElement("a"); a.href = ho.adapter; a.textContent = ho.adapter.replace("https://", "");
      a.rel = "noopener"; a.target = "_blank"; p.appendChild(a); el.appendChild(p);
    }
  }
}

async function renderRuns() {
  const el = $("sheet");
  el.innerHTML = '<p class="empty">Loading runs…</p>';
  const data = await (await fetch(STATIC ? "data/runs.json" : "api/runs")).json();
  el.innerHTML = "";
  const h = document.createElement("h3"); h.textContent = "Model comparison: speed and answer quality"; el.appendChild(h);
  const note = document.createElement("p"); note.className = "muted";
  note.textContent = "Quality is the Gym verifier's reward (0–1) for questions in the evaluation bank: correct facts, grounding, tool choice, valid calls, efficiency. Model time is time spent waiting on the model; tool time is the deterministic analytics.";
  el.appendChild(note);
  if (!data.runs.length) { el.insertAdjacentHTML("beforeend", '<p class="empty">No runs yet. Ask a question, or run python -m demo_app.bench.</p>'); return; }
  el.appendChild(simpleTable([
    ["Label", (r) => r.label], ["Model", (r) => r.model], ["Runs", (r) => r.runs, 1],
    ["Median time (s)", (r) => r.median_duration_s, 1], ["Median model time (s)", (r) => r.median_llm_s, 1],
    ["Model turns", (r) => r.mean_llm_turns, 1], ["Tool calls", (r) => r.mean_tool_calls, 1],
    ["Tokens / s", (r) => r.median_tokens_per_s, 1], ["No unsupported numbers %", (r) => r.grounded_pct, 1],
    ["Mean quality", (r) => r.mean_reward, 1], ["Scored runs", (r) => r.scored_runs, 1],
    ["Failed runs", (r) => r.failed_runs, 1], ["Truncated prompts", (r) => r.truncated_runs, 1],
  ], data.summary));
  const h4 = document.createElement("h4"); h4.textContent = "Individual runs (newest first)"; el.appendChild(h4);
  el.appendChild(simpleTable([
    ["When (UTC)", (r) => r.ts.replace("T", " ").replace("+00:00", "")], ["Label", (r) => r.label],
    ["Question", (r) => r.question], ["Time (s)", (r) => r.duration_s, 1], ["Model (s)", (r) => r.llm_s, 1],
    ["Tools (s)", (r) => r.tool_s, 1], ["Turns", (r) => r.llm_turns, 1], ["Tools used", (r) => r.tools_used.join(", ")],
    ["Grounding", (r) => r.grounding], ["Quality", (r) => r.quality ? r.quality.reward : null, 1],
    ["Facts correct", (r) => r.quality ? `${r.quality.checks_passed}/${r.quality.checks_total}` : null, 1],
  ], data.runs));
}

function renderSheet() {
  const s = state.pack.sheets.find((x) => x.name === state.tab);
  const sourceTools = Object.fromEntries(state.pack.source_calls.map((c) => [c.call_id, c.tool_name]));
  const el = $("sheet");
  el.innerHTML = "";
  const h = document.createElement("h3"); h.textContent = s.title; el.appendChild(h);
  if (s.chart) el.insertAdjacentHTML("beforeend", waterfall(s.chart));
  s.tables.forEach((t) => el.appendChild(table(t, sourceTools)));
  if (s.name === "Sources") el.querySelectorAll("tbody tr").forEach((tr) => { tr.dataset.callId = tr.cells[0].textContent; });
}

function openSource(callId) {
  state.tab = "Sources"; renderTabs(); renderSheet();
  const row = [...$("sheet").querySelectorAll("tbody tr")].find((tr) => tr.dataset.callId === callId);
  if (row) { row.classList.add("flash"); row.scrollIntoView({ block: "center" }); }
}

function renderHeadline() {
  const h = state.pack.headline;
  $("headline").innerHTML = [
    ["Previous allowance", short(h.previous_allowance)],
    ["Current allowance", short(h.current_allowance)],
    ["Change", (h.change >= 0 ? "+" : "−") + short(Math.abs(h.change))],
    ["Residual", FMT.money(h.residual) === "-" ? "$0" : FMT.money(h.residual)],
    ["Reconciliation", `<span class="pill ${h.reconciliation}">${h.reconciliation}</span>`],
  ].map(([l, v]) => `<div class="stat"><div class="l">${l}</div><div class="v">${v}</div></div>`).join("");
}

async function loadPack(ev) {
  if (ev) ev.preventDefault();
  const q = new URLSearchParams({ previous_date: $("previous").value, current_date: $("current").value, framework: $("framework").value });
  $("export").href = STATIC ? packFile(q, "xlsx") : "api/export.xlsx?" + q;
  $("load").disabled = true;
  $("sheet").innerHTML = '<p class="empty">Building the pack from the tools… (first build warms the attribution cache)</p>';
  try {
    const r = await fetch(STATIC ? packFile(q, "json") : "api/pack?" + q);
    if (!r.ok) {
      if (STATIC) throw new Error("this period pair is not in the snapshot");
      throw new Error((await r.json()).detail || r.statusText);
    }
    state.pack = await r.json();
    renderHeadline(); renderTabs(); renderCurrent();
  } catch (e) {
    $("sheet").innerHTML = `<p class="error">Could not build the pack: ${esc(e.message)}</p>`;
  } finally { $("load").disabled = false; }
}

function backendOverrides() {
  const b = $("backend").value;
  if (b === "ollama") return { provider: "vllm", base_url: "http://localhost:11434/v1", model: $("ollamamodel").value.trim() };
  if (b === "vllm") return { provider: "vllm", base_url: $("baseurl").value, model: $("ollamamodel").value.trim() || null };
  if (b === "nvidia") return { provider: "nvidia" };   // key from NVIDIA_API_KEY (.env); model from config/agents.yaml
  return { provider: "mock" };
}

async function ask() {
  const question = $("question").value.trim() || $("question").placeholder;
  $("askbtn").disabled = true;
  $("answer").innerHTML = '<p class="muted">Agents are working… local models can take a minute.</p>';
  try {
    let a;
    if (STATIC) {
      a = ((state.answers || {})[$("mode").value] || {})[question];
      if (!a) throw new Error("This is a static snapshot, so it can only replay the suggested questions. "
        + "Run the app locally (python -m demo_app.server) to ask your own.");
    } else {
      const r = await fetch("api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, mode: $("mode").value, label: $("label").value.trim() || null, ...backendOverrides() }) });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      a = await r.json();
    }
    const g = (a.grounding || {}).status || "n/a";
    const trace = a.tool_calls.map((c) => `<li class="${c.status === "ok" ? "" : "err"}">${esc(c.tool_name)}(${esc(JSON.stringify(c.parameters || c.arguments_requested || {}))}) · ${esc(c.status)} · ${esc(c.call_id)}</li>`).join("");
    $("answer").innerHTML = `
      <div class="meta"><span class="pill ${esc(g)}">${esc(g)}</span><span>${esc(a.provider.provider)}:${esc(a.provider.model)}</span>
        <span>${esc(a.duration_s)}s${STATIC ? " (recorded)" : ""}${a.timing ? ` (model ${esc(a.timing.llm_s)}s, tools ${esc(a.timing.tool_s)}s, ${esc(a.timing.llm_turns)} turns)` : ""}</span>
        <span>${a.tool_calls.length} tool calls</span>
        ${a.context_truncated ? `<span class="error">Prompt likely truncated by Ollama's 4k context: use model nemotron-3-nano-4b-32k</span>` : ""}
        ${a.quality ? `<span>quality ${esc(a.quality.reward)} · facts ${esc(a.quality.checks_passed)}/${esc(a.quality.checks_total)}</span>` : (a.timing ? "<span>quality: not in evaluation bank</span>" : "")}
        ${a.specialists.length ? `<span>specialists: ${esc(a.specialists.join(", "))}</span>` : ""}</div>
      <div class="text"></div>
      <details><summary>Tool trace</summary><ul class="trace">${trace}</ul></details>`;
    $("answer").querySelector(".text").textContent = a.answer || "(no answer)";
  } catch (e) {
    $("answer").innerHTML = `<p class="error">${STATIC ? "" : "The agent run failed: "}${esc(e.message)}</p>`;
  } finally { $("askbtn").disabled = false; }
}

// ---------------------------------------------------------------- warm-up / keep-warm (local models)
let warmTimer = null;
const WARM_TEXT = {
  warming: "Model warming up: first read of the agent prompt (about 1-2 min on a laptop)",
  ready: "Model warm and kept warm",
  error: "Warm-up failed",
};

function showWarm(st) {
  const el = $("warmstate");
  if (!st || st.state === "not_needed" || st.state === "idle") { el.textContent = ""; el.className = "warm"; return; }
  let text = WARM_TEXT[st.state] || st.state;
  if (st.state === "ready" && st.warm_s) text += ` (warm-up took ${st.warm_s}s)`;
  if (st.state === "error" && st.error) text += `: ${st.error}`;
  el.textContent = text;
  el.className = `warm ${st.state}`;
}

async function pollWarm() {
  clearTimeout(warmTimer);
  try {
    const st = await (await fetch("api/warmup")).json();
    showWarm(st);
    warmTimer = setTimeout(pollWarm, st.state === "warming" ? 4000 : 30000);
  } catch (e) { /* server restarting; the next warm-up call will retry */ }
}

async function warmUp() {
  if (STATIC) return;
  const o = backendOverrides();
  if (o.provider === "mock") { showWarm(null); return; }
  try {
    const r = await fetch("api/warmup", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: $("mode").value, ...o }) });
    showWarm(await r.json());
    pollWarm();
  } catch (e) { showWarm({ state: "error", error: e.message }); }
}

async function init() {
  const ctx = await (await fetch(STATIC ? "data/context.json" : "api/context")).json();
  if (STATIC) {
    state.answers = await (await fetch("data/answers.json")).json();
    document.querySelector(".brand .sub").textContent = `Allowance attribution pack · static snapshot, precomputed by the tools on ${ctx.generated_at}`;
    $("backend").innerHTML = '<option value="mock">Recorded runs (mock planner)</option>';
    $("backend").disabled = true;
    $("question").readOnly = true;
    $("question").placeholder = "Pick a suggested question above";
  }
  const dates = ctx.reporting_dates;
  ["previous", "current"].forEach((id, i) => {
    $(id).innerHTML = dates.map((d) => `<option>${d}</option>`).join("");
    $(id).value = dates[i === 0 ? Math.max(0, dates.length - 2) : dates.length - 1];
  });
  $("chips").innerHTML = "";
  ctx.demo_questions.forEach((q) => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = q;
    b.addEventListener("click", () => { $("question").value = q; });
    $("chips").appendChild(b);
  });
  $("backend").addEventListener("change", () => {
    $("urlrow").hidden = $("backend").value !== "vllm";
    $("ollamarow").hidden = ["mock", "nvidia"].includes($("backend").value);
    warmUp();
  });
  ["mode", "ollamamodel", "baseurl"].forEach((id) => $(id).addEventListener("change", warmUp));
  if ($("backend").value !== "mock") {                // a restored selection: show the fields and warm up now
    $("urlrow").hidden = $("backend").value !== "vllm";
    $("ollamarow").hidden = ["mock", "nvidia"].includes($("backend").value);
    warmUp();
  }
  if (location.hash.toLowerCase() === "#scenarios") state.tab = SCEN_TAB;   // direct link to the comparison
  $("pick").addEventListener("submit", loadPack);
  $("askbtn").addEventListener("click", ask);
  loadPack();
}
init();
