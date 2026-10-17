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

function renderTabs() {
  const tabs = $("tabs");
  tabs.innerHTML = "";
  state.pack.sheets.forEach((s) => {
    const b = document.createElement("button");
    b.type = "button"; b.role = "tab"; b.textContent = s.name;
    b.setAttribute("aria-selected", String(s.name === state.tab));
    b.addEventListener("click", () => { state.tab = s.name; renderTabs(); renderSheet(); });
    tabs.appendChild(b);
  });
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
    renderHeadline(); renderTabs(); renderSheet();
  } catch (e) {
    $("sheet").innerHTML = `<p class="error">Could not build the pack: ${esc(e.message)}</p>`;
  } finally { $("load").disabled = false; }
}

function backendOverrides() {
  const b = $("backend").value;
  if (b === "ollama") return { provider: "vllm", base_url: "http://localhost:11434/v1", model: "nemotron-3-nano:4b" };
  if (b === "vllm") return { provider: "vllm", base_url: $("baseurl").value };
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
        body: JSON.stringify({ question, mode: $("mode").value, ...backendOverrides() }) });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      a = await r.json();
    }
    const g = (a.grounding || {}).status || "n/a";
    const trace = a.tool_calls.map((c) => `<li class="${c.status === "ok" ? "" : "err"}">${esc(c.tool_name)}(${esc(JSON.stringify(c.parameters || c.arguments_requested || {}))}) · ${esc(c.status)} · ${esc(c.call_id)}</li>`).join("");
    $("answer").innerHTML = `
      <div class="meta"><span class="pill ${esc(g)}">${esc(g)}</span><span>${esc(a.provider.provider)}:${esc(a.provider.model)}</span>
        <span>${esc(a.duration_s)}s${STATIC ? " (recorded)" : ""}</span><span>${a.tool_calls.length} tool calls</span>
        ${a.specialists.length ? `<span>specialists: ${esc(a.specialists.join(", "))}</span>` : ""}</div>
      <div class="text"></div>
      <details><summary>Tool trace</summary><ul class="trace">${trace}</ul></details>`;
    $("answer").querySelector(".text").textContent = a.answer || "(no answer)";
  } catch (e) {
    $("answer").innerHTML = `<p class="error">${STATIC ? "" : "The agent run failed: "}${esc(e.message)}</p>`;
  } finally { $("askbtn").disabled = false; }
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
  $("backend").addEventListener("change", () => { $("urlrow").hidden = $("backend").value !== "vllm"; });
  $("pick").addEventListener("submit", loadPack);
  $("askbtn").addEventListener("click", ask);
  loadPack();
}
init();
