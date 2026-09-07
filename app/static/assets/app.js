const CONCURRENCY = 3;

const state = {
  items: [],
  filter: "all",
  inFlight: 0,
  ledgerLoading: false,
  ledgerSignature: "",
};

const els = {
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("fileInput"),
  pickFiles: document.getElementById("pickFiles"),
  queueSection: document.getElementById("queueSection"),
  queue: document.getElementById("queue"),
  queueMeta: document.getElementById("queueMeta"),
  board: document.getElementById("board"),
  boardEmpty: document.getElementById("boardEmpty"),
  currencyList: document.getElementById("currencyList"),
  textInput: document.getElementById("textInput"),
  textSubmit: document.getElementById("textSubmit"),
  refreshBtn: document.getElementById("refreshBtn"),
  apiStatus: document.getElementById("apiStatus"),
  toast: document.getElementById("toast"),
};

function toast(message) {
  els.toast.hidden = false;
  els.toast.textContent = message;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    els.toast.hidden = true;
  }, 3200);
}

function setApiStatus(ok) {
  els.apiStatus.dataset.state = ok ? "ok" : "down";
  els.apiStatus.querySelector(".status-text").textContent = ok ? "api live" : "api down";
}

async function checkHealth() {
  try {
    const r = await fetch("/health");
    setApiStatus(r.ok);
  } catch {
    setApiStatus(false);
  }
}

function animateNumber(el, next, { money = false } = {}) {
  const span = el.querySelector("span") || el;
  const prev = Number(span.dataset.value || 0);
  const target = Number(next) || 0;
  span.dataset.value = String(target);
  const start = performance.now();
  const dur = 520;

  function frame(now) {
    const t = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    const val = prev + (target - prev) * eased;
    if (money) {
      span.textContent = target === 0 && next === "—" ? "—" : formatMoney(val);
    } else {
      span.textContent = String(Math.round(val));
    }
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function formatMoney(n) {
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function classify(row) {
  if (!row.is_receipt) return "reject";
  if (row.needs_review) return "review";
  return "receipt";
}

function ticketHTML(row) {
  const kind = classify(row);
  const flag =
    kind === "reject" ? "not a receipt" : kind === "review" ? "needs review" : "receipt";
  const merchant = row.merchant || (kind === "reject" ? "No merchant detected" : "Unknown merchant");
  const total =
    row.total == null
      ? "—"
      : `${row.currency ? row.currency + " " : ""}${row.total}`;
  return `
    <article class="ticket" data-kind="${kind}" data-id="${row.id ?? ""}">
      <div class="ticket-top">
        <span class="ticket-flag ${kind === "receipt" ? "" : kind}">${flag}</span>
        <span class="ticket-id">#${row.id ?? "—"}</span>
      </div>
      <h3 class="ticket-merchant">${escapeHtml(merchant)}</h3>
      <p class="ticket-total">${escapeHtml(String(total))}</p>
      <p class="ticket-meta">
        <span>date ${escapeHtml(row.date || "—")}</span>
        <span>tax ${escapeHtml(row.tax == null ? "—" : String(row.tax))}</span>
      </p>
    </article>
  `;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderBoard() {
  const filtered = state.items.filter((row) => {
    if (state.filter === "all") return true;
    return classify(row) === state.filter;
  });
  els.board.innerHTML = filtered.map(ticketHTML).join("");
  els.boardEmpty.hidden = state.items.length > 0;
}

function updateAnalytics() {
  const total = state.items.length;
  const receipts = state.items.filter((r) => r.is_receipt).length;
  const review = state.items.filter((r) => r.needs_review).length;
  let spend = 0;
  const byCurrency = {};

  for (const row of state.items) {
    if (!row.is_receipt || row.total == null) continue;
    const n = Number(row.total);
    if (!Number.isFinite(n)) continue;
    spend += n;
    const c = row.currency || "unknown";
    byCurrency[c] = (byCurrency[c] || 0) + n;
  }

  animateNumber(document.querySelector('[data-metric="total"]'), total);
  animateNumber(document.querySelector('[data-metric="receipts"]'), receipts);
  animateNumber(document.querySelector('[data-metric="review"]'), review);

  const spendEl = document.querySelector('[data-metric="spend"]');
  if (spend === 0) {
    spendEl.querySelector("span").textContent = "—";
    spendEl.querySelector("span").dataset.value = "0";
  } else {
    animateNumber(spendEl, spend, { money: true });
  }

  const receiptRate = total ? (receipts / total) * 100 : 0;
  const reviewRate = total ? (review / total) * 100 : 0;
  document.querySelector('[data-bar="receiptRate"]').style.width = `${receiptRate}%`;
  document.querySelector('[data-bar="reviewRate"]').style.width = `${reviewRate}%`;

  const entries = Object.entries(byCurrency).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    els.currencyList.innerHTML = `<li class="muted">No spend yet</li>`;
  } else {
    els.currencyList.innerHTML = entries
      .map(
        ([code, amount]) =>
          `<li><span>${escapeHtml(code)}</span><span>${formatMoney(amount)}</span></li>`
      )
      .join("");
  }
}

async function loadLedger() {
  if (state.ledgerLoading) return;
  state.ledgerLoading = true;
  try {
    const r = await fetch("/v1/receipts", { cache: "no-store" });
    if (!r.ok) throw new Error("Could not load receipts");
    const rows = await r.json();
    const signature = JSON.stringify(rows);
    if (signature === state.ledgerSignature) return;
    state.ledgerSignature = signature;
    state.items = rows.map(normalizeRow);
    renderBoard();
    updateAnalytics();
  } finally {
    state.ledgerLoading = false;
  }
}

function normalizeRow(row) {
  return {
    id: row.id,
    is_receipt: Boolean(row.is_receipt),
    merchant: row.merchant,
    total: row.total,
    currency: row.currency,
    date: row.date,
    tax: row.tax,
    needs_review: Boolean(row.needs_review),
  };
}

function updateQueueMeta() {
  const scanning = els.queue.querySelectorAll(".queue-item.scanning").length;
  const done = els.queue.querySelectorAll(".queue-item.done").length;
  const err = els.queue.querySelectorAll(".queue-item.error").length;
  els.queueMeta.textContent = `${scanning} scanning · ${done} done · ${err} failed`;
}

function enqueuePreview(file) {
  els.queueSection.hidden = false;
  const item = document.createElement("div");
  item.className = "queue-item scanning";
  const img = document.createElement("img");
  img.alt = file.name;
  img.src = URL.createObjectURL(file);
  const badge = document.createElement("div");
  badge.className = "badge";
  badge.textContent = "scanning…";
  item.append(img, badge);
  els.queue.prepend(item);
  updateQueueMeta();
  return { item, badge, url: img.src };
}

async function uploadImage(file, preview) {
  const body = new FormData();
  body.append("file", file, file.name);
  const r = await fetch("/v1/ingest/image", { method: "POST", body });
  if (!r.ok) {
    let detail = `Upload failed (${r.status})`;
    try {
      const j = await r.json();
      detail = j.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  const data = await r.json();
  preview.item.classList.remove("scanning");
  preview.item.classList.add("done");
  preview.badge.textContent = data.extract?.is_receipt ? "receipt" : "not receipt";
  updateQueueMeta();
  return data.extract;
}

async function runQueue(files) {
  const list = [...files];
  if (!list.length) return;
  els.dropzone.classList.add("busy");
  let idx = 0;

  async function worker() {
    while (idx < list.length) {
      const file = list[idx++];
      const preview = enqueuePreview(file);
      state.inFlight += 1;
      try {
        await uploadImage(file, preview);
        await loadLedger();
        await loadUsage();
      } catch (err) {
        preview.item.classList.remove("scanning");
        preview.item.classList.add("error");
        preview.badge.textContent = "failed";
        updateQueueMeta();
        toast(err.message || "Upload failed");
      } finally {
        state.inFlight -= 1;
        URL.revokeObjectURL(preview.url);
      }
    }
  }

  const workers = Array.from({ length: Math.min(CONCURRENCY, list.length) }, () => worker());
  await Promise.all(workers);
  els.dropzone.classList.remove("busy");
}

function onFiles(fileList) {
  const files = [...fileList].filter((f) =>
    ["image/jpeg", "image/png", "image/webp"].includes(f.type)
  );
  if (!files.length) {
    toast("Use jpeg, png, or webp images");
    return;
  }
  if (files.length < fileList.length) {
    toast("Skipped unsupported file types");
  }
  runQueue(files);
}

els.pickFiles.addEventListener("click", (e) => {
  e.stopPropagation();
  els.fileInput.click();
});

els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    els.fileInput.click();
  }
});

els.fileInput.addEventListener("change", () => {
  if (els.fileInput.files?.length) onFiles(els.fileInput.files);
  els.fileInput.value = "";
});

["dragenter", "dragover"].forEach((ev) => {
  els.dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((ev) => {
  els.dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("dragover");
  });
});

els.dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer?.files?.length) onFiles(e.dataTransfer.files);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    state.filter = chip.dataset.filter;
    renderBoard();
  });
});

els.refreshBtn.addEventListener("click", async () => {
  try {
    await loadLedger();
    toast("Board refreshed");
  } catch (err) {
    toast(err.message);
  }
});

els.textSubmit.addEventListener("click", async () => {
  const text = els.textInput.value.trim();
  if (!text) {
    toast("Paste some receipt text first");
    return;
  }
  els.textSubmit.disabled = true;
  try {
    const r = await fetch("/v1/ingest", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.detail || `Extract failed (${r.status})`);
    }
    els.textInput.value = "";
    await loadLedger();
    await loadUsage();
    toast("Text extracted");
  } catch (err) {
    toast(err.message || "Extract failed");
  } finally {
    els.textSubmit.disabled = false;
  }
});

// ── Cost analytics ──

async function loadUsage() {
  try {
    const r = await fetch("/v1/usage", { cache: "no-store" });
    if (!r.ok) return;
    const data = await r.json();
    renderUsage(data);
  } catch { /* silent */ }
}

function renderUsage(data) {
  const callsEl = document.querySelector('[data-metric="apiCalls"]');
  const totalEl = document.querySelector('[data-metric="totalUsd"]');
  const avgEl = document.querySelector('[data-metric="avgUsd"]');
  const tokensEl = document.querySelector('[data-metric="totalTokens"]');
  const costList = document.getElementById("costList");

  if (callsEl) animateNumber(callsEl, data.total_calls);
  if (totalEl) {
    const span = totalEl.querySelector("span");
    span.textContent = "$" + (data.total_usd || 0).toFixed(4);
  }
  if (avgEl) {
    const span = avgEl.querySelector("span");
    span.textContent = "$" + (data.avg_usd_per_call || 0).toFixed(4);
  }
  if (tokensEl) {
    const span = tokensEl.querySelector("span");
    span.textContent = (data.total_tokens || 0).toLocaleString();
  }

  if (costList && data.calls && data.calls.length > 0) {
    const recent = data.calls.slice(-10).reverse();
    costList.innerHTML = recent
      .map((c) => {
        const cost = c.usd != null ? "$" + Number(c.usd).toFixed(4) : "—";
        const model = c.model ? c.model.split("/").pop() : "—";
        const ts = c.ts ? new Date(c.ts).toLocaleTimeString() : "";
        return `<li>
          <span class="cost-kind">${escapeHtml(c.kind || "—")}</span>
          <span class="cost-model" title="${escapeHtml(c.model || "")}">${escapeHtml(model)}</span>
          <span class="cost-amount">${cost}</span>
        </li>`;
      })
      .join("");
  }
}

checkHealth();
loadLedger().catch(() => {
  els.boardEmpty.textContent = "Could not load the ledger yet. Upload to start.";
});
loadUsage();

setInterval(() => {
  loadLedger().catch(() => {});
}, 2500);
