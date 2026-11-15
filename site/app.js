/* Paradox Earth explorer. All numbers come from site/data/*.json at run time. */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const params = new URLSearchParams(location.search);
  const DATA = "data/";

  // ---------- palette (kept in sync with style.css tokens) ----------
  const C = {
    ocean: [9, 17, 31], grat: [20, 32, 54], neutral: [96, 108, 132], faded: [44, 55, 79],
    mid: [52, 62, 84], wet: [75, 163, 227], dry: [227, 154, 59], outlier: [184, 150, 226], none: [30, 40, 58],
  };
  const rgb = (a) => `rgb(${a[0] | 0},${a[1] | 0},${a[2] | 0})`;
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  // ---------- formatting ----------
  const fmt = (x, d = 1) => (x == null || !isFinite(x) ? "–" : Number(x).toFixed(d).replace("-", "−"));
  const sgn = (x, d = 1) => (x == null || !isFinite(x) ? "–" : (x > 0 ? "+" : "") + fmt(x, d));
  const pct = (x, d = 1) => (x == null ? "–" : fmt(x * 100, d) + "%");
  const pval = (p) => (p == null ? "–" : p < 0.001 ? "< 0.001" : fmt(p, p < 0.01 ? 3 : 2));
  const area = (km2) => {
    if (km2 == null || !isFinite(km2)) return "–";
    if (km2 >= 1e6) return fmt(km2 / 1e6, 2) + " million km²";
    if (km2 >= 1e4) return (Math.round(km2 / 1000) * 1000).toLocaleString("en-US") + " km²";
    return Math.round(km2).toLocaleString("en-US") + " km²";
  };
  const latStr = (v) => `${Math.abs(v)}°${v >= 0 ? "N" : "S"}`;
  const lonStr = (v) => `${Math.abs(v)}°${v >= 0 ? "E" : "W"}`;
  const yearOf = (s) => String(s || "").slice(0, 4);

  // ---------- state ----------
  const S = {
    meta: null, deg: 1, tiles: [], idx: new Map(), regions: { up: [], down: [] }, regionById: new Map(),
    cards: [], cardsFromFile: false, cardByRegion: new Map(), team: null,
    layer: "sig", view: "globe", reveal: 0, highlight: null, highlightId: null, selected: -1,
    series: new Map(), pctMax: 20, strength: null, alpha: 0.05,
  };

  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-cache" });
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return r.json();
  }

  // ---------- data ----------
  function ingestTiles(t) {
    S.deg = Number(t.tile_deg) || 1;
    const col = Object.fromEntries(t.cols.map((c, i) => [c, i]));
    S.tiles = t.rows.map((r) => ({
      id: r[col.id], w: +r[col.west], s: +r[col.south], slope: r[col.slope], pct: r[col.pct],
      p: r[col.p], q: r[col.q], sig: r[col.sig] | 0, lisa: r[col.lisa], region: r[col.region],
    }));
    S.tiles.forEach((tile, i) => S.idx.set(cellKey(tile.s, tile.w), i));
    const abs = S.tiles.map((x) => Math.abs(x.pct)).filter(isFinite).sort((a, b) => a - b);
    S.pctMax = Math.max(5, Math.ceil(abs[Math.floor(abs.length * 0.95)] / 5) * 5 || 20);
    const smax = Math.max(...S.tiles.map((x) => Math.abs(x.slope) || 0)) || 1;
    S.strength = S.tiles.map((x) => Math.sqrt(Math.abs(x.slope || 0) / smax));
  }
  const cellKey = (south, west) => `${Math.round(south / S.deg)}|${Math.round(west / S.deg)}`;
  function tileAt(lat, lng) {
    if (lat == null || lng == null) return -1;
    let L = ((lng + 180) % 360 + 360) % 360 - 180;
    const s = Math.floor(lat / S.deg) * S.deg, w = Math.floor(L / S.deg) * S.deg;
    const i = S.idx.get(cellKey(s, w));
    return i == null ? -1 : i;
  }
  function blockKey(south, west) {
    const b = (v, n) => { const k = Math.floor(v / 10) * 10; return (k < 0 ? "-" : "+") + String(Math.abs(k)).padStart(n - 1, "0"); };
    return `b_${b(south, 4)}_${b(west, 5)}`;
  }
  async function loadSeries(tile) {
    const key = blockKey(tile.s, tile.w);
    if (!S.series.has(key)) S.series.set(key, getJSON(`${DATA}series/${key}.json`).catch(() => null));
    const block = await S.series.get(key);
    return block && block.tiles[tile.id] ? { start: block.start, values: block.tiles[tile.id] } : null;
  }

  // ---------- texture ----------
  const CELL = 8; // pixels per tile in the equirectangular texture
  const tex = document.createElement("canvas");
  const tctx = tex.getContext("2d");
  function sizeTexture() { tex.width = Math.round(360 / S.deg) * CELL; tex.height = Math.round(180 / S.deg) * CELL; }

  function tileColor(t, i) {
    let target;
    if (S.layer === "sig") target = t.sig > 0 ? C.wet : t.sig < 0 ? C.dry : C.faded;
    else if (S.layer === "pct") {
      const v = clamp((t.pct || 0) / S.pctMax, -1, 1);
      target = mix(C.mid, v >= 0 ? C.wet : C.dry, Math.pow(Math.abs(v), 0.8));
    } else {
      target = t.lisa === "HH" ? C.wet : t.lisa === "LL" ? C.dry : (t.lisa === "HL" || t.lisa === "LH") ? C.outlier : t.lisa === "nn" ? C.none : C.faded;
    }
    if (S.reveal < 1) {
      const delay = (1 - S.strength[i]) * 0.55;
      const k = clamp((S.reveal - delay) / 0.45, 0, 1);
      target = mix(C.neutral, target, k * k * (3 - 2 * k));
    }
    if (S.highlight && !S.highlight.has(t.id)) target = mix(target, C.ocean, 0.62);
    return target;
  }

  function drawTexture() {
    const W = tex.width, H = tex.height, cpd = CELL / S.deg;
    tctx.fillStyle = rgb(C.ocean); tctx.fillRect(0, 0, W, H);
    tctx.strokeStyle = rgb(C.grat); tctx.lineWidth = 2; tctx.beginPath();
    for (let lon = -180; lon <= 180; lon += 30) { const x = (lon + 180) * cpd; tctx.moveTo(x, 0); tctx.lineTo(x, H); }
    for (let lat = -60; lat <= 60; lat += 30) { const y = (90 - lat) * cpd; tctx.moveTo(0, y); tctx.lineTo(W, y); }
    tctx.stroke();
    const sz = CELL; // tiles abut; no gutter so coastlines read as solid land
    for (let i = 0; i < S.tiles.length; i++) {
      const t = S.tiles[i];
      tctx.fillStyle = rgb(tileColor(t, i));
      tctx.fillRect((t.w + 180) * cpd, (90 - t.s - S.deg) * cpd, sz, sz);
    }
    if (S.highlight) outline(S.highlight, "rgba(255,255,255,.95)", 3);
    if (S.selected >= 0) {
      const t = S.tiles[S.selected];
      tctx.strokeStyle = "#ffd479"; tctx.lineWidth = 3;
      tctx.strokeRect((t.w + 180) * cpd - 1.5, (90 - t.s - S.deg) * cpd - 1.5, CELL + 3, CELL + 3);
    }
  }
  function outline(set, color, lw) {
    const cpd = CELL / S.deg, d = S.deg;
    const inSet = (s, w) => { const i = S.idx.get(cellKey(s, w)); return i != null && set.has(S.tiles[i].id); };
    tctx.strokeStyle = color; tctx.lineWidth = lw; tctx.beginPath();
    for (const id of set) {
      const i = S.idx.get(idToKey(id)); if (i == null) continue;
      const t = S.tiles[i], x = (t.w + 180) * cpd, y = (90 - t.s - d) * cpd;
      const wrapW = (w) => (w >= 180 ? w - 360 : w < -180 ? w + 360 : w);
      if (!inSet(t.s + d, t.w)) { tctx.moveTo(x, y); tctx.lineTo(x + CELL, y); }
      if (!inSet(t.s - d, t.w)) { tctx.moveTo(x, y + CELL); tctx.lineTo(x + CELL, y + CELL); }
      if (!inSet(t.s, wrapW(t.w - d))) { tctx.moveTo(x, y); tctx.lineTo(x, y + CELL); }
      if (!inSet(t.s, wrapW(t.w + d))) { tctx.moveTo(x + CELL, y); tctx.lineTo(x + CELL, y + CELL); }
    }
    tctx.stroke();
  }
  const idKeyCache = new Map();
  function idToKey(id) {
    if (!idKeyCache.size) S.tiles.forEach((t) => idKeyCache.set(t.id, cellKey(t.s, t.w)));
    return idKeyCache.get(id);
  }

  let frameReq = 0;
  function render() {
    if (frameReq) return;
    frameReq = requestAnimationFrame(() => { frameReq = 0; drawTexture(); pushGlobe(); drawFlat(); });
  }

  // ---------- 3D globe ----------
  let globe = null, globeMapReady = false;
  function webglOK() {
    try { const c = document.createElement("canvas"); return !!(window.WebGLRenderingContext && (c.getContext("webgl2") || c.getContext("webgl"))); } catch { return false; }
  }
  function initGlobe() {
    if (typeof window.Globe !== "function" || !webglOK()) return false;
    const el = $("#globe");
    try {
      globe = new window.Globe(el, { animateIn: !reduceMotion, rendererConfig: { antialias: true, alpha: true, preserveDrawingBuffer: params.has("capture") } });
    } catch (e) { console.warn("Globe init failed", e); globe = null; return false; }
    globe
      .backgroundColor("rgba(0,0,0,0)")
      .showAtmosphere(true).atmosphereColor("#5f9fd6").atmosphereAltitude(0.16)
      .showGraticules(false)
      .width(el.clientWidth).height(el.clientHeight)
      .pointOfView({ lat: 18, lng: 10, altitude: 2.3 })
      .onGlobeClick(({ lat, lng }) => { stopSpin(); inspect(tileAt(lat, lng), true); });
    const mat = globe.globeMaterial();
    if (mat) { mat.shininess = 4; }
    const ctr = globe.controls();
    ctr.autoRotate = !reduceMotion; ctr.autoRotateSpeed = 0.35; ctr.enableDamping = true;
    ctr.minDistance = 120; ctr.maxDistance = 600;
    el.addEventListener("pointerdown", stopSpin);
    el.addEventListener("wheel", stopSpin, { passive: true });
    el.addEventListener("pointermove", (e) => hover(e, () => {
      const r = el.getBoundingClientRect();
      return globe.toGlobeCoords(e.clientX - r.left, e.clientY - r.top);
    }));
    el.addEventListener("pointerleave", hideTip);
    globe.globeImageUrl(tex.toDataURL());
    const wait = setInterval(() => {
      const m = globe.globeMaterial();
      if (m && m.map && m.map.image) {
        clearInterval(wait); globeMapReady = true;
        if (m.emissive) { m.emissiveMap = m.map; m.emissive.setRGB(0.55, 0.55, 0.55); m.needsUpdate = true; }
        pushGlobe();
      }
    }, 150);
    return true;
  }
  function pushGlobe() {
    if (!globe || !globeMapReady || S.view !== "globe") return;
    const m = globe.globeMaterial().map;
    if (m.image !== tex) m.image = tex;
    m.needsUpdate = true;
  }
  function stopSpin() { if (globe) globe.controls().autoRotate = false; }

  // ---------- 2D flat map ----------
  const flat = $("#flat"), fctx = flat.getContext("2d");
  const F = { k: 3, cx: 10, cy: 12, minLat: -60, maxLat: 85 };
  function fitFlat() {
    const W = flat.clientWidth, H = flat.clientHeight;
    F.kMin = Math.min(W / 360, H / (F.maxLat - F.minLat));
    F.k = Math.max(F.kMin, H / (F.maxLat - F.minLat));
    F.cy = (F.maxLat + F.minLat) / 2;
  }
  function clampFlat() {
    const H = flat.clientHeight, half = H / 2 / F.k, span = F.maxLat - F.minLat;
    const padS = 110 / F.k; // allow panning a little past 60°S so southern regions can sit above the legend
    F.cy = span / 2 <= half ? (F.maxLat + F.minLat) / 2 : clamp(F.cy, Math.min(F.minLat - padS + half, F.maxLat - half), F.maxLat - half);
    F.cx = ((F.cx + 180) % 360 + 360) % 360 - 180;
    F.k = clamp(F.k, F.kMin || 1, 60);
  }
  function drawFlat() {
    if (S.view !== "flat") return;
    const dpr = Math.min(devicePixelRatio || 1, 2), W = flat.clientWidth, H = flat.clientHeight;
    if (flat.width !== Math.round(W * dpr) || flat.height !== Math.round(H * dpr)) { flat.width = Math.round(W * dpr); flat.height = Math.round(H * dpr); }
    clampFlat();
    fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fctx.fillStyle = rgb(C.ocean); fctx.fillRect(0, 0, W, H);
    fctx.imageSmoothingEnabled = F.k < CELL / S.deg;
    for (const off of [-360, 0, 360]) {
      const dx = W / 2 + (-180 + off - F.cx) * F.k, dy = H / 2 - (90 - F.cy) * F.k;
      if (dx > W || dx + 360 * F.k < 0) continue;
      fctx.drawImage(tex, 0, 0, tex.width, tex.height, dx, dy, 360 * F.k, 180 * F.k);
    }
  }
  const flatToGeo = (x, y) => ({ lng: F.cx + (x - flat.clientWidth / 2) / F.k, lat: F.cy - (y - flat.clientHeight / 2) / F.k });
  (() => {
    let drag = null;
    flat.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, cx: F.cx, cy: F.cy, moved: false }; flat.setPointerCapture(e.pointerId); });
    flat.addEventListener("pointermove", (e) => {
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; flat.classList.add("dragging"); hideTip(); }
        if (drag.moved) { F.cx = drag.cx - dx / F.k; F.cy = drag.cy + dy / F.k; drawFlat(); }
        return;
      }
      hover(e, () => { const r = flat.getBoundingClientRect(); return flatToGeo(e.clientX - r.left, e.clientY - r.top); });
    });
    flat.addEventListener("pointerup", (e) => {
      if (drag && !drag.moved) { const r = flat.getBoundingClientRect(); const g = flatToGeo(e.clientX - r.left, e.clientY - r.top); inspect(tileAt(g.lat, g.lng), true); }
      drag = null; flat.classList.remove("dragging");
    });
    flat.addEventListener("pointercancel", () => { drag = null; flat.classList.remove("dragging"); });
    flat.addEventListener("pointerleave", hideTip);
    flat.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = flat.getBoundingClientRect(), before = flatToGeo(e.clientX - r.left, e.clientY - r.top);
      F.k *= Math.exp(-e.deltaY * 0.0015); clampFlat();
      const after = flatToGeo(e.clientX - r.left, e.clientY - r.top);
      F.cx += before.lng - after.lng; F.cy += before.lat - after.lat; drawFlat();
    }, { passive: false });
  })();

  // ---------- view switching, zoom, keyboard ----------
  function setView(v) {
    if (v === "globe" && !globe && !initGlobe()) v = "flat";
    S.view = v;
    $("#globe").hidden = v !== "globe";
    flat.hidden = v !== "flat";
    const btn = $("#viewToggle");
    btn.textContent = v === "globe" ? "Flat map" : "3D globe";
    btn.setAttribute("aria-pressed", String(v === "flat"));
    btn.setAttribute("aria-label", v === "globe" ? "Switch to flat map" : "Switch to 3D globe");
    btn.hidden = !globe && !canGlobe;
    if (globe) { v === "globe" ? globe.resumeAnimation() : globe.pauseAnimation(); }
    if (v === "flat") { if (!F.kMin) fitFlat(); }
    $("#stageHint").textContent = v === "globe"
      ? "Drag to rotate · scroll to zoom · click a tile for its evidence"
      : "Drag to pan · scroll or use + − to zoom · tap a tile for its evidence";
    resize(); render();
  }
  let canGlobe = false;
  function resize() {
    const el = $("#stage");
    if (globe && S.view === "globe") globe.width(el.clientWidth).height(el.clientHeight);
    if (S.view === "flat") { const prevMin = F.kMin; F.kMin = Math.min(flat.clientWidth / 360, flat.clientHeight / (F.maxLat - F.minLat)); if (!prevMin) fitFlat(); drawFlat(); }
  }
  new ResizeObserver(() => resize()).observe($("#stage"));

  function zoom(dir) {
    stopSpin();
    if (S.view === "globe" && globe) {
      const p = globe.pointOfView(); globe.pointOfView({ altitude: clamp(p.altitude * (dir > 0 ? 0.75 : 1.33), 0.25, 4) }, reduceMotion ? 0 : 300);
    } else { F.k *= dir > 0 ? 1.4 : 1 / 1.4; drawFlat(); }
  }
  function pan(dLat, dLng) {
    stopSpin();
    if (S.view === "globe" && globe) {
      const p = globe.pointOfView(); globe.pointOfView({ lat: clamp(p.lat + dLat, -80, 80), lng: p.lng + dLng }, reduceMotion ? 0 : 250);
    } else { F.cx += dLng; F.cy += dLat; drawFlat(); }
  }
  function centerGeo() {
    if (S.view === "globe" && globe) { const p = globe.pointOfView(); return { lat: p.lat, lng: p.lng }; }
    return { lat: F.cy, lng: F.cx };
  }
  $("#zoomBtns").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) zoom(b.dataset.zoom === "in" ? 1 : -1); });
  $("#stageFocus").addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 20 : 6;
    const map = { ArrowLeft: [0, -step], ArrowRight: [0, step], ArrowUp: [step, 0], ArrowDown: [-step, 0] };
    if (map[e.key]) { e.preventDefault(); pan(...map[e.key]); }
    else if (e.key === "+" || e.key === "=") { e.preventDefault(); zoom(1); }
    else if (e.key === "-" || e.key === "_") { e.preventDefault(); zoom(-1); }
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); const g = centerGeo(); const i = tileAt(g.lat, g.lng); if (i < 0) announceNoTile(); else inspect(i, true); }
  });
  function announceNoTile() { const p = $("#panel"); const n = document.createElement("p"); n.className = "sr-only"; n.textContent = "No land tile under the crosshair."; p.append(n); setTimeout(() => n.remove(), 2000); }
  $("#viewToggle").addEventListener("click", () => setView(S.view === "globe" ? "flat" : "globe"));
  document.querySelectorAll('input[name="layer"]').forEach((r) => r.addEventListener("change", () => { S.layer = r.value; renderLegend(); render(); }));

  // ---------- tooltip ----------
  const tip = $("#tooltip");
  let hoverRaf = 0;
  function hover(e, geoFn) {
    if (hoverRaf || e.pointerType === "touch") return;
    hoverRaf = requestAnimationFrame(() => {
      hoverRaf = 0;
      const g = geoFn(); const i = g ? tileAt(g.lat, g.lng) : -1;
      if (i < 0) { hideTip(); return; }
      const t = S.tiles[i], r = $("#stage").getBoundingClientRect();
      tip.innerHTML = `<b>${latStr(t.s)} ${lonStr(t.w)}</b><br>${esc(verdictShort(t))}<br><span class="mono">${sgn(t.pct)}% / decade · q ${pval(t.q)}</span>`;
      tip.hidden = false;
      const x = e.clientX - r.left, y = e.clientY - r.top, tw = tip.offsetWidth, th = tip.offsetHeight;
      tip.style.left = clamp(x + 14, 6, r.width - tw - 6) + "px";
      tip.style.top = clamp(y + 14, 6, r.height - th - 6) + "px";
      ($("#globe").style.cursor = "pointer");
    });
  }
  function hideTip() { tip.hidden = true; $("#globe").style.cursor = ""; }

  // ---------- legend ----------
  function renderLegend() {
    const sw = (c, label) => `<div class="row"><span class="sw" style="background:${rgb(c)}"></span>${label}</div>`;
    let h = "";
    if (S.layer === "sig") {
      h = `<div class="legend-title">Trend after FDR correction</div>${sw(C.wet, "Wetting (q &lt; " + S.alpha + ")")}${sw(C.dry, "Drying (q &lt; " + S.alpha + ")")}${sw(C.faded, "Not significant")}`;
    } else if (S.layer === "pct") {
      h = `<div class="legend-title">Trend, % of mean per decade</div><div class="ramp" style="background:linear-gradient(90deg,${rgb(C.dry)},${rgb(C.mid)},${rgb(C.wet)})"></div><div class="ramp-lab"><span>−${S.pctMax}</span><span>0</span><span>+${S.pctMax}</span></div>`;
    } else {
      h = `<div class="legend-title">Spatial clusters (local Moran’s I)</div>${sw(C.wet, "Cluster of rising trends")}${sw(C.dry, "Cluster of falling trends")}${sw(C.outlier, "Outlier vs neighbours")}${sw(C.faded, "No cluster")}`;
    }
    $("#legend").innerHTML = h;
  }

  // ---------- wording ----------
  function verdictShort(t) {
    if (t.sig > 0) return "Supported wetting trend";
    if (t.sig < 0) return "Supported drying trend";
    if (t.p != null && t.p < S.alpha) return "Suggestive, fails correction";
    return "No detectable trend";
  }
  function confidence(t) {
    const n = S.tiles.length.toLocaleString("en-US");
    if (t.sig !== 0) return `<strong>Statistically supported ${t.sig > 0 ? "wetting" : "drying"} trend.</strong> It survives the false-discovery correction for testing ${n} tiles at once (q = ${pval(t.q)}), so it is unlikely to be a fluke of testing so many places. It describes ${periodText()} and does not identify a cause.`;
    if (t.p != null && t.p < S.alpha) return `<strong>Suggestive, but not supported.</strong> On its own this trend looks significant (p = ${pval(t.p)}), but after correcting for ${n} simultaneous tests (q = ${pval(t.q)}) it could be chance. The detector does not count it.`;
    return `<strong>No detectable trend.</strong> Year-to-year swings here are larger than any steady change over ${periodText()} (p = ${pval(t.p)}).`;
  }
  const LISA = {
    HH: "In a cluster of rising trends", LL: "In a cluster of falling trends", HL: "Rising faster than its neighbours (outlier)",
    LH: "Falling against its neighbours (outlier)", ns: "No significant spatial cluster", nn: "No neighbouring land tiles",
  };
  function periodText() { const p = S.meta?.dataset?.period || []; return p.length ? `${yearOf(p[0])}–${yearOf(p[1])}` : "the study period"; }

  // ---------- panel: intro story ----------
  const panel = $("#panel");
  function aggregateSVG() {
    const a = S.meta.headline.aggregate, w = a.window || S.meta.dataset.period;
    const y0 = +yearOf(w[0]), y1 = +yearOf(w[1]) + 1, mid = (y0 + y1) / 2, mean = a.mean;
    const W = 360, H = 170, L = 34, R = 10, T = 12, B = 24;
    const ymax = Math.ceil(mean * 1.5 / 10) * 10;
    const X = (yr) => L + (yr - y0) / (y1 - y0) * (W - L - R), Y = (v) => T + (1 - v / ymax) * (H - T - B);
    const at = (s, yr) => mean + s * (yr - mid) / 10;
    const [lo, hi] = a.ci95 || [a.slope, a.slope];
    const band = `M${X(y0)},${Y(at(hi, y0))} L${X(y1)},${Y(at(hi, y1))} L${X(y1)},${Y(at(lo, y1))} L${X(y0)},${Y(at(lo, y0))} Z`;
    const band2 = `M${X(y0)},${Y(at(lo, y0))} L${X(y1)},${Y(at(lo, y1))} L${X(y1)},${Y(at(hi, y1))} L${X(y0)},${Y(at(hi, y0))} Z`;
    let grid = "";
    for (let v = 0; v <= ymax; v += ymax / 3) grid += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#24324d"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${Math.round(v)}</text>`;
    let ticks = "";
    for (let yr = Math.ceil(y0 / 5) * 5; yr <= y1; yr += 5) ticks += `<text x="${X(yr)}" y="${H - 6}" text-anchor="middle">${yr}</text>`;
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Global land-average precipitation trend line, ${fmt(a.slope, 2)} mm per month per decade${a.p_value >= S.alpha ? ", not statistically detectable" : ""}, with 95 percent confidence band.">
      <g font-family="IBM Plex Mono, monospace" font-size="10" fill="#8696ae">${grid}${ticks}</g>
      <path d="${band}" fill="rgba(169,182,202,.16)"/><path d="${band2}" fill="rgba(169,182,202,.16)"/>
      <line x1="${X(y0)}" y1="${Y(at(a.slope, y0))}" x2="${X(y1)}" y2="${Y(at(a.slope, y1))}" stroke="#e8eef6" stroke-width="2.5" stroke-linecap="round" class="agg-line"/>
      <text x="${W - R}" y="${Y(at(a.slope, y1)) - 10}" text-anchor="end" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="#e8eef6">${fmt(mean, 0)} mm/month, ${sgn(a.slope_pct_per_decade)}%/decade</text>
    </svg>`;
  }
  function showIntro(step) {
    S.highlight = null; S.highlightId = null; S.selected = -1;
    const m = S.meta, a = m.headline.aggregate, g = m.headline.regional;
    const n = (g.n_tiles_tested || S.tiles.length).toLocaleString("en-US");
    const notDetect = a.p_value >= S.alpha;
    const head = `<p class="eyebrow">Global land average · ${esc(m.dataset.variable_label?.split(",")[0] || "precipitation")} · ${periodText()}</p>
      <h1>${taglineHTML(m.tagline)}</h1>`;
    if (step === 1) {
      panel.innerHTML = `${head}
        <div class="agg">${aggregateSVG()}<p class="agg-caption">Fitted trend of the land-area-weighted average with its 95% confidence band (seasonal regression; autocorrelation-robust test).</p></div>
        <span class="verdict">${notDetect ? "No detectable global trend" : "Detectable global trend"}</span>
        <p>The land-average fitted trend is <span class="num">${fmt(a.slope, 2)}</span> mm/month per decade (<span class="num">${sgn(a.slope_pct_per_decade)}%</span>/decade; 95% CI <span class="num">${fmt(a.ci95[0], 2)} to ${fmt(a.ci95[1], 2)}</span> mm/month per decade, p = <span class="num">${pval(a.p_value)}</span>). The average says: nothing much is happening.</p>
        <div class="btn-row"><button class="btn" type="button" id="revealBtn">Reveal what the average hides</button></div>`;
      $("#revealBtn").addEventListener("click", () => runReveal());
    } else {
      panel.innerHTML = `${head}
        <p>Tile by tile, across ${n} one-degree land cells, the detector finds:</p>
        <div class="split">
          <div class="w"><span class="big">${pct(g.share_area_up, 1)}</span><small>of land area with a supported <b>wetting</b> trend</small></div>
          <div class="d"><span class="big">${pct(g.share_area_down, 1)}</span><small>of land area with a supported <b>drying</b> trend</small></div>
        </div>
        <p class="detective">Without correcting for testing ${n} tiles at once, <strong>${pct(g.share_tiles_raw_p_below_alpha, 0)}</strong> of tiles would have looked significant. The detector keeps only those that survive a false-discovery-rate test (q &lt; ${g.alpha_fdr ?? S.alpha}).</p>
        <p class="muted small">Most land shows no detectable trend; the few regions that do point in opposite directions. Click or tap any tile to see its monthly record for ${periodText()}, or open a case file.</p>
        <div class="btn-row"><a class="btn" href="#cases">Open the case files</a><button class="btn ghost" type="button" id="replayBtn">Replay</button></div>`;
      $("#replayBtn").addEventListener("click", () => { S.reveal = 0; render(); showIntro(1); $("#revealBtn")?.focus(); });
    }
    render();
  }
  function taglineHTML(t) {
    const parts = String(t || "").split(/(?<=\.)\s+/);
    return parts.length > 1 ? `${esc(parts[0])}<br><em>${esc(parts.slice(1).join(" "))}</em>` : esc(t);
  }
  function runReveal() {
    stopSpin();
    if (globe && S.view === "globe") globe.pointOfView({ lat: 15, lng: 20, altitude: 2.1 }, reduceMotion ? 0 : 1200);
    showIntro(2);
    if (reduceMotion) { S.reveal = 1; render(); return; }
    const t0 = performance.now(), dur = 1800;
    let last = 0;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / dur);
      if (now - last > 45 || k === 1) { last = now; S.reveal = k; render(); }
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // ---------- panel: tile evidence ----------
  async function inspect(i, fromMap) {
    if (i < 0) return;
    if (S.reveal < 1) S.reveal = 1;
    S.selected = i;
    const t = S.tiles[i];
    if (S.highlight && !S.highlight.has(t.id)) { S.highlight = null; S.highlightId = null; }
    render();
    const cls = t.sig > 0 ? "wet" : t.sig < 0 ? "dry" : "";
    const reg = t.region ? S.regionById.get(t.region) : null;
    const card = t.region ? S.cardByRegion.get(t.region) : null;
    const regLine = reg ? `<p class="small">Part of ${t.region.startsWith("up") ? "wetting" : "drying"} region <span class="pill ${t.region.startsWith("up") ? "wet" : "dry"}">${esc(t.region)}</span> (${reg.n_tiles} tiles). ${card ? `<button class="linkish" type="button" data-case="${esc(card.id)}">Open its case file</button>` : ""}</p>` : "";
    panel.innerHTML = `<button class="back" type="button" id="backBtn">← Back to the overview</button>
      <p class="eyebrow">Tile ${latStr(t.s)}–${latStr(t.s + S.deg)}, ${lonStr(t.w)}–${lonStr(t.w + S.deg)}</p>
      <span class="verdict ${cls}">${esc(verdictShort(t))}</span>
      <div class="series" id="seriesBox"><p class="muted small">Loading monthly record…</p></div>
      <dl class="stats">
        <div><dt>Trend</dt><dd>${sgn(t.slope, 2)} <span class="small muted">mm/month per decade</span></dd></div>
        <div><dt>Relative trend</dt><dd>${sgn(t.pct)}% <span class="small muted">per decade</span></dd></div>
        <div><dt>p (single test)</dt><dd>${pval(t.p)}</dd></div>
        <div><dt>q (FDR-corrected)</dt><dd>${pval(t.q)}</dd></div>
      </dl>
      <p class="small">${confidence(t)}</p>
      <p class="small muted">Spatial pattern: ${esc(LISA[t.lisa] || t.lisa)}.</p>
      ${regLine}`;
    $("#backBtn").addEventListener("click", () => { showIntro(2); });
    panel.querySelector("[data-case]")?.addEventListener("click", (e) => openCase(e.target.dataset.case));
    if (fromMap && matchMedia("(max-width: 820px)").matches) panel.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    const ser = await loadSeries(t);
    if (S.selected !== i) return;
    $("#seriesBox").innerHTML = ser ? seriesSVG(ser, t) : `<p class="muted small">The monthly record for this tile could not be loaded.</p>`;
  }

  function seriesSVG(ser, t) {
    const v = ser.values, n = v.length;
    const [sy, sm] = ser.start.split("-").map(Number);
    const W = 380, H = 200, L = 36, R = 8, T = 10, B = 22;
    const ok = v.filter((x) => x != null && isFinite(x));
    const mean = ok.reduce((a, b) => a + b, 0) / (ok.length || 1);
    const sorted = [...ok].sort((a, b) => a - b);
    let lo = sorted[Math.floor(sorted.length * 0.01)] ?? -1, hi = sorted[Math.ceil(sorted.length * 0.99) - 1] ?? 1;
    const ext = Math.max(Math.abs(lo), Math.abs(hi), 1) * 1.08; lo = -ext; hi = ext;
    const X = (k) => L + k / (n - 1) * (W - L - R), Y = (y) => T + (1 - (clamp(y, lo, hi) - lo) / (hi - lo)) * (H - T - B);
    let raw = "", run = "";
    v.forEach((y, k) => { if (y == null) return; raw += `${raw ? "L" : "M"}${X(k).toFixed(1)},${Y(y).toFixed(1)}`; });
    for (let k = 11; k < n; k++) {
      const win = v.slice(k - 11, k + 1).filter((x) => x != null);
      if (win.length < 9) continue;
      const m = win.reduce((a, b) => a + b, 0) / win.length;
      run += `${run ? "L" : "M"}${X(k - 5.5).toFixed(1)},${Y(m).toFixed(1)}`;
    }
    const midK = (n - 1) / 2, perMonth = (t.slope || 0) / 120;
    const trendC = t.sig > 0 ? "#4ba3e3" : t.sig < 0 ? "#e39a3b" : "#a9b6ca";
    let ticks = "";
    for (let yr = Math.ceil(sy / 5) * 5; yr <= sy + n / 12; yr += 5) {
      const k = (yr - sy) * 12 - (sm - 1); if (k < 0 || k > n - 1) continue;
      ticks += `<text x="${X(k)}" y="${H - 6}" text-anchor="middle">${yr}</text>`;
    }
    const yt = [lo * 0.75, 0, hi * 0.75].map((y) => `<line x1="${L}" x2="${W - R}" y1="${Y(y)}" y2="${Y(y)}" stroke="#24324d"/><text x="${L - 5}" y="${Y(y) + 3}" text-anchor="end">${Math.round(y)}</text>`).join("");
    return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Monthly precipitation anomalies for this tile with 12-month running mean and fitted trend of ${sgn(t.slope, 2)} millimetres per month per decade.">
      <g font-family="IBM Plex Mono, monospace" font-size="9.5" fill="#8696ae">${yt}${ticks}</g>
      <path d="${raw}" fill="none" stroke="#5c6b86" stroke-width="0.8"/>
      <path d="${run}" fill="none" stroke="#e8eef6" stroke-width="1.8" stroke-linejoin="round"/>
      <line x1="${X(0)}" y1="${Y(mean + perMonth * (0 - midK))}" x2="${X(n - 1)}" y2="${Y(mean + perMonth * (n - 1 - midK))}" stroke="${trendC}" stroke-width="2.4" stroke-dasharray="7 4" stroke-linecap="round"/>
    </svg>
    <div class="series-legend"><span><i style="border-color:#5c6b86"></i>Monthly anomaly (mm/month)</span><span><i style="border-color:#e8eef6"></i>12-month running mean</span><span><i style="border-color:${trendC};border-top-style:dashed"></i>Trend</span></div>`;
  }

  // ---------- cases ----------
  const STATUS_ICON = {
    pass: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M1.5 5.2 4 7.5 8.5 2.5" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>',
    partial: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M1.5 5h7" stroke="currentColor" stroke-width="1.8"/><circle cx="5" cy="5" r="1" fill="currentColor"/></svg>',
    fail: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M2 2l6 6M8 2 2 8" stroke="currentColor" stroke-width="1.8"/></svg>',
    not_run: '<svg viewBox="0 0 10 10" aria-hidden="true"><circle cx="5" cy="5" r="3" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>',
  };
  const STATUS_LABEL = { pass: "Pass", partial: "Partial", fail: "Fail", not_run: "Not run" };
  const KIND_LABEL = { wetting: "Wetting region", drying: "Drying region", retraction: "Caught a false trend", local: "Local region" };
  const normStatus = (s) => (STATUS_LABEL[s] ? s : "not_run");

  function gradeHTML(checks) {
    // the selection criterion (FDR significance) passes by construction, so it is not part of the score
    const graded = checks.filter((c) => c.role !== "selection");
    const selected = graded.length < checks.length;
    const pts = graded.reduce((a, c) => a + (c.status === "pass" ? 1 : c.status === "partial" ? 0.5 : 0), 0);
    const bars = graded.map((c) => `<i class="${normStatus(c.status)}"></i>`).join("");
    return `<div class="grade" aria-label="Evidence grade ${pts} of ${graded.length} checks passed"><span class="grade-bars" aria-hidden="true">${bars}</span>${pts.toString().replace(".5", "½")} / ${graded.length} checks pass${selected ? " · selected by FDR" : ""}</div>`;
  }
  function checksHTML(checks) {
    return `<ul class="checks">${checks.map((c) => { const s = normStatus(c.status); return `<li><span class="status ${s}">${STATUS_ICON[s]}${STATUS_LABEL[s]}</span><span>${esc(c.check)}</span>${c.detail ? `<span class="detail">${esc(c.detail)}</span>` : ""}</li>`; }).join("")}</ul>`;
  }
  function litHTML(lit) {
    if (!lit || !lit.length) return "";
    const ag = (a) => (a === true ? '<span class="agree yes">agrees</span>' : a === "partly" ? '<span class="agree partly">partly</span>' : a === false ? '<span class="agree no">disagrees</span>' : "");
    const link = (u) => { if (!u) return ""; const href = /^10\./.test(u) ? `https://doi.org/${u}` : u; return /^https?:\/\//.test(href) ? ` <a href="${esc(href)}" target="_blank" rel="noopener">source</a>` : ` <span class="mono">${esc(u)}</span>`; };
    return `<details><summary>Independent literature (${lit.length})</summary><ul>${lit.map((l) => `<li>${esc(l.citation)}${ag(l.agrees)}${link(l.url_or_doi)}${l.note ? `<br><span class="small">${esc(l.note)}</span>` : ""}</li>`).join("")}</ul></details>`;
  }
  function placeholderHTML(c) {
    const clean = (x) => String(x || "").replace(/^TODO:\s*/i, "");
    const t = clean(c.title) || "Your local region";
    const checks = (c.robustness || []).map((x) => ({ check: x.check, status: "not_run" }));
    return `<article class="card local" id="case-${esc(c.id)}" aria-labelledby="t-${esc(c.id)}">
      <div class="card-kind"><span class="dot"></span>${esc(KIND_LABEL.local)} <span class="pill">Placeholder</span></div>
      <h3 class="card-title" id="t-${esc(c.id)}">${esc(t.charAt(0).toUpperCase() + t.slice(1))}</h3>
      <p class="headline">This slot is reserved for the team’s own region. Once a bounding box is chosen, the detector fills in the numbers, runs the same four checks, and names who would act on the result.</p>
      ${checks.length ? gradeHTML(checks) + checksHTML(checks) : ""}
      <div class="card-foot"><span class="evidence">${esc(c.evidence_level || "not yet evaluated")}</span></div>
    </article>`;
  }
  function cardHTML(c, featured) {
    if (c.kind === "local" && !c.region_id && /^TODO/i.test(String(c.title || ""))) return placeholderHTML(c);
    const kind = KIND_LABEL[c.kind] ? c.kind : "wetting";
    const n = c.numbers || {};
    const nums = [];
    if (n.median_pct_per_decade != null) nums.push(`<span><b>${sgn(n.median_pct_per_decade)}%</b>median per decade</span>`);
    else if (n.median_slope_per_decade != null) nums.push(`<span><b>${sgn(n.median_slope_per_decade)}</b>mm/month per decade</span>`);
    if (n.area_km2 != null) nums.push(`<span><b>${area(n.area_km2)}</b>area</span>`);
    if (n.n_tiles != null) nums.push(`<span><b>${n.n_tiles}</b>tiles</span>`);
    if (n.share_tiles_fdr_significant != null) nums.push(`<span><b>${pct(n.share_tiles_fdr_significant, 0)}</b>FDR-significant</span>`);
    const checks = c.robustness || [];
    const u = c.user || {};
    const showBtn = c.region_id && S.regionById.has(c.region_id);
    const colA = `<div class="card-kind"><span class="dot"></span>${esc(KIND_LABEL[kind])}${kind === "local" && !c.region_id ? ` <span class="pill">Placeholder</span>` : ""}</div>
      ${kind === "retraction" ? `<span class="stamp">Rejected by its own detector</span>` : ""}
      <h3 class="card-title" id="t-${esc(c.id)}">${esc(c.title)}</h3>
      ${c.place ? `<p class="place">${esc(c.place)}</p>` : ""}
      <p class="headline">${esc(c.headline)}</p>
      ${nums.length ? `<div class="nums">${nums.join("")}</div>` : ""}
      ${checks.length ? gradeHTML(checks) : ""}`;
    const colB = `${checks.length ? checksHTML(checks) : ""}
      ${u.who ? `<div class="user"><div class="user-label">Who acts on this</div><div class="who">${esc(u.who)}</div>${u.decision ? `<p class="decision">${esc(u.decision)}</p>` : ""}${u.why_it_matters ? `<p class="why">${esc(u.why_it_matters)}</p>` : ""}</div>` : ""}
      ${litHTML(c.literature)}
      ${c.caveats && c.caveats.length ? `<details><summary>Caveats (${c.caveats.length})</summary><ul>${c.caveats.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></details>` : ""}
      <div class="card-foot">${c.evidence_level ? `<span class="evidence">${esc(c.evidence_level)}</span>` : "<span></span>"}${showBtn ? `<button class="btn ghost" type="button" data-case="${esc(c.id)}">Show on the map</button>` : ""}</div>`;
    return `<article class="card ${kind}${featured ? " is-featured" : ""}" id="case-${esc(c.id)}" aria-labelledby="t-${esc(c.id)}">
      <div class="col">${colA}</div><div class="col">${colB}</div>
    </article>`;
  }

  function fallbackCards() {
    const cards = [];
    if (S.meta.retraction) cards.push({
      id: "retraction", kind: "retraction", region_id: null, title: "The detector caught a false trend",
      place: "Global soil water", headline: S.meta.retraction,
      robustness: [{ check: "Break detection on the input record", status: "fail", detail: "The apparent decline is a step change, not a gradual trend." }],
      user: { who: "Anyone reusing GLDAS-2.1 soil moisture for trend work", decision: "Whether to report a multi-decade soil-water decline", why_it_matters: "A change in the model’s inputs can masquerade as a climate signal." },
      evidence_level: "retracted", literature: [], caveats: [],
    });
    const mk = (r, kind) => {
      const ids = new Set(r.tiles); let sig = 0;
      S.tiles.forEach((t) => { if (ids.has(t.id) && t.sig !== 0) sig++; });
      return {
        id: r.id, kind, region_id: r.id, title: `${kind === "wetting" ? "Wetting" : "Drying"} region ${r.id}`,
        place: `Centred near ${latStr(Math.round(r.centroid_lat))}, ${lonStr(Math.round(r.centroid_lon))}`,
        headline: `${r.n_tiles} connected tiles share a ${kind} trend with a median of ${sgn(r.median_slope_per_decade)} mm/month per decade.`,
        numbers: { median_pct_per_decade: r.median_slope_pct_per_decade, median_slope_per_decade: r.median_slope_per_decade, area_km2: r.area_km2, n_tiles: r.n_tiles, share_tiles_fdr_significant: r.n_tiles ? sig / r.n_tiles : null },
        robustness: [
          { check: "FDR-corrected significance", status: sig === r.n_tiles ? "pass" : sig ? "partial" : "fail", detail: `${sig} of ${r.n_tiles} tiles have q < ${S.alpha}.` },
          { check: "Stable across time windows", status: "not_run" },
          { check: "Neighbour definitions agree (queen vs kNN)", status: "not_run" },
          { check: "Independent literature", status: "not_run" },
        ],
        user: { who: "Teachers and students using the explorer", decision: "Which region to study as an example of a trend the global average hides", why_it_matters: "Shows how local change and a flat global mean can both be true." },
        evidence_level: "statistically supported", literature: [], caveats: [],
      };
    };
    S.regions.up.slice(0, 3).forEach((r) => cards.push(mk(r, "wetting")));
    S.regions.down.slice(0, 3).forEach((r) => cards.push(mk(r, "drying")));
    return cards;
  }

  function renderCases() {
    const featured = S.cards.filter((c) => c.kind === "retraction");
    const rest = S.cards.filter((c) => c.kind !== "retraction");
    const order = { wetting: 1, drying: 1, local: 3 };
    rest.sort((a, b) => (order[a.kind] || 2) - (order[b.kind] || 2));
    $("#featured").innerHTML = featured.map((c) => cardHTML(c, true)).join("");
    $("#cardGrid").innerHTML = rest.map((c) => cardHTML(c, false)).join("");
    const note = $("#caseNotice");
    if (!S.cardsFromFile) {
      note.hidden = false;
      note.textContent = "The graded case files are not published yet. Below is the detector’s raw output: the retraction and the largest regions, with checks that have not run marked as such.";
    }
    document.querySelectorAll("#cases [data-case]").forEach((b) => b.addEventListener("click", () => openCase(b.dataset.case)));
  }

  function openCase(id, fromLoad) {
    const c = S.cards.find((x) => x.id === id);
    if (!c) return;
    const r = c.region_id ? S.regionById.get(c.region_id) : null;
    S.reveal = 1; S.selected = -1;
    if (r) {
      S.highlight = new Set(r.tiles); S.highlightId = r.id;
      flyTo(r);
    }
    render();
    const checks = c.robustness || [], u = c.user || {};
    panel.innerHTML = `<button class="back" type="button" id="backBtn">← Back to the overview</button>
      <p class="eyebrow">Case file · ${esc(KIND_LABEL[c.kind] || c.kind)}</p>
      <h1>${esc(c.title)}</h1>
      ${c.place ? `<p class="muted">${esc(c.place)}</p>` : ""}
      <p>${esc(c.headline)}</p>
      ${checks.length ? gradeHTML(checks) + checksHTML(checks) : ""}
      ${u.who ? `<div class="user" style="margin-top:1rem"><div class="user-label">Who acts on this</div><div class="who">${esc(u.who)}</div>${u.decision ? `<p class="decision">${esc(u.decision)}</p>` : ""}</div>` : ""}
      <p class="small muted" style="margin-top:1rem">${r ? "Highlighted tiles are the region’s members. Click any tile for its monthly record." : ""}</p>
      <div class="btn-row"><a class="btn ghost" href="#case-${esc(c.id)}">Read the full case file</a></div>`;
    $("#backBtn").addEventListener("click", () => showIntro(2));
    history.replaceState(null, "", `#case=${encodeURIComponent(id)}`);
    $("#explorer").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    panel.scrollTop = 0;
    if (!fromLoad) setTimeout(() => $("#backBtn")?.focus({ preventScroll: true }), reduceMotion ? 0 : 500);
  }
  function flyTo(r) {
    const [w, s, e, n] = r.bbox || [r.centroid_lon - 5, r.centroid_lat - 5, r.centroid_lon + 5, r.centroid_lat + 5];
    const span = Math.max(e - w, (n - s) * 1.4, 8);
    const fspan = Math.max(span, 18); // flat map: keep surrounding context visible
    if (globe && S.view === "globe") {
      stopSpin();
      globe.pointOfView({ lat: r.centroid_lat, lng: r.centroid_lon, altitude: clamp(span / 18, 0.9, 2.4) }, reduceMotion ? 0 : 1400);
    }
    F.cx = (w + e) / 2; F.cy = (s + n) / 2;
    if (flat.clientWidth && flat.clientWidth < 600) {
      // phones: fit the region itself so it is not pushed under the legend by the latitude clamp
      F.k = clamp(Math.min(flat.clientWidth / (Math.max(e - w, 6) * 1.8), flat.clientHeight / ((n - s + 4) * 1.6)), F.kMin || 1, 30);
      F.cy -= 20 / F.k; // nudge the region up, clear of the legend
    } else if (flat.clientWidth) F.k = clamp(Math.min(flat.clientWidth / (fspan * 1.6), flat.clientHeight / (Math.max(n - s + 6, fspan * 0.6) * 1.6)), F.kMin || 1, 30);
  }

  // ---------- method & credits ----------
  function renderMethod() {
    const m = S.meta, md = m.methods || {};
    const labels = { trend: "Trend test", multiple_testing: "Multiple testing", spatial: "Spatial clustering", regions: "Region rule" };
    const pretty = (v) => String(v)
      .replace(/\btrend_slope\b/g, "trend slope");
    $("#methodList").innerHTML = Object.entries(md).map(([k, v]) => `<div><dt>${esc(labels[k] || k)}</dt><dd>${esc(pretty(v))}</dd></div>`).join("")
      + `<div><dt>Grid</dt><dd>${S.deg}° land tiles, ${S.tiles.length.toLocaleString("en-US")} tested; areas weighted by ${esc(m.headline?.weights || "cos(latitude)")}</dd></div>`;
    const a = m.headline.aggregate, g = m.headline.regional;
    const rows = [[`${yearOf(a.window?.[0])}–${yearOf(a.window?.[1])} (main)`, a.slope, a.p_value, g.share_area_up, g.share_area_down]];
    for (const [name, w] of Object.entries(m.headline.windows || {})) rows.push([name, w.aggregate_slope, w.aggregate_p, w.share_area_up, w.share_area_down]);
    $("#windowTable").innerHTML += `<thead><tr><th scope="col">Window</th><th scope="col">Global slope<br>mm/mo/dec</th><th scope="col">p</th><th scope="col">Area wetting</th><th scope="col">Area drying</th></tr></thead><tbody>`
      + rows.map((r) => `<tr><td>${esc(r[0])}</td><td>${sgn(r[1], 2)}</td><td>${pval(r[2])}</td><td>${pct(r[3])}</td><td>${pct(r[4])}</td></tr>`).join("") + "</tbody>";
    $("#caveatList").innerHTML = (m.caveats || []).map((c) => `<li>${esc(c)}</li>`).join("");
    const d = m.dataset;
    $("#dataCredit").innerHTML = `${esc(d.name)} (${esc(d.short_name)} v${esc(d.version)}), variable <span class="mono">${esc(d.variable)}</span>, ${esc(d.period?.[0])} to ${esc(d.period?.[1])}. Distributed by ${esc(d.provider)}: <a href="https://doi.org/${esc(d.doi)}" target="_blank" rel="noopener">doi:${esc(d.doi)}</a>. Without this freely available, consistently processed global record, a tile-by-tile search of every land area would not be possible.`;
  }
  async function loadTeam() {
    try {
      const t = await getJSON("../demo/team.json");
      const members = (t.members || []).map(esc).join(", ");
      $("#teamLine").innerHTML = `<strong>${esc(t.team_name)}</strong>${members ? ` · ${members}` : ""}<br>${esc(t.event || "")}${t.local_event ? `, ${esc(t.local_event)}` : ""}<br>Challenge: ${esc(t.challenge || "")}${t.project_url ? `<br><a href="${esc(t.project_url)}" target="_blank" rel="noopener">Source code</a>` : ""}`;
    } catch {
      $("#teamLine").textContent = `Built for ${S.meta.challenge}.`;
    }
  }

  // ---------- boot ----------
  async function boot() {
    try {
      const [meta, tiles, regions] = await Promise.all([getJSON(DATA + "meta.json"), getJSON(DATA + "tiles.json"), getJSON(DATA + "regions.json")]);
      S.meta = meta; S.alpha = meta.headline?.regional?.alpha_fdr ?? 0.05;
      ingestTiles(tiles);
      S.regions = { up: regions.up || [], down: regions.down || [] };
      [...S.regions.up, ...S.regions.down].forEach((r) => S.regionById.set(r.id, r));
    } catch (e) {
      console.error(e);
      $("#loading").textContent = "The trend data could not be loaded. Serve the site folder over HTTP (for example with a static file server) and reload.";
      return;
    }
    try {
      const cj = await getJSON(params.get("cards") || DATA + "cards.json");
      if (Array.isArray(cj.cards) && cj.cards.length) { S.cards = cj.cards; S.cardsFromFile = true; }
    } catch { /* cards.json not published yet: fall back below */ }
    if (!S.cardsFromFile) S.cards = fallbackCards();
    S.cards.forEach((c) => { if (c.region_id) S.cardByRegion.set(c.region_id, c); });

    sizeTexture();
    canGlobe = webglOK() && typeof window.Globe === "function";
    const narrow = matchMedia("(max-width: 820px)").matches;
    const want = params.get("view") || (narrow || !canGlobe ? "flat" : "globe");
    setView(want === "flat" ? "flat" : "globe");
    renderLegend(); renderCases(); renderMethod(); loadTeam();

    const hash = decodeURIComponent(location.hash || "");
    if (params.has("reveal") || hash.startsWith("#case=")) { S.reveal = 1; showIntro(2); } else showIntro(1);
    if (hash.startsWith("#case=")) openCase(hash.slice(6), true);
    if (params.has("tile")) { const t = S.tiles.findIndex((x) => x.id === params.get("tile")); if (t >= 0) inspect(t, false); }
    // remove the overlay at once: a fade leaves a half-dimmed map in screenshots taken right after load
    const ld = $("#loading"); ld.classList.add("done"); ld.hidden = true;
    window.__paradox = { state: S, get globe() { return globe; }, openCase, inspect };
    window.__paradoxReady = true;
  }
  boot();
})();
