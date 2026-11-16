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
    ocean: [17, 18, 20], grat: [40, 42, 46], neutral: [108, 110, 115], faded: [62, 64, 69],
    // outlier is a pale lilac so it cannot be mistaken for the blue wetting clusters under red-green colour blindness
    mid: [66, 68, 73], wet: [90, 162, 216], dry: [222, 155, 74], outlier: [230, 220, 245], none: [38, 40, 44],
  };
  const rgb = (a) => `rgb(${a[0] | 0},${a[1] | 0},${a[2] | 0})`;
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  // ---------- formatting ----------
  const fmt = (x, d = 1) => (x == null || !isFinite(x) ? "–" : Number(x).toFixed(d).replace("-", "−"));
  const sgn = (x, d = 1) => (x == null || !isFinite(x) ? "–" : +Number(x).toFixed(d) === 0 ? fmt(0, d) : (x > 0 ? "+" : "") + fmt(x, d));
  const pct = (x, d = 1) => (x == null ? "–" : fmt(x * 100, d) + "%");
  // three decimals below 0.01 and next to the 0.05 threshold, so a q of 0.047 never reads as "0.05";
  // a value just under 0.05 gets as many more digits as it needs not to round up to "0.050"
  const pval = (p) => {
    if (p == null) return "–";
    if (p < 0.001) return "< 0.001";
    let d = p < 0.01 || (p >= 0.045 && p < 0.055) ? 3 : 2;
    while (d < 6 && p < 0.05 && +Number(p).toFixed(d) >= 0.05) d++;
    return fmt(p, d);
  };
  const area = (km2) => {
    if (km2 == null || !isFinite(km2)) return "–";
    if (km2 >= 1e6) return fmt(km2 / 1e6, 2) + " million km²";
    if (km2 >= 1e4) return (Math.round(km2 / 1000) * 1000).toLocaleString("en-US") + " km²";
    return Math.round(km2).toLocaleString("en-US") + " km²";
  };
  // display-only typography for text that comes from the data files: year ranges get an en dash,
  // negative numbers a true minus, "->" an arrow. No digit changes.
  const typo = (s) => String(s ?? "")
    .replace(/\b((?:19|20)\d{2})-((?:19|20)\d{2})\b/g, "$1–$2")
    .replace(/ -> /g, " → ")
    .replace(/ - /g, " – ")
    .replace(/(^|[\s(\[/:,;=])-(?=\d)/g, "$1−")
    .replace(/\b(\d{1,3})-(\d{1,3})(?=\s?(?:°|cm\b|mm\b|km\b|%))/g, "$1–$2")
    .replace(/(\w)'(\w)/g, "$1’$2")
    .replace(/(^|[\s(])'(?=\w)/g, "$1‘")
    .replace(/(\w[.,;:!?]?)'(?=[\s.,;:)]|$)/g, "$1’")
    // keep "q < 0.05", "p = 0.03" and "λ = 10" on one line
    .replace(/(^|[\s(])([pqλ]) ([<=≤]) (?=[−\d.])/g, "$1$2\u00a0$3\u00a0");
  const et = (s) => esc(typo(s));
  const nw = (s) => `<span class="nw">${s}</span>`;
  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  const coarse = matchMedia("(pointer: coarse)").matches;
  // a nice round axis step so gridlines read 0, 25, 50… rather than 0, 37, 73
  function niceStep(max, n = 4) {
    const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * mag >= raw) return m * mag;
    return 10 * mag;
  }
  // the largest round value (1, 2, 2.5 or 5 × 10^k) that does not exceed x
  function niceBelow(x) {
    const mag = Math.pow(10, Math.floor(Math.log10(x || 1)));
    for (const m of [5, 2.5, 2, 1]) if (m * mag <= x) return m * mag;
    return mag;
  }
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
    if (S.layer === "sig") {
      const possible = t.sig === 0 && t.p != null && t.p < S.alpha && t.slope != null;
      target = t.sig > 0 ? C.wet : t.sig < 0 ? C.dry : possible ? mix(C.faded, t.slope > 0 ? C.wet : C.dry, 0.38) : C.faded;
    }
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
    // dim the rest of the land, but keep it lighter than the ocean so coastlines stay readable
    if (S.highlight && !S.highlight.has(t.id)) target = mix(target, C.ocean, 0.4);
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
  // globe.gl (~500 KB) is loaded only when the 3D view is actually used, so phones on the flat map never fetch it
  const GLOBE_SRC = "https://cdn.jsdelivr.net/npm/globe.gl@2.46.2/dist/globe.gl.min.js";
  let globeLib = null;
  function loadGlobeLib(timeoutMs = 15000) {
    if (typeof window.Globe === "function") return Promise.resolve(true);
    if (!globeLib) globeLib = new Promise((res) => {
      const s = document.createElement("script");
      s.src = GLOBE_SRC; s.async = true;
      const t = setTimeout(() => { globeLib = null; res(false); }, timeoutMs);
      s.onload = () => { clearTimeout(t); res(typeof window.Globe === "function"); };
      s.onerror = () => { clearTimeout(t); globeLib = null; res(false); };
      document.head.append(s);
    });
    return globeLib;
  }
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
      .showAtmosphere(false)
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
    // a plain wheel scrolls the page; Ctrl/⌘ + wheel (and trackpad pinch, which sets ctrlKey) zooms the globe.
    // This capture listener on the container runs before OrbitControls' listener on the canvas inside it.
    el.addEventListener("wheel", (e) => { if (!(e.ctrlKey || e.metaKey)) e.stopPropagation(); else stopSpin(); }, { capture: true, passive: true });
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
  // stop the auto-rotation at the first key press anywhere, and stop rendering the globe while it is scrolled out of view
  addEventListener("keydown", () => { if (globe && globe.controls().autoRotate) stopSpin(); });
  new IntersectionObserver(([e]) => {
    if (globe && S.view === "globe") e.isIntersecting ? globe.resumeAnimation() : globe.pauseAnimation();
  }).observe($("#stage"));

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
    const padS = (H < 600 ? 180 : 110) / F.k; // allow panning a little past 60°S so southern regions can sit above the legend
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
    // touch: at the fitted zoom a vertical swipe scrolls the page (the whole latitude range is already in view);
    // once zoomed in, dragging in any direction pans the map
    const ta = F.k > H / (F.maxLat - F.minLat) * 1.05 ? "none" : "pan-y pinch-zoom";
    if (flat.style.touchAction !== ta) flat.style.touchAction = ta;
  }
  const flatToGeo = (x, y) => ({ lng: F.cx + (x - flat.clientWidth / 2) / F.k, lat: F.cy - (y - flat.clientHeight / 2) / F.k });
  function nearestTile(lat, lng, rDeg) {
    let best = -1, bd = Infinity;
    const n = Math.ceil(rDeg / S.deg);
    for (let a = -n; a <= n; a++) for (let b = -n; b <= n; b++) {
      const i = tileAt(lat + a * S.deg, lng + b * S.deg); if (i < 0) continue;
      const t = S.tiles[i], dLat = t.s + S.deg / 2 - lat, dLng = t.w + S.deg / 2 - lng, d = dLat * dLat + dLng * dLng;
      if (d < bd && d <= (rDeg + S.deg) * (rDeg + S.deg)) { bd = d; best = i; }
    }
    return best;
  }
  function noTileTip(x, y) {
    tip.textContent = "No land tile here";
    tip.hidden = false;
    const r = $("#stage").getBoundingClientRect();
    tip.style.left = clamp(x + 10, 6, r.width - tip.offsetWidth - 6) + "px";
    tip.style.top = clamp(y + 10, 6, r.height - tip.offsetHeight - 6) + "px";
    announce("No land tile here.");
    clearTimeout(noTileTip.t); noTileTip.t = setTimeout(hideTip, 1600);
  }
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
      if (drag && !drag.moved) {
        const r = flat.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, g = flatToGeo(x, y);
        let i = tileAt(g.lat, g.lng);
        // a fingertip is wider than a 1° tile on a phone: take the nearest land tile within about 12 px
        if (i < 0 && e.pointerType !== "mouse") i = nearestTile(g.lat, g.lng, 12 / F.k);
        if (i < 0) noTileTip(x, y); else inspect(i, true);
      }
      drag = null; flat.classList.remove("dragging");
    });
    flat.addEventListener("pointercancel", () => { drag = null; flat.classList.remove("dragging"); });
    flat.addEventListener("pointerleave", hideTip);
    flat.addEventListener("wheel", (e) => {
      // plain wheel scrolls the page; Ctrl/⌘ + wheel or a trackpad pinch (ctrlKey) zooms the map
      if (!(e.ctrlKey || e.metaKey)) return;
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
    btn.innerHTML = v === "globe" ? `Flat<span class="lbl-long"> map</span>` : `3D<span class="lbl-long"> globe</span>`;
    btn.setAttribute("aria-label", v === "globe" ? "Switch to flat map" : "Switch to 3D globe");
    btn.hidden = !globe && !canGlobe;
    if (globe) { v === "globe" ? globe.resumeAnimation() : globe.pauseAnimation(); }
    if (v === "flat") { if (!F.kMin) fitFlat(); }
    const zoomHow = coarse ? "+ − to zoom" : `${isMac ? "⌘" : "Ctrl"} + scroll or + − to zoom`;
    $("#stageHint").textContent = v === "globe"
      ? `Drag to rotate · ${zoomHow} · ${coarse ? "tap" : "click"} a tile for its evidence`
      : `Drag to pan · ${zoomHow} · ${coarse ? "tap" : "click"} a tile for its evidence`;
    resize(); render();
  }
  async function toggleView() {
    const btn = $("#viewToggle");
    if (toggleView.busy) return;
    if (S.view === "flat" && typeof window.Globe !== "function") {
      toggleView.busy = true; btn.setAttribute("aria-busy", "true"); btn.classList.add("busy");
      const ok = await loadGlobeLib();
      toggleView.busy = false; btn.removeAttribute("aria-busy"); btn.classList.remove("busy");
      if (!ok) { announce("The 3D globe could not be loaded. The flat map stays on."); return; }
    }
    setView(S.view === "globe" ? "flat" : "globe");
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
  $("#stageFocus").addEventListener("focus", stopSpin);
  // one short polite announcement per panel change, instead of reading the whole panel aloud
  function announce(msg) {
    const el = $("#panelStatus"); if (!el) return;
    el.textContent = "";
    setTimeout(() => { el.textContent = msg; }, 60);
  }
  function announceNoTile() { announce("No land tile under the crosshair."); }
  $("#viewToggle").addEventListener("click", toggleView);
  $("#stageCta").addEventListener("click", () => runReveal());
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
      tip.innerHTML = `<b>${latStr(t.s)} ${lonStr(t.w)}</b><br>${esc(verdictShort(t))}<br><span class="mono">${sgn(t.pct)}% / decade · q&nbsp;=&nbsp;${pval(t.q)}</span>`;
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
    const sw = (c, label) => `<div class="row"><span class="sw" style="background:${rgb(c)}"></span><span>${label}</span></div>`;
    const lt = (a) => `&lt;&nbsp;${a}`;
    let title = "", body = "";
    if (S.layer === "sig") {
      title = `Trend evidence, ${periodText()}`;
      body = `${sw(C.wet, `Detectable wetting (q&nbsp;${lt(S.alpha)})`)}${sw(C.dry, `Detectable drying (q&nbsp;${lt(S.alpha)})`)}${sw(mix(C.faded, C.wet, 0.38), `Possible wetting (p&nbsp;${lt(S.alpha)} alone)`)}${sw(mix(C.faded, C.dry, 0.38), `Possible drying (p&nbsp;${lt(S.alpha)} alone)`)}${sw(C.faded, "No trend detected")}<p class="legend-note q-note">q is p after correcting for testing many tiles</p>`;
    } else if (S.layer === "pct") {
      title = "Trend, % of mean per decade";
      body = `<div class="ramp" style="background:linear-gradient(90deg,${rgb(C.dry)},${rgb(C.mid)},${rgb(C.wet)})"></div><div class="ramp-lab"><span>≤−${S.pctMax}</span><span>0</span><span>≥+${S.pctMax}</span></div><p class="legend-note">All tiles, whether or not the trend is detectable</p>`;
    } else {
      title = "Spatial clusters (local Moran’s I)";
      body = `${sw(C.wet, "Cluster of rising trends")}${sw(C.dry, "Cluster of falling trends")}${sw(C.outlier, "Outlier vs neighbours")}${sw(C.faded, "No cluster")}`;
    }
    if (S.preReveal) body = `<p class="legend-note legend-pre">Colours appear after you reveal the trends</p>` + body;
    const lg = $("#legend");
    lg.classList.toggle("pre", !!S.preReveal);
    lg.classList.toggle("collapsed", !!S.legendCollapsed);
    lg.innerHTML = `<button class="legend-title" type="button" aria-expanded="${!S.legendCollapsed}" aria-controls="legendBody">${title}<span class="chev" aria-hidden="true"></span></button><div class="legend-body" id="legendBody">${body}</div>`;
  }
  $("#legend").addEventListener("click", (e) => {
    if (!e.target.closest(".legend-title")) return;
    S.legendCollapsed = !S.legendCollapsed;
    renderLegend();
    $("#legend .legend-title").focus();
  });
  // before the reveal the map is all grey: on narrow screens a reveal button takes the legend's place on the map
  function setPreReveal(on) {
    if (S.preReveal === on) return;
    S.preReveal = on;
    $("#stage").classList.toggle("pre", on);
    const cta = $("#stageCta"); if (cta) cta.hidden = !on;
    renderLegend();
  }

  // ---------- wording ----------
  function verdictShort(t) {
    if (t.sig > 0) return "Detectable wetting trend";
    if (t.sig < 0) return "Detectable drying trend";
    if (t.p != null && t.p < S.alpha) return `Possible ${t.slope > 0 ? "wetting" : "drying"}, not confirmed`;
    return "No detectable trend";
  }
  function confidence(t) {
    const n = S.tiles.length.toLocaleString("en-US");
    if (t.sig !== 0) return `<strong>Detectable ${t.sig > 0 ? "wetting" : "drying"} trend.</strong> It survives the false-discovery correction (FDR) for testing ${n} tiles at once (q&nbsp;=&nbsp;${pval(t.q)}), so it is unlikely to be a fluke of testing so many places. It describes ${periodText()} and does not identify a cause.`;
    if (t.p != null && t.p < S.alpha) return `<strong>Possible ${t.slope > 0 ? "wetting" : "drying"} trend, not confirmed at this tile.</strong> Tested alone it passes (p&nbsp;=&nbsp;${pval(t.p)}). But when ${n} tiles are tested, about 1 in 20 pass by chance. ${periodText()} is also short next to natural multi-year swings such as El Niño. So after the false-discovery correction (q&nbsp;=&nbsp;${pval(t.q)}) the detector cannot single this tile out. Many such tiles are probably real; the map shows them faintly, and the headline numbers do not count them.`;
    return `<strong>No detectable trend.</strong> Year-to-year swings here are larger than any steady change over ${periodText()} (p&nbsp;=&nbsp;${pval(t.p)}).`;
  }
  const LISA = {
    HH: "In a cluster of rising trends", LL: "In a cluster of falling trends", HL: "Rising faster than its neighbours (outlier)",
    LH: "Falling against its neighbours (outlier)", ns: "No significant spatial cluster", nn: "No neighbouring land tiles",
  };
  function periodText() { const p = S.meta?.dataset?.period || []; return p.length ? `${yearOf(p[0])}–${yearOf(p[1])}` : "the study period"; }

  // ---------- panel: intro story ----------
  const panel = $("#panel");
  // chart width in CSS pixels, so one viewBox unit is one pixel and axis text renders at its stated size
  function boxWidth(el, fallback) {
    if (!el || !el.clientWidth) return fallback;
    const cs = getComputedStyle(el);
    return el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  }
  function aggregateSVG() {
    const a = S.meta.headline.aggregate, w = a.window || S.meta.dataset.period;
    const y0 = +yearOf(w[0]), y1 = +yearOf(w[1]) + 1, mid = (y0 + y1) / 2, mean = a.mean;
    const W = Math.round(clamp(boxWidth(panel, 360), 280, 560)), H = 178, L = 40, R = 10, T = 22, B = 26;
    const step = niceStep(mean * 1.5, 5), ymax = Math.ceil(mean * 1.5 / step) * step;
    const X = (yr) => L + (yr - y0) / (y1 - y0) * (W - L - R), Y = (v) => T + (1 - v / ymax) * (H - T - B);
    const at = (s, yr) => mean + s * (yr - mid) / 10;
    const [lo, hi] = a.ci95 || [a.slope, a.slope];
    const band = `M${X(y0)},${Y(at(hi, y0))} L${X(y1)},${Y(at(hi, y1))} L${X(y1)},${Y(at(lo, y1))} L${X(y0)},${Y(at(lo, y0))} Z`;
    const band2 = `M${X(y0)},${Y(at(lo, y0))} L${X(y1)},${Y(at(lo, y1))} L${X(y1)},${Y(at(hi, y1))} L${X(y0)},${Y(at(hi, y0))} Z`;
    let grid = "";
    for (let v = 0; v <= ymax + 1e-9; v += step) grid += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${Math.round(v)}</text>`;
    let ticks = "";
    const yrStep = W < 340 ? 10 : 5;
    for (let yr = Math.ceil(y0 / yrStep) * yrStep; yr <= y1; yr += yrStep) ticks += `<text x="${X(yr)}" y="${H - 7}" text-anchor="middle">${yr}</text>`;
    return `<svg class="chart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Global land-average precipitation trend line, ${fmt(a.slope, 2)} mm per month per decade${a.p_value >= S.alpha ? ", not statistically detectable" : ""}, with 95 percent confidence band.">
      <g class="axis">${grid}${ticks}<text x="${L}" y="${T - 10}" text-anchor="start">mm/month</text></g>
      <path class="band" d="${band}"/><path class="band" d="${band2}"/>
      <line class="agg-line" x1="${X(y0)}" y1="${Y(at(a.slope, y0))}" x2="${X(y1)}" y2="${Y(at(a.slope, y1))}"/>
      <text class="end-label" x="${W - R}" y="${Y(at(a.slope, y1)) - 10}" text-anchor="end">${fmt(mean, 0)} mm/month, ${sgn(a.slope_pct_per_decade)}%/decade</text>
    </svg>`;
  }
  function showIntro(step) {
    S.highlight = null; S.highlightId = null; S.selected = -1;
    const m = S.meta, a = m.headline.aggregate, g = m.headline.regional;
    const n = (g.n_tiles_tested || S.tiles.length).toLocaleString("en-US");
    const notDetect = a.p_value >= S.alpha;
    const d = m.dataset || {};
    const src = d.version ? `NASA GLDAS-${esc(d.version)}` : "NASA GLDAS";
    const head = `<p class="eyebrow">${nw(src + " ·")} ${nw("precipitation ·")} ${nw(periodText())}</p>
      <h1 tabindex="-1">${taglineHTML(m.tagline)}</h1>`;
    setPreReveal(step === 1 && S.reveal < 1);
    if (step === 1) {
      panel.innerHTML = `${head}
        <div class="agg">${aggregateSVG()}<p class="agg-caption">Fitted trend (seasonal regression) of the land-area-weighted average. Its shaded 95% confidence band is about as wide as the line at this scale; the p-value allows for autocorrelation.</p></div>
        <span class="verdict">${notDetect ? "No detectable global trend" : "Detectable global trend"}</span>
        <p>The land-average fitted trend is <span class="num">${fmt(a.slope, 2)}</span>&nbsp;mm/month per decade (<span class="num">${sgn(a.slope_pct_per_decade)}%</span>/decade; 95%&nbsp;CI <span class="num">${fmt(a.ci95[0], 2)}&nbsp;to&nbsp;${fmt(a.ci95[1], 2)}</span>&nbsp;mm/month per decade, p&nbsp;=&nbsp;<span class="num">${pval(a.p_value)}</span>). By this average, little is changing.</p>
        <div class="btn-row"><button class="btn" type="button" id="revealBtn">Reveal what the average hides</button></div>`;
      $("#revealBtn").addEventListener("click", () => runReveal(true));
    } else {
      panel.innerHTML = `${head}
        <p>Tile by tile, across ${n} one-degree land tiles, the detector finds:</p>
        <div class="split">
          <div class="w"><span class="big">${pct(g.share_area_up, 1)}</span><small>of land area with a detectable wetting trend</small></div>
          <div class="d"><span class="big">${pct(g.share_area_down, 1)}</span><small>of land area with a detectable drying trend</small></div>
        </div>
        <p class="detective">Without correcting for testing ${n} tiles at once, <span class="num">${pct(g.share_tiles_raw_p_below_alpha, 0)}</span> of tiles would have looked significant. The detector keeps only those that survive the false-discovery correction (FDR, q&nbsp;&lt;&nbsp;${g.alpha_fdr ?? S.alpha}).</p>
        <p class="muted small">Most land shows no detectable trend; where one is detectable, some regions are wetting and others drying. ${coarse ? "Tap" : "Click"} any tile to see its monthly record for ${periodText()}, or open a case file.</p>
        <div class="btn-row"><a class="btn" href="#cases">Open the case files</a><button class="btn ghost" type="button" id="replayBtn">Replay</button></div>`;
      $("#replayBtn").addEventListener("click", () => { S.reveal = 0; setURL(null, null); render(); showIntro(1); $("#revealBtn")?.focus(); });
    }
    render();
  }
  function taglineHTML(t) {
    const parts = String(t || "").split(/(?<=\.)\s+/);
    return parts.length > 1 ? `${esc(parts[0])}<br><em>${esc(parts.slice(1).join(" "))}</em>` : esc(t);
  }
  // keep the address bar in step with what the panel shows, so a reload or a shared link opens the same view
  function setURL(tileId, caseId) {
    try {
      const u = new URL(location.href), had = u.searchParams.has("tile");
      if (tileId) u.searchParams.set("tile", tileId); else if (had) u.searchParams.delete("tile");
      // only re-serialise the query when the tile changes, so flags like ?reveal stay exactly as written
      const next = location.pathname + (tileId || had ? u.search : location.search) + (caseId ? `#case=${encodeURIComponent(caseId)}` : "");
      if (next !== location.pathname + location.search + location.hash) history.replaceState(null, "", next);
    } catch { /* file:// or sandboxed: leave the URL alone */ }
  }
  const focusPanelHeading = () => panel.querySelector("h1")?.focus({ preventScroll: true });
  function backToOverview() {
    showIntro(2);
    setURL(null, null);
    focusPanelHeading();
    announce("Back to the overview.");
  }
  function runReveal(fromPanel) {
    stopSpin();
    if (globe && S.view === "globe") globe.pointOfView({ lat: 15, lng: 20, altitude: 2.1 }, reduceMotion ? 0 : 1200);
    showIntro(2);
    setPreReveal(false);
    // the Reveal button was replaced: move focus to the new heading instead of dropping it on <body>
    focusPanelHeading();
    // one-column layouts: the new heading can sit above the viewport, focus ring and all (for instance when a
    // smooth scroll to the Reveal button is still running). Check now and again once that scroll has settled.
    if (fromPanel && matchMedia("(max-width: 820px)").matches) {
      const h = panel.querySelector("h1"), bar = $(".topbar")?.offsetHeight || 0;
      const keep = () => { const r = h?.getBoundingClientRect(); if (r && (r.top < bar || r.bottom > innerHeight)) h.scrollIntoView({ block: "nearest" }); };
      keep(); setTimeout(keep, reduceMotion ? 0 : 700);
    }
    { const g = S.meta.headline.regional; announce(`Trends revealed: ${pct(g.share_area_up, 1)} of land area with a detectable wetting trend, ${pct(g.share_area_down, 1)} with a detectable drying trend.`); }
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
    setPreReveal(false);
    S.selected = i;
    const t = S.tiles[i];
    if (S.highlight && !S.highlight.has(t.id)) { S.highlight = null; S.highlightId = null; }
    render();
    const cls = t.sig > 0 ? "wet" : t.sig < 0 ? "dry" : "";
    const reg = t.region ? S.regionById.get(t.region) : null;
    const card = t.region ? S.cardByRegion.get(t.region) : null;
    const regKind = t.region && t.region.startsWith("up") ? "wetting" : "drying";
    const regLine = reg ? `<p class="small">${card
      ? `Part of a ${reg.n_tiles}-tile ${regKind} region, written up in the case file “${et(card.title)}”.</p><div class="btn-row"><button class="btn ghost" type="button" data-case="${esc(card.id)}">Open its case file</button></div>`
      : `Part of a ${reg.n_tiles}-tile ${regKind} region.</p>`}` : "";
    const label = `Tile ${latStr(t.s)}–${latStr(t.s + S.deg)}, ${lonStr(t.w)}–${lonStr(t.w + S.deg)}`;
    panel.innerHTML = `<button class="back" type="button" id="backBtn">← Back to the overview</button>
      <h1 class="tile-h" tabindex="-1">${label}</h1>
      <span class="verdict ${cls}">${esc(verdictShort(t))}</span>
      <div class="series" id="seriesBox"><p class="muted small">Loading monthly record…</p></div>
      <dl class="stats">
        <div><dt>Trend</dt><dd>${sgn(t.slope, 2)}<span class="unit">mm/month per decade</span></dd></div>
        <div><dt>Relative trend</dt><dd>${sgn(t.pct)}%<span class="unit">per decade</span></dd></div>
        <div><dt>p (single test)</dt><dd>${pval(t.p)}</dd></div>
        <div><dt>q (FDR-corrected)</dt><dd>${pval(t.q)}</dd></div>
      </dl>
      <p class="small">${confidence(t)}</p>
      <p class="small muted">Spatial pattern: ${esc(LISA[t.lisa] || t.lisa)}.</p>
      ${regLine}`;
    $("#backBtn").addEventListener("click", backToOverview);
    panel.querySelector("[data-case]")?.addEventListener("click", (e) => openCase(e.currentTarget.dataset.case));
    setURL(t.id, null);
    announce(`${label}: ${verdictShort(t)}.`);
    if (fromMap && matchMedia("(max-width: 820px)").matches) panel.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    const ser = await loadSeries(t);
    if (S.selected !== i) return;
    $("#seriesBox").innerHTML = ser ? seriesSVG(ser, t) : `<p class="muted small">The monthly record for this tile could not be loaded. Try another tile, or reload the page.</p>`;
  }

  function seriesSVG(ser, t) {
    const v = ser.values, n = v.length;
    const [sy, sm] = ser.start.split("-").map(Number);
    const W = Math.round(clamp(boxWidth($("#seriesBox"), 380), 280, 560)), H = 210, L = 42, R = 8, T = 10, B = 26;
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
    const trendCls = t.sig > 0 ? "wet" : t.sig < 0 ? "dry" : "none";
    let ticks = "";
    const yrStep = W < 340 ? 10 : 5;
    for (let yr = Math.ceil(sy / yrStep) * yrStep; yr <= sy + n / 12; yr += yrStep) {
      const k = (yr - sy) * 12 - (sm - 1); if (k < 0 || k > n - 1) continue;
      ticks += `<text x="${X(k)}" y="${H - 7}" text-anchor="middle">${yr}</text>`;
    }
    const ys = niceBelow(ext * 0.9), lab = (y) => (y > 0 ? "+" : "") + fmt(y, 0);
    const yt = [-ys, 0, ys].filter((y) => y >= lo && y <= hi).map((y) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(y)}" y2="${Y(y)}"/><text x="${L - 6}" y="${Y(y) + 4}" text-anchor="end">${lab(y)}</text>`).join("");
    return `<svg class="chart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Monthly precipitation anomalies for this tile with 12-month running mean and fitted trend of ${sgn(t.slope, 2)} millimetres per month per decade.">
      <g class="axis">${yt}${ticks}</g>
      <path class="raw" d="${raw}"/>
      <path class="run" d="${run}"/>
      <line class="trend ${trendCls}" x1="${X(0)}" y1="${Y(mean + perMonth * (0 - midK))}" x2="${X(n - 1)}" y2="${Y(mean + perMonth * (n - 1 - midK))}"/>
    </svg>
    <div class="series-legend"><span><i class="raw"></i>Monthly anomaly (mm/month)</span><span><i class="run"></i>12-month running mean</span><span><i class="trend ${trendCls}"></i>Trend</span></div>`;
  }

  // ---------- cases ----------
  const STATUS_ICON = {
    pass: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M1.5 5.2 4 7.5 8.5 2.5" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>',
    partial: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M1.5 5h7" stroke="currentColor" stroke-width="1.8"/><circle cx="5" cy="5" r="1" fill="currentColor"/></svg>',
    fail: '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M2 2l6 6M8 2 2 8" stroke="currentColor" stroke-width="1.8"/></svg>',
    not_run: '<svg viewBox="0 0 10 10" aria-hidden="true"><circle cx="5" cy="5" r="3" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>',
  };
  const STATUS_LABEL = { pass: "Pass", partial: "Partial", fail: "Fail", not_run: "Not run" };
  const KIND_LABEL = { wetting: "Wetting region", drying: "Drying region", retraction: "Retraction", observation: "Satellite observation", local: "Local region" };
  const normStatus = (s) => (STATUS_LABEL[s] ? s : "not_run");

  function gradeHTML(checks) {
    // the selection criterion (FDR significance) passes by construction, so it is not part of the score
    const graded = checks.filter((c) => c.role !== "selection");
    const selected = graded.length < checks.length;
    // show the raw counts, so the line always agrees with the evidence chip ("passes 2 of 4 extra checks")
    const cnt = { pass: 0, partial: 0, fail: 0, not_run: 0 };
    graded.forEach((c) => { cnt[normStatus(c.status)]++; });
    const extra = [cnt.partial && `${cnt.partial} partial`, cnt.fail && `${cnt.fail} failed`, cnt.not_run && `${cnt.not_run} not run`].filter(Boolean);
    const bars = graded.map((c) => `<i class="${normStatus(c.status)}"></i>`).join("");
    const parts = [`${cnt.pass} of ${graded.length} ${selected ? "extra checks" : "checks"} pass`, ...extra];
    // the separator ends the previous no-wrap chunk, so a wrapped line never starts with "·"
    return `<div class="grade"><span class="grade-bars" aria-hidden="true">${bars}</span><span>${parts.map((x, i) => nw(i < parts.length - 1 ? x + " ·" : x)).join(" ")}</span></div>`
      + (selected ? `<p class="grade-note">The significance check is not scored, because it is how the region was selected.</p>` : "");
  }
  // passing checks collapse to one line; partial, failed and not-run checks stay open so their caveats are always visible
  function checksHTML(checks, allOpen) {
    return `<ul class="checks">${checks.map((c) => {
      const s = normStatus(c.status);
      const head = `<span class="status ${s}">${STATUS_ICON[s]}${STATUS_LABEL[s]}</span><span class="chk-name">${et(c.check)}</span>`;
      if (!c.detail) return `<li><div class="chk-row">${head}</div></li>`;
      return `<li><details class="chk"${allOpen || s !== "pass" ? " open" : ""}><summary>${head}</summary><p class="detail">${et(c.detail)}</p></details></li>`;
    }).join("")}</ul>`;
  }
  function litHTML(lit) {
    if (!lit || !lit.length) return "";
    const ag = (a) => (a === true ? '<span class="agree yes">agrees</span>' : a === "partly" ? '<span class="agree partly">partly</span>' : a === false ? '<span class="agree no">disagrees</span>' : "");
    const link = (u, cit) => { if (!u) return ""; const href = /^10\./.test(u) ? `https://doi.org/${u}` : u; return /^https?:\/\//.test(href) ? ` <a class="src" href="${esc(href)}" target="_blank" rel="noopener" aria-label="Source: ${esc(String(cit || "").slice(0, 60))} (opens in a new tab)">source&nbsp;↗</a>` : ` <span class="mono">${esc(u)}</span>`; };
    return `<details class="more"><summary>Independent literature (${lit.length})</summary><ul>${lit.map((l) => `<li>${et(l.citation)}${ag(l.agrees)}${link(l.url_or_doi, l.citation)}${l.note ? `<br><span class="small">${et(l.note)}</span>` : ""}</li>`).join("")}</ul></details>`;
  }
  function placeholderHTML(c) {
    const clean = (x) => String(x || "").replace(/^TODO:\s*/i, "");
    const t = clean(c.title) || "Your local region";
    const checks = (c.robustness || []).map((x) => ({ check: x.check, status: "not_run" }));
    return `<article class="card local" id="case-${esc(c.id)}" aria-labelledby="t-${esc(c.id)}">
      <div class="card-kind"><span class="dot"></span>${esc(KIND_LABEL.local)} <span class="pill">Placeholder</span></div>
      <h3 class="card-title" id="t-${esc(c.id)}">${esc(t.charAt(0).toUpperCase() + t.slice(1))}</h3>
      <p class="headline">This slot is reserved for the team’s own region. Once a bounding box is chosen, the detector fills in the numbers and runs the same checks; the team names who would act on the result.</p>
      ${checks.length ? gradeHTML(checks) + checksHTML(checks) : ""}
      <div class="card-foot"><span class="evidence">${et(c.evidence_level || "not yet evaluated")}</span></div>
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
    if (n.grace_mean_slope_mm_per_decade != null) nums.push(`<span><b>${sgn(n.grace_mean_slope_mm_per_decade, 0)}</b>GRACE mm/decade</span>`);
    if (n.model_mean_slope_mm_per_decade != null) nums.push(`<span><b>${sgn(n.model_mean_slope_mm_per_decade, 0)}</b>land model mm/decade</span>`);
    if (n.share_tiles_fdr_significant != null) nums.push(`<span><b>${pct(n.share_tiles_fdr_significant, 0)}</b>${n.n_boxes != null ? "boxes " : ""}FDR-significant</span>`);
    const checks = c.robustness || [];
    const u = c.user || {};
    const showBtn = c.region_id && S.regionById.has(c.region_id);
    const colA = `<div class="card-kind"><span class="dot"></span>${esc(KIND_LABEL[kind])}${kind === "local" && !c.region_id ? ` <span class="pill">Placeholder</span>` : ""}</div>
      ${kind === "retraction" ? `<span class="stamp">Rejected by its own detector</span>` : ""}
      <h3 class="card-title" id="t-${esc(c.id)}">${et(c.title)}</h3>
      ${c.place ? `<p class="place">${et(c.place)}</p>` : ""}
      <p class="headline">${et(c.headline)}</p>
      ${nums.length ? `<div class="nums">${nums.join("")}</div>` : ""}
      ${checks.length ? gradeHTML(checks) : ""}`;
    const colB = `${checks.length ? checksHTML(checks, featured) : ""}
      ${u.who ? `<div class="user"><div class="user-label">Who acts on this</div><div class="who">${et(u.who)}</div>${u.decision ? `<p class="decision">${et(u.decision)}</p>` : ""}${u.why_it_matters ? `<p class="why">${et(u.why_it_matters)}</p>` : ""}</div>` : ""}
      ${litHTML(c.literature)}
      ${c.caveats && c.caveats.length ? `<details class="more"><summary>Caveats (${c.caveats.length})</summary><ul>${c.caveats.map((x) => `<li>${et(x)}</li>`).join("")}</ul></details>` : ""}
      <div class="card-foot">${c.evidence_level ? `<span class="evidence">${et(c.evidence_level)}</span>` : "<span></span>"}${showBtn ? `<button class="btn ghost" type="button" data-case="${esc(c.id)}">Show on the map</button>` : ""}</div>`;
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

  // an unfilled "TODO" local-region slot is not shown to visitors (placeholderHTML stays for when a region is chosen)
  const isPlaceholder = (c) => c.kind === "local" && !c.region_id && /^TODO/i.test(String(c.title || ""));
  function renderCases() {
    const shown = S.cards.filter((c) => !isPlaceholder(c));
    const featured = shown.filter((c) => c.kind === "retraction");
    const observed = shown.filter((c) => c.kind === "observation");
    const rest = shown.filter((c) => c.kind !== "retraction" && c.kind !== "observation");
    const order = { wetting: 1, drying: 1, local: 3 };
    rest.sort((a, b) => (order[a.kind] || 2) - (order[b.kind] || 2));
    $("#featured").innerHTML = featured.map((c) => cardHTML(c, true)).join("");
    $("#cardGrid").innerHTML = rest.map((c) => cardHTML(c, false)).join("");
    $("#observed").innerHTML = observed.map((c) => cardHTML(c, true)).join("");
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
    setPreReveal(false);
    if (r) {
      S.highlight = new Set(r.tiles); S.highlightId = r.id;
      flyTo(r);
    } else { S.highlight = null; S.highlightId = null; }
    render();
    const checks = c.robustness || [], u = c.user || {};
    panel.innerHTML = `<button class="back" type="button" id="backBtn">← Back to the overview</button>
      <p class="eyebrow">Case file · ${esc(KIND_LABEL[c.kind] || c.kind)}</p>
      <h1 tabindex="-1">${et(c.title)}</h1>
      ${c.place ? `<p class="muted">${et(c.place)}</p>` : ""}
      <p>${et(c.headline)}</p>
      ${checks.length ? gradeHTML(checks) + checksHTML(checks, true) : ""}
      ${u.who ? `<div class="user" style="margin-top:1rem"><div class="user-label">Who acts on this</div><div class="who">${et(u.who)}</div>${u.decision ? `<p class="decision">${et(u.decision)}</p>` : ""}</div>` : ""}
      ${r ? `<p class="small muted" style="margin-top:1rem">Highlighted tiles are the region’s members. ${coarse ? "Tap" : "Click"} any tile for its monthly record.</p>` : ""}
      <div class="btn-row"><a class="btn ghost" href="#case-${esc(c.id)}">Read the full case file</a></div>`;
    $("#backBtn").addEventListener("click", backToOverview);
    setURL(null, id);
    announce(`Case file opened: ${typo(c.title)}`);
    $("#explorer").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    panel.scrollTop = 0;
    if (!fromLoad) setTimeout(focusPanelHeading, reduceMotion ? 0 : 500);
  }
  function flyTo(r) {
    const [w, s, e, n] = r.bbox || [r.centroid_lon - 5, r.centroid_lat - 5, r.centroid_lon + 5, r.centroid_lat + 5];
    const span = Math.max(e - w, (n - s) * 1.4, 8);
    const fspan = Math.max(span * 1.8, 40); // flat map: keep surrounding coastlines in view
    if (globe && S.view === "globe") {
      stopSpin();
      globe.resumeAnimation(); // it may be paused while the stage was scrolled out of view
      globe.pointOfView({ lat: r.centroid_lat, lng: r.centroid_lon, altitude: clamp(span / 14, 1.4, 2.4) }, reduceMotion ? 0 : 1400);
    }
    F.cx = (w + e) / 2; F.cy = (s + n) / 2;
    if (flat.clientWidth && flat.clientWidth < 600) {
      // phones: fit the region with room around it, so at least one coastline stays in view
      F.k = clamp(Math.min(flat.clientWidth / (Math.max(e - w, 8) * 2.6), flat.clientHeight / ((n - s + 6) * 2)), F.kMin || 1, 10);
      F.cy -= 30 / F.k; // nudge the region up, clear of the legend
    } else if (flat.clientWidth) F.k = clamp(Math.min(flat.clientWidth / (fspan * 1.6), flat.clientHeight / (Math.max(n - s + 6, fspan * 0.6) * 1.6)), F.kMin || 1, 20);
  }


  // ---------- soil water: the decline that wasn't ----------
  const openFull = (src) => `<a class="fig-open" href="${esc(src)}" target="_blank" rel="noopener">Open full size<span class="sr-only"> (opens in a new tab)</span>&nbsp;↗</a>`;
  const stepCI = (ci, d = 1) => (Array.isArray(ci) && ci.length === 2 ? ` (95% CI ${sgn(ci[0], d)} to ${sgn(ci[1], d)})` : "");
  function caseLink(kind, label) {
    const c = S.cards.find((x) => x.kind === kind);
    return c ? `<a class="btn ghost" href="#case-${esc(c.id)}">${esc(label)}</a>` : "";
  }
  function renderSoil() {
    const sw = S.meta?.soil_water;
    if (!sw || typeof sw !== "object") return;
    const ac = sw.artifact_control, fl = sw.fldas, gr = sw.grace;
    if (!ac && !fl && !gr) return;
    const r = S.meta.retraction || {};
    const per = S.meta.dataset?.period || [];
    const bps = (r.breakpoints || []).map(esc);
    $("#soilLede").textContent = `GLDAS-2.1 soil water first looked like a ${per.length ? `${+yearOf(per[1]) - +yearOf(per[0]) + 1}-year ` : ""}global decline. Paradox Earth ran three NASA cross-checks on its own headline: a second version of the same model, a NASA land model with consistent forcing (the weather data fed into the model), and satellite gravity. The second version and the consistently forced model show the decline is not a detectable climate signal, so it was withdrawn${bps.length ? ` (level shifts at ${bps.join(" and ")})` : ""}. Satellite gravity weighs all water, so it can neither confirm nor refute a soil-only trend.`;
    if (ac?.figure) {
      const img = $("#soilImg");
      img.src = ac.figure;
      img.alt = "Two stacked time series, 2000 to 2014, comparing GLDAS-2.1 and GLDAS-2.0. Top: incoming solar radiation in the model forcing; only GLDAS-2.1 steps up after August 2010. Bottom: soil water 0 to 200 cm; both versions dip in 2011 to 2013, GLDAS-2.1 more deeply.";
      $("#soilImgLink").href = ac.figure;
      $("#soilCap").innerHTML = `${et(ac.reading || "")} ${openFull(ac.figure)}`;
      img.addEventListener("error", () => { $("#soilFig").hidden = true; }, { once: true });
      $("#soilFig").hidden = false;
    }
    const steps = [];
    if (ac) {
      const h = ac.headline || {};
      const s21 = ac.swdown_step_21_W_m2 ?? h["swdown_step_2.1_W_m2"], s20 = ac.swdown_step_20_W_m2 ?? h["swdown_step_2.0_W_m2"];
      const w = h.common_window || [];
      // the heading already names the versions and years: show only the rest of the description
      let what = String(ac.what || "").replace(/^GLDAS-2\.1 vs GLDAS-2\.0:\s*/i, "").replace(/,\s*(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}\s*\.?\s*$/, "");
      if (what) what = what.charAt(0).toUpperCase() + what.slice(1) + (/[.!?]$/.test(what) ? "" : ".");
      steps.push(`<article class="step"><p class="step-n">1 · Same model, different forcing</p>
        <h3 class="step-t">GLDAS-2.1 vs GLDAS-2.0${w.length ? `, ${nw(`${esc(yearOf(w[0]))}–${esc(yearOf(w[1]))}`)}` : ""}</h3>
        ${what ? `<p class="small muted">${et(what)}</p>` : ""}
        <div class="split"><div class="v21"><div class="big">${sgn(s21)}</div><small>W/m² solar-radiation step at ${esc(ac.step_date || "")} in 2.1${stepCI(ac.swdown_step_21_ci95 ?? h["swdown_step_2.1_ci95"])}</small></div>
        <div class="v20"><div class="big">${sgn(s20)}</div><small>W/m² in 2.0, same date${stepCI(ac.swdown_step_20_ci95 ?? h["swdown_step_2.0_ci95"])}</small></div></div>
        ${h["soil_trend_2.1_common_mm_per_decade"] != null ? `<p class="small">Soil-water trend over the common window: <span class="num">${sgn(h["soil_trend_2.1_common_mm_per_decade"])}</span>&nbsp;mm/decade in 2.1 (p&nbsp;=&nbsp;${pval(h["soil_trend_2.1_common_p"])}) vs <span class="num">${sgn(h["soil_trend_2.0_common_mm_per_decade"])}</span> in 2.0 (p&nbsp;=&nbsp;${pval(h["soil_trend_2.0_common_p"])}).</p>` : ""}
      </article>`);
    }
    if (fl) {
      const w = fl.window || [], ci = fl.aggregate_ci95;
      const detectable = fl.aggregate_p != null && fl.aggregate_p < 0.05;
      steps.push(`<article class="step"><p class="step-n">2 · A consistently forced NASA model</p>
        <h3 class="step-t">NASA FLDAS, ${nw(`${esc(yearOf(w[0]))}–${esc(yearOf(w[1]))}`)}</h3>
        <p class="small muted">Noah land model driven by MERRA-2 and CHIRPS, with no radiation-forcing switch.</p>
        <dl class="stats">
          <div class="wide"><dt>Global land soil water</dt><dd>${sgn(fl.aggregate_slope_mm_per_decade)}<span class="unit">mm/decade</span></dd></div>
          <div><dt>95% interval</dt><dd>${Array.isArray(ci) ? `${sgn(ci[0])} to ${sgn(ci[1])}` : "–"}</dd></div>
          <div><dt>p</dt><dd>${pval(fl.aggregate_p)}</dd></div>
          <div><dt>Land detectably wetting</dt><dd>${pct(fl.share_area_wetting)}</dd></div>
          <div><dt>Land detectably drying</dt><dd>${pct(fl.share_area_drying)}</dd></div>
        </dl>
        <p class="verdict ${detectable ? (fl.aggregate_slope_mm_per_decade > 0 ? "wet" : "dry") : ""}"><span>${detectable ? "Detectable global trend" : "No detectable global trend, and no decline"}</span></p>
      </article>`);
    }
    const v = gr?.vs_fldas_soil_water;
    if (gr) {
      const w = gr.window || [];
      const hs = Object.entries(v?.hotspots || {});
      const short = (n) => String(n).replace(/\s*\(.*\)\s*$/, "");
      steps.push(`<article class="step"><p class="step-n">3 · Satellite gravity</p>
        <h3 class="step-t">GRACE/GRACE-FO, ${esc(yearOf(w[0]))}–${esc(yearOf(w[1]))}</h3>
        <p class="small">GRACE weighs all water, including groundwater, lakes and ice. It shows FDR-significant storage trends over <span class="num">${pct(gr.land_area_share_fdr_significant, 0)}</span> of land${v ? `; where GRACE or FLDAS shows a significant trend, the two agree in sign over <span class="num">${pct(v.sign_match_area_sig_either, 0)}</span> of the area (rank correlation of the trends <span class="num">${fmt(v.spearman_slopes, 2)}</span>, where 1 means the same ordering)` : ""}. The biggest mismatches line up with places where groundwater is pumped or lakes and inland seas are shrinking, which the land model does not include.</p>
        ${hs.length ? `<div class="table-wrap" tabindex="0" role="region" aria-label="GRACE hotspots table"><table class="mini"><caption class="sr-only">GRACE hotspots: water-storage trend from GRACE versus the FLDAS soil-water trend, mm per decade</caption>
          <thead><tr><th scope="col">Hotspot</th><th scope="col">GRACE<br>mm/dec</th><th scope="col">Land model<br>mm/dec</th></tr></thead>
          <tbody>${hs.map(([n, h]) => `<tr><th scope="row">${esc(short(n))}</th><td class="dry-t">${sgn(h.grace_mean_slope_mm_decade, 0)}</td><td>${sgn(h.model_mean_slope_mm_decade, 0)}</td></tr>`).join("")}</tbody></table></div>` : ""}
        ${v?.note ? `<p class="small muted" style="margin-top:.6rem">${et(v.note.charAt(0).toUpperCase() + v.note.slice(1))}.</p>` : ""}
      </article>`);
    }
    $("#soilSteps").innerHTML = steps.join("");
    $("#soilLinks").innerHTML = caseLink("retraction", "Read the retraction case file") + caseLink("observation", "Read the GRACE case file");
    $("#soil").hidden = false; $("#navSoil").hidden = false;
  }

  // ---------- forecast lab: physics-informed emulator ----------
  function renderLab() {
    const P = S.pinn;
    if (!P || !Array.isArray(P.rows) || !P.rows.length) return;
    const d = P.design || {}, sp = d.splits_by_target_month || {};
    const g = (o, path) => path.reduce((a, k) => (a == null ? a : a[k]), o);
    const ms = (o) => (o && o.mean != null ? `${sgn(o.mean, 2)}<span class="sd"> ± ${fmt(o.sd, 2)}</span>` : "–");
    const mm = (o) => (o && o.mean != null ? `${fmt(o.mean, 1)}<span class="sd"> ± ${fmt(o.sd, o.sd < 0.05 ? 2 : 1)}</span>` : "–");
    const lam = (l) => (Number.isInteger(+l) ? String(+l) : String(+l));
    const row = (arm, l) => P.rows.find((r) => r.arm === arm && +r.lambda === +l);
    const arms = d.arms || [...new Set(P.rows.map((r) => r.arm))];
    const nRuns = P.rows.reduce((a, r) => a + (g(r, ["test", "in_region", "msess_vs_linear_ar", "n"]) || 0), 0);
    const yr = (a) => (a ? `${esc(a[0])} to ${esc(a[1])}` : "");
    $("#labLede").textContent = "Can a small neural network learn how NASA’s land model moves water, and does adding a water-budget physics term help? We trained one to predict next month’s change in stored land water, then tested it on months and a region it never saw.";
    $("#labExplain").innerHTML = `<h3>How it works</h3>
      <ol class="how">
        <li><strong>Target.</strong> The change in the GLDAS-2.1 land water store from one month to the next: soil water 0–200 cm plus snow and canopy water, on a ${d.coarsen ? `${fmt(0.25 * d.coarsen, 0)}°` : "coarse"} grid. This is the model’s store, not total water storage.</li>
        <li><strong>Inputs.</strong> This month’s store plus the month’s rain and temperature (one input set also adds solar and longwave radiation).</li>
        <li><strong>Physics term.</strong> A penalty with weight λ when the predicted change breaks the water budget: change in storage = rain − evaporation − runoff. λ = 0 is a plain network.</li>
        <li><strong>Skill.</strong> Mean-squared-error skill against a linear autoregression that uses the same inputs: 0 means no better than the linear model, above 0 means better.</li>
        <li><strong>Tests.</strong> Training months ${yr(sp.train)}; test months ${yr(sp.test)}; a held-out region (90°E–180°E) never used in training; and ${yr(sp.oor)}, after a documented change in the model’s radiation forcing (out-of-regime).</li>
      </ol>
      <p class="small muted">${nRuns ? `${nRuns} training runs` : "Training runs"}: ${arms.length} input sets × ${(d.lambdas || []).length} values of λ × ${(d.seeds || []).length} random seeds, ${esc(d.epochs || "")} epochs each, on a rented RunPod RTX 4090 GPU. Numbers below are means ± standard deviation across seeds.</p>`;

    const b0 = row("base", 0), b1 = row("base", 1);
    const floor = g(P, ["truth_floor_mm", "test", "in_region"]);
    const tiles = [];
    if (b1) {
      tiles.push(`<div><dt>Skill vs linear model, test months</dt><dd>${ms(g(b1, ["test", "in_region", "msess_vs_linear_ar"]))}</dd></div>`);
      tiles.push(`<div><dt>Skill vs linear model, held-out region</dt><dd>${ms(g(b1, ["test", "spatial_holdout", "msess_vs_linear_ar"]))}</dd></div>`);
    }
    if (b0 && b1) {
      tiles.push(`<div><dt>Budget residual, <span class="nc nw">λ = 0 → 1</span></dt><dd>${fmt(g(b0, ["test", "in_region", "budget_residual_own_fluxes_mm", "mean"]))} → ${fmt(g(b1, ["test", "in_region", "budget_residual_own_fluxes_mm", "mean"]))} mm/mo</dd></div>`);
      tiles.push(`<div><dt>Mismatch with GLDAS fluxes, <span class="nc nw">λ = 0 → 1</span></dt><dd>${fmt(g(b0, ["test", "in_region", "budget_residual_gldas_fluxes_mm", "mean"]))} → ${fmt(g(b1, ["test", "in_region", "budget_residual_gldas_fluxes_mm", "mean"]))} mm/mo</dd></div>`);
    }
    // radiation arm vs base under the regime shift (computed, phrased from the data)
    const drop = (arm) => { const r = row(arm, 1); const a = g(r, ["test", "in_region", "msess_vs_linear_ar", "mean"]), o = g(r, ["oor", "in_region", "msess_vs_linear_ar", "mean"]); return a != null && o != null ? { a, o, d: a - o } : null; };
    const db = drop("base"), dr = drop("radiation");
    $("#labStats").innerHTML = `<h3>What we found (rain + temperature inputs, <span class="nc">λ</span> = 1)</h3>
      ${tiles.length ? `<dl class="stats">${tiles.join("")}</dl>` : ""}
      <ul class="findings">
        <li>The physics term at λ = 1 brings the network’s water budget much closer to closing, with no loss of accuracy.${floor != null ? ` For reference, GLDAS’s own budget residual under the same monthly-mean approximation is <span class="num">${fmt(floor)}</span>&nbsp;mm/month (the “GLDAS reference” row in the table below).` : ""}</li>
        <li>At λ = 10 the physics term dominates and skill collapses to about the linear model’s level.</li>
        ${db && dr && dr.a <= db.a ? `<li>Adding radiation inputs does not help: skill is ${sgn(dr.a, 2)} with radiation against ${sgn(db.a, 2)} without on test months, and ${sgn(dr.o, 2)} against ${sgn(db.o, 2)} after the 2024 change in the model’s radiation forcing.</li>` : ""}
      </ul>`;

    // results table
    const lambdas = d.lambdas || [...new Set(P.rows.map((r) => +r.lambda))];
    const body = [];
    arms.forEach((arm) => lambdas.forEach((l) => {
      const r = row(arm, l); if (!r) return;
      const hl = +l === 1;
      body.push(`<tr${hl ? ' class="hl"' : ""}><th scope="row">${esc(r.arm_label || arm)}</th><td>${lam(l)}${hl ? '<span class="sr-only"> (highlighted)</span>' : ""}</td>
        <td>${ms(g(r, ["test", "in_region", "msess_vs_linear_ar"]))}</td><td>${ms(g(r, ["test", "spatial_holdout", "msess_vs_linear_ar"]))}</td><td>${ms(g(r, ["oor", "in_region", "msess_vs_linear_ar"]))}</td>
        <td>${mm(g(r, ["test", "in_region", "budget_residual_own_fluxes_mm"]))}</td><td>${mm(g(r, ["test", "in_region", "budget_residual_gldas_fluxes_mm"]))}</td></tr>`);
    }));
    $("#labTable").innerHTML = `<caption class="sr-only">Emulator results for each input set and physics weight λ: skill against a linear autoregression on test months, the held-out region and out-of-regime months, and water-budget residuals in mm per month. Mean ± standard deviation across seeds.</caption>
      <thead><tr><th scope="col">Inputs</th><th scope="col"><span class="nc">λ</span></th><th scope="col">Skill<br>test months</th><th scope="col">Skill<br>held-out region</th><th scope="col">Skill<br>out-of-regime</th><th scope="col">Budget residual<br>mm/mo</th><th scope="col">vs GLDAS fluxes<br>mm/mo</th></tr></thead>
      <tbody>${body.join("")}</tbody>
      ${floor != null ? `<tfoot><tr><th scope="row" colspan="5">GLDAS reference: its own budget residual (test months)</th><td>${fmt(floor)}</td><td></td></tr></tfoot>` : ""}`;
    $("#labNote").textContent = "Skill is mean-squared-error skill on next-month storage change against a linear autoregression with the same inputs (0 = no better, 1 = perfect). Budget residual is |ΔS − (P − ET − R)| using the network’s own predicted fluxes; lower is closer to the physics. Highlighted rows: λ = 1.";

    const figs = P.figures || {};
    const figHTML = [];
    if (figs.tradeoff) figHTML.push(`<figure class="fig"><img src="${esc(figs.tradeoff)}" width="1600" height="960" loading="lazy" decoding="async" alt="Scatter plot of skill against water-budget residual for every training run. Runs with λ = 1 keep the same skill as λ = 0 and 0.1 while roughly halving the residual; λ = 10 runs have the lowest residual but skill near zero. All runs sit left of GLDAS’s own residual line."><figcaption>Raising λ shrinks the water-budget residual; skill holds through λ&nbsp;=&nbsp;1 and drops to about the linear model’s level at λ&nbsp;=&nbsp;10. ${openFull(figs.tradeoff)}</figcaption></figure>`);
    if (figs.skill_map) figHTML.push(`<figure class="fig"><img src="${esc(figs.skill_map)}" width="1920" height="864" loading="lazy" decoding="async" alt="World map of the emulator’s skill against the linear model on test months, including the held-out region from 90°E to 180°E."><figcaption>Where the emulator beats the linear model on test months${P.best_for_display ? ` (${esc(row(P.best_for_display.arm, P.best_for_display.lambda)?.arm_label || P.best_for_display.arm)}, λ = ${lam(P.best_for_display.lambda)}, mean of seeds)` : ""}, including the held-out region east of 90°E. In deserts storage barely changes, so per-cell skill there swings to either extreme (deep blue Sahara, red Australian interior); the area-weighted numbers above are the fair summary. ${openFull(figs.skill_map)}</figcaption></figure>`);
    $("#labFigs").innerHTML = figHTML.join("");
    $("#labFigs").querySelectorAll("img").forEach((im) => im.addEventListener("error", () => { im.closest("figure").hidden = true; }, { once: true }));

    $("#labLimits").innerHTML = `<h3>Limits</h3><ul class="caveats">
      <li>This is emulation skill against a model (GLDAS-2.1), not skill at forecasting real-world water.</li>
      <li>The target is the model’s land water store (soil 0–200 cm, snow, canopy). It is not total water storage and leaves out groundwater, lakes and rivers.</li>
      <li>The rain and temperature for the month being predicted are given to the network, so this is not a forecast that could be issued in advance.</li>
      <li>Uses data from ${yr(d.window)}. Training and test months share one radiation-forcing regime; after the 2024 change in the model’s radiation inputs (${yr(sp.oor)}) skill ${P.rows.every((r) => g(r, ["oor", "in_region", "msess_vs_linear_ar", "mean"]) < g(r, ["test", "in_region", "msess_vs_linear_ar", "mean"])) ? "drops for every setting" : "drops for most settings"}.</li>
    </ul>`;
    $("#lab").hidden = false; $("#navLab").hidden = false;
  }

  // ---------- method & credits ----------
  function renderMethod() {
    const m = S.meta, md = m.methods || {};
    const labels = { trend: "Trend test", multiple_testing: "Multiple testing", spatial: "Spatial clustering", regions: "Region rule" };
    // a plain-language line first; the exact technical recipe follows in small print
    const plain = {
      trend: "Straight-line trend per tile; the p-value is the larger (more cautious) of two tests that allow for persistence from one month or year to the next (one follows Santer et al. 2000).",
      multiple_testing: "Correction for testing thousands of tiles at once (false-discovery rate, FDR).",
      spatial: "Checks whether a tile’s trend clusters with similar trends in neighbouring tiles.",
      regions: "At least 5 touching tiles, each significant after false-discovery correction and in a same-sign cluster.",
    };
    const pretty = (v) => typo(String(v))
      .replace(/\btrend_slope_pct\b/g, "percent trend")
      .replace(/\btrend_slope\b/g, "trend slope")
      .replace(/\bAND\b/g, "and")
      .replace(/>=\s*/g, "≥ ")
      .replace(/\bq<0\.05\b/g, "q < 0.05")
      .replace(/\bBH-FDR\b/g, "false-discovery correction (BH)")
      .replace(/\brobustness\b/g, "cross-checks");
    $("#methodList").innerHTML = Object.entries(md).map(([k, v]) => `<div><dt>${esc(labels[k] || k)}</dt><dd>${plain[k] ? `${esc(plain[k])}<span class="tech">${esc(pretty(v))}</span>` : esc(pretty(v))}</dd></div>`).join("")
      + `<div><dt>Grid</dt><dd>${S.deg}° land tiles, ${S.tiles.length.toLocaleString("en-US")} tested; areas weighted by ${esc(String(m.headline?.weights || "cos(latitude)").replace(/\s*\(area_km2__\w+\)/, ""))}</dd></div>`;
    const a = m.headline.aggregate, g = m.headline.regional;
    const rows = [[`${yearOf(a.window?.[0])}–${yearOf(a.window?.[1])} (main)`, a.slope, a.p_value, g.share_area_up, g.share_area_down]];
    for (const [name, w] of Object.entries(m.headline.windows || {})) rows.push([name, w.aggregate_slope, w.aggregate_p, w.share_area_up, w.share_area_down]);
    $("#windowTable").innerHTML += `<thead><tr><th scope="col">Window</th><th scope="col">Global slope<br>mm/mo/dec</th><th scope="col">p</th><th scope="col">Area wetting</th><th scope="col">Area drying</th></tr></thead><tbody>`
      + rows.map((r) => `<tr><th scope="row">${et(r[0])}</th><td>${sgn(r[1], 2)}</td><td>${pval(r[2])}</td><td>${pct(r[3])}</td><td>${pct(r[4])}</td></tr>`).join("") + "</tbody>";
    $("#caveatList").innerHTML = (m.caveats || []).map((c) => `<li>${et(c)}</li>`).join("");
    const x = m.cross_checks?.precip_chirps_fldas;
    if (x && x.sign_match_area_sig_either != null) {
      const w = x.window || [];
      $("#crossText").innerHTML = `GLDAS-2.1 precipitation is GPCP-based model forcing, so the detector repeats the trend test on NASA FLDAS precipitation, which is built from CHIRPS (${esc(w[0] || "")} to ${esc(w[1] || "")}${x.n_tiles ? `, ${x.n_tiles.toLocaleString("en-US")} tiles` : ""}). `
        + `Where either product shows an FDR-significant trend, the signs match over <span class="num">${pct(x.sign_match_area_sig_either, 0)}</span> of the land area`
        + `${x.sign_match_area_sig_both != null ? `, and over <span class="num">${pct(x.sign_match_area_sig_both, 0)}</span> where both do${x.n_sig_both != null ? ` (${x.n_sig_both} tiles)` : ""}` : ""}. `
        + `${x.spearman_slopes_all != null ? `Across all tiles the trend slopes have a Spearman rank correlation of <span class="num">${fmt(x.spearman_slopes_all, 2)}</span>${x.sign_match_area_all != null ? ` and the same sign over <span class="num">${pct(x.sign_match_area_all, 0)}</span> of land` : ""}. ` : ""}`
        + `Most tiles have no detectable trend in either product, which lowers the all-tile figures.`;
      $("#crossBlock").hidden = false;
    }
    const d = m.dataset;
    const doiLink = (doi) => `<a href="https://doi.org/${esc(doi)}" target="_blank" rel="noopener">doi:${esc(doi)}<span class="sr-only"> (opens in a new tab)</span></a>`;
    // DOIs written inside dataset strings ("…, doi:10.5067/…") become links too
    const withDois = (s) => esc(s).replace(/,?\s*doi:(10\.\d{4,9}\/[\w.\-]+)/g, (_, x) => `; ${doiLink(x)}`);
    let credit = `<p><strong>${esc(d.name)}</strong> (${esc(d.short_name)} v${esc(d.version)}), variable <span class="mono">${esc(d.variable)}</span>, ${esc(d.period?.[0])} to ${esc(d.period?.[1])}. Distributed by ${esc(d.provider)}: ${doiLink(d.doi)}.</p>`;
    {
      const sw = m.soil_water || {}, extra = [];
      if (sw.artifact_control) extra.push(`GLDAS-2.0 Noah: the same model driven by ${esc(sw.artifact_control.forcing?.["2.0"] || "different forcing")}, used as a forcing control`);
      if (sw.fldas?.dataset) extra.push(`NASA FLDAS soil water and CHIRPS-based precipitation: ${withDois(sw.fldas.dataset)}`);
      if (sw.grace?.product) extra.push(`NASA JPL satellite gravity: ${esc(sw.grace.product)}`);
      if (extra.length) credit += `<p class="credit-sub">Cross-checks use:</p><ul class="credit-list">${extra.map((x) => `<li>${x}</li>`).join("")}</ul>`;
    }
    $("#dataCredit").innerHTML = credit;
  }
  async function loadTeam() {
    // never show unconfirmed placeholder values ("TEAM NAME (confirm)", "LOCAL EVENT (confirm)") to visitors
    const ok = (v) => typeof v === "string" && v.trim() && !/\(confirm\)|^TEAM NAME|^LOCAL EVENT|^TODO/i.test(v.trim());
    let t = null;
    try { t = await getJSON("../demo/team.json"); } catch { t = null; }
    const challenge = (t && ok(t.challenge) && t.challenge) || S.meta.challenge;
    const src = t && t.project_url ? `<br><a href="${esc(t.project_url)}" target="_blank" rel="noopener">Source code<span class="sr-only"> (opens in a new tab)</span></a>` : "";
    if (t && ok(t.team_name)) {
      const members = (t.members || []).filter(ok).map(esc).join(", ");
      const ev = [t.event, t.local_event].filter(ok).map(esc).join(", ");
      $("#teamLine").innerHTML = `<strong>${esc(t.team_name)}</strong>${members ? ` · ${members}` : ""}${ev ? `<br>${ev}` : ""}<br>Challenge: ${esc(challenge || "")}${src}`;
    } else {
      const ev = t && ok(t.event) ? `${esc(t.event)}<br>` : "";
      $("#teamLine").innerHTML = `${ev}Challenge: ${esc(challenge || "")}${src}`;
    }
  }

  // ---------- page helpers ----------
  // wide tables scroll sideways inside their wrapper: say so, but only when they actually overflow
  function markScrollables() {
    document.querySelectorAll(".table-wrap").forEach((w) => {
      let note = w.previousElementSibling;
      if (!note || !note.classList.contains("scroll-note")) {
        note = document.createElement("p");
        note.className = "scroll-note"; note.setAttribute("aria-hidden", "true");
        note.textContent = `${coarse ? "Swipe" : "Scroll"} sideways to see every column →`;
        w.before(note);
      }
      const over = w.scrollWidth > w.clientWidth + 1;
      note.hidden = !over;
      w.classList.toggle("is-scrollable", over);
    });
  }
  let scrollTimer = 0;
  addEventListener("resize", () => { clearTimeout(scrollTimer); scrollTimer = setTimeout(markScrollables, 200); });
  // mark the section in view in the top bar (the last section whose top has passed a line 30% down the screen)
  function trackSections() {
    const links = [...document.querySelectorAll(".topbar nav a[href^='#']")]
      .map((a) => ({ a, el: document.getElementById(a.getAttribute("href").slice(1)) })).filter((x) => x.el);
    let raf = 0;
    const update = () => {
      raf = 0;
      const line = innerHeight * 0.3;
      let cur = null;
      links.forEach((x) => { if (!x.el.hidden && x.el.getBoundingClientRect().top <= line) cur = x; });
      links.forEach((x) => { if (x === cur) x.a.setAttribute("aria-current", "true"); else x.a.removeAttribute("aria-current"); });
    };
    addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(update); }, { passive: true });
    update();
  }

  // ---------- boot ----------
  async function boot() {
    const narrow = matchMedia("(max-width: 820px)").matches;
    canGlobe = webglOK();
    const want = params.get("view") || (narrow || !canGlobe ? "flat" : "globe");
    // fetch globe.gl alongside the data, and only when the 3D view is going to be shown
    const libP = want !== "flat" && canGlobe ? loadGlobeLib() : null;
    try {
      const [meta, tiles, regions] = await Promise.all([getJSON(DATA + "meta.json"), getJSON(DATA + "tiles.json"), getJSON(DATA + "regions.json")]);
      S.meta = meta; S.alpha = meta.headline?.regional?.alpha_fdr ?? 0.05;
      ingestTiles(tiles);
      S.regions = { up: regions.up || [], down: regions.down || [] };
      [...S.regions.up, ...S.regions.down].forEach((r) => S.regionById.set(r.id, r));
    } catch (e) {
      console.error(e);
      const ld = $("#loading");
      ld.setAttribute("role", "alert");
      ld.innerHTML = `<div class="load-err"><p>The NASA trend data could not be loaded. Check your connection and try again.</p><button class="btn ghost" type="button" id="retryBtn">Retry</button></div>`;
      $("#retryBtn").addEventListener("click", () => location.reload());
      return;
    }
    try {
      const cj = await getJSON(params.get("cards") || DATA + "cards.json");
      if (Array.isArray(cj.cards) && cj.cards.length) { S.cards = cj.cards; S.cardsFromFile = true; }
    } catch { /* cards.json not published yet: fall back below */ }
    if (!S.cardsFromFile) S.cards = fallbackCards();
    try { S.pinn = await getJSON(DATA + "pinn.json"); } catch { S.pinn = null; /* emulator results not published: section stays hidden */ }
    S.cards.forEach((c) => { if (c.region_id) S.cardByRegion.set(c.region_id, c); });

    sizeTexture();
    if (libP) await libP; // if the CDN fails, setView falls back to the flat map
    setView(want === "flat" ? "flat" : "globe");
    renderLegend(); renderCases(); renderSoil(); renderLab(); renderMethod(); loadTeam();
    markScrollables(); trackSections();

    const hash = decodeURIComponent(location.hash || "");
    if (params.has("reveal") || hash.startsWith("#case=")) { S.reveal = 1; showIntro(2); } else showIntro(1);
    if (hash.startsWith("#case=")) openCase(hash.slice(6), true);
    if (params.has("tile")) { const t = S.tiles.findIndex((x) => x.id === params.get("tile")); if (t >= 0) inspect(t, false); }
    // remove the overlay at once: a fade leaves a half-dimmed map in screenshots taken right after load
    const ld = $("#loading"); ld.classList.add("done"); ld.hidden = true;
    // a #case= link pasted into an already open tab
    addEventListener("hashchange", () => { const h = decodeURIComponent(location.hash || ""); if (h.startsWith("#case=")) openCase(h.slice(6)); });
    window.__paradox = { state: S, get globe() { return globe; }, openCase, inspect };
    window.__paradoxReady = true;
  }
  boot();
})();
