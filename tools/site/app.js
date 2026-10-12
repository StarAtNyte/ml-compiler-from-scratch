// Site behaviour for the book: theme, chapter drawer, code copy, anchors, outline, progress, keys, search.
(function () {
  "use strict";
  const $ = (s, el) => (el || document).querySelector(s);
  const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));
  const root = document.documentElement;
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch (e) { /* storage blocked */ } },
  };

  // ---- theme: system -> light -> dark -> system
  const themeBtn = $(".theme-toggle");
  function applyTheme(t) {
    if (t === "light" || t === "dark") root.setAttribute("data-theme", t); else root.removeAttribute("data-theme");
    if (themeBtn) { const label = "Theme: " + (t || "system"); themeBtn.title = label; themeBtn.setAttribute("aria-label", label + " (click to change)"); }
  }
  applyTheme(store.get("theme"));
  if (themeBtn) themeBtn.addEventListener("click", () => {
    const next = { null: "light", light: "dark", dark: null }[String(store.get("theme"))] ?? "light";
    store.set("theme", next); applyTheme(next);
  });

  // ---- chapter drawer (phones)
  const navBtn = $(".nav-toggle"), scrim = $(".scrim");
  function setNav(open) {
    document.body.classList.toggle("nav-open", open);
    if (scrim) scrim.hidden = !open;
    if (navBtn) navBtn.setAttribute("aria-expanded", String(open));
  }
  if (navBtn) navBtn.addEventListener("click", () => setNav(!document.body.classList.contains("nav-open")));
  if (scrim) scrim.addEventListener("click", () => setNav(false));
  const cur = $(".sidebar a.cur");
  if (cur) cur.scrollIntoView({ block: "center" });

  // ---- copy buttons on code blocks
  $$(".prose .codehilite").forEach((block) => {
    const btn = document.createElement("button");
    btn.className = "copy-btn"; btn.type = "button"; btn.textContent = "Copy";
    btn.addEventListener("click", () => {
      const text = $("pre", block).innerText;
      const done = () => { btn.textContent = "Copied"; btn.classList.add("done"); setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("done"); }, 1500); };
      const fallback = () => { const r = document.createRange(); r.selectNodeContents($("pre", block)); const s = getSelection(); s.removeAllRanges(); s.addRange(r); btn.textContent = "Press Ctrl+C"; };
      if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, fallback); else fallback();
    });
    block.appendChild(btn);
  });

  // ---- heading anchors
  $$(".prose h2[id], .prose h3[id]").forEach((h) => {
    const a = document.createElement("a");
    a.className = "anchor"; a.href = "#" + h.id; a.textContent = "#"; a.setAttribute("aria-label", "Link to this section");
    h.appendChild(a);
  });

  // ---- outline scroll-spy
  const links = $$(".outline a");
  if (links.length && "IntersectionObserver" in window) {
    const byId = new Map(links.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
    const visible = new Set();
    const heads = $$(".prose h2[id], .prose h3[id]").filter((h) => byId.has(h.id));
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => (e.isIntersecting ? visible.add(e.target) : visible.delete(e.target)));
      let active = heads.filter((h) => visible.has(h))[0];
      if (!active) active = heads.filter((h) => h.getBoundingClientRect().top < 120).pop();
      links.forEach((a) => a.classList.remove("active"));
      if (active) byId.get(active.id).classList.add("active");
    }, { rootMargin: "-60px 0px -65% 0px" });
    heads.forEach((h) => io.observe(h));
  }

  // ---- reading progress
  const bar = $(".progress span");
  if (bar) {
    const update = () => {
      const max = document.documentElement.scrollHeight - innerHeight;
      bar.style.width = (max > 0 ? Math.min(100, (scrollY / max) * 100) : 0) + "%";
    };
    addEventListener("scroll", update, { passive: true }); update();
  }

  // ---- search over assets/search.json
  const input = $("#search"), box = $("#search-results");
  let index = null, loading = null, sel = -1;
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function load() {
    if (index || loading) return loading;
    loading = fetch("assets/search.json").then((r) => r.json()).then((d) => {
      index = d.map((r) => ({ ...r, lt: r.t.toLowerCase(), lx: r.x.toLowerCase(), lc: r.c.toLowerCase() }));
    }).catch(() => { index = []; box.innerHTML = '<div class="sr-empty">Search needs the site to be served over http(s).</div>'; });
    return loading;
  }
  function snippet(r, terms) {
    const i = Math.max(0, r.lx.indexOf(terms[0]));
    let s = r.x.slice(Math.max(0, i - 60), i + 140);
    s = (i > 60 ? "…" : "") + esc(s) + "…";
    terms.forEach((t) => { if (t.length > 1) s = s.replace(new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>"); });
    return s;
  }
  function run() {
    const q = input.value.trim().toLowerCase();
    if (!q) { box.hidden = true; return; }
    const terms = q.split(/\s+/).filter(Boolean);
    const scored = [];
    for (const r of index || []) {
      let score = 0, ok = true;
      for (const t of terms) {
        const inT = r.lt.includes(t), inC = r.lc.includes(t), n = r.lx.split(t).length - 1;
        if (!inT && !n && !inC) { ok = false; break; }
        score += (inT ? 20 : 0) + (inC ? 6 : 0) + Math.min(n, 10);
      }
      if (ok) scored.push([score, r]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    sel = -1;
    box.innerHTML = scored.length
      ? scored.slice(0, 12).map(([, r]) => `<a href="${r.u}"><div class="sr-c">${esc(r.c)}</div><div class="sr-t">${esc(r.t)}</div><div class="sr-x">${snippet(r, terms)}</div></a>`).join("")
      : '<div class="sr-empty">No matches. Try a shorter word: <code>rangeify</code>, <code>barrier</code>, <code>softmax</code>.</div>';
    box.hidden = false;
  }
  if (input) {
    input.addEventListener("focus", load);
    input.addEventListener("input", () => (index ? run() : load().then(run)));
    input.addEventListener("keydown", (e) => {
      const items = $$("a", box);
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        sel = (sel + (e.key === "ArrowDown" ? 1 : -1) + items.length) % Math.max(items.length, 1);
        items.forEach((a, i) => a.classList.toggle("sel", i === sel));
        if (items[sel]) items[sel].scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter" && items.length) { location.href = (items[sel] || items[0]).href; }
      else if (e.key === "Escape") { input.value = ""; box.hidden = true; input.blur(); }
    });
    document.addEventListener("click", (e) => { if (!e.target.closest(".search")) box.hidden = true; });
  }

  // ---- keys: "/" search, arrows for previous/next chapter, Escape closes the drawer
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (e.key === "Escape") setNav(false);
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "/" && input) { e.preventDefault(); input.focus(); }
    if (e.key === "ArrowLeft") { const a = $(".pager-card.prev"); if (a) location.href = a.href; }
    if (e.key === "ArrowRight") { const a = $(".pager-card.next"); if (a) location.href = a.href; }
  });
})();
