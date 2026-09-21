/**
 * Collapsible + resizable sidebar.
 *
 * Fully self-contained and independent of app.js -- same pattern as
 * tips.js: it only touches its own DOM elements, a --rail-width CSS
 * custom property, and its own localStorage keys, so it can be dropped
 * in or pulled out without touching chat logic at all.
 */
(() => {
  const MIN_WIDTH = 200;
  const MAX_WIDTH = 440;
  const DEFAULT_WIDTH = 260;
  const WIDTH_KEY = "connectx_rail_width";
  const COLLAPSED_KEY = "connectx_rail_collapsed";

  const handle = document.getElementById("railResizeHandle");
  const collapseBtn = document.getElementById("railCollapseBtn");
  const expandTab = document.getElementById("railExpandTab");

  // Decorative/convenience feature -- if any expected element is missing,
  // bail out quietly rather than throwing.
  if (!handle || !collapseBtn || !expandTab) return;

  const root = document.documentElement;

  function applyWidth(px) {
    const clamped = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, px));
    root.style.setProperty("--rail-width", `${clamped}px`);
    return clamped;
  }

  function setCollapsed(collapsed, persist) {
    document.body.classList.toggle("rail-collapsed", collapsed);
    collapseBtn.setAttribute("aria-expanded", String(!collapsed));
    expandTab.setAttribute("aria-hidden", String(!collapsed));
    if (persist) {
      try {
        localStorage.setItem(COLLAPSED_KEY, collapsed ? "1" : "0");
      } catch (e) {
        /* private browsing or storage disabled -- fine, just won't persist */
      }
    }
  }

  // --- restore saved width + collapsed state on load ----------------------
  try {
    const savedWidth = parseInt(localStorage.getItem(WIDTH_KEY), 10);
    applyWidth(Number.isFinite(savedWidth) ? savedWidth : DEFAULT_WIDTH);
  } catch (e) {
    applyWidth(DEFAULT_WIDTH);
  }

  try {
    setCollapsed(localStorage.getItem(COLLAPSED_KEY) === "1", false);
  } catch (e) {
    /* default: expanded */
  }

  // --- collapse / expand ----------------------------------------------------
  collapseBtn.addEventListener("click", () => setCollapsed(true, true));
  expandTab.addEventListener("click", () => setCollapsed(false, true));

  // --- drag to resize ---------------------------------------------------------
  let dragging = false;

  handle.addEventListener("pointerdown", (e) => {
    if (document.body.classList.contains("rail-collapsed")) return;
    e.preventDefault();
    dragging = true;
    handle.setPointerCapture(e.pointerId);
    document.body.classList.add("rail-resizing");
  });

  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    e.preventDefault();
    applyWidth(e.clientX);
  });

  function stopDragging() {
    if (!dragging) return;
    dragging = false;
    document.body.classList.remove("rail-resizing");
    try {
      const current = getComputedStyle(root).getPropertyValue("--rail-width").trim();
      const px = parseInt(current, 10);
      localStorage.setItem(WIDTH_KEY, Number.isFinite(px) ? String(px) : String(DEFAULT_WIDTH));
    } catch (e) {
      /* ignore -- just won't persist this session's width */
    }
  }

  handle.addEventListener("pointerup", stopDragging);
  handle.addEventListener("pointercancel", stopDragging);
})();
