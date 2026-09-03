/**
 * Feature tip index card -- a small, dismissible rotating tip widget in
 * the bottom-right corner, styled after the app's own "card index"
 * language (see the course chips in style.css). Not a generic toast:
 * it reads like flipping through a card catalog of what the assistant
 * can do, which doubles as a quick feature tour during a live demo.
 *
 * Fully self-contained and independent of app.js -- it only reads/writes
 * its own localStorage key and never touches chat state, so it can be
 * dropped in or pulled out without touching the chat logic at all.
 */
(() => {
  const TIPS = [
    "Every answer is grounded in the real syllabus \u2014 no guessing, no invented dates.",
    "Ask \u201cwhat's my GPA if I get a B+ in a 4-credit class?\u201d \u2014 real math, not a vibe.",
    "Set a target GPA and ask what grades you'd need \u2014 the planner works backward from it.",
    "Give a course code and ask for a study plan \u2014 topics and dates are pulled straight from the syllabus.",
    "Don't see your course? Add it in the sidebar \u2014 paste the text or upload a PDF.",
    "Try Exam Mode or Teach Me This Chapter from the bar above the chat for a different kind of answer.",
    "The whole conversation is remembered, so follow-ups don't need repeated context.",
    "If it's not in the syllabus, the answer says so instead of making something up.",
  ];

  const ROTATE_MS = 7000;
  const SWAP_MS = 250;
  const STORAGE_KEY = "connectx_tip_index_collapsed";

  const card = document.getElementById("tipIndexCard");
  const tab = document.getElementById("tipIndexTab");
  const closeBtn = document.getElementById("tipIndexClose");
  const textEl = document.getElementById("tipIndexText");
  const countEl = document.getElementById("tipIndexCount");
  const progressFill = document.getElementById("tipIndexProgressFill");

  // If any expected element is missing, bail out quietly rather than
  // throwing -- this widget is decorative and should never be able to
  // break the rest of the page.
  if (!card || !tab || !closeBtn || !textEl || !countEl || !progressFill) return;

  let index = 0;
  let timer = null;

  function renderTip() {
    textEl.textContent = TIPS[index];
    countEl.textContent = `${index + 1} / ${TIPS.length}`;
  }

  function swapTip() {
    textEl.classList.add("swap");
    setTimeout(() => {
      index = (index + 1) % TIPS.length;
      renderTip();
      textEl.classList.remove("swap");
    }, SWAP_MS);
  }

  function restartProgress() {
    // Reset-then-reflow-then-reapply is the standard trick for
    // re-triggering a CSS animation on the same element.
    progressFill.style.animation = "none";
    void progressFill.offsetWidth;
    progressFill.style.animation = `tip-progress-fill ${ROTATE_MS}ms linear forwards`;
  }

  function startRotation() {
    stopRotation();
    restartProgress();
    timer = setInterval(() => {
      swapTip();
      restartProgress();
    }, ROTATE_MS);
  }

  function stopRotation() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  function collapse(persist) {
    card.classList.add("hidden");
    tab.classList.add("visible");
    stopRotation();
    if (persist) {
      try {
        localStorage.setItem(STORAGE_KEY, "1");
      } catch (e) {
        /* private browsing or storage disabled -- fine, just won't persist */
      }
    }
  }

  function expand() {
    card.classList.remove("hidden");
    tab.classList.remove("visible");
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      /* ignore */
    }
    startRotation();
  }

  closeBtn.addEventListener("click", () => collapse(true));
  tab.addEventListener("click", expand);

  renderTip();

  let startCollapsed = false;
  try {
    startCollapsed = localStorage.getItem(STORAGE_KEY) === "1";
  } catch (e) {
    /* ignore -- default to shown */
  }

  if (startCollapsed) {
    card.classList.add("hidden");
    tab.classList.add("visible");
  } else {
    startRotation();
  }
})();
