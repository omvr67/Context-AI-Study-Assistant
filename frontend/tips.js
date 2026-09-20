/**
 * Tip strip -- a thin, ghosted line above the composer that cycles once
 * through a handful of feature tips, then collapses away and doesn't
 * come back on later visits. Not a persistent widget: just a brief
 * orientation pass for a first-time visitor.
 *
 * Fully self-contained and independent of app.js -- it only reads/writes
 * its own localStorage key and never touches chat state.
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

  const ROTATE_MS = 6000;
  const SWAP_MS = 250;
  const SEEN_KEY = "connectx_tips_seen";

  const strip = document.getElementById("tipStrip");
  const textEl = document.getElementById("tipStripText");

  // Decorative feature -- if the markup isn't there, bail out quietly.
  if (!strip || !textEl) return;

  let alreadySeen = false;
  try {
    alreadySeen = localStorage.getItem(SEEN_KEY) === "1";
  } catch (e) {
    /* private browsing or storage disabled -- default to showing once */
  }

  if (alreadySeen) {
    strip.classList.add("done");
    return;
  }

  let index = 0;
  textEl.textContent = TIPS[0];

  function finish() {
    strip.classList.add("done");
    try {
      localStorage.setItem(SEEN_KEY, "1");
    } catch (e) {
      /* ignore -- worst case it shows once more on a later visit */
    }
  }

  function showNext() {
    index += 1;
    if (index >= TIPS.length) {
      clearInterval(timer);
      finish();
      return;
    }
    textEl.classList.add("swap");
    setTimeout(() => {
      textEl.textContent = TIPS[index];
      textEl.classList.remove("swap");
    }, SWAP_MS);
  }

  const timer = setInterval(showNext, ROTATE_MS);
})();
