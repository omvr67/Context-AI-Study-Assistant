/**
 * Tip strip -- a thin, ghosted line above the composer that cycles once
 * through a handful of feature tips each time the page loads, then
 * collapses away for the rest of that visit.
 *
 * v1.3: previously "seen" was persisted forever in localStorage, so the
 * strip played once ever per browser and then silently never came back --
 * which defeats its purpose now that it's the one place slash commands are
 * advertised (the mode-bar hint text was removed). It now just replays on
 * every fresh page load instead of remembering across visits.
 *
 * Fully self-contained and independent of app.js -- it only touches its
 * own DOM nodes and never touches chat state.
 */
(() => {
  const TIPS = [
    "Every answer is grounded in the real syllabus \u2014 no guessing, no invented dates.",
    "Ask \u201cwhat's my GPA if I get a B+ in a 4-credit class?\u201d \u2014 real math, not a vibe.",
    "Set a target GPA and ask what grades you'd need \u2014 the planner works backward from it.",
    "Give a course code and ask for a study plan \u2014 topics and dates are pulled straight from the syllabus.",
    "Don't see your course? Add it in the sidebar \u2014 paste the text or upload a PDF.",
    "Type \u201c/\u201d in the chatbox to see commands \u2014 /depth, /exam, /flashcards \u2014 with autocomplete.",
    "Try Explain Like I'm 5 or Teach Me This Chapter from the bar above the chat for a different kind of answer.",
    "Type /flashcards CS301 (or just /flashcards with a course selected) for a 10-card study deck.",
    "The whole conversation is remembered, so follow-ups don't need repeated context.",
    "If it's not in the syllabus, the answer says so instead of making something up.",
  ];

  const ROTATE_MS = 6000;
  const SWAP_MS = 250;

  const strip = document.getElementById("tipStrip");
  const textEl = document.getElementById("tipStripText");

  // Decorative feature -- if the markup isn't there, bail out quietly.
  if (!strip || !textEl) return;

  let index = 0;
  textEl.textContent = TIPS[0];

  function finish() {
    strip.classList.add("done");
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
