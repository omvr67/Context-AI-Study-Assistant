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
    "Every answer is grounded in the real syllabus. No guessing, no invented dates.",
    "Ask \u201cwhat's my GPA if I get a B+ in a 4-credit class?\u201d. Real math, not just a vibe.",
    "Set a target GPA and ask what grades you'd need. The planner works backward from it.",
    "Give a course code and ask for a study plan; Topics and dates are pulled straight from the syllabus.",
    "Don't see your course? Add it in the sidebar, then paste the text or upload a PDF.",
    "Type \u201c/\u201d in the chatbox to see commands - /depth, /exam, /flashcards, /quiz, /retry - with autocomplete.",
    "Click a PDF in your notebook to pin it. The chat grounds itself in just that document.",
    "Chat about a topic for a bit, then type /flashcards for a 10-card deck built from that conversation.",
    "Type /quiz for a graded multiple-choice quiz on what we've discussed. then /retry to review anything you missed.",
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
