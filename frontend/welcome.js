/*
 * Welcome -- the rotating greeting shown while the thread is empty.
 *
 * Picks a fresh line from a roster on every page load and after every
 * conversation reset, never repeating the one shown last. The greeting is
 * set in the logo's brand italic (see .welcome-greeting in style.css).
 *
 * Self-contained like tips.js: it watches #thread and shows itself only
 * while nothing but the (optional) reset recap is in it, so app.js doesn't
 * need to know it exists.
 */
(function () {
  const thread = document.getElementById("thread");
  if (!thread) return;

  const LAST_KEY = "contextai.lastGreeting";

  const ROSTER = [
    "What are we studying today?",
    "Ready when you are.",
    "Let's get ahead of that exam.",
    "What's on the syllabus?",
    "Back to the books?",
    "Where should we start?",
    "One topic at a time.",
    "Finals won't study themselves.",
    "Got a question? Let's dig in.",
    "What's the plan for today?",
    "Let's make this semester count.",
    "Ask away \u2014 the syllabus has answers.",
  ];

  function timeGreeting() {
    const h = new Date().getHours();
    if (h >= 5 && h < 12) return "Good morning. What are we tackling first?";
    if (h >= 12 && h < 17) return "Good afternoon. Ready to study?";
    if (h >= 17 && h < 22) return "Good evening. Let's review something.";
    return "Burning the midnight oil?";
  }

  function readLast() {
    try { return sessionStorage.getItem(LAST_KEY); } catch (e) { return null; }
  }
  function writeLast(value) {
    try { sessionStorage.setItem(LAST_KEY, value); } catch (e) { /* private mode */ }
  }

  function pickGreeting() {
    const pool = ROSTER.concat(timeGreeting());
    const last = readLast();
    const options = pool.filter((g) => g !== last);
    const choice = options[Math.floor(Math.random() * options.length)];
    writeLast(choice);
    return choice;
  }

  const el = document.createElement("div");
  el.className = "welcome";

  const greeting = document.createElement("h2");
  greeting.className = "welcome-greeting";

  const sub = document.createElement("p");
  sub.className = "welcome-sub";
  sub.textContent =
    "I'm grounded strictly in your syllabi. Ask about grading, exam dates, attendance, " +
    "GPA projections, or a study plan. Don't see your course? Add its syllabus in the " +
    "sidebar \u2014 paste the text or upload a PDF.";

  el.append(greeting, sub);

  function hasConversation() {
    return Array.from(thread.children).some(
      (child) => child !== el && !child.classList.contains("recap")
    );
  }

  function sync() {
    const shouldShow = !hasConversation();
    if (shouldShow && !el.isConnected) {
      greeting.textContent = pickGreeting();
      thread.appendChild(el);
    } else if (!shouldShow && el.isConnected) {
      el.remove();
    }
  }

  new MutationObserver(sync).observe(thread, { childList: true });
  sync();
})();
