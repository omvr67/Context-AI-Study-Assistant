const API_BASE = "http://127.0.0.1:8000";

const sessionId = crypto.randomUUID();
let activeCourse = null; // null = search across all courses
// Notebook doc_ids the student has pinned via the sidebar (see
// renderNotebookChip) -- when non-empty, every chat turn is scoped to
// exactly these PDFs (search_syllabus excluded, search_notebook restricted
// to just these doc_ids; see backend/main.py's _notebook_scope). Sticky
// like activeCourse: it stays set across messages until toggled off.
let selectedNotebookDocIds = new Set();
let activeController = null; // AbortController for the in-flight stream, if any

// Identifies this browser for the private notebook feature (see
// backend/notebook_store.py) -- generated once and kept in localStorage so
// notebook uploads survive a page reload but never leave this device. Not
// an auth token: it's the entire basis for keeping one device's notebook
// away from every other device, so it's sent as plain request data, never
// something the model can see or set itself (see backend/tools.py's
// make_notebook_tool).
function getOrCreateDeviceId() {
  const KEY = "connectx_device_id";
  try {
    let id = localStorage.getItem(KEY);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(KEY, id);
    }
    return id;
  } catch (e) {
    // Private browsing or storage disabled -- fall back to a per-tab id.
    // The notebook feature just won't persist across a reload in that case.
    return crypto.randomUUID();
  }
}
const deviceId = getOrCreateDeviceId();

const thread = document.getElementById("thread");
const composer = document.getElementById("composer");
const composerInputWrap = document.getElementById("composerInputWrap");
const composerHighlight = document.getElementById("composerHighlight");
const commandSuggestions = document.getElementById("commandSuggestions");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const stopBtn = document.getElementById("stopBtn");
const courseList = document.getElementById("courseList");
const activeCourseTab = document.getElementById("activeCourseTab");
const notebookScopeTab = document.getElementById("notebookScopeTab");
const resetBtn = document.getElementById("resetBtn");

const addCourseBtn = document.getElementById("addCourseBtn");
const addCourseForm = document.getElementById("addCourseForm");
const addCourseStatus = document.getElementById("addCourseStatus");

const tabTextBtn = document.getElementById("tabTextBtn");
const tabPdfBtn = document.getElementById("tabPdfBtn");
const addCourseTextForm = document.getElementById("addCourseTextForm");
const addCoursePdfForm = document.getElementById("addCoursePdfForm");
const cancelAddCourse = document.getElementById("cancelAddCourse");
const cancelAddCoursePdf = document.getElementById("cancelAddCoursePdf");

const newCourseCode = document.getElementById("newCourseCode");
const newCourseName = document.getElementById("newCourseName");
const newCourseContent = document.getElementById("newCourseContent");

const pdfCourseCode = document.getElementById("pdfCourseCode");
const pdfCourseName = document.getElementById("pdfCourseName");
const pdfFile = document.getElementById("pdfFile");
const pdfAppend = document.getElementById("pdfAppend");

const notebookList = document.getElementById("notebookList");
const addNotebookBtn = document.getElementById("addNotebookBtn");
const addNotebookForm = document.getElementById("addNotebookForm");
const addNotebookStatus = document.getElementById("addNotebookStatus");
const cancelAddNotebook = document.getElementById("cancelAddNotebook");
const notebookTitle = document.getElementById("notebookTitle");
const notebookFile = document.getElementById("notebookFile");

// Rotated randomly so repeated questions don't feel like a canned response.
const THINKING_PHRASES = [
  "Checking the syllabus…",
  "Thinking it through…",
  "Running the numbers…",
  "Working on it…",
];

const TOOL_LABELS = {
  search_syllabus: "📖 Searched syllabus",
  search_notebook: "📓 Searched notebook",
  gpa_impact_simulator: "🧮 Calculated GPA",
  gpa_target_planner: "🎯 Planned target GPA",
  generate_study_schedule: "🗓️ Built a schedule",
  build_ai_study_plan: "🗓️ Auto-built study plan",
};

// Real markdown rendering (headings, lists, code blocks, tables, quotes,
// links, bold/italic) via vendored marked + DOMPurify -- see
// frontend/vendor/. Falls back to the old escape-and-<br> behavior if
// either vendor script failed to load, so a rendering hiccup can't take
// the whole chat down.
if (typeof marked !== "undefined") {
  marked.setOptions({ gfm: true, breaks: true });
}

function renderMarkdown(text) {
  if (typeof marked === "undefined" || typeof DOMPurify === "undefined") {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\n/g, "<br>");
  }
  const rawHtml = marked.parse(text);
  return DOMPurify.sanitize(rawHtml);
}

function addCard(role, text) {
  const card = document.createElement("div");
  card.className = `card ${role}`;
  card.textContent = text;
  thread.appendChild(card);
  thread.scrollTop = thread.scrollHeight;
  return card;
}

function addThinkingCard() {
  const phrase = THINKING_PHRASES[Math.floor(Math.random() * THINKING_PHRASES.length)];
  const card = document.createElement("div");
  card.className = "card assistant thinking";
  card.innerHTML = `<span class="thinking-text">${phrase}</span><span class="dots"><span></span><span></span><span></span></span>`;
  thread.appendChild(card);
  thread.scrollTop = thread.scrollHeight;
  return card;
}

// --- Streaming assistant card -------------------------------------------
// Built incrementally as SSE events arrive: startStreamingCard() creates an
// empty, still-"thinking"-styled card; appendStreamToken() fills it in as
// text tokens arrive; markToolActive() shows a live badge while a tool runs;
// finalizeStreamingCard() swaps in the final grounded/tools/sources meta
// exactly like the old one-shot addAssistantCard() used to build up front.

function startStreamingCard() {
  const card = document.createElement("div");
  card.className = "card assistant streaming";
  card.innerHTML = `<div class="streamed-text"></div><span class="stream-cursor"></span>`;
  thread.appendChild(card);
  thread.scrollTop = thread.scrollHeight;
  card._fullText = "";
  return card;
}

function appendStreamToken(card, text) {
  card._fullText += text;
  scheduleStreamRender(card);
}

// The expensive part of a token update -- reparsing + sanitizing the WHOLE
// accumulated response and replacing the DOM subtree -- costs roughly
// O(current length), so calling it on every single token (the old
// behavior) makes a long response cost O(n^2) in total output length:
// each new token re-renders everything that came before it too. That's
// what caused visible, worsening lag specifically on /depth's much longer
// answers -- ordinary short replies were never big enough for the
// difference to be noticeable. Coalescing to at most one render per
// animation frame (browsers paint ~60x/sec) bounds the number of full
// re-renders by elapsed time instead of by token count; card._fullText
// itself is still updated synchronously above, so nothing is lost, only
// the expensive re-render is deferred and batched.
function scheduleStreamRender(card) {
  if (card._renderScheduled) return;
  card._renderScheduled = true;
  requestAnimationFrame(() => {
    card._renderScheduled = false;
    renderStreamedCard(card);
  });
}

function renderStreamedCard(card) {
  const textEl = card.querySelector(".streamed-text");
  if (!textEl) return;
  const cursor = card.querySelector(".stream-cursor");
  textEl.innerHTML = renderMarkdown(card._fullText);
  // Keep the cursor flowing right after the last rendered character,
  // wherever that now lives (end of a paragraph, a list item, etc.)
  // rather than parked after the whole block.
  if (cursor) textEl.appendChild(cursor);
  thread.scrollTop = thread.scrollHeight;
}

function markToolActive(card, name) {
  let liveMeta = card.querySelector(".card-meta.live");
  if (!liveMeta) {
    liveMeta = document.createElement("div");
    liveMeta.className = "card-meta live";
    card.appendChild(liveMeta);
  }
  const tag = document.createElement("span");
  tag.className = "badge tool pending";
  tag.textContent = `${TOOL_LABELS[name] || name}…`;
  liveMeta.appendChild(tag);
  thread.scrollTop = thread.scrollHeight;
}

function finalizeStreamingCard(card, data) {
  card.classList.remove("streaming");
  card.querySelector(".stream-cursor")?.remove();

  const liveMeta = card.querySelector(".card-meta.live");
  if (liveMeta) liveMeta.remove();

  // One final, synchronous render straight from the server's authoritative
  // text -- a token render might still be waiting on its coalesced
  // animation frame (see scheduleStreamRender) when "done" arrives, and
  // this also covers the no-tokens-ever-streamed case (e.g. an empty
  // reply) in one place instead of two.
  card.querySelector(".streamed-text").innerHTML = renderMarkdown(data.response || "(no response)");

  const meta = document.createElement("div");
  meta.className = "card-meta";

  if (data.grounded) {
    const badge = document.createElement("span");
    badge.className = "badge grounded";
    badge.textContent = "✓ Grounded in syllabus";
    meta.appendChild(badge);
  }

  if (data.mode) {
    const modeBadge = document.createElement("span");
    modeBadge.className = "badge mode";
    modeBadge.textContent = MODE_LABELS[data.mode] || data.mode;
    meta.appendChild(modeBadge);
  }

  (data.tools_used || []).forEach((name) => {
    const tag = document.createElement("span");
    tag.className = "badge tool";
    tag.textContent = TOOL_LABELS[name] || name;
    meta.appendChild(tag);
  });

  if (meta.childNodes.length) card.appendChild(meta);

  if (data.sources && data.sources.length) {
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `${data.sources.length} source${data.sources.length > 1 ? "s" : ""}`;
    details.appendChild(summary);
    data.sources.forEach((src) => {
      const p = document.createElement("p");
      p.className = "source-item";
      const pageTag = src.page ? ` p.${src.page}` : "";
      p.innerHTML = `<span class="source-code">${src.course_code}${pageTag}</span> ${src.snippet}`;
      details.appendChild(p);
    });
    card.appendChild(details);
  }

  thread.scrollTop = thread.scrollHeight;
}

const MODE_LABELS = {
  exam: "📝 Exam Mode",
  depth: "🧠 Deep Explainer",
};

// --- Solutions PDF attach: shared helper used by both course chips and
// notebook chips (see backend's POST /courses/{code}/solutions and
// POST /notebook/{doc_id}/solutions). A small button opens a native file
// picker and uploads the chosen PDF immediately -- no separate form UI,
// consistent with how compact the rest of a chip row already is.
function addSolutionsButton(chip, { hasSolutions, label, uploadFn, onDone }) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "course-chip-solutions" + (hasSolutions ? " attached" : "");
  btn.title = hasSolutions ? `Replace the solutions PDF for ${label}` : `Attach a solutions PDF for ${label}`;
  // Icon-only: this sits in a fixed-width slot next to the delete button
  // (see .course-chip-solutions in style.css) -- the old "🔑 Add solutions"
  // text label crowded that slot, so the tooltip above carries the label
  // instead.
  btn.textContent = hasSolutions ? "✓" : "🔑";

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "application/pdf,.pdf";
  fileInput.className = "hidden";

  const resetLabel = () => {
    btn.disabled = false;
    btn.textContent = hasSolutions ? "✓" : "🔑";
  };

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    fileInput.value = "";
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      alert("Please choose a .pdf file.");
      return;
    }
    btn.disabled = true;
    btn.textContent = "Uploading…";
    try {
      await uploadFn(file);
      onDone();
    } catch (err) {
      alert(err.message || "Couldn't upload that solutions PDF — is the backend running?");
      resetLabel();
    }
  });

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  chip.appendChild(btn);
  chip.appendChild(fileInput);
}

function renderChip(course, isAll) {
  const chip = document.createElement("div");
  chip.className = "course-chip" + (isAll ? " active" : "");

  const main = document.createElement("button");
  main.type = "button";
  main.className = "course-chip-main";
  const sourceTag = course.source === "pdf" ? ` <span class="pdf-tag">PDF${course.page_count ? ` · ${course.page_count}p` : ""}</span>` : "";
  main.innerHTML = `<span class="code">${course.course_code}</span><span class="name">${course.course_name}${sourceTag}</span>`;
  main.addEventListener("click", () => {
    document.querySelectorAll(".course-chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    activeCourse = isAll ? null : course.course_code;
    activeCourseTab.textContent = isAll
      ? "All Courses"
      : `${course.course_code} — ${course.course_name}`;
  });
  chip.appendChild(main);

  if (course.custom) {
    addSolutionsButton(chip, {
      hasSolutions: course.has_solutions,
      label: course.course_code,
      uploadFn: async (file) => {
        const formData = new FormData();
        formData.append("file", file);
        const res = await fetch(`${API_BASE}/courses/${encodeURIComponent(course.course_code)}/solutions`, {
          method: "POST",
          body: formData,
        });
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.detail || `status ${res.status}`);
        }
      },
      onDone: loadCourses,
    });

    const del = document.createElement("button");
    del.type = "button";
    del.className = "course-chip-delete";
    del.title = `Remove ${course.course_code}`;
    del.textContent = "×";
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Remove ${course.course_code} from your syllabi?`)) return;
      try {
        const res = await fetch(`${API_BASE}/courses/${encodeURIComponent(course.course_code)}`, {
          method: "DELETE",
        });
        if (!res.ok) throw new Error(`status ${res.status}`);
        if (activeCourse === course.course_code) {
          activeCourse = null;
          activeCourseTab.textContent = "All Courses";
        }
        loadCourses();
      } catch (err) {
        alert("Couldn't remove that course — is the backend running?");
      }
    });
    chip.appendChild(del);
  }

  return chip;
}

async function loadCourses() {
  try {
    const res = await fetch(`${API_BASE}/courses`);
    if (!res.ok) throw new Error(`status ${res.status}`);
    const courses = await res.json();

    courseList.innerHTML = "";
    courseList.appendChild(renderChip({ course_code: "ALL", course_name: "All Courses" }, true));
    for (const course of courses) {
      courseList.appendChild(renderChip(course, false));
    }
  } catch (err) {
    courseList.innerHTML = `<p class="loading">Couldn't reach the backend at ${API_BASE}. Is uvicorn running?</p>`;
  }
}

// --- Notebook: private-per-device PDFs, separate from the shared course
// list above -- see backend/notebook_store.py. Reuses the same .course-chip
// styling (just its own list/click behavior: clicking a notebook doc has
// no "search this one" filter the way a course chip does, since
// search_notebook always searches the whole notebook), with every entry
// deletable since none of them are built-in.

// Updates the small "📓 N PDF(s) only" pill in the deck header to reflect
// selectedNotebookDocIds -- purely a display of that Set, never a source
// of truth itself.
function updateNotebookScopeTab() {
  const n = selectedNotebookDocIds.size;
  if (!n) {
    notebookScopeTab.classList.add("hidden");
    notebookScopeTab.textContent = "";
    return;
  }
  notebookScopeTab.textContent = `📓 ${n} PDF${n === 1 ? "" : "s"} only`;
  notebookScopeTab.classList.remove("hidden");
}

function renderNotebookChip(doc) {
  const chip = document.createElement("div");
  chip.className = "course-chip" + (selectedNotebookDocIds.has(doc.doc_id) ? " notebook-selected" : "");

  // Clicking pins/unpins this PDF for the chat context (see
  // selectedNotebookDocIds and backend/main.py's _notebook_scope) -- unlike
  // a course chip's single-select "which course am I filtering by", this
  // is multi-select: any number of notebook docs can be pinned at once,
  // toggled independently.
  const main = document.createElement("button");
  main.type = "button";
  main.className = "course-chip-main";
  main.title = selectedNotebookDocIds.has(doc.doc_id)
    ? `Pinned for this chat -- click to unpin "${doc.title}"`
    : `Click to pin "${doc.title}" as the only source for this chat`;
  main.innerHTML = `<span class="code">📓 ${doc.page_count} page${doc.page_count === 1 ? "" : "s"}</span><span class="name">${doc.title}</span>`;
  main.addEventListener("click", () => {
    if (selectedNotebookDocIds.has(doc.doc_id)) {
      selectedNotebookDocIds.delete(doc.doc_id);
    } else {
      selectedNotebookDocIds.add(doc.doc_id);
    }
    chip.classList.toggle("notebook-selected");
    main.title = selectedNotebookDocIds.has(doc.doc_id)
      ? `Pinned for this chat -- click to unpin "${doc.title}"`
      : `Click to pin "${doc.title}" as the only source for this chat`;
    updateNotebookScopeTab();
  });
  chip.appendChild(main);

  addSolutionsButton(chip, {
    hasSolutions: doc.has_solutions,
    label: doc.title,
    uploadFn: async (file) => {
      const formData = new FormData();
      formData.append("device_id", deviceId);
      formData.append("file", file);
      const res = await fetch(`${API_BASE}/notebook/${encodeURIComponent(doc.doc_id)}/solutions`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `status ${res.status}`);
      }
    },
    onDone: loadNotebook,
  });

  const del = document.createElement("button");
  del.type = "button";
  del.className = "course-chip-delete";
  del.title = `Remove ${doc.title} from your notebook`;
  del.textContent = "×";
  del.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!confirm(`Remove "${doc.title}" from your notebook?`)) return;
    try {
      const res = await fetch(
        `${API_BASE}/notebook/${encodeURIComponent(doc.doc_id)}?device_id=${encodeURIComponent(deviceId)}`,
        { method: "DELETE" }
      );
      if (!res.ok) throw new Error(`status ${res.status}`);
      selectedNotebookDocIds.delete(doc.doc_id);
      updateNotebookScopeTab();
      loadNotebook();
    } catch (err) {
      alert("Couldn't remove that document — is the backend running?");
    }
  });
  chip.appendChild(del);

  return chip;
}

async function loadNotebook() {
  try {
    const res = await fetch(`${API_BASE}/notebook?device_id=${encodeURIComponent(deviceId)}`);
    if (!res.ok) throw new Error(`status ${res.status}`);
    const docs = await res.json();

    // A pinned doc that no longer exists server-side (deleted from another
    // tab, say) shouldn't silently keep restricting the chat -- drop it.
    const liveIds = new Set(docs.map((d) => d.doc_id));
    let pruned = false;
    for (const id of selectedNotebookDocIds) {
      if (!liveIds.has(id)) {
        selectedNotebookDocIds.delete(id);
        pruned = true;
      }
    }
    if (pruned) updateNotebookScopeTab();

    notebookList.innerHTML = "";
    if (!docs.length) {
      notebookList.innerHTML = `<p class="loading">Nothing here yet — add a PDF below.</p>`;
      return;
    }
    for (const doc of docs) {
      notebookList.appendChild(renderNotebookChip(doc));
    }
  } catch (err) {
    notebookList.innerHTML = `<p class="loading">Couldn't reach the backend at ${API_BASE}. Is uvicorn running?</p>`;
  }
}

// --- Slash commands: highlight overlay + "/" suggestion palette ---------
// Five commands are recognized when typed as the leading token of the
// composer: /depth and /exam are one-shot mode overrides for that single
// message (handled in the submit handler below); /flashcards, /quiz, and
// /retry are each their own command entirely (see FLASHCARDS_COMMAND_RE /
// QUIZ_COMMAND_RE / RETRY_COMMAND_RE further down) and are only listed
// here so they show up in the suggestion box too.
const SLASH_COMMANDS = [
  { cmd: "depth", desc: "Deep, in-depth explanation for this message" },
  { cmd: "exam", desc: "Frame this message in exam-prep mode" },
  { cmd: "flashcards", desc: "Generate a flashcard deck from what we've discussed so far" },
  { cmd: "quiz", desc: "Generate a graded multiple-choice quiz from what we've discussed" },
  { cmd: "retry", desc: "A new quiz focused on what you've gotten wrong so far" },
];

// A fully-typed, recognized command word at the very start of the message --
// used both to color it blue in the overlay and, in the submit handler, to
// decide whether this particular send carries a one-shot mode override.
const LEADING_COMMAND_RE = /^\/(depth|exam|flashcards|quiz|retry)\b/i;

// "/" plus an in-progress word and nothing else yet -- while this matches,
// the student is still choosing a command, so the suggestion box stays open.
const PARTIAL_COMMAND_RE = /^\/([a-zA-Z]*)$/;

let suggestionItems = [];
let activeSuggestionIndex = -1;

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Keeps #composerHighlight showing the same text as #messageInput (whose own
// text is transparent -- see style.css), with a recognized leading command
// wrapped in a blue span, and its horizontal scroll position matched so long
// lines still line up correctly while the input is scrolled.
function renderComposerHighlight() {
  const value = messageInput.value;
  const match = value.match(LEADING_COMMAND_RE);
  if (!match) {
    composerHighlight.textContent = value;
  } else {
    const rest = value.slice(match[0].length);
    composerHighlight.innerHTML = `<span class="cmd-token">${escapeHtml(match[0])}</span>${escapeHtml(rest)}`;
  }
  composerHighlight.scrollLeft = messageInput.scrollLeft;
}

function closeSuggestions() {
  commandSuggestions.classList.add("hidden");
  commandSuggestions.innerHTML = "";
  suggestionItems = [];
  activeSuggestionIndex = -1;
}

function renderSuggestions() {
  commandSuggestions.innerHTML = "";
  suggestionItems.forEach((entry, i) => {
    const item = document.createElement("div");
    item.className = "command-suggestion-item" + (i === activeSuggestionIndex ? " active" : "");
    item.innerHTML = `<span class="cmd">/${entry.cmd}</span><span class="desc">${entry.desc}</span>`;
    // mousedown (not click) fires before the input's blur handler would
    // otherwise close the box first and swallow the selection.
    item.addEventListener("mousedown", (e) => {
      e.preventDefault();
      acceptSuggestion(entry.cmd);
    });
    commandSuggestions.appendChild(item);
  });
  commandSuggestions.classList.remove("hidden");
}

function acceptSuggestion(cmd) {
  messageInput.value = `/${cmd} `;
  renderComposerHighlight();
  closeSuggestions();
  messageInput.focus();
  const end = messageInput.value.length;
  messageInput.setSelectionRange(end, end);
}

function updateSuggestions() {
  const match = messageInput.value.match(PARTIAL_COMMAND_RE);
  if (!match) {
    closeSuggestions();
    return;
  }
  const fragment = match[1].toLowerCase();
  suggestionItems = SLASH_COMMANDS.filter((c) => c.cmd.startsWith(fragment));
  if (!suggestionItems.length) {
    closeSuggestions();
    return;
  }
  activeSuggestionIndex = 0;
  renderSuggestions();
}

messageInput.addEventListener("input", () => {
  renderComposerHighlight();
  updateSuggestions();
});

messageInput.addEventListener("scroll", () => {
  composerHighlight.scrollLeft = messageInput.scrollLeft;
});

messageInput.addEventListener("keydown", (e) => {
  if (commandSuggestions.classList.contains("hidden")) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    activeSuggestionIndex = (activeSuggestionIndex + 1) % suggestionItems.length;
    renderSuggestions();
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    activeSuggestionIndex = (activeSuggestionIndex - 1 + suggestionItems.length) % suggestionItems.length;
    renderSuggestions();
  } else if (e.key === "Enter" || e.key === "Tab") {
    if (activeSuggestionIndex >= 0) {
      e.preventDefault();
      acceptSuggestion(suggestionItems[activeSuggestionIndex].cmd);
    }
  } else if (e.key === "Escape") {
    closeSuggestions();
  }
});

messageInput.addEventListener("blur", () => {
  setTimeout(closeSuggestions, 100);
});

// --- Sending a message, streamed ----------------------------------------

function setSending(isSending) {
  messageInput.disabled = isSending;
  sendBtn.disabled = isSending;
  sendBtn.classList.toggle("hidden", isSending);
  stopBtn.classList.toggle("hidden", !isSending);
}

async function sendMessage(message, oneShotMode) {
  addCard("user", message);
  messageInput.value = "";
  renderComposerHighlight();
  setSending(true);

  const pending = addThinkingCard();
  activeController = new AbortController();

  let streamCard = null;
  let sawAnyEvent = false;

  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message,
        course_code: activeCourse,
        // "/depth"/"/exam" typed into this message (see the composer's
        // submit handler) is one-shot -- it applies to this request only,
        // with no sticky mode state to fall back to otherwise.
        mode: oneShotMode || null,
        device_id: deviceId,
        // Non-empty only when the student has pinned specific notebook
        // PDFs in the sidebar (see selectedNotebookDocIds) -- see
        // backend/main.py's _notebook_scope for what this does server-side.
        notebook_doc_ids: Array.from(selectedNotebookDocIds),
      }),
      signal: activeController.signal,
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `status ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // last part may be incomplete -- keep it for the next read

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        let event;
        try {
          event = JSON.parse(line.slice(5).trim());
        } catch {
          continue;
        }

        if (!sawAnyEvent) {
          pending.remove();
          streamCard = startStreamingCard();
          sawAnyEvent = true;
        }

        if (event.type === "token") {
          appendStreamToken(streamCard, event.text);
        } else if (event.type === "tool") {
          markToolActive(streamCard, event.name);
        } else if (event.type === "done") {
          finalizeStreamingCard(streamCard, {
            response: event.content,
            grounded: event.grounded,
            sources: event.sources,
            tools_used: event.tools_used,
            mode: event.mode,
          });
        } else if (event.type === "error") {
          throw new Error(event.error || "Streaming error");
        }
      }
    }

    if (!sawAnyEvent) {
      pending.remove();
      addCard("assistant", "(no response)");
    }
  } catch (err) {
    if (err.name === "AbortError") {
      if (streamCard) {
        streamCard.classList.remove("streaming");
        streamCard.querySelector(".stream-cursor")?.remove();
        const note = document.createElement("p");
        note.className = "stopped-note";
        note.textContent = "— stopped —";
        streamCard.appendChild(note);
      } else {
        pending.className = "card assistant";
        pending.textContent = "— stopped —";
      }
    } else {
      pending.remove();
      if (streamCard) {
        streamCard.classList.remove("streaming");
        streamCard.querySelector(".stream-cursor")?.remove();
        streamCard.classList.add("error");
      } else {
        addCard("error", err.message);
      }
    }
  } finally {
    activeController = null;
    setSending(false);
    messageInput.focus();
  }
}

// --- Flashcards: a /command, not a button -- see backend's POST /flashcards.
// Typing "/flashcards" is intercepted in the composer's submit handler
// below and never reaches the normal chat loop at all; it hits its own
// endpoint and renders its own interactive card type instead of a
// streamed reply. v1.3 round 2: the deck's topic is inferred entirely
// from what's been discussed in the session so far (see backend's
// recent_transcript()/generate_flashcards_from_conversation()) -- any
// text typed after "/flashcards" is ignored rather than treated as a
// course code or notebook title.
const FLASHCARDS_COMMAND_RE = /^\/flashcards\b/i;

function addFlashcardDeckCard(deck) {
  const card = document.createElement("div");
  card.className = "card assistant flashcard-deck";

  const header = document.createElement("div");
  header.className = "flashcard-header";
  header.innerHTML = `<span class="title">🗂️ ${deck.title}</span><span>${deck.cards.length} cards</span>`;
  card.appendChild(header);

  const face = document.createElement("div");
  face.className = "flashcard-face";
  face.title = "Click to flip";
  card.appendChild(face);

  const nav = document.createElement("div");
  nav.className = "flashcard-nav";
  const prevBtn = document.createElement("button");
  prevBtn.type = "button";
  prevBtn.textContent = "← Prev";
  const counter = document.createElement("span");
  counter.className = "counter";
  const nextBtn = document.createElement("button");
  nextBtn.type = "button";
  nextBtn.textContent = "Next →";
  nav.append(prevBtn, counter, nextBtn);
  card.appendChild(nav);

  const state = { index: 0, flipped: false };

  function render() {
    const c = deck.cards[state.index];
    const label = state.flipped ? "Answer" : "Question";
    const text = state.flipped ? c.back : c.front;
    face.innerHTML = `<div><span class="side-label">${label}</span>${renderMarkdown(text)}</div>`;
    counter.textContent = `${state.index + 1} / ${deck.cards.length}`;
    prevBtn.disabled = state.index === 0;
    nextBtn.disabled = state.index === deck.cards.length - 1;
  }

  face.addEventListener("click", () => {
    state.flipped = !state.flipped;
    render();
  });
  prevBtn.addEventListener("click", () => {
    if (state.index === 0) return;
    state.index -= 1;
    state.flipped = false;
    render();
  });
  nextBtn.addEventListener("click", () => {
    if (state.index === deck.cards.length - 1) return;
    state.index += 1;
    state.flipped = false;
    render();
  });

  render();
  thread.appendChild(card);
  thread.scrollTop = thread.scrollHeight;
  return card;
}

async function handleFlashcardsCommand(raw) {
  addCard("user", raw);
  messageInput.value = "";
  renderComposerHighlight();
  setSending(true);

  const pending = addThinkingCard();
  try {
    const res = await fetch(`${API_BASE}/flashcards`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `status ${res.status}`);

    pending.remove();
    addFlashcardDeckCard(data);
  } catch (err) {
    pending.className = "card error";
    pending.textContent = `Couldn't generate flashcards: ${err.message}`;
  } finally {
    setSending(false);
    messageInput.focus();
  }
}

// --- Quiz mode + retry-exam: /quiz and /retry, not buttons -- see backend's
// POST /quiz, POST /quiz/results, POST /retry-quiz. Same /command shape as
// /flashcards above: intercepted in the submit handler below, never reaches
// the normal chat loop, hits its own endpoint(s) and renders its own
// interactive card instead of a streamed reply. Unlike flashcards, this one
// talks to the backend a second time after it renders: once every question
// is answered (grading itself already happened client-side, immediately,
// the instant each one was answered), POST /quiz/results records what was
// actually gotten right vs wrong so a later /retry can generate a
// follow-up quiz targeting exactly that.
const QUIZ_COMMAND_RE = /^\/quiz\b/i;
const RETRY_COMMAND_RE = /^\/retry\b/i;

function addQuizCard(deck) {
  const card = document.createElement("div");
  card.className = "card assistant quiz-deck";

  const header = document.createElement("div");
  header.className = "quiz-header";
  const icon = deck.source === "retry" ? "🔁" : "📝";
  header.innerHTML = `<span class="title">${icon} ${escapeHtml(deck.title)}</span><span>${deck.questions.length} questions</span>`;
  card.appendChild(header);

  const body = document.createElement("div");
  card.appendChild(body);

  const nav = document.createElement("div");
  nav.className = "quiz-nav";
  const prevBtn = document.createElement("button");
  prevBtn.type = "button";
  prevBtn.textContent = "← Prev";
  const counter = document.createElement("span");
  counter.className = "counter";
  const nextBtn = document.createElement("button");
  nextBtn.type = "button";
  nextBtn.textContent = "Next →";
  nav.append(prevBtn, counter, nextBtn);
  card.appendChild(nav);

  const finishBtn = document.createElement("button");
  finishBtn.type = "button";
  finishBtn.className = "quiz-finish-btn";
  card.appendChild(finishBtn);

  // answers[i] is null until question i is answered, then locked to the
  // chosen option index -- clicking an already-answered question's options
  // again is a no-op (see the click handler below), so a choice can't be
  // changed after the fact once it's been graded.
  const state = { index: 0, answers: new Array(deck.questions.length).fill(null) };

  function answeredCount() {
    return state.answers.filter((a) => a !== null).length;
  }

  function renderFinishBtn() {
    const total = deck.questions.length;
    const done = answeredCount();
    finishBtn.disabled = done < total;
    finishBtn.textContent = done < total
      ? `Answer all questions to finish (${done}/${total})`
      : "Finish quiz ✓";
  }

  function renderQuestion() {
    const q = deck.questions[state.index];
    const chosen = state.answers[state.index];

    let html = `<div class="quiz-question">${renderMarkdown(q.question)}</div><div class="quiz-options">`;
    q.options.forEach((opt, i) => {
      const letter = String.fromCharCode(65 + i);
      let cls = "quiz-option";
      if (chosen !== null) {
        if (i === chosen && i === q.correct_index) cls += " chosen-correct";
        else if (i === chosen) cls += " chosen-wrong";
        else if (i === q.correct_index) cls += " reveal-correct";
      }
      html += `<button type="button" class="${cls}" data-i="${i}" ${chosen !== null ? "disabled" : ""}><span class="letter">${letter}</span><p>${renderMarkdown(opt)}</p></button>`;
    });
    html += `</div>`;
    if (chosen !== null) {
      html += `<div class="quiz-explanation">${renderMarkdown(q.explanation)}</div>`;
    }
    body.innerHTML = html;

    body.querySelectorAll(".quiz-option").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (state.answers[state.index] !== null) return;
        state.answers[state.index] = Number(btn.dataset.i);
        renderQuestion();
        renderFinishBtn();
      });
    });

    counter.textContent = `${state.index + 1} / ${deck.questions.length}`;
    prevBtn.disabled = state.index === 0;
    nextBtn.disabled = state.index === deck.questions.length - 1;
  }

  prevBtn.addEventListener("click", () => {
    if (state.index === 0) return;
    state.index -= 1;
    renderQuestion();
  });
  nextBtn.addEventListener("click", () => {
    if (state.index === deck.questions.length - 1) return;
    state.index += 1;
    renderQuestion();
  });

  function renderSummary(score, answers) {
    nav.classList.add("hidden");
    finishBtn.classList.add("hidden");

    const total = deck.questions.length;
    const wrongCount = total - score;
    let html = `<div class="quiz-summary"><div class="score">${score} / ${total}</div><div class="score-detail">correct</div><ul class="quiz-summary-list">`;
    deck.questions.forEach((q, i) => {
      const correct = answers[i].chosen_index === q.correct_index;
      const mark = correct ? `<span class="mark correct">✓</span>` : `<span class="mark wrong">✗</span>`;
      html += `<li>${mark}<span>${escapeHtml(q.question)}</span></li>`;
    });
    html += `</ul>`;
    if (wrongCount > 0) {
      html += `<p class="retry-hint">Missed ${wrongCount}? Type <code>/retry</code> for a focused follow-up quiz.</p>`;
    }
    html += `</div>`;
    body.innerHTML = html;
  }

  finishBtn.addEventListener("click", async () => {
    if (answeredCount() < deck.questions.length) return;
    const answers = deck.questions.map((q, i) => ({
      question: q.question,
      options: q.options,
      correct_index: q.correct_index,
      chosen_index: state.answers[i],
    }));
    const score = answers.filter((a) => a.chosen_index === a.correct_index).length;

    try {
      await fetch(`${API_BASE}/quiz/results`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, topic: deck.title, answers }),
      });
    } catch (err) {
      // best-effort -- the score below is already known client-side
      // either way, and /retry falls back to a clear "nothing recorded
      // yet" message rather than failing outright if this never landed.
    }

    renderSummary(score, answers);
  });

  renderQuestion();
  renderFinishBtn();
  thread.appendChild(card);
  thread.scrollTop = thread.scrollHeight;
  return card;
}

async function handleQuizCommand(raw) {
  addCard("user", raw);
  messageInput.value = "";
  renderComposerHighlight();
  setSending(true);

  const pending = addThinkingCard();
  try {
    const res = await fetch(`${API_BASE}/quiz`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `status ${res.status}`);

    pending.remove();
    addQuizCard(data);
  } catch (err) {
    pending.className = "card error";
    pending.textContent = `Couldn't generate a quiz: ${err.message}`;
  } finally {
    setSending(false);
    messageInput.focus();
  }
}

async function handleRetryCommand(raw) {
  addCard("user", raw);
  messageInput.value = "";
  renderComposerHighlight();
  setSending(true);

  const pending = addThinkingCard();
  try {
    const res = await fetch(`${API_BASE}/retry-quiz`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `status ${res.status}`);

    pending.remove();
    addQuizCard(data);
  } catch (err) {
    pending.className = "card error";
    pending.textContent = `Couldn't generate a retry quiz: ${err.message}`;
  } finally {
    setSending(false);
    messageInput.focus();
  }
}

// /depth and /exam are the only way to reach either mode now (the older
// mode-bar buttons and eli5/teach modes were removed entirely, frontend
// and backend) and apply for exactly the message they're typed into --
// passed straight to sendMessage as this request's mode, with no sticky
// mode state anywhere to fall back to. The leading command is NOT
// stripped out -- the literal text the student typed is what gets shown
// in their chat bubble and what's sent as the message, exactly like
// /flashcards already works. The mode prompt itself (see agent.py) tells
// the model that a leading "/depth"/"/exam" token is just the trigger,
// not part of the question, so nothing downstream gets confused by it
// either.
const CHAT_COMMAND_RE = /^\/(depth|exam)\b/i;

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  closeSuggestions();
  const value = messageInput.value.trim();
  if (!value) return;

  if (FLASHCARDS_COMMAND_RE.test(value)) {
    handleFlashcardsCommand(value);
    return;
  }

  if (QUIZ_COMMAND_RE.test(value)) {
    handleQuizCommand(value);
    return;
  }

  if (RETRY_COMMAND_RE.test(value)) {
    handleRetryCommand(value);
    return;
  }

  const cmdMatch = value.match(CHAT_COMMAND_RE);
  sendMessage(value, cmdMatch ? cmdMatch[1].toLowerCase() : undefined);
});

stopBtn.addEventListener("click", () => {
  if (activeController) activeController.abort();
});

resetBtn.addEventListener("click", async () => {
  let summary = null;
  try {
    const res = await fetch(`${API_BASE}/chat/${sessionId}?summarize=true`, { method: "DELETE" });
    if (res.ok) {
      const data = await res.json();
      summary = data.summary;
    }
  } catch (err) {
    // best-effort -- worst case the backend keeps one orphaned session entry
  }

  thread.innerHTML = "";
  if (summary) {
    const card = document.createElement("div");
    card.className = "card recap";
    card.innerHTML = `<p class="recap-label">Before we reset — a quick recap:</p>${renderMarkdown(summary)}`;
    thread.appendChild(card);
  }
  addCard("assistant", "Conversation reset. Ask me anything about your syllabi.");
});

// --- Add-a-syllabus panel: tab switching + both submit flows -------------

addCourseBtn.addEventListener("click", () => {
  addCourseForm.classList.toggle("hidden");
  addCourseStatus.textContent = "";
});

function showTextTab() {
  tabTextBtn.classList.add("active");
  tabPdfBtn.classList.remove("active");
  addCourseTextForm.classList.remove("hidden");
  addCoursePdfForm.classList.add("hidden");
  addCourseStatus.textContent = "";
}

function showPdfTab() {
  tabPdfBtn.classList.add("active");
  tabTextBtn.classList.remove("active");
  addCoursePdfForm.classList.remove("hidden");
  addCourseTextForm.classList.add("hidden");
  addCourseStatus.textContent = "";
}

tabTextBtn.addEventListener("click", showTextTab);
tabPdfBtn.addEventListener("click", showPdfTab);

cancelAddCourse.addEventListener("click", () => {
  addCourseForm.classList.add("hidden");
  addCourseTextForm.reset();
  addCourseStatus.textContent = "";
});

cancelAddCoursePdf.addEventListener("click", () => {
  addCourseForm.classList.add("hidden");
  addCoursePdfForm.reset();
  addCourseStatus.textContent = "";
});

addCourseTextForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const course_code = newCourseCode.value.trim();
  const course_name = newCourseName.value.trim();
  const content = newCourseContent.value.trim();

  if (!course_code || !course_name || !content) {
    addCourseStatus.textContent = "Fill in all three fields.";
    addCourseStatus.className = "add-course-status error";
    return;
  }

  addCourseStatus.textContent = "Saving…";
  addCourseStatus.className = "add-course-status";

  try {
    const res = await fetch(`${API_BASE}/courses`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ course_code, course_name, content }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `status ${res.status}`);

    addCourseStatus.textContent = `${data.course_code} saved.`;
    addCourseStatus.className = "add-course-status success";
    addCourseTextForm.reset();
    setTimeout(() => addCourseForm.classList.add("hidden"), 900);
    loadCourses();
  } catch (err) {
    addCourseStatus.textContent = err.message;
    addCourseStatus.className = "add-course-status error";
  }
});

addCoursePdfForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const course_code = pdfCourseCode.value.trim();
  const course_name = pdfCourseName.value.trim();
  const file = pdfFile.files[0];

  if (!course_code || !course_name || !file) {
    addCourseStatus.textContent = "Fill in the code, name, and choose a PDF.";
    addCourseStatus.className = "add-course-status error";
    return;
  }
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    addCourseStatus.textContent = "Please choose a .pdf file.";
    addCourseStatus.className = "add-course-status error";
    return;
  }

  addCourseStatus.textContent = "Uploading & indexing…";
  addCourseStatus.className = "add-course-status";

  const formData = new FormData();
  formData.append("course_code", course_code);
  formData.append("course_name", course_name);
  formData.append("append", pdfAppend.checked ? "true" : "false");
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/courses/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `status ${res.status}`);

    addCourseStatus.textContent = `${data.course_code} indexed (${data.page_count || 0} page(s)).`;
    addCourseStatus.className = "add-course-status success";
    addCoursePdfForm.reset();
    setTimeout(() => addCourseForm.classList.add("hidden"), 1200);
    loadCourses();
  } catch (err) {
    addCourseStatus.textContent = err.message;
    addCourseStatus.className = "add-course-status error";
  }
});

// --- My Notebook panel: upload/cancel/delete -------------------------------
// Same form shape as the course-PDF upload above, but posts to /notebook
// with this browser's device_id instead of a shared course_code -- see
// backend/notebook_store.py.

addNotebookBtn.addEventListener("click", () => {
  addNotebookForm.classList.toggle("hidden");
  addNotebookStatus.textContent = "";
});

cancelAddNotebook.addEventListener("click", () => {
  addNotebookForm.classList.add("hidden");
  addNotebookForm.reset();
  addNotebookStatus.textContent = "";
});

addNotebookForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const title = notebookTitle.value.trim();
  const file = notebookFile.files[0];

  if (!title || !file) {
    addNotebookStatus.textContent = "Give it a title and choose a PDF.";
    addNotebookStatus.className = "add-course-status error";
    return;
  }
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    addNotebookStatus.textContent = "Please choose a .pdf file.";
    addNotebookStatus.className = "add-course-status error";
    return;
  }

  addNotebookStatus.textContent = "Uploading & indexing…";
  addNotebookStatus.className = "add-course-status";

  const formData = new FormData();
  formData.append("device_id", deviceId);
  formData.append("title", title);
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/notebook/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `status ${res.status}`);

    addNotebookStatus.textContent = `${data.title} indexed (${data.page_count || 0} page(s)).`;
    addNotebookStatus.className = "add-course-status success";
    addNotebookForm.reset();
    setTimeout(() => addNotebookForm.classList.add("hidden"), 1200);
    loadNotebook();
  } catch (err) {
    addNotebookStatus.textContent = err.message;
    addNotebookStatus.className = "add-course-status error";
  }
});

renderComposerHighlight();
loadCourses();
loadNotebook();
