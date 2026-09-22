const API_BASE = "http://127.0.0.1:8000";

const sessionId = crypto.randomUUID();
let activeCourse = null; // null = search across all courses
let currentMode = "";    // "" = normal, or "eli5" | "exam" | "teach"
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
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const stopBtn = document.getElementById("stopBtn");
const courseList = document.getElementById("courseList");
const activeCourseTab = document.getElementById("activeCourseTab");
const resetBtn = document.getElementById("resetBtn");
const modeBar = document.getElementById("modeBar");

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
  const textEl = card.querySelector(".streamed-text");
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

  if (!data.response) {
    card.querySelector(".streamed-text").innerHTML = renderMarkdown("(no response)");
  }

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
  eli5: "🧒 ELI5",
  exam: "📝 Exam Mode",
  teach: "🎓 Teach Me",
};

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

function renderNotebookChip(doc) {
  const chip = document.createElement("div");
  chip.className = "course-chip";

  const main = document.createElement("div");
  main.className = "course-chip-main";
  main.innerHTML = `<span class="code">📓 ${doc.page_count} page${doc.page_count === 1 ? "" : "s"}</span><span class="name">${doc.title}</span>`;
  chip.appendChild(main);

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

// --- Sending a message, streamed ----------------------------------------

function setSending(isSending) {
  messageInput.disabled = isSending;
  sendBtn.disabled = isSending;
  sendBtn.classList.toggle("hidden", isSending);
  stopBtn.classList.toggle("hidden", !isSending);
}

async function sendMessage(message) {
  addCard("user", message);
  messageInput.value = "";
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
        mode: currentMode || null,
        device_id: deviceId,
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

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const value = messageInput.value.trim();
  if (!value) return;
  sendMessage(value);
});

stopBtn.addEventListener("click", () => {
  if (activeController) activeController.abort();
});

modeBar.addEventListener("click", (e) => {
  const btn = e.target.closest(".mode-btn");
  if (!btn) return;
  document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  currentMode = btn.dataset.mode || "";
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

loadCourses();
loadNotebook();
