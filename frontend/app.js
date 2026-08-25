const API_BASE = "http://127.0.0.1:8000";

const sessionId = crypto.randomUUID();
let activeCourse = null; // null = search across all courses

const thread = document.getElementById("thread");
const composer = document.getElementById("composer");
const messageInput = document.getElementById("messageInput");
const courseList = document.getElementById("courseList");
const activeCourseTab = document.getElementById("activeCourseTab");
const resetBtn = document.getElementById("resetBtn");

const addCourseBtn = document.getElementById("addCourseBtn");
const addCourseForm = document.getElementById("addCourseForm");
const cancelAddCourse = document.getElementById("cancelAddCourse");
const addCourseStatus = document.getElementById("addCourseStatus");
const newCourseCode = document.getElementById("newCourseCode");
const newCourseName = document.getElementById("newCourseName");
const newCourseContent = document.getElementById("newCourseContent");

// Rotated randomly so repeated questions don't feel like a canned response.
const THINKING_PHRASES = [
  "Checking the syllabus…",
  "Thinking it through…",
  "Running the numbers…",
  "Working on it…",
];

const TOOL_LABELS = {
  search_syllabus: "📖 Searched syllabus",
  gpa_impact_simulator: "🧮 Calculated GPA",
  generate_study_schedule: "🗓️ Built a schedule",
};

function renderMarkdown(text) {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  return escaped
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
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

function addAssistantCard(data) {
  const card = document.createElement("div");
  card.className = "card assistant";
  card.innerHTML = renderMarkdown(data.response);

  const meta = document.createElement("div");
  meta.className = "card-meta";

  if (data.grounded) {
    const badge = document.createElement("span");
    badge.className = "badge grounded";
    badge.textContent = "✓ Grounded in syllabus";
    meta.appendChild(badge);
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
      p.innerHTML = `<span class="source-code">${src.course_code}</span> ${src.snippet}`;
      details.appendChild(p);
    });
    card.appendChild(details);
  }

  thread.appendChild(card);
  thread.scrollTop = thread.scrollHeight;
  return card;
}

function renderChip(course, isAll) {
  const chip = document.createElement("div");
  chip.className = "course-chip" + (isAll ? " active" : "");

  const main = document.createElement("button");
  main.type = "button";
  main.className = "course-chip-main";
  main.innerHTML = `<span class="code">${course.course_code}</span><span class="name">${course.course_name}</span>`;
  main.addEventListener("click", () => {
    document.querySelectorAll(".course-chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    activeCourse = isAll ? null : course.course_code;
    activeCourseTab.textContent = isAll
      ? "All courses"
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
          activeCourseTab.textContent = "All courses";
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
    courseList.appendChild(renderChip({ course_code: "ALL", course_name: "All courses" }, true));
    for (const course of courses) {
      courseList.appendChild(renderChip(course, false));
    }
  } catch (err) {
    courseList.innerHTML = `<p class="loading">Couldn't reach the backend at ${API_BASE}. Is uvicorn running?</p>`;
  }
}

async function sendMessage(message) {
  addCard("user", message);
  messageInput.value = "";
  messageInput.disabled = true;
  composer.querySelector("button").disabled = true;

  const pending = addThinkingCard();

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message,
        course_code: activeCourse,
      }),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `status ${res.status}`);
    }

    const data = await res.json();
    pending.remove();
    addAssistantCard(data);
  } catch (err) {
    pending.className = "card error";
    pending.textContent = `Request failed: ${err.message}`;
  } finally {
    messageInput.disabled = false;
    composer.querySelector("button").disabled = false;
    messageInput.focus();
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const value = messageInput.value.trim();
  if (!value) return;
  sendMessage(value);
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

addCourseBtn.addEventListener("click", () => {
  addCourseForm.classList.toggle("hidden");
  addCourseStatus.textContent = "";
});

cancelAddCourse.addEventListener("click", () => {
  addCourseForm.classList.add("hidden");
  addCourseForm.reset();
  addCourseStatus.textContent = "";
});

addCourseForm.addEventListener("submit", async (e) => {
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
    addCourseForm.reset();
    setTimeout(() => addCourseForm.classList.add("hidden"), 900);
    loadCourses();
  } catch (err) {
    addCourseStatus.textContent = err.message;
    addCourseStatus.className = "add-course-status error";
  }
});

loadCourses();
