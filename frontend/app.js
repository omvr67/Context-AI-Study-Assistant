const API_BASE = "http://127.0.0.1:8000";

const sessionId = crypto.randomUUID();
let activeCourse = null; // null = search across all courses

const thread = document.getElementById("thread");
const composer = document.getElementById("composer");
const messageInput = document.getElementById("messageInput");
const courseList = document.getElementById("courseList");
const activeCourseTab = document.getElementById("activeCourseTab");
const resetBtn = document.getElementById("resetBtn");

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

function renderChip(course, isAll) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "course-chip" + (isAll ? " active" : "");
  chip.innerHTML = `<span class="code">${course.course_code}</span><span class="name">${course.course_name}</span>`;
  chip.addEventListener("click", () => {
    document.querySelectorAll(".course-chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    activeCourse = isAll ? null : course.course_code;
    activeCourseTab.textContent = isAll
      ? "All courses"
      : `${course.course_code} — ${course.course_name}`;
  });
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

  const pending = addCard("assistant", "Thinking…");

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
    pending.innerHTML = renderMarkdown(data.response);
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
  try {
    await fetch(`${API_BASE}/chat/${sessionId}`, { method: "DELETE" });
  } catch (err) {
    // best-effort -- worst case the backend keeps one orphaned session entry
  }
  thread.innerHTML = "";
  addCard("assistant", "Conversation reset. Ask me anything about your syllabi.");
});

loadCourses();
