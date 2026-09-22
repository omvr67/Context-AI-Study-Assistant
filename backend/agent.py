"""
SyllabusAssistantAgent: a pure-LangChain, tool-calling agent with
per-session memory.

Architecture mirrors Session 3's MemoryAugmentedAgentExecutor
(SystemMessage / HumanMessage / AIMessage / ToolMessage + .bind_tools()),
generalized to serve multiple concurrent chat sessions behind a FastAPI
backend instead of a single notebook-global chat_history list.

On top of the base loop, this version:
  - tags each incoming message with a lightweight, no-LLM-call intent
    hint so the model leans toward the right tool instead of reflexively
    calling search_syllabus for GPA/schedule questions;
  - tracks which tools fired and, for search_syllabus specifically, which
    syllabus chunks were actually retrieved, so the API layer can report
    whether an answer was grounded and show its sources;
  - exposes summarize() so a session can get a short recap right before
    it's reset instead of just vanishing.

v1.1 additions:
  - explicit interaction modes (ELI5 / Exam / Teach Me This Chapter),
    injected as an ephemeral system message for the current turn only --
    never persisted into session history, so the mode applies exactly to
    the request that asked for it and nothing "sticks" silently. Grounding
    rules 1-4 in GUARDRAIL_SYSTEM_PROMPT are unaffected by any mode.
  - chat_stream(): a generator twin of chat() for the streaming endpoint.
    Tool-call resolution works exactly like chat() (same turn-by-turn
    .invoke()-equivalent loop); the one part that streams token-by-token
    is the model's final, tool-free answer, since that's the only part a
    student is actually waiting to read live. See the inline note in
    chat_stream for how it tells a "this turn is calling a tool" stream
    apart from a "this turn is the real answer" stream before deciding
    whether to forward tokens.

Rate-limiting addition:
  - every Groq call (chat()'s invoke, chat_stream()'s stream, and
    summarize()'s invoke) goes through _invoke_with_backoff /
    _stream_with_backoff, which retries with exponential backoff on a
    429 and otherwise fails immediately. See backend/rate_limit.py for
    the separate per-visitor request throttle enforced in main.py.
"""
import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

GUARDRAIL_SYSTEM_PROMPT = """You are the ConnectX Course Syllabus & Exam Assistant -- a sharp, encouraging
study partner for enrolled students, not a generic chatbot. Be confident, concise, and specific;
when you cite a policy, weave the course code naturally into the sentence (e.g. "CS301's final is
worth 35% of your grade") instead of just tagging it on at the end.

Rules you must always follow:
1. For any question about grading breakdowns, exam dates, attendance policy, or lecture topics, call the
   search_syllabus tool before answering. Never answer syllabus questions from memory alone.
2. A search_notebook tool may also be available this turn -- a private space for PDFs only this student
   has personally uploaded, kept separate from the shared syllabi above. If it's available and the
   student references "my notes", "the PDF I uploaded", "my practice exam", or similar, use it directly.
   If search_syllabus comes back NOT_FOUND for something and search_notebook is available, try that too
   before giving up -- it may be covered there instead. Only once both come back empty (or
   search_notebook isn't available at all) should you respond exactly with: "I don't see that in the
   syllabus -- please check with your Teaching Assistant." Do not guess or fall back on outside knowledge
   either way.
3. When a student asks about their GPA or how a grade would affect it, call the gpa_impact_simulator tool
   directly. When a student states a *target* GPA and asks what grades they'd need, call
   gpa_target_planner instead. Don't call search_syllabus first unless they're also asking about a
   specific course policy. Never do GPA arithmetic yourself.
4. When a student asks for a study plan or revision schedule and gives you a course code, call
   build_ai_study_plan first -- it pulls the topics (and, given an exam date, the day count) straight
   from the syllabus automatically. Only fall back to search_syllabus + generate_study_schedule if
   build_ai_study_plan reports it couldn't find a structured topic list, or the student gives you an
   explicit topic list of their own instead of a course code.
5. Messages may start with a bracketed hint like "[Likely tool: ...]" -- treat it as a suggestion from
   the interface, not a hard rule. Use your own judgment about which tool actually fits the question.
6. Keep answers concise.
"""

SUMMARY_SYSTEM_PROMPT = """Summarize the conversation below in exactly 3 short lines, each starting with
a dash. Focus on what the student asked and what was found or calculated. No preamble, no closing remarks."""

MODE_PROMPTS: dict[str, str] = {
    "eli5": """Explain Like I'm Stupid mode is ON for this turn only.
Rewrite your answer for someone with zero background: short sentences, everyday words, one idea
per sentence, and a concrete analogy if it helps. Do not skip steps or assume prior knowledge.
This changes HOW you explain, not WHAT you're allowed to say -- rules 1-4 above (grounding, tool
use) still apply exactly as written; never invent facts just to keep the explanation simple.""",

    "exam": """Exam Mode is ON for this turn. The student is actively revising, not casually
chatting. Lean toward exam-relevant framing: which topics are most heavily weighted, what's likely
to be tested, and concrete revision/practice actions. If it fits the question, offer to quiz them
with a practice question, or ask one yourself before giving the full answer. Stay grounded in the
syllabus per the rules above -- if asked for a practice question, base it on real topics/policies
you retrieved, not a fabricated specific like a made-up question number or past-paper detail.""",

    "teach": """Teach Me This Chapter mode is ON. Structure your entire response as a short
tutoring sequence, in this exact order, with clear labels:
1. Explanation -- the core idea in plain terms, grounded in the retrieved syllabus content.
2. Example -- one concrete, worked example illustrating it.
3. Understanding check -- one short question testing whether the idea landed (don't answer it
   yourself; wait for the student's reply next turn).
4. Practice -- one practice prompt or problem the student can try on their own.
Call search_syllabus first if you haven't already retrieved content for this topic -- teaching from
memory instead of the actual syllabus defeats the point of this mode.""",
}

_GPA_KEYWORDS = ("gpa", "grade point", "cumulative average", "what would my grade", "my grade be")
_GPA_TARGET_KEYWORDS = ("what gpa do i need", "reach a", "target gpa", "goal gpa", "hit a gpa")
_SCHEDULE_KEYWORDS = (
    "study plan", "study schedule", "revision plan", "revise for",
    "how should i study", "days until", "prepare for", "cram",
)

# Retry/backoff for Groq calls -- see _invoke_with_backoff and
# _stream_with_backoff. Not about cost (Groq's free tier is free); it's
# so a busy demo session doesn't turn into a hard failure the moment
# Groq's per-minute cap is briefly hit.
_MAX_RETRIES = 3
_BASE_DELAY_SECONDS = 1.0

FRIENDLY_QUOTA_MESSAGE = (
    "We've hit the shared AI quota for a moment -- please wait a few seconds and try again."
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detects a 429 from Groq without depending on the exact exception
    class the client library raises (groq's SDK, langchain's wrapper, or
    a raw httpx error can all surface this differently depending on
    version). Checking status_code plus a couple of string/name
    fallbacks catches all of them without an extra dependency.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status == 429:
        return True
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "429" in str(exc)


def _invoke_with_backoff(invoke_fn, messages, max_retries: int = _MAX_RETRIES, base_delay: float = _BASE_DELAY_SECONDS):
    """Calls invoke_fn(messages) with exponential backoff, but only for
    429s -- anything else (bad request, auth failure, etc.) fails
    immediately since retrying those just burns time for no benefit.
    Backoff schedule: base_delay, base_delay*2, base_delay*4, ...
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return invoke_fn(messages)
        except Exception as e:
            last_exc = e
            if not _is_rate_limit_error(e) or attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise last_exc  # pragma: no cover -- loop above always returns or raises


def _stream_with_backoff(stream_fn, messages, max_retries: int = _MAX_RETRIES, base_delay: float = _BASE_DELAY_SECONDS):
    """Retries the underlying .stream() call with backoff, but only while
    no chunks have been yielded yet for the current attempt -- once even
    one token has reached the caller, retrying would mean re-sending
    duplicate output, so any error after that point is left to propagate
    instead of silently duplicating partial output.
    """
    attempt = 0
    while True:
        got_any = False
        try:
            for chunk in stream_fn(messages):
                got_any = True
                yield chunk
            return
        except Exception as e:
            if got_any or not _is_rate_limit_error(e) or attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))
            attempt += 1


def _intent_hint(user_input: str) -> str | None:
    """Cheap keyword heuristic -- no LLM call -- that nudges tool choice for
    the current turn without hard-coding it. Returns None when nothing
    obvious matches, letting the model decide freely.
    """
    text = user_input.lower()
    if any(k in text for k in _GPA_TARGET_KEYWORDS):
        return "gpa_target_planner (this reads like a target-GPA planning question)"
    if any(k in text for k in _GPA_KEYWORDS):
        return "gpa_impact_simulator (this reads like a GPA question, not a syllabus lookup)"
    if any(k in text for k in _SCHEDULE_KEYWORDS):
        return "build_ai_study_plan (this reads like a study-planning question)"
    return None


def _parse_sources(observation: str) -> list[dict]:
    """Pulls {course_code, snippet, page?} entries back out of
    format_docs()'s "[CODE] content" / "[CODE|p.N] content" blocks so the
    API layer can report exactly what grounded an answer, without
    search_syllabus needing to know about HTTP responses at all.
    """
    if observation.startswith("NOT_FOUND") or observation.startswith("Syllabus search error"):
        return []

    sources = []
    for block in observation.split("\n\n"):
        block = block.strip()
        match = re.match(r"^\[([^\]]+)\]\s*(.*)$", block, re.DOTALL)
        if not match:
            continue
        tag, snippet = match.group(1).strip(), match.group(2).strip()
        if "|p." in tag:
            code, page_str = tag.split("|p.", 1)
            page = int(page_str) if page_str.isdigit() else None
        else:
            code, page = tag, None
        if len(snippet) > 220:
            snippet = snippet[:220].rsplit(" ", 1)[0] + "\u2026"
        entry: dict = {"course_code": code.strip(), "snippet": snippet}
        if page is not None:
            entry["page"] = page
        sources.append(entry)
    return sources


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    seen = {}
    for s in sources:
        seen[(s["course_code"], s.get("page"), s["snippet"])] = s
    return list(seen.values())


class SyllabusAssistantAgent:
    """Wraps an LLM + tool set with independent chat history per session_id."""

    def __init__(self, llm, tools, system_prompt: str = GUARDRAIL_SYSTEM_PROMPT, max_turns: int = 5):
        self.llm = llm  # kept tool-free for summarize()
        self.tools_map = {t.name: t for t in tools}
        self.llm_with_tools = llm.bind_tools(tools)
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._sessions: dict[str, list] = {}

    def _history(self, session_id: str) -> list:
        if session_id not in self._sessions:
            self._sessions[session_id] = [SystemMessage(content=self.system_prompt)]
        return self._sessions[session_id]

    def _tools_for_call(self, extra_tools):
        """Resolves the (bound-LLM, tools_map) pair to use for one call.

        Most turns reuse the precomputed self.llm_with_tools / self.tools_map
        built once at construction time. When extra_tools is given (e.g. a
        request-scoped search_notebook tool -- see backend/tools.py's
        make_notebook_tool and its call site in backend/main.py), a fresh
        bind is built for just this call instead, so a per-request tool
        never leaks into another session's tool set.
        """
        if not extra_tools:
            return self.llm_with_tools, self.tools_map
        combined = list(self.tools_map.values()) + list(extra_tools)
        return self.llm.bind_tools(combined), {**self.tools_map, **{t.name: t for t in extra_tools}}

    @staticmethod
    def _with_mode(history: list, mode_prompt: str | None) -> list:
        """Builds the message list actually sent to the LLM for one call:
        the persisted history, plus an ephemeral mode instruction (if any)
        injected right after the base system prompt. The mode instruction
        is never appended to `history` itself, so it applies only to the
        request that asked for it -- the next turn starts back in normal
        mode unless the caller asks for a mode again.
        """
        if not mode_prompt:
            return history
        return [history[0], SystemMessage(content=mode_prompt)] + history[1:]

    def reset(self, session_id: str) -> None:
        """Clears a session's memory; the next turn starts a fresh SystemMessage."""
        self._sessions.pop(session_id, None)

    def summarize(self, session_id: str) -> str | None:
        """Generates a short recap of a session before it's reset. Returns
        None if the session doesn't exist yet or nothing was actually said.
        """
        history = self._sessions.get(session_id)
        if not history or len(history) <= 1:
            return None

        transcript_lines = []
        for msg in history:
            if isinstance(msg, HumanMessage) and msg.content:
                transcript_lines.append(f"Student: {msg.content}")
            elif isinstance(msg, AIMessage) and msg.content:
                transcript_lines.append(f"Assistant: {msg.content}")
        if not transcript_lines:
            return None

        transcript = "\n".join(transcript_lines)[-4000:]  # keep the summarizer prompt small
        try:
            summary_msg = _invoke_with_backoff(self.llm.invoke, [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=transcript),
            ])
            return summary_msg.content.strip() or None
        except Exception:
            return None

    def chat(self, session_id: str, user_input: str, mode: str | None = None, extra_tools=None) -> dict:
        """Runs one turn of the ReAct loop.

        extra_tools: optional request-scoped tools (e.g. a device-bound
        search_notebook) that apply to this call only -- see
        _tools_for_call. Never persisted into the session's own tool set.

        Returns {"content": str, "sources": list[dict], "tools_used": list[str],
        "mode": str | None} instead of a bare string so the API layer can
        surface grounding info, which tools actually fired, and which mode
        (if any) applied to this turn.
        """
        history = self._history(session_id)

        hint = _intent_hint(user_input)
        tagged_input = f"[Likely tool: {hint}] {user_input}" if hint else user_input
        history.append(HumanMessage(content=tagged_input))

        mode_key = (mode or "").strip().lower()
        mode_prompt = MODE_PROMPTS.get(mode_key)

        llm_with_tools, tools_map = self._tools_for_call(extra_tools)

        tools_used: list[str] = []
        sources: list[dict] = []

        for _ in range(self.max_turns):
            call_messages = self._with_mode(history, mode_prompt)
            try:
                ai_msg: AIMessage = _invoke_with_backoff(llm_with_tools.invoke, call_messages)
            except Exception as e:
                content = FRIENDLY_QUOTA_MESSAGE if _is_rate_limit_error(e) else f"Sorry, I hit an error talking to the model: {e}"
                return {
                    "content": content,
                    "sources": [],
                    "tools_used": [],
                    "mode": mode_key or None,
                }

            history.append(ai_msg)

            if not getattr(ai_msg, "tool_calls", None):
                return {
                    "content": ai_msg.content,
                    "sources": _dedupe_sources(sources),
                    "tools_used": tools_used,
                    "mode": mode_key or None,
                }

            for tool_call in ai_msg.tool_calls:
                name = tool_call["name"]
                tool_obj = tools_map.get(name)
                if tool_obj is None:
                    observation = f"Error: tool '{name}' is not registered."
                else:
                    observation = tool_obj.invoke(tool_call["args"])
                    tools_used.append(name)
                    if name in ("search_syllabus", "search_notebook"):
                        sources.extend(_parse_sources(str(observation)))
                history.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

        return {
            "content": "I'm having trouble finishing that request -- could you rephrase or ask one thing at a time?",
            "sources": _dedupe_sources(sources),
            "tools_used": tools_used,
            "mode": mode_key or None,
        }

    def chat_stream(self, session_id: str, user_input: str, mode: str | None = None, extra_tools=None):
        """Generator twin of chat(): yields small event dicts as the turn
        progresses instead of returning one final dict, so the API layer can
        forward them to the client live over SSE.

        extra_tools: same meaning as in chat() -- request-scoped tools
        (e.g. a device-bound search_notebook) applying to this call only.

        Event shapes:
          {"type": "tool", "name": ...}                                  -- a tool started running
          {"type": "token", "text": ...}                                 -- a piece of the final answer
          {"type": "done", "content", "sources", "tools_used", "mode"}   -- turn finished normally
          {"type": "error", "error": ...}                                -- turn failed; stream ends

        Tool-call turns and the final tool-free answer turn are both run via
        .stream() so a network/API error surfaces mid-turn instead of only
        after a full non-streaming call would have completed -- but tokens
        are only forwarded to the caller once we're confident this turn is
        the real answer and not a tool call: the very first chunk of a turn
        tells us which, since providers emit tool_call_chunks (not prose)
        from the first chunk of a turn that's invoking a tool. That's a
        pragmatic heuristic, not a guarantee for every possible provider
        behavior -- a model that emitted a little preamble text before
        deciding to call a tool could leak that fragment -- but it matches
        how this project's guardrail prompt already tells the model to
        behave (call the tool, don't narrate first), and it avoids the far
        more fragile alternative of trying to detect and retroactively
        "unsend" already-streamed tokens.
        """
        history = self._history(session_id)

        hint = _intent_hint(user_input)
        tagged_input = f"[Likely tool: {hint}] {user_input}" if hint else user_input
        history.append(HumanMessage(content=tagged_input))

        mode_key = (mode or "").strip().lower()
        mode_prompt = MODE_PROMPTS.get(mode_key)

        llm_with_tools, tools_map = self._tools_for_call(extra_tools)

        tools_used: list[str] = []
        sources: list[dict] = []

        for turn in range(self.max_turns):
            call_messages = self._with_mode(history, mode_prompt)
            is_tool_turn: bool | None = None
            chunks = []

            try:
                for chunk in _stream_with_backoff(llm_with_tools.stream, call_messages):
                    chunks.append(chunk)
                    if is_tool_turn is None:
                        is_tool_turn = bool(getattr(chunk, "tool_call_chunks", None))
                    if not is_tool_turn and chunk.content:
                        yield {"type": "token", "text": chunk.content}
            except Exception as e:
                error_msg = FRIENDLY_QUOTA_MESSAGE if _is_rate_limit_error(e) else f"Sorry, I hit an error talking to the model: {e}"
                yield {"type": "error", "error": error_msg}
                return

            if not chunks:
                yield {"type": "error", "error": "No response was generated."}
                return

            ai_msg = chunks[0]
            for c in chunks[1:]:
                ai_msg = ai_msg + c

            history.append(ai_msg)

            if not getattr(ai_msg, "tool_calls", None):
                yield {
                    "type": "done",
                    "content": ai_msg.content,
                    "sources": _dedupe_sources(sources),
                    "tools_used": tools_used,
                    "mode": mode_key or None,
                }
                return

            for tool_call in ai_msg.tool_calls:
                name = tool_call["name"]
                yield {"type": "tool", "name": name}
                tool_obj = tools_map.get(name)
                if tool_obj is None:
                    observation = f"Error: tool '{name}' is not registered."
                else:
                    observation = tool_obj.invoke(tool_call["args"])
                    tools_used.append(name)
                    if name in ("search_syllabus", "search_notebook"):
                        sources.extend(_parse_sources(str(observation)))
                history.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

        yield {
            "type": "done",
            "content": "I'm having trouble finishing that request -- could you rephrase or ask one thing at a time?",
            "sources": _dedupe_sources(sources),
            "tools_used": tools_used,
            "mode": mode_key or None,
        }
