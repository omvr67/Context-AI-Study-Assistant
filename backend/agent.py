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
"""
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

GUARDRAIL_SYSTEM_PROMPT = """You are the ConnectX Course Syllabus & Exam Assistant -- a sharp, encouraging
study partner for enrolled students, not a generic chatbot. Be confident, concise, and specific;
when you cite a policy, weave the course code naturally into the sentence (e.g. "CS301's final is
worth 35% of your grade") instead of just tagging it on at the end.

Rules you must always follow:
1. For any question about grading breakdowns, exam dates, attendance policy, or lecture topics, call the
   search_syllabus tool before answering. Never answer syllabus questions from memory alone.
2. If search_syllabus returns "NOT_FOUND" or content that does not actually answer the question, respond
   exactly with: "I don't see that in the syllabus -- please check with your Teaching Assistant." Do not
   guess or fall back on outside knowledge.
3. When a student asks about their GPA or how a grade would affect it, call the gpa_impact_simulator tool
   directly -- don't call search_syllabus first unless they're also asking about a specific course policy.
   Never do GPA arithmetic yourself.
4. When a student asks for a study plan or revision schedule: first call search_syllabus to find the
   relevant lecture topics AND the grading breakdown if you don't already have them, then call
   generate_study_schedule with those topics -- and whenever the grading breakdown gives you percentages
   for the relevant exam/topics, pass them as topic_weights so heavier-weighted material gets more days.
5. Messages may start with a bracketed hint like "[Likely tool: ...]" -- treat it as a suggestion from
   the interface, not a hard rule. Use your own judgment about which tool actually fits the question.
6. Keep answers concise.
"""

SUMMARY_SYSTEM_PROMPT = """Summarize the conversation below in exactly 3 short lines, each starting with
a dash. Focus on what the student asked and what was found or calculated. No preamble, no closing remarks."""

_GPA_KEYWORDS = ("gpa", "grade point", "cumulative average", "what would my grade", "my grade be")
_SCHEDULE_KEYWORDS = (
    "study plan", "study schedule", "revision plan", "revise for",
    "how should i study", "days until", "prepare for", "cram",
)


def _intent_hint(user_input: str) -> str | None:
    """Cheap keyword heuristic -- no LLM call -- that nudges tool choice for
    the current turn without hard-coding it. Returns None when nothing
    obvious matches, letting the model decide freely.
    """
    text = user_input.lower()
    if any(k in text for k in _GPA_KEYWORDS):
        return "gpa_impact_simulator (this reads like a GPA question, not a syllabus lookup)"
    if any(k in text for k in _SCHEDULE_KEYWORDS):
        return "generate_study_schedule (this reads like a study-planning question)"
    return None


def _parse_sources(observation: str) -> list[dict]:
    """Pulls {course_code, snippet} pairs back out of format_docs()'s
    "[CODE] content" blocks so the API layer can report exactly what
    grounded an answer, without search_syllabus needing to know about
    HTTP responses at all.
    """
    if observation.startswith("NOT_FOUND") or observation.startswith("Syllabus search error"):
        return []

    sources = []
    for block in observation.split("\n\n"):
        block = block.strip()
        match = re.match(r"^\[([^\]]+)\]\s*(.*)$", block, re.DOTALL)
        if not match:
            continue
        code, snippet = match.group(1).strip(), match.group(2).strip()
        if len(snippet) > 220:
            snippet = snippet[:220].rsplit(" ", 1)[0] + "\u2026"
        sources.append({"course_code": code, "snippet": snippet})
    return sources


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
            summary_msg = self.llm.invoke([
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=transcript),
            ])
            return summary_msg.content.strip() or None
        except Exception:
            return None

    def chat(self, session_id: str, user_input: str) -> dict:
        """Runs one turn of the ReAct loop.

        Returns {"content": str, "sources": list[dict], "tools_used": list[str]}
        instead of a bare string so the API layer can surface grounding
        info and which tools actually fired for this turn.
        """
        history = self._history(session_id)

        hint = _intent_hint(user_input)
        tagged_input = f"[Likely tool: {hint}] {user_input}" if hint else user_input
        history.append(HumanMessage(content=tagged_input))

        tools_used: list[str] = []
        sources: list[dict] = []

        for _ in range(self.max_turns):
            try:
                ai_msg: AIMessage = self.llm_with_tools.invoke(history)
            except Exception as e:
                return {
                    "content": f"Sorry, I hit an error talking to the model: {e}",
                    "sources": [],
                    "tools_used": [],
                }

            history.append(ai_msg)

            if not getattr(ai_msg, "tool_calls", None):
                return {"content": ai_msg.content, "sources": _dedupe_sources(sources), "tools_used": tools_used}

            for tool_call in ai_msg.tool_calls:
                name = tool_call["name"]
                tool_obj = self.tools_map.get(name)
                if tool_obj is None:
                    observation = f"Error: tool '{name}' is not registered."
                else:
                    observation = tool_obj.invoke(tool_call["args"])
                    tools_used.append(name)
                    if name == "search_syllabus":
                        sources.extend(_parse_sources(str(observation)))
                history.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

        return {
            "content": "I'm having trouble finishing that request -- could you rephrase or ask one thing at a time?",
            "sources": _dedupe_sources(sources),
            "tools_used": tools_used,
        }


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    seen = {}
    for s in sources:
        seen[(s["course_code"], s["snippet"])] = s
    return list(seen.values())
