"""
SyllabusAssistantAgent: a pure-LangChain, tool-calling agent with
per-session memory.

Architecture mirrors Session 3's MemoryAugmentedAgentExecutor
(SystemMessage / HumanMessage / AIMessage / ToolMessage + .bind_tools()),
generalized to serve multiple concurrent chat sessions behind a FastAPI
backend instead of a single notebook-global chat_history list.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

GUARDRAIL_SYSTEM_PROMPT = """You are the ConnectX Course Syllabus & Exam Assistant, a study companion for enrolled students.

Rules you must always follow:
1. For any question about grading breakdowns, exam dates, attendance policy, or lecture topics, call the
   search_syllabus tool before answering. Never answer syllabus questions from memory alone.
2. If search_syllabus returns "NOT_FOUND" or content that does not actually answer the question, respond
   exactly with: "I don't see that in the syllabus -- please check with your Teaching Assistant." Do not
   guess or fall back on outside knowledge.
3. When a student asks about their GPA or how a grade would affect it, call the gpa_impact_simulator tool
   rather than doing the arithmetic yourself.
4. When a student asks for a study plan or revision schedule, first use search_syllabus to find the
   relevant lecture topics if they weren't already given to you, then call generate_study_schedule with
   those topics.
5. Keep answers concise and cite the course code (e.g. "[CS301]") when quoting a policy.
"""


class SyllabusAssistantAgent:
    """Wraps an LLM + tool set with independent chat history per session_id."""

    def __init__(self, llm, tools, system_prompt: str = GUARDRAIL_SYSTEM_PROMPT, max_turns: int = 5):
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

    def chat(self, session_id: str, user_input: str) -> str:
        history = self._history(session_id)
        history.append(HumanMessage(content=user_input))

        for _ in range(self.max_turns):
            try:
                ai_msg: AIMessage = self.llm_with_tools.invoke(history)
            except Exception as e:
                return f"Sorry, I hit an error talking to the model: {e}"

            history.append(ai_msg)

            if not getattr(ai_msg, "tool_calls", None):
                return ai_msg.content

            for tool_call in ai_msg.tool_calls:
                tool_obj = self.tools_map.get(tool_call["name"])
                if tool_obj is None:
                    observation = f"Error: tool '{tool_call['name']}' is not registered."
                else:
                    observation = tool_obj.invoke(tool_call["args"])
                history.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

        return "I'm having trouble finishing that request -- could you rephrase or ask one thing at a time?"
