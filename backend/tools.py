"""
Agent tools for the Syllabus & Exam Assistant.

Follows the @tool + docstring pattern from Session 3's `calculator` and
`wikipedia_search` tools: the docstring IS the description the LLM
reads to decide when and how to call the function, and every tool
returns a plain string "observation" rather than raising.
"""
from langchain_core.tools import tool

from .rag import format_docs

GRADE_POINTS = {
    "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0,
    "F": 0.0,
}


def make_tools(vectorstore):
    """
    Builds the tool list for a given FAISS vectorstore.

    The vectorstore is captured via closure so `search_syllabus` can
    query it without a global variable -- each tool stays a plain,
    inspectable function, same shape as Session 3's @tool functions.
    """

    @tool
    def search_syllabus(query: str, course_code: str = "") -> str:
        """Searches the indexed course syllabi for grading policy, exam dates,
        attendance rules, or lecture topics.

        Args:
            query: What to look up, e.g. "final exam date" or "attendance policy".
            course_code: Optional course code to restrict the search to a single
                course (e.g. "CS301"). Leave empty to search all courses.
        """
        search_filter = {"course_code": course_code} if course_code else None
        try:
            docs = vectorstore.similarity_search(query, k=3, filter=search_filter)
        except Exception as e:
            return f"Syllabus search error: {e}"

        if not docs:
            return "NOT_FOUND: No matching syllabus content was retrieved for this query."
        return format_docs(docs)

    @tool
    def gpa_impact_simulator(
        current_gpa: float,
        completed_credits: float,
        course_credits: float,
        expected_grade: str,
    ) -> str:
        """Calculates a student's updated cumulative GPA after adding one more
        course's expected grade. Always use this instead of doing the
        arithmetic yourself.

        Args:
            current_gpa: The student's current cumulative GPA (e.g. 3.4).
            completed_credits: Total credit hours already completed.
            course_credits: Credit hours for the course being simulated.
            expected_grade: Anticipated letter grade -- one of A, A-, B+, B,
                B-, C+, C, C-, D+, D, F.
        """
        grade = expected_grade.strip().upper()
        if grade not in GRADE_POINTS:
            valid = ", ".join(GRADE_POINTS.keys())
            return f"Error: '{expected_grade}' is not a recognized letter grade. Use one of: {valid}."
        if completed_credits < 0 or course_credits <= 0:
            return "Error: credit hours must be positive numbers."

        current_quality_points = current_gpa * completed_credits
        new_quality_points = GRADE_POINTS[grade] * course_credits
        total_credits = completed_credits + course_credits
        new_gpa = (current_quality_points + new_quality_points) / total_credits

        return (
            f"Projected cumulative GPA: {new_gpa:.3f} "
            f"(was {current_gpa:.3f} over {completed_credits:g} credits; "
            f"adding a {grade} in a {course_credits:g}-credit course, "
            f"now {total_credits:g} total credits)."
        )

    @tool
    def generate_study_schedule(
        topics: str,
        days_until_exam: int,
        hours_per_day: float = 2.0,
    ) -> str:
        """Builds a day-by-day revision plan that spreads a list of exam
        topics across the days remaining, allocating a fixed number of
        study hours per day. If the topics aren't already known, call
        search_syllabus first to find the course's lecture topics.

        Args:
            topics: Comma-separated lecture topics to revise, e.g.
                "Trees & BSTs, Hash Tables, Graphs & Traversal".
            days_until_exam: Number of days left before the exam.
            hours_per_day: Study hours available per day (default 2.0).
        """
        topic_list = [t.strip() for t in topics.split(",") if t.strip()]
        if not topic_list:
            return "Error: no topics were provided."
        if days_until_exam <= 0:
            return "Error: days_until_exam must be at least 1."
        if hours_per_day <= 0:
            return "Error: hours_per_day must be a positive number."

        total_hours = days_until_exam * hours_per_day
        hours_per_topic = total_hours / len(topic_list)

        # Cycle topics across the available days so every day has a focus block;
        # topics repeat if there are more days than topics, which is fine for revision.
        schedule_lines = []
        for day in range(1, days_until_exam + 1):
            topic = topic_list[(day - 1) % len(topic_list)]
            schedule_lines.append(f"Day {day}: {topic} ({hours_per_day:g}h)")

        summary = (
            f"Study plan across {days_until_exam} day(s), {hours_per_day:g}h/day "
            f"({total_hours:g}h total), ~{hours_per_topic:.1f}h per topic across "
            f"{len(topic_list)} topic(s):\n"
        )
        return summary + "\n".join(schedule_lines)

    return [search_syllabus, gpa_impact_simulator, generate_study_schedule]
