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


def _parse_weights(topic_weights: str, n_topics: int) -> list[float]:
    """Parses a comma-separated weight string parallel to the topic list.
    Falls back to equal weights (1.0 each) for anything malformed --
    wrong count, non-numeric, negative, or all-zero.
    """
    if not topic_weights.strip():
        return [1.0] * n_topics
    raw = [w.strip() for w in topic_weights.split(",") if w.strip()]
    try:
        values = [float(w) for w in raw]
    except ValueError:
        return [1.0] * n_topics
    if len(values) != n_topics or any(v < 0 for v in values) or sum(values) == 0:
        return [1.0] * n_topics
    return values


def _allocate_days(days_until_exam: int, weights: list[float]) -> list[int]:
    """Largest-remainder allocation: hands out whole days to topics roughly
    proportional to their weight, so heavier-weighted topics reliably get
    more days than lighter ones instead of every topic getting an equal
    round-robin slice.
    """
    n = len(weights)
    total_weight = sum(weights)
    raw_shares = [days_until_exam * (w / total_weight) for w in weights]

    if days_until_exam >= n:
        # Guarantee every topic at least one day when there's room for it.
        base = [max(1, int(share)) for share in raw_shares]
    else:
        # Not enough days to cover every topic -- some may get zero.
        base = [int(share) for share in raw_shares]

    remainders = [share - int(share) for share in raw_shares]
    order = sorted(range(n), key=lambda i: remainders[i], reverse=True)

    remaining = days_until_exam - sum(base)
    i = 0
    while remaining > 0:
        base[order[i % n]] += 1
        remaining -= 1
        i += 1
    while sum(base) > days_until_exam:
        idx = max(range(n), key=lambda i: base[i])
        if base[idx] > 0:
            base[idx] -= 1
        else:
            break
    return base


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
        topic_weights: str = "",
    ) -> str:
        """Builds a day-by-day revision plan that spreads a list of exam
        topics across the days remaining, allocating a fixed number of
        study hours per day. If the topics aren't already known, call
        search_syllabus first to find the course's lecture topics -- and,
        if available, the grading breakdown, so heavier-weighted material
        can get proportionally more days via topic_weights.

        Args:
            topics: Comma-separated lecture topics to revise, e.g.
                "Trees & BSTs, Hash Tables, Graphs & Traversal".
            days_until_exam: Number of days left before the exam.
            hours_per_day: Study hours available per day (default 2.0).
            topic_weights: Optional comma-separated numbers, parallel to
                topics, representing each topic's relative importance --
                e.g. exam-section percentages pulled from the grading
                breakdown like "40,35,25". Heavier-weighted topics get
                proportionally more days. Leave empty to split days
                evenly across topics.
        """
        topic_list = [t.strip() for t in topics.split(",") if t.strip()]
        if not topic_list:
            return "Error: no topics were provided."
        if days_until_exam <= 0:
            return "Error: days_until_exam must be at least 1."
        if hours_per_day <= 0:
            return "Error: hours_per_day must be a positive number."

        weights = _parse_weights(topic_weights, len(topic_list))
        day_counts = _allocate_days(days_until_exam, weights)
        weighted = bool(topic_weights.strip()) and weights != [1.0] * len(topic_list)

        total_hours = days_until_exam * hours_per_day
        summary_lines = [
            f"Study plan across {days_until_exam} day(s), {hours_per_day:g}h/day "
            f"({total_hours:g}h total), {'weighted by exam importance' if weighted else 'split evenly'}:"
        ]
        for topic, day_count, weight in zip(topic_list, day_counts, weights):
            share_pct = 100 * weight / sum(weights)
            summary_lines.append(
                f"  - {topic}: {day_count} day(s), {day_count * hours_per_day:g}h ({share_pct:.0f}% of plan)"
            )

        schedule_lines = []
        day_number = 1
        for topic, day_count in zip(topic_list, day_counts):
            for _ in range(day_count):
                schedule_lines.append(f"Day {day_number}: {topic} ({hours_per_day:g}h)")
                day_number += 1

        return "\n".join(summary_lines) + "\n\n" + "\n".join(schedule_lines)

    return [search_syllabus, gpa_impact_simulator, generate_study_schedule]
