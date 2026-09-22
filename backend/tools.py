"""
Agent tools for the Syllabus & Exam Assistant.

Follows the @tool + docstring pattern from Session 3's `calculator` and
`wikipedia_search` tools: the docstring IS the description the LLM
reads to decide when and how to call the function, and every tool
returns a plain string "observation" rather than raising.

v1.1 additions:
  - search_syllabus now goes through rag.retrieve_relevant_chunks() instead
    of raw top-k similarity search, so it can actually say "not found"
    instead of always returning its k-nearest chunks regardless of how
    weak the match is.
  - gpa_target_planner: given a goal GPA, works out the average grade
    needed across remaining credits (target-GPA planning), complementing
    gpa_impact_simulator (which only simulates one already-decided grade).
  - build_ai_study_plan: an auto-populated sibling to generate_study_schedule
    that pulls a course's topics (and, given an exam date, the day count)
    straight from its indexed syllabus instead of requiring them to be
    retyped by hand.

v1.2 addition:
  - make_notebook_tool(vectorstore, device_id): unlike make_tools() below,
    built fresh per request (see backend/main.py) once the caller's
    device_id is known, so the resulting search_notebook tool is hard-
    scoped to that one device's private uploads via closure -- never a
    model-settable argument.
"""
import re
from datetime import date, datetime

from langchain_core.tools import tool

from .rag import format_docs, get_course_full_text, retrieve_relevant_chunks

GRADE_POINTS = {
    "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0,
    "F": 0.0,
}

_TOPIC_LINE_RE = re.compile(r"(?im)^\s*week\s*\d+\s*[:\-]\s*(.+?)\s*$")


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


def _extract_topics(full_text: str) -> list[str]:
    """Pulls topic names out of a syllabus's "Week N: Topic" lines, in
    order, de-duplicated. Deliberately narrow (rather than trying to guess
    topics from arbitrary prose) so it either finds a real structured list
    or clearly reports that it didn't -- no fuzzy guessing that could
    hand the agent a made-up-looking topic list.
    """
    topics = [m.group(1).strip() for m in _TOPIC_LINE_RE.finditer(full_text)]
    seen: set[str] = set()
    ordered: list[str] = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _nearest_grade_for(required_points: float) -> str | None:
    """Returns the lowest letter grade whose GPA point value meets or
    exceeds required_points, or None if even a straight-A average isn't
    enough (caller should already have checked required_points <= 4.0).
    """
    for grade, points in sorted(GRADE_POINTS.items(), key=lambda kv: kv[1]):
        if points >= required_points - 1e-9:
            return grade
    return None


def make_tools(vectorstore):
    """
    Builds the tool list for a given FAISS vectorstore.

    The vectorstore is captured via closure so `search_syllabus` can
    query it without a global variable -- each tool stays a plain,
    inspectable function, same shape as Session 3's @tool functions.
    """

    @tool
    def search_syllabus(query: str, course_code: str = "") -> str:
        """Searches the indexed course syllabi and any uploaded PDF material
        for grading policy, exam dates, attendance rules, or lecture topics.
        Results below a minimum relevance threshold are dropped rather than
        returned as if they were a real match, and each result is tagged
        with its course code (and page number, for PDF-sourced material) so
        an answer can point back to exactly where it came from.

        Args:
            query: What to look up, e.g. "final exam date" or "attendance policy".
            course_code: Optional course code to restrict the search to a single
                course (e.g. "CS301"). Leave empty to search all courses.
        """
        try:
            docs, _best_score = retrieve_relevant_chunks(vectorstore, query, course_code=course_code, k=4)
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
    def gpa_target_planner(
        current_gpa: float,
        completed_credits: float,
        target_gpa: float,
        remaining_credits: float,
    ) -> str:
        """Calculates the average grade needed across all remaining credit
        hours to reach a target cumulative GPA. Use this whenever a student
        states a GPA goal and asks what grades they need -- as opposed to
        gpa_impact_simulator, which only simulates the effect of one
        already-decided grade rather than working backward from a goal.

        Args:
            current_gpa: The student's current cumulative GPA (e.g. 3.4).
            completed_credits: Total credit hours already completed.
            target_gpa: The cumulative GPA the student wants to reach (e.g. 3.6).
            remaining_credits: Total credit hours left before graduation (or
                before the point they want to hit the target by).
        """
        if completed_credits < 0 or remaining_credits <= 0:
            return "Error: completed_credits must be >= 0 and remaining_credits must be positive."
        if not (0.0 <= target_gpa <= 4.0):
            return "Error: target_gpa must be between 0.0 and 4.0."

        total_credits = completed_credits + remaining_credits
        current_points = current_gpa * completed_credits
        target_points = target_gpa * total_credits
        needed_points = target_points - current_points
        required_avg = needed_points / remaining_credits

        if required_avg <= 0:
            return (
                f"Your current GPA already secures at least a {target_gpa:.2f} cumulative GPA over "
                f"{total_credits:g} total credits -- even a 0.0 average across your remaining "
                f"{remaining_credits:g} credits wouldn't bring you below it."
            )
        if required_avg > 4.0:
            return (
                f"Not achievable: reaching a {target_gpa:.2f} GPA would require averaging "
                f"{required_avg:.3f} grade points across your remaining {remaining_credits:g} credits, "
                "which is above the 4.0 (straight A's) ceiling. Consider a lower target GPA, or "
                "spreading the goal across more remaining credits."
            )

        grade = _nearest_grade_for(required_avg)
        return (
            f"To reach a {target_gpa:.2f} cumulative GPA over {total_credits:g} total credits, you need "
            f"to average {required_avg:.3f} grade points across your remaining {remaining_credits:g} "
            f"credits -- roughly a {grade} average or better in everything remaining."
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
        study hours per day. Use this when you already have (or the student
        gave you) an explicit topic list; if a course_code is known instead,
        prefer build_ai_study_plan, which discovers the topics automatically.

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

    @tool
    def build_ai_study_plan(
        course_code: str,
        hours_per_day: float = 2.0,
        exam_date: str = "",
        days_until_exam: int = 0,
        topic_weights: str = "",
    ) -> str:
        """Builds a day-by-day revision plan for a specific indexed course by
        automatically discovering its lecture topics from the syllabus,
        instead of requiring them to be retyped by hand. Prefer this over
        generate_study_schedule whenever a course_code is available -- call
        search_syllabus first only if you need to double check the course
        exists or find its exam date.

        Args:
            course_code: Course to plan for, e.g. "CS301". Must already be
                indexed (a built-in course, or one added via the syllabus
                upload/paste feature).
            hours_per_day: Study hours available per day (default 2.0).
            exam_date: Optional exam date as YYYY-MM-DD. If given, the
                number of days remaining is computed automatically from
                today's date.
            days_until_exam: Days remaining before the exam. Required if
                exam_date is not given.
            topic_weights: Optional comma-separated weights parallel to the
                discovered topics (in syllabus order), same convention as
                generate_study_schedule. Leave empty for even weighting.
        """
        code = course_code.strip().upper()
        full_text = get_course_full_text(code)
        if full_text is None:
            return f"Error: '{code}' isn't an indexed course. Check the course code, or add its syllabus first."

        topics = _extract_topics(full_text)
        if not topics:
            return (
                f"Error: couldn't find a structured 'Week N: Topic' list in {code}'s syllabus, "
                "so a plan can't be auto-built from it. Use generate_study_schedule directly with "
                "topics you supply instead."
            )

        if exam_date.strip():
            try:
                parsed = datetime.strptime(exam_date.strip(), "%Y-%m-%d").date()
            except ValueError:
                return "Error: exam_date must be in YYYY-MM-DD format."
            days_until_exam = (parsed - date.today()).days
            if days_until_exam <= 0:
                return f"Error: {exam_date} has already passed (or is today) relative to today's date."
        elif days_until_exam <= 0:
            return "Error: provide either exam_date (YYYY-MM-DD) or a positive days_until_exam."

        if hours_per_day <= 0:
            return "Error: hours_per_day must be a positive number."

        weights = _parse_weights(topic_weights, len(topics))
        day_counts = _allocate_days(days_until_exam, weights)
        weighted = bool(topic_weights.strip()) and weights != [1.0] * len(topics)

        total_hours = days_until_exam * hours_per_day
        summary_lines = [
            f"Auto-built study plan for {code} -- {len(topics)} topic(s) found in the syllabus, "
            f"{days_until_exam} day(s), {hours_per_day:g}h/day ({total_hours:g}h total), "
            f"{'weighted by exam importance' if weighted else 'split evenly'}:"
        ]
        for topic, day_count, weight in zip(topics, day_counts, weights):
            share_pct = 100 * weight / sum(weights)
            summary_lines.append(
                f"  - {topic}: {day_count} day(s), {day_count * hours_per_day:g}h ({share_pct:.0f}% of plan)"
            )

        schedule_lines = []
        day_number = 1
        for topic, day_count in zip(topics, day_counts):
            for _ in range(day_count):
                schedule_lines.append(f"Day {day_number}: {topic} ({hours_per_day:g}h)")
                day_number += 1

        return "\n".join(summary_lines) + "\n\n" + "\n".join(schedule_lines)

    return [
        search_syllabus,
        gpa_impact_simulator,
        gpa_target_planner,
        generate_study_schedule,
        build_ai_study_plan,
    ]


def make_notebook_tool(vectorstore, device_id: str):
    """Builds a search_notebook tool scoped to exactly one device's private
    PDFs. Unlike make_tools() above (called once at startup, shared across
    every session), this is built fresh per chat turn in main.py, once the
    request's device_id is known -- and device_id is captured here via
    closure, not exposed as a tool argument the model could set itself. A
    tool's argument schema is something the model controls; a Python
    closure over a request-scoped value is not, so there is no way for a
    model call -- confused, or steered by adversarial content in a
    retrieved document -- to search a different device's notebook than the
    one that actually made this request.
    """

    @tool
    def search_notebook(query: str) -> str:
        """Searches the student's own private notebook -- PDFs only they
        have personally uploaded, kept separate from the shared course
        syllabi search_syllabus covers. Use this when they reference "my
        notes", "the PDF I uploaded", "my practice exam", "my document", or
        similar. If search_syllabus came back NOT_FOUND for the same topic,
        try this too before telling the student you don't see it anywhere --
        it may be covered in their own notebook instead.

        Args:
            query: What to look up in the student's own uploaded PDFs.
        """
        search_filter = {"device_id": device_id, "notebook": True}
        try:
            docs = vectorstore.similarity_search(query, k=3, filter=search_filter, fetch_k=40)
        except Exception as e:
            return f"Notebook search error: {e}"

        if not docs:
            return "NOT_FOUND: No matching content was retrieved from this device's notebook."
        return format_docs(docs)

    return search_notebook
