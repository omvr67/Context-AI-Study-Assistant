"""
Sample syllabus knowledge base for the demo.

In a real deployment these would be ingested from uploaded PDFs/docx
files. Here they're inline strings so the whole project runs end-to-end
with zero uploads, mirroring the inline `raw_docs` dict from the
Assignment 2 grounded-RAG exercise.

Each Document's metadata carries `course_code` / `course_name` so the
retriever can filter to a single course when the frontend has one
selected (see `search_syllabus` in tools.py).
"""
from langchain_core.documents import Document

_CS301 = """
CS301 -- Data Structures & Algorithms

Grading breakdown: Homework 20%, Midterm Exam 25%, Final Exam 35%, Participation 10%, Final Project 10%.

Exam schedule: Midterm Exam is on October 14. Final Exam is on December 10, cumulative, covering all
lecture topics.

Attendance policy: Two unexcused absences are allowed with no penalty. Each additional unexcused absence
deducts 1% from the final grade. Excused absences require an email to the TA at least 24 hours in advance.

Lecture topics, in order:
Week 1: Arrays & Linked Lists
Week 2: Stacks & Queues
Week 3: Trees & Binary Search Trees
Week 4: Heaps & Priority Queues
Week 5: Hash Tables
Week 6: Graphs & Traversal (BFS/DFS)
Week 7: Sorting Algorithms
Week 8: Dynamic Programming
Week 9: Greedy Algorithms
Week 10: NP-Completeness
"""

_MATH210 = """
MATH210 -- Linear Algebra

Grading breakdown: Problem Sets 15%, In-Class Quizzes 15%, Midterm Exam 30%, Final Exam 40%.

Exam schedule: Midterm Exam is on October 21. Final Exam is on December 15.

Attendance policy: Attendance is not directly graded but strongly recommended. Quizzes are given in the
first 10 minutes of class and cannot be made up if a student arrives late or is absent.

Lecture topics, in order:
Week 1: Vector Spaces
Week 2: Matrix Operations
Week 3: Systems of Linear Equations
Week 4: Determinants
Week 5: Eigenvalues & Eigenvectors
Week 6: Linear Transformations
Week 7: Orthogonality & Least Squares
Week 8: Diagonalization
Week 9: Singular Value Decomposition
"""

_PSY101 = """
PSY101 -- Introduction to Psychology

Grading breakdown: Weekly Reflections 10%, Two Papers 30% combined, Midterm Exam 25%, Final Exam 35%.

Exam schedule: Midterm Exam is on October 7. Final Exam is on December 12.

Attendance policy: Attendance is mandatory and tracked via sign-in sheet. More than three absences results
in a full letter grade deduction from the final course grade.

Lecture topics, in order:
Week 1: History & Perspectives in Psychology
Week 2: Biological Bases of Behavior
Week 3: Sensation & Perception
Week 4: Learning & Conditioning
Week 5: Memory
Week 6: Cognition & Language
Week 7: Motivation & Emotion
Week 8: Personality Theories
Week 9: Psychological Disorders
Week 10: Social Psychology
"""

SYLLABUS_DOCUMENTS = [
    Document(
        page_content=_CS301.strip(),
        metadata={"course_code": "CS301", "course_name": "Data Structures & Algorithms"},
    ),
    Document(
        page_content=_MATH210.strip(),
        metadata={"course_code": "MATH210", "course_name": "Linear Algebra"},
    ),
    Document(
        page_content=_PSY101.strip(),
        metadata={"course_code": "PSY101", "course_name": "Introduction to Psychology"},
    ),
]
