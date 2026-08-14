Smart Attendance AI Facade

The backend imports only SmartAttendanceAI. It does not import detector,recognizer, gallery, enrollment, or session-manager internals.

Initialize once when the backend process starts

from pathlib import Path
from smart_attendance_ai.ai_facade import SmartAttendanceAI

AI = SmartAttendanceAI(project_root=Path(__file__).resolve().parents[1])

The SCRFD and ArcFace models are loaded once and shared by enrollment andattendance.

Enroll all images for one student in one call

result = AI.enroll_student(
    student_id="STU_016",
    full_name="Student Name",
    image_paths=image_paths,
    mode="NEW",
)

Store result.student_template and every record returned byresult.iter_embedding_records() in the database. Each embedding is anumpy.float32 vector with 512 values.

This call does not modify the active Gallery cache.

Build a cache version after the database transaction succeeds

cache = AI.build_gallery_version(
    enrollment=result,
    update_policy="APPEND",
    approve_reviewed_package=False,
    activate_after_build=False,
)

Run regression tests before activation. Then activate explicitly:

AI.activate_gallery_version(cache.version_id)

Start attendance

snapshot = AI.start_attendance(
    session_id="SESSION_001",
    class_id="CLASS_A",
    course_id="COURSE_101",
    source="rtsp://camera/stream",
    roster_student_ids=["STU_001", "STU_002"],
)

Poll:

status = AI.get_attendance_status("SESSION_001")

Read final structured result:

result = AI.get_attendance_result("SESSION_001")

The backend stores the returned attendance records and events.

Shutdown

AI.shutdown()