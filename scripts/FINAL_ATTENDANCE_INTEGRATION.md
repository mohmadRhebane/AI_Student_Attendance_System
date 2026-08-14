Final Attendance AI Contract

SmartAttendanceAI is the only class the backend needs to call. The backendpasses dynamic session data and receives typed, JSON-friendly results. There isno FastAPI or database dependency inside the AI package.

Create the AI runtime once

from pathlib import Path
from smart_attendance_ai.ai_facade import SmartAttendanceAI

AI = SmartAttendanceAI(project_root=Path("D:/Project2_ArcFace_MobileFaceNet"))

Start a dynamic session

from smart_attendance_ai.ai_contracts import AttendanceSessionAIRequest

request = AttendanceSessionAIRequest(
    session_id=session_id,
    class_id=class_id,
    course_id=course_id,
    camera_id=camera_id,
    source=video_path_or_rtsp_or_camera_index,
    roster_student_ids=student_ids,
    gallery_path=None,  # None means use active_gallery.json
    max_frames=0,
    max_seconds=0.0,
    metadata={"semester_id": semester_id},
    extensions={},
)

snapshot = AI.start_attendance(request)

start_attendance returns immediately. Poll with:

snapshot = AI.get_attendance_status(session_id)

Stop with:

snapshot = AI.stop_attendance(session_id)

Read the final result after COMPLETED, FAILED, or CANCELLED:

result = AI.get_attendance_result(session_id)
payload = result.to_dict()

Backend storage helpers

for record in result.iter_attendance_records():
    database.save_attendance(record)

for event in result.iter_event_records():
    database.save_recognition_event(event)

Extensibility

metadata: business information owned by the backend.

extensions: optional future AI fields.

contract_version: identifies the public result schema.

facade_version: identifies the integration implementation.

Existing dict-style access remains available: result["status"].

Prefer attribute access for new code: result.status.

The source, roster, class, course, camera, limits, gallery path, metadata, andextensions are all provided at runtime; none are fixed in the production AIfacade.