from pathlib import Path
from smart_attendance_ai.ai_facade import SmartAttendanceAI


def create_ai(project_root: str) -> SmartAttendanceAI:
    return SmartAttendanceAI(project_root=Path(project_root))