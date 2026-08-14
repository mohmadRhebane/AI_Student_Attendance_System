import enum

class AttendanceMode(enum.Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"

class AttendanceStatus(enum.Enum):
    PRESENT = "present"
    ABSENT = "absent"
    LATE = "late"