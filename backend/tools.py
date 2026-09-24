from langchain_core.tools import tool
import os
import json 
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
DB_PATH = (
    CURRENT_DIR.parent / "database" / "schedule.json"
)

def load_schedule_data():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

DAY_MAPPING = {
    "thứ 2": "Monday", "thứ hai": "Monday", "thu hai": "Monday", "t2": "Monday", "mon": "Monday", "monday": "Monday",
    "thứ 3": "Tuesday", "thứ ba": "Tuesday", "thu ba": "Tuesday", "t3": "Tuesday", "tue": "Tuesday", "tuesday": "Tuesday",
    "thứ 4": "Wednesday", "thứ tư": "Wednesday", "thứ bốn": "Wednesday", "thu tu": "Wednesday", "t4": "Wednesday", "wed": "Wednesday", "wednesday": "Wednesday",
    "thứ 5": "Thursday", "thứ năm": "Thursday", "thu nam": "Thursday", "t5": "Thursday", "thu": "Thursday", "thursday": "Thursday",
    "thứ 6": "Friday", "thứ sáu": "Friday", "thu sau": "Friday", "t6": "Friday", "fri": "Friday", "friday": "Friday",
    "thứ 7": "Saturday", "thứ bảy": "Saturday", "thu bay": "Saturday", "t7": "Saturday", "sat": "Saturday", "saturday": "Saturday",
    "chủ nhật": "Sunday", "chu nhat": "Sunday", "cn": "Sunday", "sun": "Sunday", "sunday": "Sunday",
}

@tool
def check_ticket_schedule(day: str) -> list | dict:
    """Tra cứu danh sách chuyến bay theo thứ hoặc ngày trong tuần.

    Args:
        day (str): Tên thứ trong tuần bằng tiếng Việt hoặc tiếng Anh (ví dụ: 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', hoặc 'Thứ 2', 'Thứ 3', 'Thứ ba', 'Chủ nhật').
    Returns:
        list | dict: Danh sách các chuyến bay của ngày đó hoặc thông báo không tìm thấy.
    """

    try:
        data = load_schedule_data()
        day_raw = day.strip().lower()
        target_day = DAY_MAPPING.get(day_raw, day_raw)

        for entry in data:
            day_en = entry.get("day_of_week", "").lower()
            day_vn = entry.get("day_vn", "").lower()

            if target_day in [day_en, day_vn] or day_raw in [day_en, day_vn]:
                return entry.get("flights", [])
            
        return {"status": "not_found", "message": f"Không tìm thấy lịch bay cho: {day}"}

    except FileNotFoundError:
        return {"status": "error", "message": f"Không tìm thấy file database tại {DB_PATH}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}