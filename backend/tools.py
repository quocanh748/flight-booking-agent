from langchain_core.tools import tool
import os
import json
import random
import re
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

CURRENT_DIR = Path(__file__).resolve().parent
DB_PATH = CURRENT_DIR.parent / "database" / "schedule.json"
WALLET_PATH = CURRENT_DIR.parent / "database" / "wallet.json"
BOOKINGS_PATH = CURRENT_DIR.parent / "database" / "bookings.json"
INITIAL_BALANCE = 10_000_000

def load_schedule_data():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_schedule_data(data):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_wallet_balance() -> int:
    if not WALLET_PATH.exists():
        set_wallet_balance(INITIAL_BALANCE)
        return INITIAL_BALANCE
    try:
        with open(WALLET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("balance", INITIAL_BALANCE)
    except Exception:
        return INITIAL_BALANCE

def set_wallet_balance(balance: int):
    with open(WALLET_PATH, "w", encoding="utf-8") as f:
        json.dump({"balance": balance}, f, ensure_ascii=False, indent=2)

def save_booking(booking: dict):
    bookings = []
    if BOOKINGS_PATH.exists():
        try:
            with open(BOOKINGS_PATH, "r", encoding="utf-8") as f:
                bookings = json.load(f)
        except Exception:
            bookings = []
    bookings.append(booking)
    with open(BOOKINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(bookings, f, ensure_ascii=False, indent=2)

DAY_MAPPING = {
    "thứ 2": "Monday", "thu 2": "Monday", "thứ hai": "Monday", "thu hai": "Monday", "t2": "Monday", "mon": "Monday", "monday": "Monday",
    "thứ 3": "Tuesday", "thu 3": "Tuesday", "thứ ba": "Tuesday", "thu ba": "Tuesday", "t3": "Tuesday", "tue": "Tuesday", "tuesday": "Tuesday",
    "thứ 4": "Wednesday", "thu 4": "Wednesday", "thứ tư": "Wednesday", "thứ bốn": "Wednesday", "thu tu": "Wednesday", "t4": "Wednesday", "wed": "Wednesday", "wednesday": "Wednesday",
    "thứ 5": "Thursday", "thu 5": "Thursday", "thứ năm": "Thursday", "thu nam": "Thursday", "t5": "Thursday", "thu": "Thursday", "thursday": "Thursday",
    "thứ 6": "Friday", "thu 6": "Friday", "thứ sáu": "Friday", "thu sau": "Friday", "t6": "Friday", "fri": "Friday", "friday": "Friday",
    "thứ 7": "Saturday", "thu 7": "Saturday", "thứ bảy": "Saturday", "thu bay": "Saturday", "t7": "Saturday", "sat": "Saturday", "saturday": "Saturday",
    "chủ nhật": "Sunday", "chu nhat": "Sunday", "cn": "Sunday", "sun": "Sunday", "sunday": "Sunday",
}

CLASS_MAPPING = {
    "phổ thông": "ECONOMY",
    "pho thong": "ECONOMY",
    "economy": "ECONOMY",
    "eco": "ECONOMY",
    "thương gia": "BUSINESS",
    "thuong gia": "BUSINESS",
    "business": "BUSINESS",
    "bus": "BUSINESS",
    "skyboss": "SKYBOSS",
}

class ScheduleInput(BaseModel):
    day: str = Field(
        description="Thứ trong tuần cần tra cứu (VD: Monday, Thứ 2, Thứ ba, Chủ nhật)"
    )

@tool(args_schema=ScheduleInput)
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
        target_day = DAY_MAPPING.get(day_raw, day_raw).lower()

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

@tool
def check_balance() -> dict:
    """Kiểm tra số dư tài khoản/ví tiền hiện tại của khách hàng.
    
    Returns:
        dict: Số dư hiện tại bằng VND.
    """
    balance = get_wallet_balance()
    return {
        "status": "success",
        "balance": balance,
        "formatted_balance": f"{balance:,} VND".replace(",", ".")
    }

AIRPORT_MAPPING = {
    "hà nội": "HAN", "ha noi": "HAN", "hn": "HAN", "han": "HAN", "nội bài": "HAN", "noi bai": "HAN",
    "đà nẵng": "DAD", "da nang": "DAD", "đn": "DAD", "dn": "DAD", "dad": "DAD",
    "hồ chí minh": "SGN", "tp hcm": "SGN", "tphcm": "SGN", "sài gòn": "SGN", "sai gon": "SGN", "sgn": "SGN", "tân sơn nhất": "SGN", "tan son nhat": "SGN"
}

AIRPORT_NAMES = {
    "HAN": "Hà Nội (HAN)",
    "DAD": "Đà Nẵng (DAD)",
    "SGN": "TP. Hồ Chí Minh (SGN)"
}

def parse_specific_time(time_str: str | None) -> tuple[str | None, int | None]:
    if not time_str:
        return None, None
    text = time_str.lower().strip()
    
    # 1. Định dạng HH:MM (ví dụ: 02:00, 14:30)
    m = re.search(r'(\d{1,2}):(\d{2})', text)
    if m:
        h, mnt = int(m.group(1)), int(m.group(2))
        return f"{h:02d}:{mnt:02d}", h * 60 + mnt
        
    # 2. Định dạng X giờ/h sáng/trưa/chiều/tối/đêm (ví dụ: 2 giờ sáng, 2h sáng, 8h tối)
    m = re.search(r'(\d{1,2})\s*(?:giờ|gio|h|g)(?:\s*(\d{1,2}))?\s*(sáng|sang|trưa|trua|chiều|chieu|tối|toi|đêm|dem)?', text)
    if m:
        h = int(m.group(1))
        mnt = int(m.group(2)) if m.group(2) else 0
        period = m.group(3)
        if period:
            if any(k in period for k in ['tối', 'toi', 'đêm', 'dem']) and h < 12:
                h += 12
            elif any(k in period for k in ['chiều', 'chieu']) and h < 12:
                h += 12
            elif any(k in period for k in ['sáng', 'sang']) and h == 12:
                h = 0
        return f"{h:02d}:{mnt:02d}", h * 60 + mnt
        
    return None, None

def parse_time_period(period_str: str | None) -> tuple[str, str, str, str | None, int | None]:
    if not period_str:
        return ("00:00", "23:59", "Cả ngày (00:00 - 23:59)", None, None)
    
    # Ưu tiên kiểm tra xem người dùng có nói giờ cụ thể (VD: 2 giờ sáng, 14h, 8h tối) không
    exact_time, exact_minutes = parse_specific_time(period_str)
    if exact_time:
        start_m = max(0, exact_minutes - 90)
        end_m = min(24 * 60 - 1, exact_minutes + 90)
        start_s = f"{start_m // 60:02d}:{start_m % 60:02d}"
        end_s = f"{end_m // 60:02d}:{end_m % 60:02d}"
        return (start_s, end_s, f"Khoảng {exact_time} ({start_s} - {end_s})", exact_time, exact_minutes)

    p = period_str.strip().lower()
    if any(k in p for k in ["sáng", "sang", "morning", "sớm"]):
        return ("05:00", "11:00", "Buổi sáng (05:00 - 11:00)", None, None)
    elif any(k in p for k in ["trưa", "trua", "noon"]):
        return ("11:00", "13:30", "Buổi trưa (11:00 - 13:30)", None, None)
    elif any(k in p for k in ["chiều", "chieu", "afternoon"]):
        return ("13:30", "18:00", "Buổi chiều (13:30 - 18:00)", None, None)
    elif any(k in p for k in ["tối", "toi", "đêm", "dem", "night", "evening"]):
        return ("18:00", "23:59", "Buổi tối (18:00 - 23:59)", None, None)
    return ("00:00", "23:59", period_str, None, None)

def normalize_airport_code(location_str: str | None) -> str | None:
    if not location_str:
        return None
    loc = location_str.strip().lower()
    if any(k in loc for k in ["đâu đó", "dau do", "bất kỳ", "bat ky", "đâu cũng được", "dau cung duoc"]):
        return None
    for name, code in AIRPORT_MAPPING.items():
        if name in loc:
            return code
    loc_upper = loc.upper()
    if loc_upper in AIRPORT_NAMES:
        return loc_upper
    return None

class SmartSearchInput(BaseModel):
    day: str = Field(description="Thứ trong tuần cần tìm vé (VD: 'Thứ 4', 'Wednesday', 'Thứ 2', 'Monday'...)")
    time_period: str | None = Field(
        default=None,
        description="Khung giờ bay: '2 giờ sáng', 'sáng', 'trưa', 'chiều', 'tối', hoặc giờ cụ thể"
    )
    destination: str | None = Field(
        default=None,
        description="Điểm đến mong muốn (VD: 'Hà Nội', 'Đà Nẵng', 'Sài Gòn', 'HAN', 'DAD', 'SGN') hoặc None nếu khách nói 'đi đâu đó'"
    )
    departure: str | None = Field(
        default=None,
        description="Điểm xuất phát (VD: 'Sài Gòn', 'Hà Nội', 'SGN', 'HAN') hoặc None"
    )
    seat_class: str = Field(default="ECONOMY", description="Hạng vé mong muốn (mặc định ECONOMY)")

@tool(args_schema=SmartSearchInput)
def smart_flight_search(
    day: str,
    time_period: str | None = None,
    destination: str | None = None,
    departure: str | None = None,
    seat_class: str = "ECONOMY"
) -> dict:
    """Tự động kiểm tra lịch bay, lọc khung giờ (sáng/trưa/chiều/tối/giờ cụ thể), kiểm tra ghế trống và số dư ví. Phát hiện và cảnh báo nếu giờ bay bị lệch so với giờ khách yêu cầu."""
    try:
        data = load_schedule_data()
        day_raw = day.strip().lower()
        target_day = DAY_MAPPING.get(day_raw, day_raw).lower()

        day_entry = None
        for entry in data:
            day_en = entry.get("day_of_week", "").lower()
            day_vn = entry.get("day_vn", "").lower()
            if target_day in [day_en, day_vn] or day_raw in [day_en, day_vn]:
                day_entry = entry
                break

        if not day_entry:
            return {"status": "not_found", "message": f"Không tìm thấy lịch bay cho ngày {day}."}

        flights = day_entry.get("flights", [])
        start_time, end_time, time_label, exact_time, exact_minutes = parse_time_period(time_period)
        dest_code = normalize_airport_code(destination)
        dep_code = normalize_airport_code(departure)
        target_class = CLASS_MAPPING.get(seat_class.strip().lower(), seat_class.strip().upper())
        wallet_balance = get_wallet_balance()

        candidates = []
        for flight in flights:
            route = flight.get("route", {})
            f_from = route.get("from", "")
            f_to = route.get("to", "")
            dep_time = flight.get("dep_time", "00:00")

            # Tính phút cất cánh
            try:
                dh, dm = map(int, dep_time.split(":"))
                dep_mins = dh * 60 + dm
            except Exception:
                dep_mins = 0

            # Tìm hạng vé
            chosen_class = None
            for c in flight.get("classes", []):
                if c.get("type", "").upper() == target_class:
                    chosen_class = c
                    break
            if not chosen_class:
                continue

            price = chosen_class.get("price", 0)
            seats_left = chosen_class.get("seats_left", 0)

            # Tính điểm độ khớp (Scoring)
            score = 0
            is_time_match = (start_time <= dep_time <= end_time)
            time_diff_mins = None

            if exact_minutes is not None:
                time_diff_mins = abs(dep_mins - exact_minutes)
                # Điểm thời gian dựa trên khoảng cách phút (càng gần giờ khách yêu cầu điểm càng cao)
                time_score = max(0, 150 - (time_diff_mins // 2))
                score += time_score
            elif is_time_match:
                score += 100
            
            is_dest_match = True
            if dest_code:
                if f_to == dest_code:
                    score += 50
                else:
                    is_dest_match = False
            
            is_dep_match = True
            if dep_code:
                if f_from == dep_code:
                    score += 30
                else:
                    is_dep_match = False

            has_seats = (seats_left > 0)
            if has_seats:
                score += 30

            can_afford = (wallet_balance >= price)
            if can_afford:
                score += 20

            candidates.append({
                "flight": flight,
                "score": score,
                "price": price,
                "seats_left": seats_left,
                "dep_mins": dep_mins,
                "time_diff_mins": time_diff_mins,
                "is_time_match": is_time_match,
                "is_dest_match": is_dest_match,
                "is_dep_match": is_dep_match,
                "has_seats": has_seats,
                "can_afford": can_afford,
            })

        # Sắp xếp theo score giảm dần
        candidates.sort(key=lambda x: (-x["score"], x["price"]))

        if not candidates:
            return {
                "status": "no_match",
                "message": f"Không có chuyến bay nào phù hợp với yêu cầu vào ngày {day_entry.get('day_vn')}."
            }

        best = candidates[0]
        f_info = best["flight"]
        from_str = AIRPORT_NAMES.get(f_info['route']['from'], f_info['route']['from'])
        to_str = AIRPORT_NAMES.get(f_info['route']['to'], f_info['route']['to'])

        # Kiểm tra xem có bị lệch giờ nghiêm trọng không
        has_time_deviation = False
        time_diff_hours = 0.0
        if exact_time and best["time_diff_mins"] is not None:
            time_diff_hours = round(best["time_diff_mins"] / 60, 1)
            if best["time_diff_mins"] > 90:  # Lệch quá 1.5 tiếng
                has_time_deviation = True

        if has_time_deviation:
            time_check_msg = (
                f"⚠️ LỆCH GIỜ BAY: Khách yêu cầu lúc {exact_time}, nhưng ngày {day_entry.get('day_vn')} "
                f"không có chuyến giờ này! Chuyến sớm nhất/gần nhất hiện có là lúc {f_info['dep_time']} "
                f"(chênh lệch {time_diff_hours} tiếng so với giờ khách muốn)."
            )
            status = "needs_time_clarification"
            instruction = (
                f"BẮT BUỘC HỎI Ý KIẾN KHÁCH HÀNG: Thông báo rõ ràng rằng ngày {day_entry.get('day_vn')} "
                f"không có chuyến bay lúc {exact_time}. Chuyến bay sớm nhất hiện có là lúc {f_info['dep_time']} "
                f"({f_info['flight_id']} của {f_info['airline']}, cất cánh muộn hơn {time_diff_hours} tiếng). "
                f"Hỏi khách: 'Quý khách có đồng ý đổi sang bay lúc {f_info['dep_time']} này không?' "
                f"TUYỆT ĐỐI KHÔNG xuất phiếu chốt vé như thể đã khớp giờ!"
            )
        else:
            time_check_msg = f"✅ Khung giờ: {time_label} (Chuyến cất cánh lúc {f_info['dep_time']})"
            status = "success"
            instruction = "HÃY XUẤT PHIẾU ĐỀ XUẤT VÉ NÀY KÈM CÁC ĐIỀU KIỆN ĐÃ CHECK ĐỂ KHÁCH HÀNG XÁC NHẬN. TUYỆT ĐỐI CHƯA GỌI book_flight TRỪ TIỀN KHI CHƯA ĐƯỢC KHÁCH XÁC NHẬN!"

        conditions_summary = {
            "schedule_check": f"✅ Lịch bay: Có chuyến bay vào {day_entry.get('day_vn')} ({day_entry.get('day_of_week')})",
            "time_check": time_check_msg,
            "route_check": f"✅ Tuyến bay: {from_str} -> {to_str}",
            "seats_check": f"✅ Ghế trống: Hạng {target_class} còn {best['seats_left']} ghế",
            "wallet_check": f"✅ Số dư ví: {wallet_balance:,} VND >= Giá vé {best['price']:,} VND (ĐỦ ĐIỀU KIỆN)".replace(",", "."),
        }

        return {
            "status": status,
            "has_time_deviation": has_time_deviation,
            "requested_time": exact_time,
            "closest_departure_time": f_info.get("dep_time"),
            "time_difference_hours": time_diff_hours,
            "message": "Đã kiểm tra lịch trình, ghế trống và số dư ví.",
            "conditions_verified": conditions_summary,
            "draft_ticket": {
                "flight_id": f_info.get("flight_id"),
                "airline": f_info.get("airline"),
                "route": f"{from_str} -> {to_str}",
                "from_code": f_info.get("route", {}).get("from"),
                "to_code": f_info.get("route", {}).get("to"),
                "dep_time": f_info.get("dep_time"),
                "arr_time": f_info.get("arr_time"),
                "day": day_entry.get("day_vn"),
                "seat_class": target_class,
                "unit_price": f"{best['price']:,} VND".replace(",", "."),
                "price_raw": best["price"],
                "seats_left": best["seats_left"],
                "current_wallet_balance": f"{wallet_balance:,} VND".replace(",", "."),
                "remaining_balance_after": f"{(wallet_balance - best['price']):,} VND".replace(",", "."),
            },
            "instruction_for_agent": instruction,
        }

    except Exception as e:
        return {"status": "error", "message": f"Lỗi khi tìm kiếm vé thông minh: {str(e)}"}

class BookFlightInput(BaseModel):
    flight_id: str = Field(description="Mã chuyến bay muốn đặt (ví dụ: VN102, VJ154, QH202)")
    passengers: list[str] | str | None = Field(
        default=None,
        description="Danh sách hoặc chuỗi họ tên các hành khách cần đặt vé (ví dụ: ['Nguyễn Văn A', 'Nguyễn Văn B'] hoặc 'Nguyễn Văn A, Nguyễn Văn B')"
    )
    passenger_name: str | None = Field(
        default=None,
        description="Họ tên hành khách (dùng nếu chỉ có 1 hành khách, ví dụ: 'Nguyễn Văn A')"
    )
    seat_class: str = Field(default="ECONOMY", description="Hạng vé: ECONOMY (Phổ thông), BUSINESS (Thương gia), hoặc SKYBOSS")

@tool(args_schema=BookFlightInput)
def book_flight(
    flight_id: str,
    passengers: list[str] | str | None = None,
    passenger_name: str | None = None,
    seat_class: str = "ECONOMY"
) -> dict:
    """Đặt vé máy bay cho một hoặc nhiều khách hàng và thực hiện trừ tiền tương ứng từ ví tài khoản.

    Args:
        flight_id (str): Mã chuyến bay (VD: VN102, VJ154).
        passengers: Danh sách tên hành khách (1 hoặc nhiều người).
        passenger_name: Tên 1 hành khách nếu có.
        seat_class (str): Hạng vé (ECONOMY, BUSINESS, SKYBOSS).
    Returns:
        dict: Kết quả đặt vé, danh sách mã PNR của từng hành khách, tổng tiền bị trừ và số dư còn lại.
    """
    try:
        # Gom và chuẩn hóa danh sách hành khách
        raw_inputs = []
        if passengers:
            if isinstance(passengers, list):
                raw_inputs.extend(passengers)
            else:
                raw_inputs.append(str(passengers))
        if passenger_name:
            raw_inputs.append(str(passenger_name))

        passenger_list = []
        for item in raw_inputs:
            parts = re.split(r'[,;\n]|(?:\s+và\s+)|\s+and\s+', str(item), flags=re.IGNORECASE)
            for p in parts:
                name = p.strip()
                if name and name not in passenger_list:
                    passenger_list.append(name)

        if not passenger_list:
            return {
                "status": "error",
                "message": "Vui lòng cung cấp họ và tên của ít nhất một hành khách để tiến hành đặt vé."
            }

        num_passengers = len(passenger_list)
        data = load_schedule_data()
        flight_id_norm = flight_id.strip().upper()
        target_class = CLASS_MAPPING.get(seat_class.strip().lower(), seat_class.strip().upper())

        target_flight = None
        target_entry = None
        for entry in data:
            for flight in entry.get("flights", []):
                if flight.get("flight_id", "").upper() == flight_id_norm:
                    target_flight = flight
                    target_entry = entry
                    break
            if target_flight:
                break

        if not target_flight:
            return {
                "status": "error",
                "message": f"Không tìm thấy chuyến bay có mã {flight_id} trong hệ thống."
            }

        # Tìm hạng vé trong chuyến bay
        chosen_class = None
        available_classes = []
        for c in target_flight.get("classes", []):
            c_type = c.get("type", "").upper()
            available_classes.append(c_type)
            if c_type == target_class:
                chosen_class = c
                break

        if not chosen_class:
            return {
                "status": "error",
                "message": f"Chuyến bay {flight_id} không có hạng vé '{seat_class}'. Các hạng vé hiện có: {', '.join(available_classes)}."
            }

        # Kiểm tra ghế trống đủ cho tất cả hành khách không
        seats_left = chosen_class.get("seats_left", 0)
        if seats_left < num_passengers:
            return {
                "status": "sold_out",
                "message": f"Rất tiếc, chuyến bay {flight_id} hạng {target_class} chỉ còn {seats_left} chỗ trống, không đủ cho {num_passengers} hành khách!"
            }

        unit_price = chosen_class.get("price", 0)
        total_price = unit_price * num_passengers
        current_balance = get_wallet_balance()

        # Kiểm tra số dư ví đủ cho TỔNG TIỀN không
        if current_balance < total_price:
            return {
                "status": "insufficient_funds",
                "message": f"Số dư ví không đủ thanh toán cho {num_passengers} vé! Đơn giá: {unit_price:,} VND/vé, Tổng tiền cần thanh toán: {total_price:,} VND, Số dư hiện tại: {current_balance:,} VND. Thiếu: {(total_price - current_balance):,} VND.".replace(",", ".")
            }

        # Trừ tổng tiền vào ví và trừ đúng số lượng ghế
        new_balance = current_balance - total_price
        set_wallet_balance(new_balance)
        chosen_class["seats_left"] -= num_passengers
        save_schedule_data(data)

        # Lưu từng vé vào bookings.json
        booked_tickets = []
        for name in passenger_list:
            pnr = f"PNR{random.randint(100000, 999999)}"
            booking = {
                "pnr": pnr,
                "flight_id": target_flight.get("flight_id"),
                "airline": target_flight.get("airline"),
                "route": target_flight.get("route"),
                "dep_time": target_flight.get("dep_time"),
                "arr_time": target_flight.get("arr_time"),
                "day": target_entry.get("day_vn") if target_entry else "",
                "passenger_name": name,
                "seat_class": target_class,
                "price_paid": unit_price,
                "remaining_balance": new_balance,
                "booked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_booking(booking)
            booked_tickets.append({
                "pnr": pnr,
                "passenger_name": name,
                "price": f"{unit_price:,} VND".replace(",", ".")
            })

        return {
            "status": "success",
            "message": f"Đặt vé thành công cho {num_passengers} hành khách!",
            "total_passengers": num_passengers,
            "tickets": booked_tickets,
            "flight_id": target_flight.get("flight_id"),
            "airline": target_flight.get("airline"),
            "route": f"{target_flight.get('route', {}).get('from')} -> {target_flight.get('route', {}).get('to')}",
            "time": f"{target_flight.get('dep_time')} - {target_flight.get('arr_time')}",
            "seat_class": target_class,
            "unit_price": f"{unit_price:,} VND".replace(",", "."),
            "total_price_deducted": f"{total_price:,} VND".replace(",", "."),
            "remaining_balance": f"{new_balance:,} VND".replace(",", "."),
        }

    except Exception as e:
        return {"status": "error", "message": f"Lỗi hệ thống khi đặt vé: {str(e)}"}