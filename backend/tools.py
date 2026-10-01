from langchain_core.tools import tool
import json
import random
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

@tool
def add_balance(tien: int) -> dict:
    """Nạp tiền vào "balance" trong wallet.json bằng số tiền "tien" khi user gọi nạp tiền.

    Returns:
        dict: Số dư sao khi đã nạp bằng VND.
    """
    balance = get_wallet_balance()

    with open(WALLET_PATH, "w", encoding="utf-8") as f:
        json.dump({"balance": balance + tien}, f, ensure_ascii=False, indent=2)

    return {
        "status": "success",
        "balance": balance + tien,
        "formatted_balance": f"{balance + tien:,} VND".replace(",", ".")
    }

AIRPORT_MAPPING = {
    "hà nội": "HAN", "ha noi": "HAN", "hn": "HAN", "han": "HAN", "nội bài": "HAN", "noi bai": "HAN",
    "đà nẵng": "DAD", "da nang": "DAD", "đn": "DAD", "dn": "DAD", "dad": "DAD",
    "hồ chí minh": "SGN", "tp hcm": "SGN", "tphcm": "SGN", "sài gòn": "SGN", "sai gon": "SGN", "sgn": "SGN", "tân sơn nhất": "SGN", "tan son nhat": "SGN"
}

def find_schedule_entry(data, query: str | None):
    if not query:
        return None
    q = str(query).strip().lower()
    target_mapped = DAY_MAPPING.get(q, q).lower()

    for entry in data:
        e_date = entry.get("date", "").lower()
        e_fdate = entry.get("formatted_date", "").lower()
        e_sdate = entry.get("short_date", "").lower()
        e_day_en = entry.get("day_of_week", "").lower()
        e_day_vn = entry.get("day_vn", "").lower()

        date_candidates = [e_date, e_fdate, e_sdate, e_fdate[:5], e_sdate.replace("/2026", "")]
        if q in date_candidates:
            return entry
        if target_mapped in [e_day_en, e_day_vn] or q in [e_day_en, e_day_vn]:
            return entry

    return None

class SmartSearchInput(BaseModel):
    date: str | None = Field(default=None, description="Ngày bay cụ thể (VD: '01/09/2026', '2026-09-01', '1/9/2026', '1/9')")
    day: str | None = Field(default=None, description="Ngày hoặc thứ cần tìm (VD: '01/09/2026', '1/9', 'Thứ 3')")
    time_period: str | None = Field(default=None, description="Khung giờ bay mong muốn nếu có")
    destination: str | None = Field(default=None, description="Điểm đến (VD: 'HAN', 'SGN', 'Đà Nẵng')")
    departure: str | None = Field(default=None, description="Điểm xuất phát (VD: 'HAN', 'SGN', 'Hà Nội')")
    seat_class: str = Field(default="ECONOMY", description="Hạng vé (mặc định ECONOMY)")

@tool(args_schema=SmartSearchInput)
def smart_flight_search(
    date: str | None = None,
    day: str | None = None,
    time_period: str | None = None,
    destination: str | None = None,
    departure: str | None = None,
    seat_class: str = "ECONOMY"
) -> dict:
    """Tra cứu danh sách các chuyến bay theo ngày bay (VD: 01/09/2026, 1/9/2026), điểm đi, điểm đến và hạng vé."""
    try:
        data = load_schedule_data()
        query_date = date or day
        if not query_date:
            return {"status": "error", "message": "Vui lòng cung cấp ngày bay cần tra cứu (ví dụ: '01/09/2026', '1/9/2026')."}

        day_entry = find_schedule_entry(data, query_date)
        if not day_entry:
            return {"status": "not_found", "message": f"Không tìm thấy lịch bay cho ngày {query_date} trong hệ thống."}

        flights = day_entry.get("flights", [])
        dep_code = AIRPORT_MAPPING.get(departure.strip().lower(), departure.strip().upper()) if departure else None
        dest_code = AIRPORT_MAPPING.get(destination.strip().lower(), destination.strip().upper()) if destination else None
        target_class = CLASS_MAPPING.get(seat_class.strip().lower(), seat_class.strip().upper())

        matched_flights = []
        for flight in flights:
            route = flight.get("route", {})
            f_from = route.get("from", "")
            f_to = route.get("to", "")

            if dep_code and f_from != dep_code:
                continue
            if dest_code and f_to != dest_code:
                continue

            chosen_class = None
            for c in flight.get("classes", []):
                if c.get("type", "").upper() == target_class:
                    chosen_class = c
                    break

            matched_flights.append({
                "flight_id": flight.get("flight_id"),
                "airline": flight.get("airline"),
                "route": f"{f_from} -> {f_to}",
                "date": day_entry.get("formatted_date"),
                "day": day_entry.get("day_vn"),
                "dep_time": flight.get("dep_time"),
                "arr_time": flight.get("arr_time"),
                "seat_class": target_class,
                "price": chosen_class.get("price") if chosen_class else None,
                "seats_left": chosen_class.get("seats_left") if chosen_class else 0,
            })

        return {
            "status": "success",
            "date": day_entry.get("formatted_date"),
            "day": day_entry.get("day_vn"),
            "total_flights": len(matched_flights),
            "flights": matched_flights,
        }
    except Exception as e:
        return {"status": "error", "message": f"Lỗi khi tra cứu chuyến bay: {str(e)}"}

class BookFlightInput(BaseModel):
    flight_id: str = Field(description="Mã chuyến bay muốn đặt (ví dụ: VN102, VJ154, QH202)")
    date: str | None = Field(default=None, description="Ngày bay nếu có (VD: '01/09/2026', '2026-09-01', '1/9')")
    passengers: list[str] | str | None = Field(
        default=None,
        description="Danh sách hoặc chuỗi họ tên các hành khách cần đặt vé (ví dụ: ['Nguyễn Văn A', 'Nguyễn Văn B'])"
    )
    passenger_name: str | None = Field(
        default=None,
        description="Họ tên hành khách (dùng nếu chỉ có 1 hành khách, ví dụ: 'Nguyễn Văn A')"
    )
    seat_class: str = Field(default="ECONOMY", description="Hạng vé: ECONOMY (Phổ thông), BUSINESS (Thương gia), hoặc SKYBOSS")

@tool(args_schema=BookFlightInput)
def book_flight(
    flight_id: str,
    date: str | None = None,
    passengers: list[str] | str | None = None,
    passenger_name: str | None = None,
    seat_class: str = "ECONOMY"
) -> dict:
    """Đặt vé máy bay cho một hoặc nhiều khách hàng và thực hiện trừ tiền tương ứng từ ví tài khoản.

    Args:
        flight_id (str): Mã chuyến bay (VD: VN102, VJ154).
        date: Ngày bay (VD: '01/09/2026', '2026-09-01', '1/9').
        passengers: Danh sách tên hành khách (1 hoặc nhiều người).
        passenger_name: Tên 1 hành khách nếu có.
        seat_class (str): Hạng vé (ECONOMY, BUSINESS, SKYBOSS).
    Returns:
        dict: Kết quả đặt vé, danh sách mã PNR của từng hành khách, tổng tiền bị trừ và số dư còn lại.
    """
    try:
        passenger_list = []
        if passengers:
            if isinstance(passengers, list):
                passenger_list.extend([str(p).strip() for p in passengers if str(p).strip()])
            else:
                passenger_list.extend([p.strip() for p in str(passengers).split(",") if p.strip()])
        if passenger_name and passenger_name.strip() and passenger_name.strip() not in passenger_list:
            passenger_list.append(passenger_name.strip())

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

        if date:
            target_entry = find_schedule_entry(data, date)
            if target_entry:
                for flight in target_entry.get("flights", []):
                    if flight.get("flight_id", "").upper() == flight_id_norm:
                        target_flight = flight
                        break

        if not target_flight:
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

        seats_left = chosen_class.get("seats_left", 0)
        if seats_left < num_passengers:
            return {
                "status": "sold_out",
                "message": f"Rất tiếc, chuyến bay {flight_id} hạng {target_class} chỉ còn {seats_left} chỗ trống, không đủ cho {num_passengers} hành khách!"
            }

        unit_price = chosen_class.get("price", 0)
        total_price = unit_price * num_passengers
        current_balance = get_wallet_balance()

        if current_balance < total_price:
            return {
                "status": "insufficient_funds",
                "message": f"Số dư ví không đủ thanh toán cho {num_passengers} vé! Đơn giá: {unit_price:,} VND/vé, Tổng tiền cần thanh toán: {total_price:,} VND, Số dư hiện tại: {current_balance:,} VND. Thiếu: {(total_price - current_balance):,} VND.".replace(",", ".")
            }

        new_balance = current_balance - total_price
        set_wallet_balance(new_balance)
        chosen_class["seats_left"] -= num_passengers
        save_schedule_data(data)

        booked_tickets = []
        for name in passenger_list:
            pnr = f"PNR{random.randint(100000, 999999)}"
            booking = {
                "pnr": pnr,
                "flight_id": target_flight.get("flight_id"),
                "airline": target_flight.get("airline"),
                "route": target_flight.get("route"),
                "date": target_entry.get("formatted_date", target_entry.get("date", "")) if target_entry else "",
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
            "date": target_entry.get("formatted_date") if target_entry else "",
            "day": target_entry.get("day_vn") if target_entry else "",
            "route": f"{target_flight.get('route', {}).get('from')} -> {target_flight.get('route', {}).get('to')}",
            "time": f"{target_flight.get('dep_time')} - {target_flight.get('arr_time')}",
            "seat_class": target_class,
            "unit_price": f"{unit_price:,} VND".replace(",", "."),
            "total_price_deducted": f"{total_price:,} VND".replace(",", "."),
            "remaining_balance": f"{new_balance:,} VND".replace(",", "."),
        }

    except Exception as e:
        return {"status": "error", "message": f"Lỗi hệ thống khi đặt vé: {str(e)}"}