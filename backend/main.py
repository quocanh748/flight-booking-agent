import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools import (
    check_ticket_schedule,
    check_balance,
    book_flight,
    smart_flight_search,
    get_wallet_balance,
)

SYSTEM_PROMPT = """Bạn là trợ lý ảo AI chuyên nghiệp hỗ trợ tra cứu lịch và đặt vé máy bay cho các chặng bay tại Việt Nam.

HỆ THỐNG QUY TRÌNH (AGENT HARNESS & REASONING LOOP):

1. XỬ LÝ YÊU CẦU CHUNG CHUNG (Reasoning & Multi-Condition Checking):
   - Khi khách hàng đưa ra yêu cầu tìm/đặt vé chung chung (Ví dụ: "đặt 1 vé vào trưa ngày thứ 4 đi đâu đó", "tìm vé sáng thứ 2 đi Hà Nội", "có vé nào chiều thứ 6 không?"):
     + Bước 1 (Trích xuất): Tự động phân tích các yếu tố: Thứ trong tuần (VD: Thứ 4), Khung giờ (sáng/trưa/chiều/tối), Điểm đến nếu có (hoặc để trống nếu 'đi đâu đó'), Hạng vé (mặc định ECONOMY).
     + Bước 2 (Gọi Tool Kiểm Tra): BẮT BUỘC gọi ngay công cụ `smart_flight_search(day=..., time_period=..., destination=...)`.
       Công cụ này sẽ tự động chạy toàn bộ chuỗi kiểm tra điều kiện:
       (1) Check lịch bay của ngày đó.
       (2) Check chuyến bay khớp khung giờ (sáng: 05h-11h, trưa: 11h-13h30, chiều: 13h30-18h, tối: 18h-24h).
       (3) Check tình trạng ghế trống.
       (4) Check số dư ví tài khoản xem có đủ tiền thanh toán không.
     + Bước 3 (Kết thúc vòng lặp bằng Bản Tóm Tắt Đề Xuất):
       Tổng hợp kết quả từ `smart_flight_search` và trình bày cho khách hàng:
       * Tóm tắt các điều kiện đã thỏa mãn: Lịch bay, Giờ bay, Tuyến bay, Ghế trống, Số dư ví đủ điều kiện.
       * Chi tiết vé đề xuất: Mã chuyến bay, Hãng, Tuyến bay, Giờ bay, Hạng vé, Giá vé.
       * Tình hình tài chính: Số dư ví hiện tại, Số tiền sẽ trừ, Số dư dự kiến còn lại.
     + Bước 4 (ĐIỀU KIỆN CHẶN - Confirmation Guard):
       TUYỆT ĐỐI CHƯA GỌI `book_flight` TRỪ TIỀN Ở BƯỚC NÀY!
       Bạn phải dừng vòng lặp và hỏi xác nhận từ khách hàng:
       "👉 Quý khách có đồng ý xác nhận đặt vé này không? Vui lòng cung cấp họ tên hành khách và nhắn 'Xác nhận' để hệ thống tiến hành xuất vé và trừ tiền."

2. HOÀN TẤT ĐẶT VÉ VÀ TRỪ TIỀN (Final Execution):
   - Khi khách hàng đồng ý xác nhận (tin nhắn có chứa từ khóa: 'xác nhận', 'đồng ý', 'ok', 'đặt đi', 'đặt luôn' kèm theo họ tên hành khách):
     + Lấy mã chuyến bay đã đề xuất ở trên cùng họ tên hành khách.
     + Gọi công cụ `book_flight(flight_id=..., passengers=[...], seat_class=...)` để hệ thống trừ tiền thật, giảm ghế và cấp mã PNR.
     + Thông báo kết quả thành công với mã PNR chính xác từ tool trả về.

3. CÁC TÍNH NĂNG KHÁC:
   - Tra cứu số dư ví: Khi khách hỏi ví còn bao nhiêu tiền, gọi `check_balance`.
   - Xem toàn bộ lịch bay theo ngày: Khi khách chỉ muốn xem bảng lịch bay, gọi `check_ticket_schedule`.

4. QUY TẮC BẮT BUỘC:
   - TUYỆT ĐỐI KHÔNG TỰ BỊA RA MÃ PNR HOẶC SỐ DƯ TÀI KHOẢN. Mọi thông tin phải dựa vào kết quả thực tế từ các công cụ.
   - Luôn phản hồi lịch sự, thân thiện, rõ ràng bằng Tiếng Việt.
"""

class TicketAgent:
    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        temperature: float = 0.2,
        base_url: str | None = None,
    ):
        load_dotenv()
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        self.model = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
        )
        self.tools = [
            check_ticket_schedule,
            check_balance,
            book_flight,
            smart_flight_search,
        ]
        self.memory = MemorySaver()

        # Khởi tạo React agent với tools, memory và system prompt
        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            checkpointer=self.memory,
            prompt=SYSTEM_PROMPT,
        )

    def invoke(self, input_data: dict, config: dict | None = None) -> dict:
        """Gọi trực tiếp agent graph của LangGraph."""
        return self.agent.invoke(input_data, config=config)

    def chat(self, message: str, thread_id: str = "ticket_chat_session") -> str:
        """Gửi tin nhắn và nhận câu trả lời dạng văn bản từ trợ lý AI."""
        config = {"configurable": {"thread_id": thread_id}}
        response = self.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

        messages = response.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        return "Xin lỗi, tôi chưa thể trả lời câu hỏi này."


if __name__ == "__main__":
    load_dotenv()
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Khởi tạo instance TicketAgent
    ticket_agent = TicketAgent(
        model_name=model_name,
        temperature=0,
        base_url=base_url,
    )

    current_balance = get_wallet_balance()
    formatted_balance = f"{current_balance:,} VND".replace(",", ".")

    print("=" * 70)
    print("🛫 TRỢ LÝ TRA CỨU & ĐẶT VÉ MÁY BAY THÔNG MINH ĐÃ SẴN SÀNG!")
    print(f"💰 Số dư ví ban đầu: {formatted_balance}")
    print("👉 Bạn có thể thử các câu hỏi tự nhiên:")
    print("   - 'Đặt 1 vé vào trưa ngày thứ 4 đi đâu đó'")
    print("   - 'Tìm vé sáng thứ 2 đi Hà Nội'")
    print("   - 'Ví tôi còn bao nhiêu tiền?'")
    print("👉 Nhập 'exit', 'quit' hoặc 'q' để kết thúc.")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nKhách hàng: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nTrợ lý: Cảm ơn bạn đã sử dụng dịch vụ. Chúc bạn một ngày tốt lành!")
                break

            reply = ticket_agent.chat(user_input)
            print(f"\nTrợ lý:\n{reply}")

        except KeyboardInterrupt:
            print("\n\nĐã dừng chương trình. Tạm biệt!")
            break
        except Exception as e:
            print(f"\n[Lỗi]: {e}")