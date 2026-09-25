import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

# Sử dụng create_agent và Middleware native từ thư viện LangChain
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ToolCallLimitMiddleware,
    ModelCallLimitMiddleware,
)

from tools import (
    check_ticket_schedule,
    check_balance,
    book_flight,
    smart_flight_search,
)

SYSTEM_PROMPT = """Bạn là trợ lý ảo AI chuyên nghiệp hỗ trợ tra cứu lịch và đặt vé máy bay theo chu trình ReAct (Reasoning + Acting).

CHU TRÌNH TƯ DUY VÀ HÀNH ĐỘNG (ReAct Loop):
Trước khi thực hiện hoặc đưa ra câu trả lời, bạn luôn tư duy từng bước:
1. Suy luận (Reasoning / Thought):
   - Phân tích yêu cầu của khách: Ngày nào trong tuần? Khung giờ nào (giờ cụ thể như '2 giờ sáng', '8h tối', hoặc khoảng thời gian như sáng/trưa/chiều/tối)? Điểm đến nếu có? Đã có tên hành khách và lệnh chốt vé chưa?
   - Xác định rõ hành động (Action) cần gọi công cụ nào.

2. Hành động (Acting / Action):
   - Khi khách đưa yêu cầu tìm vé: Gọi ngay `smart_flight_search(day=..., time_period=..., destination=...)` để hệ thống tự động kiểm tra toàn bộ (lịch bay, giờ bay, độ lệch giờ, ghế trống và số dư ví tài khoản).
   - Nếu khách hỏi số dư ví: Gọi `check_balance()`.
   - Nếu khách muốn xem bảng chuyến bay: Gọi `check_ticket_schedule(day=...)`.
   - Nếu khách đã XÁC NHẬN ('đồng ý', 'xác nhận', 'đặt luôn') kèm họ tên: Gọi `book_flight(flight_id=..., passengers=[...])` để hoàn tất thanh toán.

3. Quan sát & Đánh giá (Observation & Verification):
   - Đọc kỹ kết quả từ công cụ trả về, ĐẶC BIỆT LƯU Ý TRƯỜNG HỢP LỆCH GIỜ BAY (`has_time_deviation=True` hoặc `status='needs_time_clarification'`).

4. Đưa ra phản hồi (Decision / Response):
   - QUY TẮC BẮT BUỘC KHI LỆCH GIỜ BAY (Ví dụ: khách hỏi 2 giờ sáng nhưng chuyến sớm nhất là 06:00):
     + BẮT BUỘC PHẢI THÔNG BÁO RÕ VÀ HỎI Ý KIẾN KHÁCH HÀNG:
       "Rất tiếc ngày [Thứ] không có chuyến bay lúc [Giờ khách yêu cầu]. Chuyến bay sớm nhất hiện có là lúc [Giờ bay thực tế] ([Mã chuyến], cất cánh muộn hơn [X] tiếng so với giờ bạn mong muốn). Bạn có đồng ý đổi sang chuyến [Giờ bay thực tế] này không?"
     + TUYỆT ĐỐI KHÔNG xuất phiếu chốt vé như thể đã khớp giờ! Khách hàng không thể chấp nhận bay lúc 6 giờ sáng nếu họ cần bay lúc 2 giờ sáng mà không được hỏi ý kiến trước!
   - Khi giờ bay phù hợp hoặc khách chỉ hỏi khoảng giờ chung chung:
     + Xuất PHIẾU ĐỀ XUẤT VÉ HOÀN CHỈNH (gồm mã chuyến, hãng, tuyến bay, giờ bay, giá vé, số dư ví hiện tại và dự kiến còn lại).
     + Dừng lại và hỏi khách hàng: "Quý khách có xác nhận đặt vé này không? Vui lòng cung cấp họ tên hành khách và nhắn 'Xác nhận' để tiến hành xuất vé và trừ tiền."
   - QUY TẮC BẢO VỆ: TUYỆT ĐỐI CHƯA GỌI `book_flight` TRỪ TIỀN KHI CHƯA ĐƯỢC KHÁCH XÁC NHẬN!
   - TUYỆT ĐỐI KHÔNG TỰ BỊA RA MÃ PNR HOẶC SỐ TIỀN. Luôn phản hồi lịch sự bằng Tiếng Việt.
"""


class ConsecutiveToolLimitMiddleware(AgentMiddleware):
    """Middleware ngắt khẩn cấp nếu cùng 1 tool bị gọi lặp lại liên tiếp quá số lần cho phép."""

    def __init__(self, max_consecutive: int = 5):
        super().__init__()
        self.max_consecutive = max_consecutive
        self.last_tool = None
        self.consecutive_count = 0

    def wrap_tool_call(self, request, handler):
        tool_name = request.tool.name if hasattr(request, "tool") and request.tool else "unknown"
        if tool_name == self.last_tool:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 1
            self.last_tool = tool_name

        if self.consecutive_count >= self.max_consecutive:
            return ToolMessage(
                content=(
                    f"[CIRCUIT BREAKER]: Công cụ '{tool_name}' đã bị gọi lặp lại "
                    f"{self.max_consecutive} lần liên tiếp! Middleware đã ngắt vòng lặp an toàn."
                ),
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        return handler(request)


class TicketAgent:
    """Agent cốt lõi quản lý mô hình, prompt, tools và middlewares bảo vệ."""

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

        # Cấu hình danh sách Middleware chuẩn từ thư viện
        self.middlewares = [
            ConsecutiveToolLimitMiddleware(max_consecutive=5),
            ToolCallLimitMiddleware(run_limit=5, exit_behavior="end"),
            ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
        ]

        # Khởi tạo Agent sử dụng create_agent và Middleware native
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.memory,
            middleware=self.middlewares,
        )

    def invoke(self, input_data: dict, config: dict | None = None) -> dict:
        """Gọi trực tiếp agent graph của LangGraph."""
        config = config or {"configurable": {"thread_id": "ticket_chat_session"}}
        return self.agent.invoke(input_data, config=config)

