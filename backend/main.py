import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools import check_ticket_schedule

SYSTEM_PROMPT = """Bạn là trợ lý ảo AI chuyên nghiệp hỗ trợ tra cứu lịch và đặt vé máy bay cho các chặng bay tại Việt Nam.

Nhiệm vụ của bạn:
1. Khi khách hàng hỏi về lịch chuyến bay / lịch chiếu / lịch trình theo thứ trong tuần (ví dụ: Thứ 2, Thứ 3, Thứ ba, Thứ tư, Thứ 4, Thứ năm, Thứ 5, Thứ sáu, Thứ 6, Thứ bảy, Thứ 7, Chủ nhật, hoặc tên tiếng Anh như Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday,...):
   - Bạn BẮT BUỘC phải gọi công cụ `check_ticket_schedule` với tham số `day` tương ứng (ví dụ: day="Monday" hoặc day="Thứ 2").
   - Sau khi nhận được kết quả từ công cụ, hãy tổng hợp và trình bày danh sách các chuyến bay (gồm: Mã chuyến bay, Hãng hàng không, Giờ bay, Tuyến bay, Giá vé các hạng ghế còn trống) một cách rõ ràng, đẹp mắt bằng Tiếng Việt.
2. Nếu không tìm thấy chuyến bay, hãy thông báo lịch sự cho khách hàng.
3. Luôn phản hồi lịch sự, thân thiện bằng Tiếng Việt.
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
        self.tools = [check_ticket_schedule]
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

    print("=" * 60)
    print("🛫 Trợ lý tra cứu vé máy bay đã sẵn sàng!")
    print("👉 Nhập 'exit', 'quit' hoặc 'q' để kết thúc.")
    print("=" * 60)

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