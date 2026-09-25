import os
import sys
import re
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent import TicketAgent
from tools import get_wallet_balance


class HarnessStatus(str, Enum):
    RUNNING = "RUNNING"
    AWAITING_TIME_CLARIFICATION = "AWAITING_TIME_CLARIFICATION"  # Lệch giờ bay, dừng chờ user đồng ý đổi giờ
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"  # Đã ra vé đề xuất hoàn chỉnh, dừng chờ user chốt
    COMPLETED = "COMPLETED"                          # Đã đặt vé thành công và có mã PNR
    FAILED = "FAILED"                                # Không tìm thấy vé phù hợp hoặc số dư ví không đủ
    MAX_TURNS_EXCEEDED = "MAX_TURNS_EXCEEDED"        # Vượt quá số vòng lặp tối đa
    TOOL_LIMIT_TRIGGERED = "TOOL_LIMIT_TRIGGERED"    # Middleware ngắt do chạm ngưỡng giới hạn tool


class HarnessResult(BaseModel):
    status: HarnessStatus
    turns_taken: int
    final_response: str
    pnr: str | None = None
    wallet_balance_start: int
    wallet_balance_end: int


class AgentHarness:
    """Khung điều khiển vòng lặp hội thoại đa lượt và kiểm soát điều kiện dừng.
    
    Lưu ý: Các chốt chặn in-flight (ngắt lặp tool, giới hạn model calls) đã được
    giao trọn gói cho Middleware trong main.py quản lý.
    """

    def __init__(self, max_turns: int = 5, agent: TicketAgent | None = None):
        self.max_turns = max_turns
        self.agent = agent or TicketAgent()

    def step(
        self,
        user_input: str,
        thread_id: str = "ticket_chat_session",
        verbose: bool = True,
    ) -> HarnessResult:
        """Xử lý một lượt tương tác người dùng, hiển thị ReAct và thẩm định trạng thái dừng."""
        balance_start = get_wallet_balance()
        config = {"configurable": {"thread_id": thread_id}}

        graph_output = self.agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        messages = graph_output.get("messages", [])
        middleware_interrupted = False

        if verbose:
            print("\n🔄 [TIẾN TRÌNH ReAct (REASONING & ACTING)]:")

        for msg in messages:
            if isinstance(msg, AIMessage):
                if verbose and msg.content and getattr(msg, "tool_calls", None):
                    print(f"   🧠 [REASONING / THOUGHT]: {msg.content}")

                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        t_name = tc.get("name", "unknown_tool")
                        t_args = json.dumps(tc.get("args", {}), sort_keys=True, ensure_ascii=False)
                        if verbose:
                            print(f"   ⚡ [ACTING / TOOL CALL]: {t_name}({t_args})")

            elif isinstance(msg, ToolMessage):
                content_str = str(msg.content)
                obs_preview = content_str[:160] + "..." if len(content_str) > 160 else content_str
                if verbose:
                    print(f"   🔍 [OBSERVATION]: {msg.name} -> {obs_preview}")

                if "CIRCUIT BREAKER" in content_str or "limit exceeded" in content_str.lower():
                    middleware_interrupted = True

        ai_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                ai_text = msg.content
                break

        # Đánh giá trạng thái
        pnr_match = re.search(r"\bPNR\d{6}\b", ai_text)
        pnr = pnr_match.group(0) if pnr_match else None

        if middleware_interrupted:
            status = HarnessStatus.TOOL_LIMIT_TRIGGERED
        elif pnr:
            status = HarnessStatus.COMPLETED
        elif any(
            k in ai_text.lower()
            for k in ["không có chuyến bay lúc", "chuyến bay sớm nhất", "chuyến sớm nhất", "lệch", "cất cánh muộn hơn", "đổi sang", "chuyển sang"]
        ) and any(k in ai_text.lower() for k in ["đồng ý", "quý khách có", "bạn có", "muốn đổi"]):
            status = HarnessStatus.AWAITING_TIME_CLARIFICATION
        elif any(k in ai_text.lower() for k in ["xác nhận", "đồng ý", "quý khách có", "đề xuất", "chốt vé"]):
            status = HarnessStatus.AWAITING_CONFIRMATION
        elif any(k in ai_text.lower() for k in ["không tìm thấy", "không đủ tiền", "hết chỗ"]):
            status = HarnessStatus.FAILED
        else:
            status = HarnessStatus.RUNNING

        return HarnessResult(
            status=status,
            turns_taken=1,
            final_response=ai_text,
            pnr=pnr,
            wallet_balance_start=balance_start,
            wallet_balance_end=get_wallet_balance(),
        )

    def chat(self, message: str, thread_id: str = "ticket_chat_session") -> str:
        """Hàm rút gọn cho phép chat trực tiếp và nhận phản hồi văn bản."""
        result = self.step(user_input=message, thread_id=thread_id)
        return result.final_response

    def run(
        self,
        prompt: str,
        auto_confirm: bool = False,
        passenger_name: str = "Nguyễn Văn A",
        thread_id: str = "harness_test_session",
    ) -> HarnessResult:
        """Thực thi vòng lặp hội thoại có kiểm soát điều kiện dừng."""
        balance_start = get_wallet_balance()
        current_input = prompt
        turn = 0
        last_response = ""
        pnr = None

        config = {"configurable": {"thread_id": thread_id}}

        print("=" * 70)
        print(f"🚀 [HARNESS KHỞI ĐỘNG] Yêu cầu: '{prompt}'")
        print(f"💰 Số dư ví ban đầu: {balance_start:,} VND".replace(",", "."))
        print(f"🔄 Giới hạn hội thoại tối đa: {self.max_turns} lượt (Turns)")
        print("=" * 70)

        while turn < self.max_turns:
            turn += 1
            print(f"\n--- [LƯỢT HỘI THOẠI {turn}/{self.max_turns}] Đang xử lý... ---")

            # 1. Gọi Agent Graph (Được bảo vệ bởi Middleware bên trong)
            graph_output = self.agent.invoke(
                {"messages": [HumanMessage(content=current_input)]},
                config=config,
            )

            messages = graph_output.get("messages", [])

            # 2. HIỂN THỊ TIẾN TRÌNH ReAct (REASONING & ACTING)
            print("\n🔄 [TIẾN TRÌNH ReAct (REASONING & ACTING)]:")
            middleware_interrupted = False

            for msg in messages:
                if isinstance(msg, AIMessage):
                    # Lời suy nghĩ / reasoning của AI
                    if msg.content and getattr(msg, "tool_calls", None):
                        print(f"   🧠 [REASONING / THOUGHT]: {msg.content}")

                    # Lệnh gọi công cụ (ACTING)
                    if getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            t_name = tc.get("name", "unknown_tool")
                            t_args = json.dumps(tc.get("args", {}), sort_keys=True, ensure_ascii=False)
                            print(f"   ⚡ [ACTING / TOOL CALL]: {t_name}({t_args})")

                elif isinstance(msg, ToolMessage):
                    # Quan sát kết quả từ công cụ trả về (OBSERVATION)
                    content_str = str(msg.content)
                    obs_preview = content_str[:160] + "..." if len(content_str) > 160 else content_str
                    print(f"   🔍 [OBSERVATION]: {msg.name} -> {obs_preview}")

                    # Kiểm tra nếu Middleware đã ngắt mạch an toàn
                    if "CIRCUIT BREAKER" in content_str or "limit exceeded" in content_str.lower():
                        middleware_interrupted = True

            # 3. Lấy phản hồi văn bản cuối cùng của AI
            ai_text = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    ai_text = msg.content
                    break

            last_response = ai_text
            print(f"\n💬 [PHẢN HỒI CUỐI CÙNG TỪ AGENT]:\n{ai_text}")

            # 4. KIỂM TRA ĐIỀU KIỆN DỪNG: Middleware đã ngắt do lặp tool
            if middleware_interrupted:
                print("\n🛑 [ĐIỀU KIỆN DỪNG ĐẠT ĐƯỢC]: Middleware đã kích hoạt ngắt an toàn.")
                return HarnessResult(
                    status=HarnessStatus.TOOL_LIMIT_TRIGGERED,
                    turns_taken=turn,
                    final_response=ai_text,
                    pnr=None,
                    wallet_balance_start=balance_start,
                    wallet_balance_end=get_wallet_balance(),
                )

            # 5. KIỂM TRA ĐIỀU KIỆN DỪNG: Đã đặt vé thành công và có mã PNR
            pnr_match = re.search(r"\bPNR\d{6}\b", ai_text)
            if pnr_match:
                pnr = pnr_match.group(0)
                print(f"\n🎯 [ĐIỀU KIỆN DỪNG ĐẠT ĐƯỢC]: Đã xuất vé thành công! Mã PNR: {pnr}")
                return HarnessResult(
                    status=HarnessStatus.COMPLETED,
                    turns_taken=turn,
                    final_response=ai_text,
                    pnr=pnr,
                    wallet_balance_start=balance_start,
                    wallet_balance_end=get_wallet_balance(),
                )

            # 6. KIỂM TRA ĐIỀU KIỆN DỪNG: Lệch giờ bay (Chờ User đồng ý đổi giờ)
            is_time_clarification = any(
                k in ai_text.lower()
                for k in ["không có chuyến bay lúc", "chuyến bay sớm nhất", "chuyến sớm nhất", "lệch", "cất cánh muộn hơn", "đổi sang", "chuyển sang"]
            ) and any(k in ai_text.lower() for k in ["đồng ý", "quý khách có", "bạn có", "muốn đổi"])

            if is_time_clarification:
                print("\n🛑 [ĐIỀU KIỆN DỪNG ĐẠT ĐƯỢC]: Phát hiện lệch giờ bay! Dừng vòng lặp để hỏi ý kiến khách hàng xem có đổi giờ không.")
                if auto_confirm:
                    print(f"🤖 [Harness Auto-Confirm]: Tự động gửi lệnh 'Đồng ý đổi sang chuyến sớm nhất và đặt vé cho {passenger_name}'...")
                    current_input = f"Đồng ý đổi sang chuyến sớm nhất, đặt vé cho hành khách {passenger_name}"
                    continue

                return HarnessResult(
                    status=HarnessStatus.AWAITING_TIME_CLARIFICATION,
                    turns_taken=turn,
                    final_response=ai_text,
                    pnr=None,
                    wallet_balance_start=balance_start,
                    wallet_balance_end=get_wallet_balance(),
                )

            # 7. KIỂM TRA ĐIỀU KIỆN DỪNG: Đã ra vé đề xuất hoàn chỉnh (Dừng chờ User xác nhận)
            is_awaiting = any(
                keyword in ai_text.lower()
                for keyword in ["xác nhận", "đồng ý", "quý khách có", "đề xuất", "chốt vé"]
            )
            if is_awaiting:
                print("\n🛑 [ĐIỀU KIỆN DỪNG ĐẠT ĐƯỢC]: Đã ra phiếu đề xuất vé hoàn chỉnh. Dừng vòng lặp chờ User xác nhận!")

                if auto_confirm:
                    print(f"🤖 [Harness Auto-Confirm]: Tự động gửi lệnh 'Xác nhận đặt vé cho {passenger_name}'...")
                    current_input = f"Đồng ý, xác nhận đặt vé này cho hành khách {passenger_name}"
                    continue

                return HarnessResult(
                    status=HarnessStatus.AWAITING_CONFIRMATION,
                    turns_taken=turn,
                    final_response=ai_text,
                    pnr=None,
                    wallet_balance_start=balance_start,
                    wallet_balance_end=get_wallet_balance(),
                )

            # 8. KIỂM TRA ĐIỀU KIỆN DỪNG: Thất bại (Không tìm thấy chuyến hoặc không đủ tiền)
            if any(k in ai_text.lower() for k in ["không tìm thấy", "không đủ tiền", "hết chỗ"]):
                print("\n❌ [ĐIỀU KIỆN DỪNG ĐẠT ĐƯỢC]: Không đủ điều kiện đặt vé.")
                return HarnessResult(
                    status=HarnessStatus.FAILED,
                    turns_taken=turn,
                    final_response=ai_text,
                    pnr=None,
                    wallet_balance_start=balance_start,
                    wallet_balance_end=get_wallet_balance(),
                )

        # 9. ĐIỀU KIỆN DỪNG: Đạt giới hạn số lượt hội thoại tối đa
        print(f"\n⚠️ [CẢNH BÁO]: Đã đạt giới hạn {self.max_turns} lượt hội thoại mà chưa hoàn thành mục tiêu.")
        return HarnessResult(
            status=HarnessStatus.MAX_TURNS_EXCEEDED,
            turns_taken=turn,
            final_response=last_response,
            pnr=None,
            wallet_balance_start=balance_start,
            wallet_balance_end=get_wallet_balance(),
        )


if __name__ == "__main__":
    # Test chạy thử nghiệm Harness tinh gọn
    harness = AgentHarness(max_turns=3)

    # Chạy thử với câu hỏi lệch giờ: 'đặt 1 vé vào 2 giờ sáng ngày thứ 2 đi đâu đó'
    result = harness.run(
        prompt="đặt 1 vé vào 2 giờ sáng ngày thứ 2 đi đâu đó",
        auto_confirm=False,  # Để False để kiểm chứng Agent tự dừng hỏi ý kiến đổi giờ
    )

    print("\n" + "=" * 50)
    print(f"🏁 TRẠNG THÁI KẾT THÚC: {result.status.value}")
    print(f"📊 Số lượt đối đáp đã chạy: {result.turns_taken}")
    print(f"💰 Số dư ví sau cùng: {result.wallet_balance_end:,} VND".replace(",", "."))
    print("=" * 50)
