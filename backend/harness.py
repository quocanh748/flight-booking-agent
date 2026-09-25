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
            print("\n[TIẾN TRÌNH ReAct (REASONING & ACTING)]:")

        for msg in messages:
            if isinstance(msg, AIMessage):
                if verbose and msg.content and getattr(msg, "tool_calls", None):
                    print(f"   [REASONING]: {msg.content}")

                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        t_name = tc.get("name", "unknown_tool")
                        t_args = json.dumps(tc.get("args", {}), sort_keys=True, ensure_ascii=False)
                        if verbose:
                            print(f"   [ACTION]: {t_name}({t_args})")

            elif isinstance(msg, ToolMessage):
                content_str = str(msg.content)
                obs_preview = content_str[:160] + "..." if len(content_str) > 160 else content_str
                if verbose:
                    print(f"   [OBSERVATION]: {msg.name} -> {obs_preview}")

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