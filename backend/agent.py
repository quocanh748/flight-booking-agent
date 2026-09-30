import os
import sys
import re
import json
from typing import Annotated, TypedDict
from collections import deque
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from tools import (
    check_balance,
    add_balance,
    book_flight,
    smart_flight_search,
)

SYSTEM_PROMPT = """Bạn là trợ lý ảo AI chuyên nghiệp hỗ trợ tra cứu lịch và đặt vé máy bay theo chu trình ReAct.
Nguyên tắc phản hồi:
- Trả lời ngắn gọn, súc tích, đi thẳng vào kết quả so sánh chuyến bay tốt nhất hoặc rẻ nhất theo yêu cầu của khách, không liệt kê lan man gây dài dòng.
- Khi người dùng muốn đặt vé (các từ như 'đặt luôn', 'chốt vé', 'đặt vé', 'mua vé', 'lấy vé', 'thì lấy', 'chọn', 'mua'):
  + Nếu CHƯA CÓ họ tên hành khách: Hãy đưa ra chuyến bay tối ưu nhất kèm tổng chi phí và hỏi xin họ tên hành khách để tiến hành xuất vé.
  + Nếu ĐÃ CÓ họ tên hành khách: Thực hiện trọn vẹn: tra cứu chuyến bay -> kiểm tra ví/nạp tiền nếu cần -> BẮT BUỘC gọi book_flight để xuất mã PNR.
Tuyệt đối không dừng lại giữa chừng khi các bước của khách hàng chưa được hoàn tất!
"""

class LoopDetector:
    def __init__(self, window=5, repeat_k=3):
        self.recent = deque(maxlen=window)
        self.k = repeat_k

    def check(self, tool_name: str, args: dict) -> bool:
        fp = (tool_name, repr(sorted(args.items())))
        if self.recent.count(fp) + 1 >= self.k:
            return True
        self.recent.append(fp)
        return False


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int
    user_goal: str
    feedback_count: int
    booking_success: bool


class TicketAgent:
    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        temperature: float = 0.1,
        base_url: str | None = None,
        max_iterations: int = 25,
    ):
        load_dotenv()
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        self.tools = [check_balance, add_balance, book_flight, smart_flight_search]
        self.tool_map = {t.name: t for t in self.tools}
        self.max_iterations = max_iterations
        self.loop_detector = LoopDetector(window=5, repeat_k=3)

        self.model = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            num_predict=3072,
            num_ctx=8192
        ).bind_tools(self.tools)

        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        def agent_node(state: AgentState):
            messages = state["messages"]
            
            if not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

            response = self.model.invoke(messages)
            current_count = state.get("iteration_count", 0) + 1
            return {
                "messages": [response],
                "iteration_count": current_count
            }

        def tool_node(state: AgentState):
            last_message = state["messages"][-1]
            tool_messages = []
            booking_success = state.get("booking_success", False)

            if getattr(last_message, "tool_calls", None):
                for tc in last_message.tool_calls:
                    t_name = tc["name"]
                    t_args = tc.get("args", {})
                    t_id = tc["id"]

                    if self.loop_detector.check(t_name, t_args):
                        obs = {"status": "error", "message": f"[CẢNH BÁO LẶP]: Tool {t_name} đang bị gọi lặp. Hãy đổi hướng suy luận!"}
                    else:
                        tool_fn = self.tool_map.get(t_name)
                        obs = tool_fn.invoke(t_args) if tool_fn else {"status": "error", "message": "Tool không tồn tại"}

                    if t_name == "book_flight" and isinstance(obs, dict) and obs.get("status") == "success":
                        booking_success = True

                    tool_messages.append(ToolMessage(content=json.dumps(obs, ensure_ascii=False), tool_call_id=t_id))

            return {"messages": tool_messages, "booking_success": booking_success}

        def feedback_node(state: AgentState):
            user_goal = state.get("user_goal", "")
            current_feedback = state.get("feedback_count", 0) + 1
            prompt_reminder = (
                f"[HỆ THỐNG HARNESS NHẮC NHỞ]: Bạn chưa hoàn tất yêu cầu của khách: '{user_goal}'. "
                "Nếu đã có họ tên hành khách, hãy gọi book_flight ngay để xuất mã PNR! "
                "Nếu còn thiếu họ tên hành khách, hãy hỏi khách cung cấp họ tên ngay lập tức để tiến hành đặt vé."
            )
            return {
                "messages": [HumanMessage(content=prompt_reminder)],
                "feedback_count": current_feedback
            }

        def termination_router(state: AgentState):
            last_message = state["messages"][-1]
            iterations = state.get("iteration_count", 0)
            if iterations >= self.max_iterations:
                return END
            
            if getattr(last_message, "tool_calls", None):
                return "tools"
            content = last_message.content.lower() if isinstance(last_message, AIMessage) else ""
            needs_human_input = any(k in content for k in [
                "họ tên", "tên hành khách", "vui lòng cung cấp", "xác nhận", "quý khách có muốn", "bạn có muốn", "xin họ tên", "cho tôi biết tên"
            ])
            if needs_human_input:
                return END
            user_goal = state.get("user_goal", "").lower()
            wants_booking = any(k in user_goal for k in [
                "đặt luôn", "chốt vé", "đặt vé", "mua vé", "thì lấy", "lấy", "chọn", "mua"
            ])
            
            feedback_count = state.get("feedback_count", 0)
            if wants_booking and not state.get("booking_success", False) and feedback_count < 2:
                return "feedback"
            return END

        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("feedback", feedback_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", termination_router, {
            "tools": "tools",
            "feedback": "feedback",
            END: END
        })
        workflow.add_edge("tools", "agent")       
        workflow.add_edge("feedback", "agent")  

        return workflow.compile(checkpointer=self.memory)

    def stream(self, input_data: dict, config: dict | None = None, stream_mode: str = "updates"):
        
        messages = input_data.get("messages", [])
        user_goal = messages[-1].content if messages else ""
        payload = {
            "messages": messages,
            "user_goal": user_goal,
            "iteration_count": 0,
            "booking_success": False
        }
        return self.graph.stream(payload, config=config, stream_mode=stream_mode)