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
from pydantic import BaseModel, Field

from tools import (
    check_balance,
    add_balance,
    book_flight,
    smart_flight_search,
)

PLANNER_PROMPT = """Bạn là chuyên gia lập kế hoạch cho hệ thống vé máy bay thông minh.
Hệ thống CÓ SẴN các công cụ:
- smart_flight_search: Tra cứu danh sách chuyến bay theo ngày/tuyến/hạng vé.
- check_balance: Kiểm tra số dư ví.
- add_balance: Nạp tiền vào ví.
- book_flight: Đặt vé máy bay (cần họ tên hành khách).

NHIỆM VỤ:
Phân tích CHÍNH XÁC mục đích trong câu nói của khách hàng và xuất ra DUY NHẤT 1 JSON object dạng:
{"steps": ["nội dung bước 1...", "nội dung bước 2..."]}

QUY TẮC CỐT LÕI VỀ MỤC TIÊU (BẮT BUỘC TUÂN THỦ):
1. KHÁCH CHỈ HỎI/XEM/TRA CỨU LỊCH BAY (ví dụ: 'cho xem lịch bay', 'hôm nay có chuyến nào', 'tìm chuyến...', 'check vé...'):
   -> Kế hoạch CHỈ CÓ 1 BƯỚC DUY NHẤT: Tra cứu và hiển thị danh sách chuyến bay theo yêu cầu.
   -> TUYỆT ĐỐI KHÔNG thêm bước kiểm tra ví, nạp tiền hay đặt vé!
2. KHÁCH CHỈ HỎI SỐ DƯ HOẶC NẠP TIỀN:
   -> Kế hoạch CHỈ CÓ bước kiểm tra ví hoặc nạp tiền.
3. CHỈ KHI KHÁCH YÊU CẦU ĐẶT / MUA VÉ (ví dụ: 'đặt 2 vé', 'mua vé', 'chuyến nào rẻ thì đặt', 'chốt vé', 'lấy vé'):
   -> Mới lập các bước đầy đủ gồm: Tra cứu chuyến ➔ Xác nhận tên và đặt vé ➔ Nếu không đủ số dư, hỏi khách để nạp tiền.
4. Xuất DUY NHẤT 1 JSON hợp lệ, KHÔNG viết suy nghĩ, giải thích hay bất kỳ chữ nào ngoài JSON."""

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
    booking_success: bool
    plan: list[str]
    current_step_index: int
    completed_steps: list[dict]


def build_agent_system_prompt(state: AgentState) -> str:
    plan = state.get("plan", [])
    current_idx = state.get("current_step_index", 0)
    current_step = plan[current_idx] if current_idx < len(plan) else "Hoàn tất toàn bộ kế hoạch"
    completed_steps = state.get("completed_steps", [])

    if completed_steps:
        completed_text = "\n".join([f"- Đã xong: {item.get('step')} -> Kết quả: {item.get('result')}" for item in completed_steps])
    else:
        completed_text = "Chưa có bước nào hoàn thành."

    plan_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(plan)]) if plan else "Chưa có kế hoạch cụ thể."

    return f"""Bạn là trợ lý AI chuyên trách THỰC THI từng bước trong quy trình đặt vé máy bay theo mô hình ReAct.

KẾ HOẠCH TỔNG THỂ:
{plan_text}

CÁC BƯỚC ĐÃ HOÀN THÀNH:
{completed_text}

🎯 NHIỆM VỤ DUY NHẤT CỦA BẠN BÂY GIỜ LÀ:
👉 BƯỚC {current_idx + 1}: "{current_step}"

QUY TẮC BẮT BUỘC:
1. Chỉ tập trung thực hiện BƯỚC {current_idx + 1} bằng cách suy nghĩ và gọi các công cụ (smart_flight_search, check_balance, add_balance, book_flight) phù hợp.
2. TUYỆT ĐỐI KHÔNG làm vượt sang các bước tiếp theo khi bước hiện tại chưa hoàn tất.
3. Nếu bước hiện tại yêu cầu xuất vé nhưng chưa có họ tên hành khách, hãy hỏi khách cung cấp họ tên ngay.
4. Khi đã hoàn thành bước {current_idx + 1}, hãy tóm tắt ngắn gọn kết quả để chuyển giao cho bước kế tiếp."""


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

        self.planner_model = ChatOllama(
            model=model_name,
            temperature=0.0,
            base_url=base_url,
            format="json",
            num_predict=2048,
            num_ctx=8192
        )

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

        def planner_node(state: AgentState):
            user_goal = state.get("user_goal", "")
            existing_plan = state.get("plan", [])
            current_idx = state.get("current_step_index", 0)
            
            if existing_plan and current_idx < len(existing_plan):
                print(f"\n[TIẾP TỤC KẾ HOẠCH TỪ BƯỚC {current_idx + 1}/{len(existing_plan)}]: {existing_plan[current_idx]}")
                return {
                    "plan": existing_plan,
                    "current_step_index": current_idx
                }

            is_booking = any(k in user_goal.lower() for k in ["đặt", "mua", "chốt", "lấy", "book"])
            is_wallet = any(k in user_goal.lower() for k in ["số dư", "ví", "nạp tiền", "tiền còn"])

            if is_booking:
                default_steps = [
                    "Tra cứu các chuyến bay phù hợp theo yêu cầu",
                    "Xác nhận thông tin hành khách và tiến hành đặt vé",
                    "Nếu không đủ số dư, hỏi khách để nạp tiền"
                ]
            elif is_wallet:
                default_steps = ["Kiểm tra số dư ví hiện tại"]
            else:
                default_steps = ["Tra cứu và hiển thị danh sách chuyến bay theo yêu cầu"]

            planner_prompt_messages = [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=f"Yêu cầu của khách: '{user_goal}'. Hãy lập kế hoạch giải quyết.")
            ]
            
            steps = []
            try:
                response = self.planner_model.invoke(planner_prompt_messages)
                content = response.content.strip()
                
                # Trích xuất JSON từ phản hồi
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                        steps = data.get("steps", [])
                    except Exception:
                        pass
                
                if not steps:
                    for line in content.split("\n"):
                        m = re.match(r'^\s*\d+[\.\)\-]\s*(.+)', line)
                        if m:
                            steps.append(m.group(1).strip())
            except Exception:
                steps = []

            if not steps:
                steps = default_steps

            print("\nPLAN:")
            for i, s in enumerate(steps, 1):
                print(f"   {i}. {s}")

            return {
                "plan": steps,
                "current_step_index": 0,
                "completed_steps": [],
                "booking_success": False
            }

        def agent_node(state: AgentState):
            messages = state["messages"]
            prompt = build_agent_system_prompt(state)
            
            if messages and isinstance(messages[0], SystemMessage):
                call_messages = [SystemMessage(content=prompt)] + list(messages[1:])
            else:
                call_messages = [SystemMessage(content=prompt)] + list(messages)

            response = self.model.invoke(call_messages)
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

        def replanner_node(state: AgentState):
            plan = state.get("plan", [])
            current_idx = state.get("current_step_index", 0)
            last_message = state["messages"][-1]
            last_content = last_message.content if isinstance(last_message, AIMessage) else ""

            content_lower = last_content.lower()
            needs_human_input = any(k in content_lower for k in [
                "họ tên", "tên hành khách", "vui lòng cung cấp", "quý khách có muốn", "bạn có muốn", "xin họ tên", "cho tôi biết tên", "ngày bay", "lịch bay cho ngày nào", "thêm thông tin"
            ])

            completed_steps = list(state.get("completed_steps", []))
            current_step_desc = plan[current_idx] if current_idx < len(plan) else "Bước cuối"

            if needs_human_input:
                next_idx = current_idx
            else:
                completed_steps.append({
                    "step": current_step_desc,
                    "result": last_content[:300] if len(last_content) > 300 else last_content
                })
                next_idx = current_idx + 1

            return {
                "completed_steps": completed_steps,
                "current_step_index": next_idx,
            }

        def agent_router(state: AgentState):
            last_message = state["messages"][-1]
            iterations = state.get("iteration_count", 0)
            if iterations >= self.max_iterations:
                return "replanner"

            if getattr(last_message, "tool_calls", None):
                return "tools"

            return "replanner"

        def replanner_router(state: AgentState):
            last_message = state["messages"][-1]
            iterations = state.get("iteration_count", 0)
            if iterations >= self.max_iterations:
                return END

            content = last_message.content.lower() if isinstance(last_message, AIMessage) else ""
            needs_human_input = any(k in content for k in [
                "họ tên", "tên hành khách", "vui lòng cung cấp", "quý khách có muốn", "bạn có muốn", "xin họ tên", "cho tôi biết tên", "ngày bay", "lịch bay cho ngày nào", "thêm thông tin"
            ])
            if needs_human_input:
                return END

            if state.get("booking_success", False):
                return END

            plan = state.get("plan", [])
            current_idx = state.get("current_step_index", 0)
            if current_idx < len(plan):
                return "agent"

            return END

        workflow.add_node("planner", planner_node)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.add_node("replanner", replanner_node)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "agent")

        workflow.add_conditional_edges("agent", agent_router, {
            "tools": "tools",
            "replanner": "replanner"
        })
        workflow.add_edge("tools", "agent")

        workflow.add_conditional_edges("replanner", replanner_router, {
            "agent": "agent",
            END: END
        })

        return workflow.compile(checkpointer=self.memory)

    def stream(self, input_data: dict, config: dict | None = None, stream_mode: str = "updates"):
        messages = input_data.get("messages", [])
        user_goal = messages[-1].content if messages else ""
        payload = {
            "messages": messages,
            "user_goal": user_goal,
            "iteration_count": 0,
        }
        return self.graph.stream(payload, config=config, stream_mode=stream_mode)