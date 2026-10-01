import os
import sys
import re
import json
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent import TicketAgent
from tools import get_wallet_balance


def main():
    load_dotenv()
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    agent = TicketAgent(
        model_name=model_name,
        temperature=0.1,
        base_url=base_url,
    )

    current_balance = get_wallet_balance()
    formatted_balance = f"{current_balance:,} VND".replace(",", ".")

    print("TRỢ LÝ TRA CỨU & ĐẶT VÉ MÁY BAY THÔNG MINH")
    print(f"Số dư ví ban đầu: {formatted_balance}")
    print("Nhập 'exit', 'quit' hoặc 'q' để kết thúc.")

    session_id = "main_interactive_session"

    while True:
        try:
            user_input = input("\nKhách hàng: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nTrợ lý: Cảm ơn bạn đã sử dụng dịch vụ. Chúc bạn một ngày tốt lành!")
                break

            config = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": 25,
            }

            final_response = ""

            for update in agent.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_data in update.items():
                    if not node_data:
                        continue
                    messages = node_data.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if msg.content and getattr(msg, "tool_calls", None):
                                print(f"   [REASONING]: {msg.content}")
                            if getattr(msg, "tool_calls", None):
                                for tc in msg.tool_calls:
                                    t_name = tc.get("name", "unknown_tool")
                                    t_args = json.dumps(tc.get("args", {}), sort_keys=True, ensure_ascii=False)
                                    print(f"   [ACTION]: {t_name}({t_args})")
                            elif msg.content:
                                final_response = msg.content
                        elif isinstance(msg, ToolMessage):
                            content_str = str(msg.content)
                            obs_preview = content_str[:160] + "..." if len(content_str) > 160 else content_str
                            print(f"   [OBSERVATION]: {msg.name} -> {obs_preview}")

            print(f"\nTrợ lý:\n{final_response}")

            pnr_match = re.search(r"\bPNR\d{6}\b", final_response or "")
            if pnr_match:
                balance_now = get_wallet_balance()
                print(f"\n[THÔNG BÁO HỆ THỐNG]: Đã hoàn tất xuất vé! Mã PNR: {pnr_match.group(0)}")
                print(f"Số dư ví còn lại: {balance_now:,} VND".replace(",", "."))

        except GraphRecursionError:
            print("\n[CẢNH BÁO HỆ THỐNG]: Đã vượt quá giới hạn bước suy luận.")
        except KeyboardInterrupt:
            print("\n\nĐã dừng chương trình. Tạm biệt!")
            break
        except Exception as e:
            print(f"\n[Lỗi]: {e}")


if __name__ == "__main__":
    main()