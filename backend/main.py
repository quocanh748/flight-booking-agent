import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from harness import AgentHarness, HarnessStatus
from agent import TicketAgent
from tools import get_wallet_balance


def main():
    load_dotenv()
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    agent = TicketAgent(
        model_name=model_name,
        temperature=0.2,
        base_url=base_url,
    )

    harness = AgentHarness(max_turns=5, agent=agent)

    current_balance = get_wallet_balance()
    formatted_balance = f"{current_balance:,} VND".replace(",", ".")

    print("TRỢ LÝ TRA CỨU & ĐẶT VÉ MÁY BAY THÔNG MINH")
    print(f"Số dư ví ban đầu: {formatted_balance}")

    session_id = "main_interactive_session"

    while True:
        try:
            user_input = input("\nKhách hàng: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nTrợ lý: Cảm ơn bạn đã sử dụng dịch vụ. Chúc bạn một ngày tốt lành!")
                break

            result = harness.step(user_input=user_input, thread_id=session_id)
            print(f"\nTrợ lý:\n{result.final_response}")

            if result.status == HarnessStatus.COMPLETED and result.pnr:
                print(f"\n[THÔNG BÁO HỆ THỐNG]: Đã hoàn tất xuất vé! Mã PNR: {result.pnr}")
                print(f"Số dư ví còn lại: {result.wallet_balance_end:,} VND".replace(",", "."))
            elif result.status == HarnessStatus.TOOL_LIMIT_TRIGGERED:
                print("\n[CẢNH BÁO HỆ THỐNG]:phát hiện lặp tool.")

        except KeyboardInterrupt:
            print("\n\nĐã dừng chương trình. Tạm biệt!")
            break
        except Exception as e:
            print(f"\n[Lỗi]: {e}")


if __name__ == "__main__":
    main()