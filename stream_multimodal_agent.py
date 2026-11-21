import base64
import time
from pprint import pprint
import cv2
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()

# Create the agent once outside the loop so it stays ready
agent = create_agent(model="gpt-5-nano", tools=[])


def run_continuous_webcam_agent(interval_seconds: int = 5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check macOS camera permissions for VS Code."
        )

    print(
        f"Webcam active: Auto-capturing every {interval_seconds} seconds. Press ESC in the preview window to stop."
    )
    last_capture_time = (
        time.time() - interval_seconds
    )  # Triggers first capture immediately

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Live Webcam (Press ESC to stop)", frame)

            # Check if 5 seconds have passed since the last snapshot
            current_time = time.time()
            if current_time - last_capture_time >= interval_seconds:
                last_capture_time = current_time
                print("\n--- Capturing snapshot & invoking agent ---")

                _, buffer = cv2.imencode(".jpg", frame)
                image_base64 = base64.b64encode(buffer).decode("utf-8")

                multimodal_question = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Describe the image",
                        },
                        {
                            "type": "image",
                            "base64": image_base64,
                            "mime_type": "image/jpeg",
                        },
                    ]
                )

                response = agent.invoke({"messages": [multimodal_question]})
                pprint(response["messages"][-1].content)

            # Keeps the preview window responsive and exits on ESC (27)
            if (cv2.waitKey(1) & 0xFF) == 27:
                print("\nStopping webcam agent...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


run_continuous_webcam_agent(interval_seconds=5)