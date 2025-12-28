import json
from json import JSONDecodeError
import cv2
import numpy as np
import time
from ultralytics import YOLO


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

L_SHOULDER = 5
L_HIP = 11
L_KNEE = 13

def main():
    VIDEO_PATH = 0
    MODEL_PATH = "yolov8n-pose.pt"

    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print(f"ERROR: Can not load video/cam: {VIDEO_PATH}")
        return

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"ERROR: can not load YOLO model. Check given path: {MODEL_PATH}")
        print(f"Details: {e}")
        return

    SITTING_THRESHOLD = 145
    pose_history = []
    person_states = {}

    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            print("End of a video.")
            break

        frame = cv2.resize(frame, (640, 480))
        results = model.track(frame, stream=True, verbose=False, persist=True, tracker="tracker.yaml")

        current_time = time.time()

        for r in results:
            frame = r.plot(line_width=2,
                           labels=False,
                           conf=False,
                           kpt_radius=3)

            if r.boxes.id is None:
                continue

            boxes = r.boxes.xyxy.int().cpu().tolist()
            keypoints = r.keypoints.xy.cpu().numpy()
            person_ids = r.boxes.id.int().cpu().tolist()
            count = 0

            for i, person_id in enumerate(person_ids):
                count += 1
                kpts_data = keypoints[i]

                try:
                    shoulder = kpts_data[L_SHOULDER, :2]
                    hip = kpts_data[L_HIP, :2]
                    knee = kpts_data[L_KNEE, :2]

                    angle = calculate_angle(shoulder, hip, knee)
                    angle_text = f"{int(angle)} deg"

                    if angle > SITTING_THRESHOLD:
                        new_pose = "Standing"
                        color = (0, 255, 0)
                    else:
                        new_pose = "Sitting"
                        color = (0, 0, 255)

                    if person_id not in person_states:
                        person_states[person_id] = {
                            "pose": new_pose,
                            "start_time": current_time
                        }
                    elif person_states[person_id]["pose"] != new_pose:
                        start_time = person_states[person_id]["start_time"]
                        duration = current_time - start_time

                        pose_history.append({
                            "PersonID": person_id,
                            "Position": person_states[person_id]["pose"],
                            "Duration_sec": round(duration, 2),
                            "Start": time.ctime(start_time),
                            "End": time.ctime(current_time)
                        })

                        person_states[person_id]["pose"] = new_pose
                        person_states[person_id]["start_time"] = current_time

                    x1, y1, x2, y2 = boxes[i]
                    cv2.putText(frame, new_pose, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, angle_text, (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                except IndexError:
                    continue
                except Exception as e:
                    print(f"An error ocurred: {e}")
            cv2.putText(frame, f'Person Count in a Frame: {count}', (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Person Count Total: {len(person_states)}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f'Standing Count: {sum(1 for p in person_states.values() if p["pose"] == "Standing")}',
                        (30, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Sitting Count: {sum(1 for p in person_states.values() if p["pose"] == "Sitting")}',
                        (30, 120),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by an user.")
            break

    final_time = time.time()
    for person_id, state in person_states.items():
        start_time = state["start_time"]
        duration = final_time - start_time

        pose_history.append({
            "PersonID": person_id,
            "Position": state["pose"],
            "Duration_sec": round(duration, 2),
            "Start": time.ctime(start_time),
            "End": time.ctime(final_time)
        })

    json_file_name = "behavior_history.json"
    all_data = []
    try:
        with open(json_file_name, 'r') as file:
            all_data = json.load(file)
    except (FileNotFoundError, JSONDecodeError):
        all_data = []

    all_data.extend(pose_history)

    with open(json_file_name, 'w') as file:
        json.dump(all_data, file, indent=4)

    capture.release()
    cv2.destroyAllWindows()
    print(f"Program has been terminated successfully.'{json_file_name}' updated.")


if __name__ == "__main__":
    main()