import os, json, cv2

def save_pair(idx, img1, img2, answer):
    folder = f"output/{idx:06d}"
    os.makedirs(folder, exist_ok=True)

    p1 = f"{folder}/1.png"
    p2 = f"{folder}/2.png"
    cv2.imwrite(p1, img1)
    cv2.imwrite(p2, img2)

    data = {
        "id": str(idx),
        "category": "spot_the_difference",
        "image_list": [p1, p2],
        "answer": answer
    }

    with open(f"{folder}/answer.json", "w") as f:
        json.dump(data, f, indent=2)
