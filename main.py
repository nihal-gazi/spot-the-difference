import cv2, os
from similarity import pooled_descriptor, similarity
from align import align_images
from diff_detect import detect_diff
from llm import ask_llm
from dataset_builder import save_pair

images = [cv2.imread(f"images/{x}") for x in os.listdir("images")]

desc = [pooled_descriptor(i) for i in images]

idx = 1
for i in range(len(images)):
    for j in range(i+1, len(images)):
        if similarity(desc[i], desc[j]) < 1000:
            a1, a2 = align_images(images[i], images[j])
            cnts = detect_diff(a1, a2)

            prompt = f"""
You are creating a 'Spot the Difference' puzzle.
Describe the visual differences clearly and point by point.
Only describe what is different.

Return only the description text.
"""

            answer = ask_llm(prompt)
            save_pair(idx, a1, a2, answer)
            idx += 1
