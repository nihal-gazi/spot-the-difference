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

            m1 = draw_boxes(a1, cnts)
            m2 = draw_boxes(a2, cnts)
            
            cv2.imwrite("temp1.png", m1)
            cv2.imwrite("temp2.png", m2)
            
            prompt = """
            You are given two images with red boxes marking the only regions that differ.
            Describe exactly what is different between the two images.
            Be precise and point-by-point.
            Do not mention the boxes.
            Return only the description.
            """
            
            answer = ask_llm_with_images(prompt, "temp1.png", "temp2.png")
            save_pair(idx, m1, m2, answer)

            idx += 1
