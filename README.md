# Spot The Difference - Dataset Utility/Organizer
---
Algorithm:
1. Downsample images and compute fast similarity descriptors
2. Find near-identical image pairs
3. Align images using feature matching
4. Compute pixel-level differences
5. Cluster difference regions
6. Draw bounding boxes on difference areas
7. Send marked images to LLM for description
8. Save results as dataset entry

--- 
Essentially,
1. Have a dataset of lots of images in `/images`
2. It'll sort out the images as "spot-the-difference" types and organize them into clean folders
---

So,
- It finds near-identical images, aligns them, detects visual differences, highlights those differences, and uses an LLM to produce accurate textual answers.
- The output is structured for dataset creation and downstream ML use.


## Folder Structure:
```
spot_diff/
│
├─ main.py
├─ config.py
├─ similarity.py
├─ align.py
├─ diff_detect.py
├─ llm.py
├─ dataset_builder.py
├─ images/          # input images
└─ output/          # generated dataset
```


## Output Sample as:
```
output/
└─ 000001/
   ├─ 1.png
   ├─ 2.png
   └─ answer.json
```

NOTE: EDIT `config.py` and PUT YOUR API KEY, then set `USE_OFFLINE_LLM = False`.
