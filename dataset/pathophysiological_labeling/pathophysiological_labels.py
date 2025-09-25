import pandas as pd
import tqdm
import json
from utils import chat_gpt4o
import datetime


def generate_prompt_imaging_indication(exam_type, original_indication, radiologist_indication):
    prompt = f"""
You are a labeling assistant for imaging indications. For each imaging exam provided, assign one of the following categories:
'cancer/mass', 'infection/inflammatory', 'surgical', 'symptom-based', or 'structural'. Answer in JSON format.

Guidelines:
1. Always assess the primary intent of the imaging request, not just the presence of keywords. Determine which clinical question is being addressed overall.
2. 'cancer/mass': Use this category when the primary intent is to evaluate for abnormal growths or rule out malignancy. Even if terms like
   'cancer', 'tumor', 'malignancy', 'mass', or 'neoplasm' appear, ensure that the main focus is on a cancer-related concern.
3. 'infection/inflammatory': Choose this category when the main focus is to detect or characterize an infectious or inflammatory process.
   Although keywords like 'abscess', 'pneumonia', 'appendicitis', or 'inflammatory' might be present, confirm that the primary goal is to assess infection or inflammation.
4. 'surgical': Use this category when the imaging is ordered in a surgical context—whether for preoperative planning, intraoperative guidance, or postoperative evaluation of complications.
5. 'symptom-based': Assign this category when the imaging is driven primarily by patient-reported symptoms (such as pain or discomfort) without a specific diagnosis.
   The focus is on investigating the symptoms rather than confirming a particular pathology.
6. 'structural': Use this category when the intent is to evaluate anatomical, congenital, or degenerative abnormalities. The focus should be on the physical structure of an organ or tissue,
   rather than on diagnosing a disease process like cancer or infection.

Note: Just because a history of cancer or a tumor is mentioned does not automatically mean the imaging request should be classified as 'cancer/mass'. Always determine the primary intent of the imaging request in the given context.

### Example

**Exam Type:**
CT ABDOMEN/PELVIS

**Original Indication:**
concern for perforation.

**Radiologist Indication:**
Pancreatic cancer, status post ERCP on 09/15/2017 with common bile duct and duodenal stent placement

**Example Output:**
{{
    "generated_category": "infection/inflammatory"
}}

### Actual

**Exam Type:**
{exam_type}

**Original Indication:**  
{original_indication}  

**Radiologist Indication:**  
{radiologist_indication}  

**Output:**
"""
    return prompt

interval = "70000_77984"
parquet_path = f"/mnt/sohn2022/Adrian/rad-llm-pmhx/indication_dataset/processed/{interval}.parquet"
llm_labels = pd.read_parquet(parquet_path).reset_index(drop=True)

results = pd.DataFrame(columns=[
    "patientdurablekey",
    "radiology_deid_note_key",
    "exam_type", 
    "original_history",
    "additional_history",
    "generated_category"
])

for i in tqdm.tqdm(range(len(llm_labels))):
    row = llm_labels.iloc[i]
    exam_type = row["exam_type"]
    original_history = row["original_history"]
    additional_history = row["additional_history"]
    prompt = generate_prompt_imaging_indication(
        exam_type, 
        original_history, 
        additional_history
    )
    output = chat_gpt4o(prompt)
    output = output.replace("json", "").replace("```", "")

    # Default values to ensure a row is always added
    result = {
        "patientdurablekey": row["patientdurablekey"],
        "radiology_deid_note_key": row["radiology_deid_note_key"],
        "exam_type": exam_type,
        "original_history": original_history,
        "additional_history": additional_history,
        "generated_category": None  # Default to None in case of failure
    }

    try:
        parsed_result = json.loads(output)
        if isinstance(parsed_result, dict) and "generated_category" in parsed_result:
            result["generated_category"] = parsed_result["generated_category"]
    except Exception as e:
        print(f"Warning: Error at index {i}: {e}")

    results.loc[len(results)] = result


now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
results.to_csv(f"llm_labels_pathophysiological_{interval}_{timestamp}.csv", index=False)

