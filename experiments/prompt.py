def get_zero_shot_standard_prompt(exam_type, original_indication, clinical_notes):
    """Zero-shot prompting with standard instructions."""
    clinical_notes_string = "\n".join([n[:2000] for n in clinical_notes])
    
    return f"""Given the following set of CLINICAL NOTES and ORIGINAL INDICATION, generate a REVISED INDICATION
that supplies the relevant clinical history which may optionally include the patient's sex, age, relevant symptoms,
biopsies, surgeries, resections, and treatments related to the radiological exam of {exam_type}.

CLINICAL NOTES:
{clinical_notes_string}

ORIGINAL INDICATION:
{original_indication}

REVISED INDICATION:
"""

def get_one_shot_standard_prompt(exam_type, original_indication, clinical_notes):
    """One-shot prompting with standard instructions and exemplar."""
    clinical_notes_string = "\n".join([n[:2000] for n in clinical_notes])
    
    return f"""Given the following set of CLINICAL NOTES and ORIGINAL INDICATION, generate a REVISED INDICATION
that supplies the relevant clinical history which may optionally include the patient's sex, age, relevant symptoms,
biopsies, surgeries, resections, and treatments related to the radiological exam of {exam_type}.

(EXAMPLE) CLINICAL NOTE:
FOLLOW-UP GASTROINTESTINAL MEDICAL ONCOLOGY VISIT 
Patient name ***** ***** 
DOB 08/31/1953 Medical record number ***** Date of service 06/16/2021 Referring Provider: Dr. 
***** ***** ***** ***** Subjective ***** ***** is a 67 y.o. female who presents with the following: 
Interval history and review of systems 

Interval History: -06/30/20: CT CAP ***** Pt remains off of pembrolizumab. She is doing well. 
Only rare loose stools but generally formed stools now . Occasional fatigue. 
Seeing Dr, ***** for labs and followup. Otherwise full 14-point ROS was negative in detail 

Oncologic history ***** ***** is a very pleasant 66 y.o. female who was seen in follow-up for esophageal 
squamous cell carcinoma at our Gastrointestinal Oncology Faculty Practice. The patient had
been noticing ***** dysphagia in ***** ***** *****. An upper endoscopy was done on 
04/23/15 EGD: mass at 25 cm, biopsies demonstrate squamous cell carcinoma in situ. 
09/07/15 EGD: 1. Half circumferential mass, measuring 3 X 2cm in size, was found; 
nodular and sub-mucosal; multiple biopsies were performed 2. Hiatal hernia 3. Nodular mass in the
proximal esophagus consistent with cancer Pathology: Esophagus, mass at 25 cm, biopsy: 
At least squamous cell carcinoma in situ; see comment. 08/19/2015: PET/CT Whole Body (vertex to thighs) 
, 1. Hypermetabolic esophageal mass measuring 2.1 x 1.4 cm, corresponding to the patient's known primary malignancy. 
No evidence of hypermetabolic lymphadenopathy or distant metastatic disease. 
2. Scattered solid pulmonary nodules measuring up to 7 mm without associated hypermetabolism.
Recommend correlation to prior imaging, if available. Otherwise, further follow-up is 
recommended as per oncologic protocol.

(EXAMPLE) ORIGINAL INDICATION:
Restaging.

(EXAMPLE) REVISED INDICATION:
Esophageal SCC diagnosed 2015 s/p resection Nov 2015 with negative margins. Follow up FDG PET with 
suspicious superior mediastinal nodes, s/p adjuvant chemoradiation 2017 with radiation to mediastinal 
nodes.

CLINICAL NOTES:
{clinical_notes_string}

ORIGINAL INDICATION:
{original_indication}

REVISED INDICATION:
"""

def get_zero_shot_augmented_prompt(exam_type, original_indication, clinical_notes):
    """Zero-shot prompting with augmented comprehensiveness rules."""
    clinical_notes_string = "\n".join([n[:2000] for n in clinical_notes])
    
    return f"""Given the following set of CLINICAL NOTES and ORIGINAL INDICATION, generate a REVISED INDICATION
that supplies the relevant clinical history which may optionally include the patient's sex, age, relevant symptoms,
biopsies, surgeries, resections, and treatments related to the radiological exam of {exam_type}.

If the ORIGINAL INDICATION is for evaluation of oncological or other chronic conditions, include surgeries, biopsies, 
and treatments with dates as applicable, in 20 words or fewer.
Otherwise, generate only up to 10 words or fewer. The indication must be one sentence, with no added explanation.

CLINICAL NOTES:
{clinical_notes_string}

ORIGINAL INDICATION:
{original_indication}

REVISED INDICATION:
"""

def get_one_shot_augmented_prompt(exam_type, original_indication, clinical_notes):
    """One-shot prompting with augmented comprehensiveness rules and exemplar."""
    clinical_notes_string = "\n".join([n[:2000] for n in clinical_notes])
    
    return f"""Given the following set of CLINICAL NOTES and ORIGINAL INDICATION, generate a REVISED INDICATION
that supplies the relevant clinical history which may optionally include the patient's sex, age, relevant symptoms,
biopsies, surgeries, resections, and treatments related to the radiological exam of {exam_type}.

If the ORIGINAL INDICATION is for evaluation of oncological or other chronic conditions, include surgeries, biopsies, 
and treatments with dates as applicable, in 20 words or fewer.
Otherwise, generate only up to 10 words or fewer. The indication must be one sentence, with no added explanation.

(EXAMPLE) CLINICAL NOTE:
FOLLOW-UP GASTROINTESTINAL MEDICAL ONCOLOGY VISIT 
Patient name ***** ***** 
DOB 08/31/1953 Medical record number ***** Date of service 06/16/2021 Referring Provider: Dr. 
***** ***** ***** ***** Subjective ***** ***** is a 67 y.o. female who presents with the following: 
Interval history and review of systems 

Interval History: -06/30/20: CT CAP ***** Pt remains off of pembrolizumab. She is doing well. 
Only rare loose stools but generally formed stools now . Occasional fatigue. 
Seeing Dr, ***** for labs and followup. Otherwise full 14-point ROS was negative in detail 

Oncologic history ***** ***** is a very pleasant 66 y.o. female who was seen in follow-up for esophageal 
squamous cell carcinoma at our Gastrointestinal Oncology Faculty Practice. The patient had
been noticing ***** dysphagia in ***** ***** *****. An upper endoscopy was done on 
04/23/15 EGD: mass at 25 cm, biopsies demonstrate squamous cell carcinoma in situ. 
09/07/15 EGD: 1. Half circumferential mass, measuring 3 X 2cm in size, was found; 
nodular and sub-mucosal; multiple biopsies were performed 2. Hiatal hernia 3. Nodular mass in the
proximal esophagus consistent with cancer Pathology: Esophagus, mass at 25 cm, biopsy: 
At least squamous cell carcinoma in situ; see comment. 08/19/2015: PET/CT Whole Body (vertex to thighs) 
, 1. Hypermetabolic esophageal mass measuring 2.1 x 1.4 cm, corresponding to the patient's known primary malignancy. 
No evidence of hypermetabolic lymphadenopathy or distant metastatic disease. 
2. Scattered solid pulmonary nodules measuring up to 7 mm without associated hypermetabolism.
Recommend correlation to prior imaging, if available. Otherwise, further follow-up is 
recommended as per oncologic protocol.

(EXAMPLE) ORIGINAL INDICATION:
Restaging.

(EXAMPLE) REVISED INDICATION:
Esophageal SCC diagnosed 2015 s/p resection Nov 2015 with negative margins. Follow up FDG PET with 
suspicious superior mediastinal nodes, s/p adjuvant chemoradiation 2017 with radiation to mediastinal 
nodes.

CLINICAL NOTES:
{clinical_notes_string}

ORIGINAL INDICATION:
{original_indication}

REVISED INDICATION:
"""

def generate_prompt(exam_type, original_indication, clinical_notes, prompting_strategy="one_shot", prompt_type="standard"):
    """
    Generate prompt based on experiment configuration.
    
    Args:
        exam_type: Type of radiological exam
        original_indication: Original indication from referring physician
        clinical_notes: List of clinical notes
        prompting_strategy: "zero_shot" or "one_shot"
        prompt_type: "standard" or "augmented"
    
    Returns:
        Generated prompt string
    """
    prompt_functions = {
        ("zero_shot", "standard"): get_zero_shot_standard_prompt,
        ("one_shot", "standard"): get_one_shot_standard_prompt,
        ("zero_shot", "augmented"): get_zero_shot_augmented_prompt,
        ("one_shot", "augmented"): get_one_shot_augmented_prompt,
    }
    
    prompt_function = prompt_functions.get((prompting_strategy, prompt_type))
    if prompt_function is None:
        raise ValueError(f"Invalid combination: prompting_strategy='{prompting_strategy}', prompt_type='{prompt_type}'")
    
    return prompt_function(exam_type, original_indication, clinical_notes)
