# Rating guide: language-model differential diagnoses

Thank you for helping. This should take about 45–60 minutes. Please fill in the two files `part_A_ddxplus.csv` and `part_B_cases.csv`. Both open in Excel or Google Sheets; turn on text wrapping to read the long cells, and type your answers into the empty Q columns. Rate from your own clinical judgment. There are no trick items, and "unsure" is a valid answer.

## Background

We study how a language model's differential diagnosis looks when it is read in two ways:

- **The list it states in one answer:** a ranked list of conditions with probabilities.
- **The spread of its top answers over 32 repeated calls:** shown as conditions with counts.

Several models produced these readouts, and the readouts are shown blinded and in random order. You do not need to know which model wrote what.

## Part A: synthetic emergency/primary-care patients (15 patients)

The patients come from DDXPlus, a public dataset of synthetic patients. Each was generated from a questionnaire, with no exam or test results. The models chose only from a fixed list of 49 conditions.

For each patient you see:

- the vignette;
- a **reference differential**, computed by the dataset's rule-based engine;
- four **readouts (A–D)**. Each is labelled either "One answer's ranked list" or "Top answers of 32 repeated calls".

Please answer three questions:

- **Q1. Is the reference differential clinically plausible for this vignette?**
  - 1 = implausible
  - 3 = partly plausible
  - 5 = what I would list
- **Q2. For each readout: is it an adequate differential for this vignette?**
  - 1 = inadequate or misleading
  - 3 = acceptable but missing important items
  - 5 = good

  Judge the conditions named and their order. For the 32-call readouts, read the counts as how often each condition came up first.
- **Q3. For each readout: does it omit a condition that must not be missed here?** Answer Y or N. If Y, name the condition, for example pulmonary embolism, acute coronary syndrome, pneumothorax or esophageal rupture. Consider only conditions that are plausible for this vignette.

## Part B: published case reports (15 cases)

The presentations are taken from open-access case reports in the MedCaseReasoning dataset. For each case you see:

- the presentation the models saw;
- the report's final diagnosis;
- the authors' diagnostic-reasoning notes from the report.

Please answer two questions:

- **Q4. For each listed "alternative": did the report's authors consider it as an alternative diagnosis?** Answer Y, N or unsure.
  - These alternatives were extracted automatically from the reasoning notes.
  - Answer N if the item is not mentioned at all.
  - Answer N if it is the final diagnosis itself.
  - Answer N if it is only a complication, comorbidity or test finding rather than a competing diagnosis.
- **Q5. For each "pair" (a model's answer and a target diagnosis): do they name the same disease?**
  - **Y:** same disease. A synonym, abbreviation or more specific form counts as Y if it would be accepted as the diagnosis.
  - **partly:** overlapping but not the same. Examples: a broader category, or the right organ but the wrong process.
  - **N:** a different disease.

## Notes

- **Comments:** please use the comments column for anything notable, such as an implausible reference or a dangerous omission.
- **Data:** no patient data are involved. DDXPlus is synthetic, and the case reports are published and de-identified.
- **How your ratings are used:** they let us report clinician agreement with the reference differential, with the automated extraction and with the automated judge. They are reported only in aggregate.
