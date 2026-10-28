# Judge audit: how to grade `judge_audit.csv`

Each row pairs a case report's confirmed final diagnosis with a diagnosis a model gave as its top answer. These 200
rows are drawn from the 17–28% of top answers that string matching cannot classify. Half come from base models and
half from post-trained models, and the rows are shuffled. The judges' verdicts and which model produced each name are
in a separate key that you don't need.

Fill in the `verdict` column with one word, using the rule the judges were given:

- **same**: the candidate names the same disease as the final diagnosis. This includes a synonym, an abbreviation,
  or a more specific form of it (e.g. final "Myocarditis", candidate "Viral myocarditis").
- **different**: a broader category, a related but distinct disease, or only a complication, cause or finding
  (e.g. final "Nonfunctioning pituitary macroadenoma", candidate "Pituitary adenoma (Cushing's disease)").
- **unsure**: you can't tell without clinical knowledge you don't have. That's fine; mark it and move on.

Notes are optional. A quick web search for unfamiliar terms is fine, but judge from the two names only; you don't need
the case. When done, save the file as CSV in this folder and tell Claude. It will then compute agreement with each
judge (overall and for base vs post-trained models), and the paper will report this as an author audit, not a clinical
validation.
