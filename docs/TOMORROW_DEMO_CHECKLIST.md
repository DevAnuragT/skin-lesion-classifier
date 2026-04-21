# Tomorrow Demo Checklist

Date: 2026-04-07

## Recommended framing

Treat this as a weekly progress review.

## 5-minute presentation flow

1. Start with the aim
   - We want to build an accessible skin disease image classifier using deep learning and a Flask web app.

2. Explain why we chose this repo
   - We found an existing open-source baseline and chose it to save implementation time.
   - The repo broadly matches our project summary.

3. Show the repo structure
   - notebooks
   - data folder
   - website folder
   - pretrained-model reference

4. Show the pipeline briefly
   - business understanding
   - scraping and data cleaning
   - model training with transfer learning
   - Flask app for upload and prediction

5. Show what is done on our side
   - audited the codebase
   - understood app flow
   - identified local setup blockers

6. Show the blockers clearly
   - model file in clone is only a Git LFS pointer
   - required ML dependencies are not installed locally
   - local training data is incomplete
   - accuracy numbers need verification

7. End with the next milestone
   - install dependencies
   - fetch the actual model artifact
   - run the Flask app locally
   - test the upload flow
   - verify metrics before claiming completion

## If ma'am asks for a live demo

Say:

> The application logic is present in the repository, but we cannot honestly show it as a completed local demo yet because the actual model artifact and some runtime dependencies are still missing on this machine.

Then show:

- `website/main.py`
- `website/templates/index.html`
- `website/templates/results.html`
- `docs/REPO_AUDIT_2026-04-06.md`

## If ma'am asks whether the project is complete

Say:

> The reference project is largely built upstream, but our team has not yet fully reproduced and validated it locally, so we are presenting it as work in progress.

## If ma'am asks what exactly happened this week

Say:

> This week we selected the baseline repository, checked that it matches our proposal, understood the model and Flask app pipeline, and identified the exact technical gaps we need to close for local reproduction.

## If ma'am asks about accuracy

Say:

> The repo README mentions 88% accuracy, but the notebook contains multiple metric outputs, including lower values, so we will verify the exact final result before quoting a final number in our submission.

## Final line

Our next deliverable is a locally runnable, validated demo instead of only an audited baseline.
