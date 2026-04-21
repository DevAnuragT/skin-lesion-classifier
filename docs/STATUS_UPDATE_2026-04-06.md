# Group 19 Status Update

Date: 2026-04-06
For presentation on: 2026-04-07

## One-line status

This should be presented as a progress update, not as a completed project.

## What we proposed

Our course proposal was to build a skin lesion classification system using deep learning and transfer learning, then expose it through a Flask web app where a user uploads an image and gets a prediction.

## What we found in the cloned repo

- The upstream project already contains notebooks for business understanding, scraping, cleaning, and modelling.
- The upstream project also contains a Flask app with an image upload flow and prediction logic.
- The project scope in our summary is broadly aligned with the repo we selected as a baseline.

## What we can honestly say we completed this week

- We finalized the project direction.
- We selected an open-source baseline repo to save time.
- We audited the repo against our proposal.
- We understood the high-level pipeline:
  - data collection and cleaning notebooks
  - transfer-learning model training
  - Flask-based prediction app
- We identified the exact setup gaps blocking local reproduction.

## What is still pending

- Install required Python packages for the web app and model runtime.
- Fetch the real trained model artifact instead of the Git LFS pointer currently in the clone.
- Reproduce the app locally.
- Validate predictions using sample images.
- Verify reported accuracy using rerunnable evidence before presenting final numbers.

## What to tell ma'am

Use this wording:

> We finalized the project direction and selected an existing open-source baseline to speed up development.
>
> This week, we audited the repository against our proposal, understood the model and web app pipeline, and identified the exact environment and artifact gaps needed to reproduce it locally.
>
> The reference project is largely built upstream, but our local environment is not yet fully set up, so we are treating this as progress toward adaptation and validation rather than a finished build.

If she asks whether the project is complete:

> The reference implementation exists, but our team has not yet fully reproduced and validated it locally, so it would be inaccurate to call our project completed today.

## What not to claim

- Do not say that we built the full system from scratch in the last week.
- Do not say the project is complete on our machine.
- Do not quote 88% accuracy as a final verified result.
- Do not claim the app is locally runnable right now.

## Best closing line

Our next milestone is to complete local setup, fetch the actual model, run the app end-to-end, and validate both predictions and reported metrics before we present the project as complete.
