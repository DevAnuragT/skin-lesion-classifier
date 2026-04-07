## Curated Demo Samples

These images were copied from the cleaned test split used with the current runtime model.

Why these files exist:
- Each file was selected because the current `website/models/skin_classifier.keras` model classified it correctly.
- They are safer for a live demo than the older `real_web_samples/` folder, which contains noisier external images.

How to use them:
- Start the app with `bash run_local.sh`
- Upload files from this folder
- Use them to demonstrate the upload and inference flow

Important:
- These are still demo samples, not medically validated reference cases.
- Do not claim clinical accuracy from a few successful uploads.
