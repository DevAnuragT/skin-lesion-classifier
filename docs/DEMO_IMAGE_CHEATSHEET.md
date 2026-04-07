# Demo Image Cheat Sheet

Current app runtime uses the cleaner retrained model in:
- `artifacts/retrained_hf_strict_80/skin_classifier.keras`

Current best validated metric:
- Test accuracy: `59.375%`

Preferred sample folder:
- `sample_images/demo_curated/`

Recommended uploads by class:
- Acne:
  `sample_images/demo_curated/acne_0016.jpg`
  `sample_images/demo_curated/acne_0051.jpg`
  `sample_images/demo_curated/acne_0066.jpg`
- Actinic keratosis:
  `sample_images/demo_curated/ak_0016.jpg`
  `sample_images/demo_curated/ak_0067.jpg`
  `sample_images/demo_curated/ak_0034.jpg`
- Basal cell carcinoma:
  `sample_images/demo_curated/bcc_0051.jpg`
  `sample_images/demo_curated/bcc_0019.jpg`
  `sample_images/demo_curated/bcc_0068.jpg`
- Benign Keratosis-like Lesions (BKL):
  `sample_images/demo_curated/bkl_0066.jpg`
  `sample_images/demo_curated/bkl_0026.jpg`
  `sample_images/demo_curated/bkl_0045.jpg`
- Atopic dermatitis(Eczema):
  `sample_images/demo_curated/eczema_0006.jpg`
  `sample_images/demo_curated/eczema_0026.jpg`
  `sample_images/demo_curated/eczema_0014.jpg`
- Melanoma:
  `sample_images/demo_curated/melanoma_0026.jpg`
  `sample_images/demo_curated/melanoma_0019.jpg`
  `sample_images/demo_curated/melanoma_0014.jpg`
- Psoriasis:
  `sample_images/demo_curated/psoriasis_0014.jpg`
  `sample_images/demo_curated/psoriasis_0016.jpg`
  `sample_images/demo_curated/psoriasis_0026.jpg`
- Tinea(Ringworm):
  `sample_images/demo_curated/tinea_0006.jpg`
  `sample_images/demo_curated/tinea_0014.jpg`
  `sample_images/demo_curated/tinea_0026.jpg`

Fallback only:
- `sample_images/real_web_samples/`
  These are useful for ad hoc testing, but they are less reliable for a live demo because they were not selected against the current model.

Important speaking note:
- Use these files to demonstrate that the upload and inference flow works.
- Do not claim that a sample image is medically confirmed just because it reaches the result page.
- The project should still be presented as a working prototype with improved but not final accuracy.
