# ViT Roadmap to 85%+ Accuracy

## Current baseline summary

- Current best strict run is around 59.38% accuracy.
- Existing strict dataset has 80 images per class, with 56 train / 12 validation / 12 test per class.
- Existing trainer uses MobileNetV2 and short training schedules.

## Execution update (2026-04-13)

- ViT training pipeline is implemented in scripts/train_vit.py and validated on CPU.
- Best completed run: artifacts/vit_150_tiny160 with:
  - Test accuracy: 71.20%
  - Macro precision: 70.36%
  - Macro recall: 71.20%
- Best checkpoint is auto-promoted into website/models via scripts/promote_best_vit.py.
- Website runtime now auto-loads promoted ViT checkpoint (website/models/skin_classifier_vit.pt).
- 85% target is not reached yet; dataset scale/quality remains the primary blocker.

## Key blockers

1. Data scale is very small for 8-way lesion classification.
2. Source-to-class coupling is extreme in strict and 150 sets:
   - acne, eczema, psoriasis, tinea come only from dermnet
   - ak, bcc, bkl, melanoma come only from skin_cancer_small_dataset
3. Validation and test sets are tiny, so model selection is noisy.
4. Current augmentation and optimization are lightweight for this problem.

## Target strategy

Use a pre-trained Vision Transformer as the primary model family and improve data quality before expecting reliable 85%+ results.

## Phase 1: Establish ViT baseline (1-2 days)

Script added: scripts/train_vit.py

Recommended first run:

```bash
source .venv/bin/activate
pip install torch torchvision timm
python scripts/train_vit.py \
  --data-root datasets/hf_hybrid_150/splits \
  --output-dir artifacts/vit_150_tiny160 \
  --model-name vit_tiny_patch16_224 \
  --image-size 160 \
  --epochs 20 \
  --batch-size 8 \
  --device cpu
```

Expected outcome:
- A stronger baseline than MobileNet on the same split.
- Exported artifacts: best_model.pt, class_names.json, history.json, metrics.json, confusion_matrix.csv.

## Phase 2: Scale and clean data (highest impact, 2-5 days)

1. Increase target-per-class to at least 500-1000 (minimum 400 if constrained).
2. Ensure each class has examples from multiple sources/domains.
3. Remove ambiguous/non-dermatology samples through quick manual triage.
4. Add near-duplicate filtering beyond exact hash (perceptual hash).

Expected outcome:
- Major gain in generalization.
- More stable validation/test metrics.

## Phase 3: Stronger ViT training recipe (1-3 days)

1. Train on 224 first, then fine-tune on 384.
2. Use weighted cross-entropy or focal loss if imbalance/noise appears.
3. Add stronger augmentations: RandAugment + MixUp + CutMix.
4. Use 5-fold cross-validation and average model weights or logits.
5. Track macro-F1 and balanced accuracy, not only accuracy.

Expected outcome:
- Typical gain of 5-15 points over plain fine-tuning depending on data quality.

## Phase 4: Distillation and deployment path (1-2 days)

1. Distill ViT teacher to a smaller student for fast inference if needed.
2. Keep ViT as reference model for offline evaluation.
3. If Flask runtime must stay TensorFlow-only, convert student to ONNX/TFLite or add PyTorch inference endpoint.

## Acceptance criteria for 85% goal

Use this gate before claiming success:

1. Test accuracy >= 85.0% on a fixed, never-tuned holdout.
2. Macro-F1 >= 0.82.
3. No class recall below 0.75.
4. Repeatability: at least 3 different seeds with <= 2.0 point std.

## Risk note

With only 80-150 images per class and source-coupled labels, 85% is unlikely to be reliable. ViT helps, but data scale and quality are the deciding factors.
