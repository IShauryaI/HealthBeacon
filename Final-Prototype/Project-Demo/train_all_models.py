"""
Train all disease prediction models.
Run this once to generate .pkl files.
"""

import sys
sys.path.append('ml_models')

from ml_models.ckd_predictor import CKDPredictor

# Train CKD Model
print("=" * 70)
print("TRAINING CKD MODEL")
print("=" * 70)

ckd_predictor = CKDPredictor()
ckd_metrics = ckd_predictor.train('data/ckd_40k_with_creatinine.csv')

print(f"\n✓ CKD Model Trained!")
print(f"  Accuracy: {ckd_metrics['accuracy']*100:.2f}%")
print(f"  Recall: {ckd_metrics['recall']*100:.2f}%")

ckd_predictor.save_model('models/ckd_model.pkl')
print(f"  Saved to: models/ckd_model.pkl")

# TODO: Add diabetes and heart disease training here later
print("\n" + "=" * 70)
print("Diabetes model: Not yet implemented (placeholder)")
print("Heart Disease model: Not yet implemented (placeholder)")
print("=" * 70)