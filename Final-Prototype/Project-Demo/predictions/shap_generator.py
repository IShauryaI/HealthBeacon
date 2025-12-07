"""
SHAP (SHapley Additive exPlanations) chart generator for model explainability.
Generates waterfall and summary plots to explain predictions.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from django.conf import settings


def generate_shap_charts(model, X, feature_names, report_id):
    """
    Generate SHAP waterfall and summary plots for a prediction.

    Args:
        model: Trained scikit-learn model (Random Forest)
        X: Input features (numpy array, shape: (1, n_features))
        feature_names: List of feature names
        report_id: Unique identifier for saving charts

    Returns:
        dict: Paths to generated charts and SHAP values
    """
    try:
        print(f"\n{'=' * 60}")
        print(f"Generating SHAP charts for report: {report_id}")
        print(f"{'=' * 60}")

        # Get SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        print(f"  SHAP values type: {type(shap_values)}")
        print(f"  SHAP values shape: {np.array(shap_values).shape}")

        # Handle binary classification (2D or 3D array)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            print(f"  Binary classification (list) - using class 1 (positive)")
            shap_values_to_plot = shap_values[1]
        elif len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
            print(f"  Binary classification (3D array) - using class 1 (positive)")
            shap_values_to_plot = shap_values[:, :, 1]
        else:
            shap_values_to_plot = shap_values

        print(f"  Final SHAP values shape: {shap_values_to_plot.shape}")

        # Get base value (expected value)
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        elif isinstance(expected_value, np.ndarray) and len(expected_value) == 2:
            expected_value = expected_value[1]

        print(f"  Expected value: {expected_value}")

        # Create output directory
        media_dir = Path(settings.MEDIA_ROOT) / 'shap_charts'
        media_dir.mkdir(parents=True, exist_ok=True)

        # Generate waterfall plot
        print(f"  Creating waterfall plot...")
        waterfall_path = generate_waterfall_plot(
            shap_values_to_plot,
            X,
            feature_names,
            expected_value,
            report_id,
            media_dir
        )

        # Generate summary plot
        print(f"  Creating summary plot (bar)...")
        summary_path = generate_summary_plot(
            shap_values_to_plot,
            X,
            feature_names,
            report_id,
            media_dir
        )

        # Prepare return data
        result = {
            'waterfall': waterfall_path,
            'summary': summary_path,
            'shap_values': shap_values_to_plot.tolist()
        }

        print(f"\n{'=' * 60}")
        print(f"✓ SHAP chart generation complete")
        print(f"  Waterfall: {waterfall_path}")
        print(f"  Summary: {summary_path}")
        print(f"{'=' * 60}\n")

        return result

    except Exception as e:
        print(f"\n✗ SHAP chart generation failed:")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_waterfall_plot(shap_values, X, feature_names, base_value, report_id, output_dir):
    """
    Generate a waterfall plot showing feature contributions.
    """
    try:
        values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        data = X[0] if len(X.shape) > 1 else X

        print(f"    Values shape: {values.shape}")
        print(f"    Base value: {base_value}")
        print(f"    Data shape: {data.shape}")

        fig, ax = plt.subplots(figsize=(10, 8))

        sorted_idx = np.argsort(np.abs(values))[::-1]
        sorted_values = values[sorted_idx]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_data = data[sorted_idx]

        colors = ['#ff6b6b' if v > 0 else '#4dabf7' for v in sorted_values]
        bars = ax.barh(range(len(sorted_values)), sorted_values, color=colors, alpha=0.7)

        for i, (feat, val, d) in enumerate(zip(sorted_features, sorted_values, sorted_data)):
            label = f"{feat} = {d:.2f}"
            if val > 0:
                ax.text(val + 0.01, i, f'  {label}', va='center', fontsize=9)
            else:
                ax.text(val - 0.01, i, f'{label}  ', va='center', ha='right', fontsize=9)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
        ax.set_title('Feature Impact on Prediction', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3)

        ax.text(0.02, 0.98, f'Base value: {base_value:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        waterfall_file = output_dir / f"{report_id}_waterfall.png"
        plt.savefig(waterfall_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"    ✓ Waterfall plot saved: {waterfall_file.name}")
        return f"shap_charts/{waterfall_file.name}"

    except Exception as e:
        print(f"    ✗ Waterfall plot failed: {e}")

        try:
            print(f"    Trying bar plot instead...")
            fig, ax = plt.subplots(figsize=(10, 6))

            values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            sorted_idx = np.argsort(np.abs(values))[::-1]

            colors = ['red' if values[i] > 0 else 'blue' for i in sorted_idx]
            ax.barh([feature_names[i] for i in sorted_idx],
                    [values[i] for i in sorted_idx],
                    color=colors, alpha=0.7)

            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('SHAP Value')
            ax.set_title('Feature Impact on Prediction')
            plt.tight_layout()

            waterfall_file = output_dir / f"{report_id}_waterfall.png"
            plt.savefig(waterfall_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"    ✓ Bar plot saved instead: {waterfall_file.name}")
            return f"shap_charts/{waterfall_file.name}"

        except Exception as fallback_error:
            print(f"    ✗ Bar plot also failed: {fallback_error}")
            return None


def generate_summary_plot(shap_values, X, feature_names, report_id, output_dir):
    """
    Generate a summary plot showing feature importance.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        abs_values = np.abs(values)
        sorted_idx = np.argsort(abs_values)

        colors = ['#ff6b6b' if values[i] > 0 else '#4dabf7' for i in sorted_idx]
        ax.barh([feature_names[i] for i in sorted_idx],
                [abs_values[i] for i in sorted_idx],
                color=colors, alpha=0.7)

        ax.set_xlabel('Mean Absolute SHAP Value (Feature Importance)', fontsize=12)
        ax.set_title('Feature Importance Summary', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        summary_file = output_dir / f"{report_id}_summary.png"
        plt.savefig(summary_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"    ✓ Summary plot saved: {summary_file.name}")
        return f"shap_charts/{summary_file.name}"

    except Exception as e:
        print(f"    ✗ Summary plot failed: {e}")
        return None


def generate_feature_importance_chart(indicators, prediction_result, report_id, model=None, background_data=None):
    """
    Generate SHAP-style waterfall chart for Heart Disease.
    Creates impact scores based on deviation from healthy baseline.
    """
    try:
        print(f"\n{'=' * 60}")
        print(f"Generating Heart Disease impact chart")
        print(f"{'=' * 60}")

        # Feature importance weights (based on medical literature)
        risk_weights = {
            'BMI': 1.2,
            'Smoking': 2.5,
            'AlcoholDrinking': 1.5,
            'Stroke': 3.0,
            'PhysicalHealth': 1.3,
            'MentalHealth': 0.8,
            'DiffWalking': 1.5,
            'Sex': 0.5,
            'AgeCategory': 1.0,
            'Race': 0.3,
            'Diabetic': 2.2,
            'PhysicalActivity': 1.8,
            'GenHealth': 1.5,
            'SleepTime': 1.0,
            'Asthma': 1.2,
            'KidneyDisease': 2.0,
            'SkinCancer': 0.8
        }

        # Healthy baseline values
        healthy_baseline = {
            'BMI': 22.0,
            'Smoking': 0,
            'AlcoholDrinking': 0,
            'Stroke': 0,
            'PhysicalHealth': 0,
            'MentalHealth': 0,
            'DiffWalking': 0,
            'Sex': 1,
            'AgeCategory': '45-49',
            'Race': 'White',
            'Diabetic': 0,
            'PhysicalActivity': 1,
            'GenHealth': 'Very good',
            'SleepTime': 7.0,
            'Asthma': 0,
            'KidneyDisease': 0,
            'SkinCancer': 0
        }

        # Display names
        feature_names_map = {
            'BMI': 'bmi',
            'Smoking': 'smoking',
            'AlcoholDrinking': 'alcohol',
            'Stroke': 'stroke',
            'PhysicalHealth': 'phys_health',
            'MentalHealth': 'mental_health',
            'DiffWalking': 'diff_walk',
            'Sex': 'sex',
            'AgeCategory': 'age_cat',
            'Race': 'race',
            'Diabetic': 'diabetic',
            'PhysicalActivity': 'phys_activity',
            'GenHealth': 'gen_health',
            'SleepTime': 'sleep',
            'Asthma': 'asthma',
            'KidneyDisease': 'kidney_dis',
            'SkinCancer': 'skin_cancer'
        }

        # Calculate impact scores
        impact_scores = {}
        feature_values = {}

        for feature, value in indicators.items():
            baseline = healthy_baseline.get(feature, 0)
            weight = risk_weights.get(feature, 1.0)

            # Calculate deviation from healthy
            if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
                # Numeric features
                if feature == 'BMI':
                    deviation = max(0, value - 24.9)  # Above healthy BMI
                    impact = deviation * weight * 0.05
                elif feature in ['PhysicalHealth', 'MentalHealth']:
                    impact = value * weight * 0.1
                elif feature == 'SleepTime':
                    deviation = abs(value - 7.5)  # Deviation from optimal
                    impact = -deviation * weight * 0.08 if value >= 6 else deviation * weight * 0.08
                else:
                    impact = (value - baseline) * weight * 0.1
            else:
                # Binary/categorical features
                if feature in ['Smoking', 'AlcoholDrinking', 'Stroke', 'Diabetic',
                               'DiffWalking', 'Asthma', 'KidneyDisease', 'SkinCancer']:
                    # Risk factors (1 = bad)
                    impact = (value - baseline) * weight * 0.3
                elif feature == 'PhysicalActivity':
                    # Protective factor (1 = good)
                    impact = -(value - baseline) * weight * 0.3
                elif feature == 'GenHealth':
                    health_score = {'Excellent': -0.4, 'Very good': -0.2, 'Good': 0,
                                    'Fair': 0.3, 'Poor': 0.6}
                    impact = health_score.get(value, 0) * weight
                else:
                    impact = 0

            impact_scores[feature] = impact
            feature_values[feature] = value

        # Sort by absolute impact
        sorted_features = sorted(impact_scores.keys(),
                                 key=lambda x: abs(impact_scores[x]),
                                 reverse=True)[:11]

        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 8))

        values = [impact_scores[f] for f in sorted_features]
        display_names = [feature_names_map.get(f, f) for f in sorted_features]
        data_values = [feature_values[f] for f in sorted_features]

        # Colors: red for positive (risk), blue for negative (protective)
        colors = ['#ff6b6b' if v > 0 else '#4dabf7' for v in values]
        y_pos = range(len(values))

        ax.barh(y_pos, values, color=colors, alpha=0.7, height=0.6)

        # Add labels with feature values
        for i, (feat, val, data_val) in enumerate(zip(display_names, values, data_values)):
            # Format data value
            if isinstance(data_val, (int, float)):
                label_text = f"{feat} = {data_val:.2f}"
            else:
                label_text = f"{feat} = {data_val}"

            # Position label
            if val > 0:
                ax.text(val, i, f"  {label_text}", va='center', fontsize=10)
            else:
                ax.text(val, i, f"{label_text}  ", va='center', ha='right', fontsize=10)

        # Styling
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_yticks([])
        ax.set_xlabel('Impact on Prediction', fontsize=12)
        ax.set_title('Feature Impact on Prediction', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

        # Add base value annotation
        base_value = 0.2  # Approximate baseline risk
        ax.text(0.02, 0.98, f'Base value: {base_value:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save chart
        media_dir = Path(settings.MEDIA_ROOT) / 'shap_charts'
        media_dir.mkdir(parents=True, exist_ok=True)

        chart_path = media_dir / f"heart_{report_id}_waterfall.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  ✓ Impact chart saved: {chart_path.name}")
        print(f"{'=' * 60}\n")

        return f"shap_charts/heart_{report_id}_waterfall.png"

    except Exception as e:
        print(f"  ✗ Chart generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None