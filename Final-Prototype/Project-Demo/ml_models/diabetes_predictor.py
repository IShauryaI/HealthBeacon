"""
Diabetes Prediction Model - Production Backend
Author: Timothy Eric Dare
Integrated by: Shaurya Parshad
Version: 1.0

Random Forest model for Type 2 Diabetes prediction.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')


class DiabetesPredictor:
    """
    Diabetes prediction model using Random Forest.

    Attributes:
        model: Trained Random Forest model
        label_encoder: Label encoder for predictions
        feature_names: List of expected feature names (12 selected features)
        is_loaded: Whether model has been loaded
    """

    FEATURE_NAMES = [
        'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol_total',
        'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides',
        'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c'
    ]

    FEATURE_INFO = {
        'age': {'name': 'Age', 'unit': 'years', 'normal_range': (18, 100)},
        'bmi': {'name': 'Body Mass Index', 'unit': 'kg/m²', 'normal_range': (18.5, 24.9)},
        'systolic_bp': {'name': 'Systolic Blood Pressure', 'unit': 'mmHg', 'normal_range': (90, 120)},
        'diastolic_bp': {'name': 'Diastolic Blood Pressure', 'unit': 'mmHg', 'normal_range': (60, 80)},
        'cholesterol_total': {'name': 'Total Cholesterol', 'unit': 'mg/dL', 'normal_range': (125, 200)},
        'hdl_cholesterol': {'name': 'HDL Cholesterol', 'unit': 'mg/dL', 'normal_range': (40, 60)},
        'ldl_cholesterol': {'name': 'LDL Cholesterol', 'unit': 'mg/dL', 'normal_range': (0, 100)},
        'triglycerides': {'name': 'Triglycerides', 'unit': 'mg/dL', 'normal_range': (0, 150)},
        'glucose_fasting': {'name': 'Fasting Glucose', 'unit': 'mg/dL', 'normal_range': (70, 100)},
        'glucose_postprandial': {'name': 'Post-meal Glucose', 'unit': 'mg/dL', 'normal_range': (0, 140)},
        'insulin_level': {'name': 'Insulin Level', 'unit': 'µU/mL', 'normal_range': (2.6, 24.9)},
        'hba1c': {'name': 'HbA1c', 'unit': '%', 'normal_range': (4.0, 5.6)}
    }

    def __init__(self):
        """Initialize the Diabetes predictor."""
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.is_loaded = False

    def load_models(self, model_dir='models'):
        """
        Load trained models from disk.

        Args:
            model_dir (str): Directory containing model files
        """
        try:
            model_path = Path(model_dir)

            self.model = joblib.load(model_path / 'random_forest_model.pkl')
            self.label_encoder = joblib.load(model_path / 'label_encoder.pkl')
            self.feature_names = joblib.load(model_path / 'rf_features.pkl')

            self.is_loaded = True
            print("✓ Diabetes model loaded successfully")
            print(f"  Features: {len(self.feature_names)}")

        except Exception as e:
            raise Exception(f"Failed to load Diabetes model: {str(e)}")

    def predict(self, features, user_data=None):
        """
        Make prediction with support for missing values.

        Args:
            features: Dictionary of features (can be incomplete)
            user_data: User profile data for filling missing values

        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before making predictions")

        # Default values for missing indicators (healthy population medians)
        DEFAULT_VALUES = {
            'age': 45,  # Middle age
            'bmi': 25.0,  # Normal BMI
            'systolic_bp': 120,  # Normal systolic BP
            'diastolic_bp': 80,  # Normal diastolic BP
            'cholesterol_total': 180,  # Normal total cholesterol
            'hdl_cholesterol': 50,  # Normal HDL
            'ldl_cholesterol': 100,  # Normal LDL
            'triglycerides': 120,  # Normal triglycerides
            'glucose_fasting': 90,  # Normal fasting glucose
            'glucose_postprandial': 120,  # Normal post-meal glucose
            'insulin_level': 10.0,  # Normal insulin
            'hba1c': 5.3  # Normal HbA1c
        }

        # Override defaults with user profile data if available
        if user_data:
            if 'age' in user_data and user_data['age']:
                DEFAULT_VALUES['age'] = int(user_data['age'])
            if 'bmi' in user_data and user_data['bmi']:
                DEFAULT_VALUES['bmi'] = float(user_data['bmi'])

        # Fill in missing values with defaults
        complete_features = {}
        missing_indicators = []

        for feature in self.feature_names:
            if feature in features and features[feature] is not None:
                complete_features[feature] = float(features[feature])
            else:
                complete_features[feature] = DEFAULT_VALUES[feature]
                missing_indicators.append(feature)

        # Create DataFrame with features in correct order
        input_df = pd.DataFrame([complete_features])
        X = input_df[self.feature_names].values

        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # Get confidence (probability of predicted class)
        diabetes_probability = probabilities[1]
        confidence = diabetes_probability if prediction == 1 else probabilities[0]

        # Determine risk level
        if diabetes_probability >= 0.8:
            risk_level = 'critical'
        elif diabetes_probability >= 0.6:
            risk_level = 'high'
        elif diabetes_probability >= 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Generate recommendation
        recommendation = self._generate_recommendation(
            prediction,
            risk_level,
            complete_features,
            missing_indicators
        )

        return {
            'prediction': 'diabetes' if prediction == 1 else 'no_diabetes',
            'confidence': float(confidence),
            'probability_diabetes': float(diabetes_probability),
            'probability_no_diabetes': float(probabilities[0]),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'indicators_used': list(features.keys()),
            'indicators_missing': missing_indicators,
            'used_defaults': len(missing_indicators) > 0
        }

    def _generate_recommendation(self, prediction, risk_level, features, missing_indicators):
        """Generate clinical recommendation based on prediction."""
        base_recommendation = ""

        if prediction == 1:  # Diabetes detected
            if risk_level == 'critical':
                base_recommendation = (
                    "URGENT: High risk of Type 2 Diabetes detected. "
                    "Immediate consultation with an endocrinologist is strongly recommended. "
                    "Blood sugar monitoring and potential medication may be necessary."
                )
            elif risk_level == 'high':
                base_recommendation = (
                    "Elevated diabetes risk detected. "
                    "Please schedule an appointment with your doctor within the next few days. "
                    "Lifestyle modifications including diet and exercise are crucial."
                )
            else:
                base_recommendation = (
                    "Moderate diabetes risk detected. "
                    "Consult with your primary care physician for comprehensive evaluation. "
                    "Focus on maintaining healthy blood sugar levels through diet and exercise."
                )
        else:  # No diabetes
            base_recommendation = (
                "No significant diabetes indicators detected. "
                "Continue regular health checkups and maintain a healthy lifestyle: "
                "balanced diet, regular physical activity, and healthy weight management."
            )

        # Add warning if many indicators were missing
        if len(missing_indicators) > 6:
            base_recommendation += (
                f"\n\nNote: {len(missing_indicators)} indicators were missing and default values were used. "
                "For more accurate results, please provide a complete blood report with all diabetes-related tests."
            )

        return base_recommendation


# Example usage
if __name__ == "__main__":
    predictor = DiabetesPredictor()
    predictor.load_models()

    # Sample patient data
    patient = {
        'age': 55,
        'bmi': 27.1,
        'systolic_bp': 135,
        'diastolic_bp': 85,
        'cholesterol_total': 210,
        'hdl_cholesterol': 42,
        'ldl_cholesterol': 145,
        'triglycerides': 180,
        'glucose_fasting': 115,
        'glucose_postprandial': 168,
        'insulin_level': 8.1,
        'hba1c': 7.2
    }

    result = predictor.predict(patient)
    print("\nDiabetes Prediction:")
    print(f"Result: {result['prediction']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"Risk Level: {result['risk_level'].upper()}")
    print(f"\nRecommendation: {result['recommendation']}")