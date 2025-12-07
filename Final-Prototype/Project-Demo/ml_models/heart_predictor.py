"""
Heart Disease Prediction Model - Production Backend
Author: Sandeep Kapoor & Ayeshkant Mallick
Integrated by: Shaurya Parshad
Version: 1.0

Professional KNN and Decision Tree models for Heart Disease prediction.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')


class HeartDiseasePredictor:
    """
    Heart Disease prediction model using KNN and Decision Tree.

    Attributes:
        knn_model: K-Nearest Neighbors model
        dt_model: Decision Tree model
        transformer: Feature transformer
        scaler: Feature scaler
        feature_names: List of expected feature names
        is_loaded: Whether models have been loaded
    """

    FEATURE_NAMES = [
        'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth',
        'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race',
        'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime',
        'Asthma', 'KidneyDisease', 'SkinCancer'
    ]

    FEATURE_INFO = {
        'BMI': {'name': 'Body Mass Index', 'unit': 'kg/m²', 'normal_range': (18.5, 24.9)},
        'Smoking': {'name': 'Smoking History', 'unit': '', 'normal_range': (0, 0)},
        'AlcoholDrinking': {'name': 'Heavy Alcohol Use', 'unit': '', 'normal_range': (0, 0)},
        'Stroke': {'name': 'Previous Stroke', 'unit': '', 'normal_range': (0, 0)},
        'PhysicalHealth': {'name': 'Poor Physical Health Days', 'unit': 'days', 'normal_range': (0, 5)},
        'MentalHealth': {'name': 'Mental Health Score', 'unit': 'days', 'normal_range': (0, 5)},
        'DiffWalking': {'name': 'Difficulty Walking', 'unit': '', 'normal_range': (0, 0)},
        'Sex': {'name': 'Biological Sex', 'unit': '', 'normal_range': (0, 1)},
        'AgeCategory': {'name': 'Age Category', 'unit': '', 'normal_range': None},
        'Race': {'name': 'Ethnicity', 'unit': '', 'normal_range': None},
        'Diabetic': {'name': 'Diabetic Status', 'unit': '', 'normal_range': (0, 0)},
        'PhysicalActivity': {'name': 'Regular Exercise', 'unit': '', 'normal_range': (1, 1)},
        'GenHealth': {'name': 'General Health', 'unit': '', 'normal_range': None},
        'SleepTime': {'name': 'Sleep Hours', 'unit': 'hours', 'normal_range': (7, 9)},
        'Asthma': {'name': 'Asthma History', 'unit': '', 'normal_range': (0, 0)},
        'KidneyDisease': {'name': 'Kidney Disease', 'unit': '', 'normal_range': (0, 0)},
        'SkinCancer': {'name': 'Skin Cancer History', 'unit': '', 'normal_range': (0, 0)}
    }

    def __init__(self):
        """Initialize the Heart Disease predictor."""
        self.knn_model = None
        self.dt_model = None
        self.transformer = None
        self.scaler = None
        self.feature_names = self.FEATURE_NAMES
        self.is_loaded = False

    def load_models(self, model_dir='models'):
        """
        Load trained models from disk.

        Args:
            model_dir (str): Directory containing model files
        """
        try:
            model_path = Path(model_dir)

            self.knn_model = joblib.load(model_path / 'knn_model.pkl')
            self.dt_model = joblib.load(model_path / 'dt_model.pkl')
            self.transformer = joblib.load(model_path / 'transformer.pkl')
            self.scaler = joblib.load(model_path / 'scaler.pkl')

            self.is_loaded = True
            print("✓ Heart Disease models loaded successfully")

        except Exception as e:
            raise Exception(f"Failed to load Heart Disease models: {str(e)}")

    def predict(self, features, model_choice='knn', user_data=None):
        """
        Make prediction with support for missing values.

        Args:
            features: Dictionary of features (can be incomplete)
            model_choice: 'knn' or 'dt' for model selection
            user_data: User profile data for filling missing values

        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            raise ValueError("Models must be loaded before making predictions")

        # Default values for missing indicators (healthy population medians)
        DEFAULT_VALUES = {
            'BMI': 26.5,  # Median BMI
            'Smoking': 0,  # No smoking (most common)
            'AlcoholDrinking': 0,  # No heavy drinking (most common)
            'Stroke': 0,  # No previous stroke (most common)
            'PhysicalHealth': 3.0,  # Few poor health days
            'MentalHealth': 3.0,  # Few poor mental health days
            'DiffWalking': 0,  # No difficulty walking (most common)
            'Sex': 1,  # Male (for default)
            'AgeCategory': '45-49',  # Middle age
            'Race': 'White',  # Most common in dataset
            'Diabetic': 0,  # No diabetes (most common)
            'PhysicalActivity': 1,  # Regular exercise (recommended)
            'GenHealth': 'Good',  # Good health (median)
            'SleepTime': 7.0,  # Recommended sleep
            'Asthma': 0,  # No asthma (most common)
            'KidneyDisease': 0,  # No kidney disease (most common)
            'SkinCancer': 0  # No skin cancer (most common)
        }

        # Override defaults with user profile data if available
        if user_data:
            if 'age' in user_data:
                DEFAULT_VALUES['AgeCategory'] = self._get_age_category(user_data['age'])
            if 'bmi' in user_data and user_data['bmi']:
                DEFAULT_VALUES['BMI'] = float(user_data['bmi'])
            if 'is_smoker' in user_data:
                DEFAULT_VALUES['Smoking'] = 1 if user_data['is_smoker'] else 0
            if 'is_diabetic' in user_data:
                DEFAULT_VALUES['Diabetic'] = 1 if user_data['is_diabetic'] else 0
            if 'physical_health_days' in user_data and user_data['physical_health_days'] is not None:
                DEFAULT_VALUES['PhysicalHealth'] = float(user_data['physical_health_days'])
            if 'overall_health' in user_data and user_data['overall_health']:
                DEFAULT_VALUES['GenHealth'] = user_data['overall_health']
            if 'sex' in user_data:
                DEFAULT_VALUES['Sex'] = 1 if user_data['sex'] == 'Male' else 0

        # Fill in missing values with defaults
        complete_features = {}
        missing_indicators = []

        for feature in self.FEATURE_NAMES:
            if feature in features and features[feature] is not None:
                complete_features[feature] = features[feature]
            else:
                complete_features[feature] = DEFAULT_VALUES[feature]
                missing_indicators.append(feature)

        # Create DataFrame
        input_df = pd.DataFrame([complete_features])

        # Apply preprocessing (transform and scale)
        transformed = self.transformer.transform(input_df)
        scaled = self.scaler.transform(transformed)

        # Choose model
        model = self.knn_model if model_choice == 'knn' else self.dt_model

        # Make prediction
        prediction = model.predict(scaled)[0]
        probabilities = model.predict_proba(scaled)[0]

        # Get confidence (probability of predicted class)
        confidence = probabilities[1] if prediction == 1 else probabilities[0]

        # Determine risk level
        heart_disease_probability = probabilities[1]
        if heart_disease_probability >= 0.8:
            risk_level = 'critical'
        elif heart_disease_probability >= 0.6:
            risk_level = 'high'
        elif heart_disease_probability >= 0.4:
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
            'prediction': 'heart_disease' if prediction == 1 else 'no_heart_disease',
            'confidence': float(confidence),
            'probability_heart_disease': float(heart_disease_probability),
            'probability_no_heart_disease': float(probabilities[0]),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'model_used': 'K-Nearest Neighbors' if model_choice == 'knn' else 'Decision Tree',
            'indicators_used': list(features.keys()),
            'indicators_missing': missing_indicators,
            'used_defaults': len(missing_indicators) > 0
        }

    def _get_age_category(self, age):
        """Convert numeric age to age category."""
        if age < 25:
            return "18-24"
        elif age < 30:
            return "25-29"
        elif age < 35:
            return "30-34"
        elif age < 40:
            return "35-39"
        elif age < 45:
            return "40-44"
        elif age < 50:
            return "45-49"
        elif age < 55:
            return "50-54"
        elif age < 60:
            return "55-59"
        elif age < 65:
            return "60-64"
        elif age < 70:
            return "65-69"
        elif age < 75:
            return "70-74"
        elif age < 80:
            return "75-79"
        else:
            return "80 or older"

    def _generate_recommendation(self, prediction, risk_level, features, missing_indicators):
        """Generate clinical recommendation based on prediction."""
        base_recommendation = ""

        if prediction == 1:  # Heart disease detected
            if risk_level == 'critical':
                base_recommendation = (
                    "URGENT: High risk of heart disease detected. "
                    "Immediate consultation with a cardiologist is strongly recommended. "
                    "Consider emergency evaluation within 24-48 hours."
                )
            elif risk_level == 'high':
                base_recommendation = (
                    "Elevated heart disease risk detected. "
                    "Please schedule an appointment with a cardiologist within the next week. "
                    "Lifestyle modifications and potential medication may be necessary."
                )
            else:
                base_recommendation = (
                    "Moderate heart disease risk detected. "
                    "Consult with your primary care physician for comprehensive cardiovascular evaluation. "
                    "Focus on heart-healthy lifestyle changes."
                )
        else:  # No heart disease
            base_recommendation = (
                "No significant heart disease indicators detected. "
                "Continue regular health checkups and maintain a heart-healthy lifestyle: "
                "balanced diet, regular exercise, stress management, and adequate sleep."
            )

        # Add note about user data usage
        base_recommendation += "\n\nThese predictions were made by the user data."

        # Add warning if many indicators were missing
        if len(missing_indicators) > 8:
            base_recommendation += (
                f"\n\nNote: {len(missing_indicators)} indicators were missing and default values were used. "
                "For more accurate results, please complete your health profile with all relevant information."
            )

        return base_recommendation


# Example usage
if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    predictor.load_models()

    # Sample patient data
    patient = {
        'BMI': 28.5,
        'Smoking': 1,
        'AlcoholDrinking': 0,
        'Stroke': 0,
        'PhysicalHealth': 10.0,
        'MentalHealth': 5.0,
        'DiffWalking': 0,
        'Sex': 1,
        'AgeCategory': '55-59',
        'Race': 'White',
        'Diabetic': 1,
        'PhysicalActivity': 0,
        'GenHealth': 'Fair',
        'SleepTime': 6.0,
        'Asthma': 0,
        'KidneyDisease': 0,
        'SkinCancer': 0
    }

    result = predictor.predict(patient, model_choice='knn')
    print("\nHeart Disease Prediction:")
    print(f"Result: {result['prediction']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"Risk Level: {result['risk_level'].upper()}")
    print(f"Model: {result['model_used']}")
    print(f"\nRecommendation: {result['recommendation']}")