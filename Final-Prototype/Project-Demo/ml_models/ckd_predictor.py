"""
CKD Prediction Model - Production Backend
Author: Shaurya Parshad
Version: 1.0

Professional ensemble model for Chronic Kidney Disease prediction.
Combines Random Forest, Gradient Boosting, and Logistic Regression.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')


class CKDPredictor:
    """
    Chronic Kidney Disease prediction model using ensemble learning.

    Attributes:
        model: Trained ensemble model
        scaler: Feature scaler
        imputer: Missing value imputer
        feature_names: List of expected feature names
        is_trained: Whether model has been trained
    """

    FEATURE_NAMES = ['hemo', 'pcv', 'sg', 'gfr', 'rbcc', 'al', 'dm', 'htn', 'sod', 'bp', 'sc']

    FEATURE_INFO = {
        'hemo': {'name': 'Hemoglobin', 'unit': 'g/dL', 'normal_range': (13, 17)},
        'pcv': {'name': 'Packed Cell Volume', 'unit': '%', 'normal_range': (38, 50)},
        'sg': {'name': 'Specific Gravity', 'unit': '', 'normal_range': (1.015, 1.025)},
        'gfr': {'name': 'GFR', 'unit': 'mL/min', 'normal_range': (90, 120)},
        'rbcc': {'name': 'RBC Count', 'unit': 'millions/cmm', 'normal_range': (4.5, 6.0)},
        'al': {'name': 'Albumin', 'unit': 'g/dL', 'normal_range': (0, 1)},
        'dm': {'name': 'Diabetes', 'unit': '', 'normal_range': (0, 0)},
        'htn': {'name': 'Hypertension', 'unit': '', 'normal_range': (0, 0)},
        'sod': {'name': 'Sodium', 'unit': 'mEq/L', 'normal_range': (135, 145)},
        'bp': {'name': 'Blood Pressure', 'unit': 'mmHg', 'normal_range': (110, 130)},
        'sc': {'name': 'Serum Creatinine', 'unit': 'mg/dL', 'normal_range': (0.6, 1.3)}
    }

    def __init__(self):
        """Initialize the CKD predictor."""
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = self.FEATURE_NAMES
        self.is_trained = False

    def _create_ensemble(self):
        """Create the ensemble model architecture."""
        return VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    random_state=42,
                    n_jobs=-1
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=42
                )),
                ('lr', LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ))
            ],
            voting='soft'
        )

    def train(self, data_path, test_size=0.2, random_state=42):
        """
        Train the ensemble model.

        Args:
            data_path (str): Path to training data CSV
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility

        Returns:
            dict: Training metrics and performance
        """
        try:
            # Load data
            df = pd.read_csv(data_path)

            # Validate data
            if not all(col in df.columns for col in self.feature_names):
                raise ValueError(f"Missing required features. Expected: {self.feature_names}")

            if 'class' not in df.columns:
                raise ValueError("Missing 'class' column in training data")

            # Prepare features and target
            X = df[self.feature_names].copy()
            y = (df['class'] == 'ckd').astype(int)

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )

            # Handle missing values
            self.imputer = SimpleImputer(strategy='median')
            X_train = pd.DataFrame(
                self.imputer.fit_transform(X_train),
                columns=self.feature_names
            )
            X_test = pd.DataFrame(
                self.imputer.transform(X_test),
                columns=self.feature_names
            )

            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Create and train ensemble
            self.model = self._create_ensemble()
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True

            # Evaluate
            y_pred = self.model.predict(X_test_scaled)

            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'f1_score': float(f1_score(y_test, y_pred)),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'trained_at': datetime.now().isoformat()
            }

            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(
                self.model,
                X_train_scaled,
                y_train,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )

            metrics['cv_mean'] = float(cv_scores.mean())
            metrics['cv_std'] = float(cv_scores.std())

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = {
                'true_negative': int(cm[0, 0]),
                'false_positive': int(cm[0, 1]),
                'false_negative': int(cm[1, 0]),
                'true_positive': int(cm[1, 1])
            }

            return metrics

        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")

    def predict(self, features):
        """
        Make prediction with support for missing values

        Args:
            features: Dictionary of features (can be incomplete)

        Returns:
            Dictionary with prediction results
        """
        # Default values for missing indicators (population medians)
        DEFAULT_VALUES = {
            'hemo': 12.5,  # Hemoglobin median
            'pcv': 42.0,  # PCV median
            'sg': 1.020,  # Specific Gravity median
            'gfr': 85.0,  # GFR median
            'rbcc': 4.8,  # RBC Count median
            'al': 0.5,  # Albumin median
            'dm': 0,  # No diabetes (most common)
            'htn': 0,  # No hypertension (most common)
            'sod': 140.0,  # Sodium median
            'bp': 120.0,  # Blood Pressure median
            'sc': 1.0  # Serum Creatinine median
        }

        # Required features in order
        required_features = ['hemo', 'pcv', 'sg', 'gfr', 'rbcc', 'al', 'dm', 'htn', 'sod', 'bp', 'sc']

        # Fill in missing values with defaults
        complete_features = {}
        missing_indicators = []

        for feature in required_features:
            if feature in features and features[feature] is not None:
                complete_features[feature] = float(features[feature])
            else:
                complete_features[feature] = DEFAULT_VALUES[feature]
                missing_indicators.append(feature)

        # Create feature array in correct order
        X = np.array([[complete_features[f] for f in required_features]])

        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # Get confidence (probability of predicted class)
        confidence = probabilities[1] if prediction == 1 else probabilities[0]

        # Determine risk level
        ckd_probability = probabilities[1]
        if ckd_probability >= 0.8:
            risk_level = 'critical'
        elif ckd_probability >= 0.6:
            risk_level = 'high'
        elif ckd_probability >= 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Generate recommendation
        if prediction == 1:  # CKD detected
            if risk_level == 'critical':
                recommendation = "URGENT: High CKD risk detected. Immediate consultation with a nephrologist is strongly recommended."
            elif risk_level == 'high':
                recommendation = "Elevated CKD risk detected. Please consult with a nephrologist within the next few days."
            else:
                recommendation = "Moderate CKD indicators detected. Schedule an appointment with your doctor for further evaluation."
        else:
            recommendation = "No significant CKD indicators detected. Continue regular health checkups and maintain a healthy lifestyle."

        # Add warning if many indicators were missing
        if len(missing_indicators) > 5:
            recommendation += f"\n\nNote: {len(missing_indicators)} indicators were missing and default values were used. For more accurate results, please provide a complete blood report."

        return {
            'prediction': 'ckd' if prediction == 1 else 'not_ckd',
            'confidence': float(confidence),
            'probability_ckd': float(ckd_probability),
            'probability_not_ckd': float(probabilities[0]),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'indicators_used': list(features.keys()),
            'indicators_missing': missing_indicators,
            'used_defaults': len(missing_indicators) > 0
        }

    def predict_batch(self, patients_data):
        """
        Make predictions for multiple patients.

        Args:
            patients_data (list): List of feature dictionaries

        Returns:
            list: List of prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        results = []
        for patient in patients_data:
            try:
                result = self.predict(patient)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'prediction': None
                })

        return results

    def _identify_abnormal_indicators(self, features):
        """Identify which indicators are outside normal range."""
        abnormal = []

        for feature, value in features.items():
            if feature not in self.FEATURE_INFO:
                continue

            info = self.FEATURE_INFO[feature]
            normal_min, normal_max = info['normal_range']

            # Skip if no value
            if value is None or pd.isna(value):
                continue

            # Check if abnormal
            if value < normal_min or value > normal_max:
                status = 'low' if value < normal_min else 'high'
                abnormal.append({
                    'indicator': info['name'],
                    'value': float(value),
                    'unit': info['unit'],
                    'normal_range': f"{normal_min}-{normal_max}",
                    'status': status
                })

        return abnormal

    def _generate_recommendation(self, prediction, confidence, abnormal_indicators):
        """Generate clinical recommendation based on prediction."""
        if prediction == 1:  # CKD detected
            if confidence >= 0.8:
                return (
                    "High risk of Chronic Kidney Disease detected. "
                    "Immediate consultation with a nephrologist is strongly recommended. "
                    "Further diagnostic tests and treatment planning are necessary."
                )
            elif confidence >= 0.6:
                return (
                    "Moderate to high risk of Chronic Kidney Disease detected. "
                    "Schedule an appointment with a nephrologist for comprehensive evaluation. "
                    "Monitor kidney function regularly."
                )
            else:
                return (
                    "Possible Chronic Kidney Disease detected. "
                    "Consult with your primary care physician for further evaluation. "
                    "Follow-up testing may be necessary."
                )
        else:  # Healthy
            if len(abnormal_indicators) > 0:
                return (
                    "No significant kidney disease detected, but some indicators are outside normal range. "
                    "Routine monitoring recommended. Maintain healthy lifestyle and regular check-ups."
                )
            else:
                return (
                    "No kidney disease detected. Kidney function appears normal. "
                    "Continue regular health check-ups and maintain a healthy lifestyle."
                )

    def get_feature_importance(self):
        """
        Get feature importance from Random Forest component.

        Returns:
            dict: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        rf_model = self.model.named_estimators_['rf']
        importance = rf_model.feature_importances_

        feature_importance = {
            feature: float(importance[i])
            for i, feature in enumerate(self.feature_names)
        }

        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_importance

    def save_model(self, filepath):
        """
        Save trained model to disk.

        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'saved_at': datetime.now().isoformat()
        }

        joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        """
        Load trained model from disk.

        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.feature_names = model_data['feature_names']
        self.is_trained = True


# Training script
def train_model(data_path='ckd_40k_with_creatinine.csv', model_path='ckd_model.pkl'):
    """
    Train and save the CKD prediction model.

    Args:
        data_path (str): Path to training data
        model_path (str): Path to save trained model

    Returns:
        dict: Training metrics
    """
    predictor = CKDPredictor()

    print("Training CKD Prediction Model...")
    metrics = predictor.train(data_path)

    print("\nTraining Complete!")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Recall: {metrics['recall'] * 100:.2f}%")
    print(f"Cross-Validation: {metrics['cv_mean'] * 100:.2f}% (±{metrics['cv_std'] * 100:.2f}%)")

    predictor.save_model(model_path)
    print(f"\nModel saved to: {model_path}")

    return metrics


# Prediction example
def predict_patient(model_path='ckd_model.pkl'):
    """
    Example prediction on a sample patient.

    Args:
        model_path (str): Path to trained model
    """
    predictor = CKDPredictor()
    predictor.load_model(model_path)

    # Sample patient with CKD indicators
    patient = {
        'hemo': 6.5,
        'pcv': 28,
        'sg': 1.008,
        'gfr': 35,
        'rbcc': 3.0,
        'al': 3,
        'dm': 1,
        'htn': 1,
        'sod': 135,
        'bp': 160,
        'sc': 4.2
    }

    result = predictor.predict(patient)

    print("\nPrediction Result:")
    print(f"Diagnosis: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print(f"Risk Level: {result['risk_level'].upper()}")
    print(f"\nRecommendation: {result['recommendation']}")

    if result['abnormal_indicators']:
        print("\nAbnormal Indicators:")
        for indicator in result['abnormal_indicators']:
            print(
                f"  - {indicator['indicator']}: {indicator['value']} {indicator['unit']} ({indicator['status'].upper()})")

    return result


if __name__ == "__main__":
    # Train model
    metrics = train_model()

    # Test prediction
    predict_patient()