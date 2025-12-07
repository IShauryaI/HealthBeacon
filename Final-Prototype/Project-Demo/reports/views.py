"""
Views for blood report management
Author: Shaurya Parshad
Description: Upload, process, and manage blood reports with auto-prediction and SHAP for ALL 3 diseases
"""

from rest_framework import status, generics
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from django.utils import timezone
from django.conf import settings

from .models import BloodReport, MedicalIndicator
from .serializers import BloodReportSerializer, BloodReportUploadSerializer
from .ocr_processor import process_blood_report, get_missing_indicators
from predictions.models import Prediction

import sys
import os
import numpy as np

# Import all 3 predictors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load CKD Model
try:
    from ml_models.ckd_predictor import CKDPredictor

    ckd_model = CKDPredictor()
    ckd_model.load_model(settings.ML_MODEL_PATH)
    print("✓ CKD model loaded in reports/views.py")
except Exception as e:
    ckd_model = None
    print(f"Warning: CKD model not loaded: {e}")

# Load Diabetes Model
try:
    from ml_models.diabetes_predictor import DiabetesPredictor

    diabetes_model = DiabetesPredictor()
    models_dir = os.path.dirname(settings.ML_MODEL_PATH)
    diabetes_model.load_models(models_dir)
    print("✓ Diabetes model loaded in reports/views.py")
except Exception as e:
    diabetes_model = None
    print(f"Warning: Diabetes model not loaded: {e}")

# Load Heart Disease Model
try:
    from ml_models.heart_predictor import HeartDiseasePredictor

    heart_model = HeartDiseasePredictor()
    models_dir = os.path.dirname(settings.ML_MODEL_PATH)
    heart_model.load_models(models_dir)
    print("✓ Heart Disease model loaded in reports/views.py")
except Exception as e:
    heart_model = None
    print(f"Warning: Heart Disease model not loaded: {e}")


def _get_age_category(age):
    """Convert numeric age to age category for Heart Disease model."""
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


class BloodReportUploadView(APIView):
    """
    Upload blood report with automatic OCR processing and predictions for ALL diseases.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        if request.user.user_type != 'patient':
            return Response(
                {'error': 'Only patients can upload blood reports'},
                status=status.HTTP_403_FORBIDDEN
            )

        serializer = BloodReportUploadSerializer(data=request.data)

        if serializer.is_valid():
            blood_report = serializer.save(
                patient=request.user,
                status='processing'
            )

            file_path = blood_report.report_file.path
            print(f"\n{'=' * 60}")
            print(f"Processing blood report: {file_path}")
            print(f"{'=' * 60}")

            ocr_result = process_blood_report(file_path)

            predictions_created = []

            if ocr_result['success'] and ocr_result['indicators']:
                print(f"✓ OCR successful - Found {len(ocr_result['indicators'])} indicators")
                print(f"  Indicators: {list(ocr_result['indicators'].keys())}")

                blood_report.ocr_extracted_data = ocr_result['indicators']
                blood_report.status = 'completed'
                blood_report.processed_date = timezone.now()
                blood_report.save()

                # Save medical indicators
                ranges = {
                    'hemo': {'min': 13, 'max': 17, 'unit': 'g/dL'},
                    'pcv': {'min': 38, 'max': 50, 'unit': '%'},
                    'sg': {'min': 1.015, 'max': 1.025, 'unit': ''},
                    'gfr': {'min': 90, 'max': 120, 'unit': 'mL/min'},
                    'rbcc': {'min': 4.5, 'max': 6.0, 'unit': 'millions/cmm'},
                    'al': {'min': 0, 'max': 1, 'unit': 'g/dL'},
                    'dm': {'min': 0, 'max': 0, 'unit': ''},
                    'htn': {'min': 0, 'max': 0, 'unit': ''},
                    'sod': {'min': 135, 'max': 145, 'unit': 'mEq/L'},
                    'bp': {'min': 110, 'max': 130, 'unit': 'mmHg'},
                    'sc': {'min': 0.6, 'max': 1.3, 'unit': 'mg/dL'},
                    # Diabetes indicators
                    'age': {'min': 0, 'max': 120, 'unit': 'years'},
                    'bmi': {'min': 15, 'max': 50, 'unit': 'kg/m²'},
                    'glucose_fasting': {'min': 70, 'max': 100, 'unit': 'mg/dL'},
                    'glucose_postprandial': {'min': 0, 'max': 140, 'unit': 'mg/dL'},
                    'hba1c': {'min': 4.0, 'max': 5.6, 'unit': '%'},
                    'insulin_level': {'min': 2.6, 'max': 24.9, 'unit': 'µU/mL'},
                    'cholesterol_total': {'min': 125, 'max': 200, 'unit': 'mg/dL'},
                    'hdl_cholesterol': {'min': 40, 'max': 60, 'unit': 'mg/dL'},
                    'ldl_cholesterol': {'min': 0, 'max': 100, 'unit': 'mg/dL'},
                    'triglycerides': {'min': 0, 'max': 150, 'unit': 'mg/dL'},
                    'systolic_bp': {'min': 90, 'max': 120, 'unit': 'mmHg'},
                    'diastolic_bp': {'min': 60, 'max': 80, 'unit': 'mmHg'},
                }

                for indicator_name, value in ocr_result['indicators'].items():
                    ref_range = ranges.get(indicator_name, {'min': 0, 'max': 100, 'unit': ''})
                    is_abnormal = (value < ref_range['min'] or value > ref_range['max'])

                    MedicalIndicator.objects.create(
                        blood_report=blood_report,
                        indicator_name=indicator_name,
                        indicator_value=value,
                        unit=ref_range['unit'],
                        reference_range_min=ref_range['min'],
                        reference_range_max=ref_range['max'],
                        is_abnormal=is_abnormal
                    )

                # ========== RUN ALL 3 PREDICTIONS ==========

                # 1. CKD PREDICTION
                if ckd_model:
                    try:
                        print(f"\n{'=' * 60}")
                        print(f"Running CKD Prediction...")
                        print(f"{'=' * 60}")

                        result = ckd_model.predict(ocr_result['indicators'])

                        print(f"✓ CKD Prediction successful!")
                        print(f"  Result: {result['prediction']}")
                        print(f"  Confidence: {result['confidence']:.1%}")
                        print(f"  Risk Level: {result['risk_level']}")

                        # Generate SHAP for CKD
                        shap_chart_path = ''
                        shap_values_data = {}

                        try:
                            from predictions.shap_generator import generate_shap_charts

                            default_values = {
                                'hemo': 12.5, 'pcv': 42.0, 'sg': 1.020, 'gfr': 85.0,
                                'rbcc': 4.8, 'al': 0.5, 'dm': 0, 'htn': 0,
                                'sod': 140.0, 'bp': 120.0, 'sc': 1.0
                            }

                            all_features = ['hemo', 'pcv', 'sg', 'gfr', 'rbcc', 'al', 'dm', 'htn', 'sod', 'bp', 'sc']
                            complete = {f: ocr_result['indicators'].get(f, default_values[f]) for f in all_features}
                            X_data = np.array([[complete[f] for f in all_features]])

                            rf_model = None
                            if hasattr(ckd_model.model, 'named_estimators_'):
                                rf_model = (ckd_model.model.named_estimators_.get('rf') or
                                            ckd_model.model.named_estimators_.get('random_forest'))
                            elif hasattr(ckd_model.model, 'feature_importances_'):
                                rf_model = ckd_model.model

                            if rf_model:
                                shap_data = generate_shap_charts(rf_model, X_data, all_features,
                                                                 f"ckd_{blood_report.id}")
                                if shap_data:
                                    shap_chart_path = shap_data['waterfall']
                                    shap_values_data = shap_data['shap_values']
                                    print(f"✓ CKD SHAP charts generated")

                        except Exception as e:
                            print(f"✗ CKD SHAP generation failed: {e}")

                        # Save CKD prediction
                        ckd_prediction = Prediction.objects.create(
                            blood_report=blood_report,
                            patient=request.user,
                            disease_type='ckd',
                            prediction_result='positive' if result['prediction'] == 'ckd' else 'negative',
                            confidence_score=result['confidence'],
                            risk_level=result['risk_level'],
                            indicators=ocr_result['indicators'],
                            recommendation=result['recommendation'],
                            shap_chart_path=shap_chart_path,
                            shap_values=shap_values_data
                        )

                        predictions_created.append({
                            'disease': 'ckd',
                            'id': str(ckd_prediction.id),
                            'result': result['prediction'],
                            'confidence': result['confidence'],
                            'shap': bool(shap_chart_path)
                        })

                        print(f"✓ CKD prediction saved")

                    except Exception as e:
                        print(f"✗ CKD prediction failed: {e}")
                        import traceback
                        traceback.print_exc()

                # 2. DIABETES PREDICTION
                if diabetes_model:
                    try:
                        print(f"\n{'=' * 60}")
                        print(f"Running Diabetes Prediction...")
                        print(f"{'=' * 60}")

                        result = diabetes_model.predict(ocr_result['indicators'])

                        print(f"✓ Diabetes Prediction successful!")
                        print(f"  Result: {result['prediction']}")
                        print(f"  Confidence: {result['confidence']:.1%}")
                        print(f"  Risk Level: {result['risk_level']}")

                        # Generate SHAP for Diabetes
                        shap_chart_path = ''
                        shap_values_data = {}

                        try:
                            from predictions.shap_generator import generate_shap_charts

                            # Diabetes default values
                            default_values = {
                                'age': 45, 'bmi': 25.0, 'systolic_bp': 120, 'diastolic_bp': 80,
                                'cholesterol_total': 180, 'hdl_cholesterol': 50, 'ldl_cholesterol': 100,
                                'triglycerides': 120, 'glucose_fasting': 90, 'glucose_postprandial': 120,
                                'insulin_level': 10.0, 'hba1c': 5.3
                            }

                            complete = {f: ocr_result['indicators'].get(f, default_values[f]) for f in
                                        diabetes_model.feature_names}
                            X_data = np.array([[complete[f] for f in diabetes_model.feature_names]])

                            # Diabetes model is Random Forest
                            if hasattr(diabetes_model.model, 'feature_importances_'):
                                shap_data = generate_shap_charts(
                                    diabetes_model.model,
                                    X_data,
                                    diabetes_model.feature_names,
                                    f"diabetes_{blood_report.id}"
                                )
                                if shap_data:
                                    shap_chart_path = shap_data['waterfall']
                                    shap_values_data = shap_data['shap_values']
                                    print(f"✓ Diabetes SHAP charts generated")

                        except Exception as e:
                            print(f"✗ Diabetes SHAP generation failed: {e}")

                        # Save Diabetes prediction
                        diabetes_prediction = Prediction.objects.create(
                            blood_report=blood_report,
                            patient=request.user,
                            disease_type='diabetes',
                            prediction_result='positive' if result['prediction'] == 'diabetes' else 'negative',
                            confidence_score=result['confidence'],
                            risk_level=result['risk_level'],
                            indicators=ocr_result['indicators'],
                            recommendation=result['recommendation'],
                            shap_chart_path=shap_chart_path,
                            shap_values=shap_values_data
                        )

                        predictions_created.append({
                            'disease': 'diabetes',
                            'id': str(diabetes_prediction.id),
                            'result': result['prediction'],
                            'confidence': result['confidence'],
                            'shap': bool(shap_chart_path)
                        })

                        print(f"✓ Diabetes prediction saved")

                    except Exception as e:
                        print(f"✗ Diabetes prediction failed: {e}")
                        import traceback
                        traceback.print_exc()

                # 3. HEART DISEASE PREDICTION
                if heart_model:
                    try:
                        print(f"\n{'=' * 60}")
                        print(f"Running Heart Disease Prediction...")
                        print(f"{'=' * 60}")

                        # Heart Disease uses USER PROFILE data with EXACT feature names
                        user = blood_report.patient

                        # Build features with CORRECT names matching heart_predictor.py FEATURE_NAMES
                        heart_indicators = {
                            'BMI': float(getattr(user, 'bmi', 26.5)),
                            'Smoking': 1 if getattr(user, 'is_smoker', False) else 0,
                            'AlcoholDrinking': 1 if getattr(user, 'alcohol_drinking', False) else 0,
                            'Stroke': 1 if getattr(user, 'previous_stroke', False) else 0,
                            'PhysicalHealth': float(getattr(user, 'physical_health_days', 3)),
                            'MentalHealth': float(getattr(user, 'mental_health_days', 3)),
                            'DiffWalking': 1 if getattr(user, 'difficulty_walking', False) else 0,
                            'Sex': 1 if getattr(user, 'sex', 'Male') == 'Male' else 0,
                            'AgeCategory': _get_age_category(getattr(user, 'age', 40)),
                            'Race': getattr(user, 'race', 'White'),
                            'Diabetic': 1 if getattr(user, 'is_diabetic', False) else 0,
                            'PhysicalActivity': 1 if getattr(user, 'physical_activity', True) else 0,
                            'GenHealth': getattr(user, 'overall_health', 'Good'),
                            'SleepTime': float(getattr(user, 'sleep_hours', 7)),
                            'Asthma': 1 if getattr(user, 'has_asthma', False) else 0,
                            'KidneyDisease': 1 if getattr(user, 'has_kidney_disease', False) else 0,
                            'SkinCancer': 1 if getattr(user, 'has_skin_cancer', False) else 0
                        }

                        print(f"Collected {len(heart_indicators)} health profile indicators with CORRECT names")
                        print(
                            f"  Sample: BMI={heart_indicators['BMI']}, Smoking={heart_indicators['Smoking']}, Diabetic={heart_indicators['Diabetic']}")

                        # Make prediction using KNN model
                        result = heart_model.predict(heart_indicators, model_choice='knn')

                        print(f"✓ Heart Disease Prediction successful!")
                        print(f"  Result: {result['prediction']}")
                        print(f"  Confidence: {result['confidence']:.1%}")
                        print(f"  Heart Disease Probability: {result.get('probability_heart_disease', 0):.1%}")
                        print(f"  Risk Level: {result['risk_level']}")

                        # Generate Feature Importance chart
                        shap_chart_path = ''

                        try:
                            print(f"Generating feature importance for Heart Disease...")
                            from predictions.shap_generator import generate_feature_importance_chart

                            chart_path = generate_feature_importance_chart(
                                heart_indicators,
                                result['prediction'],
                                f"heart_{blood_report.id}"
                            )

                            if chart_path:
                                shap_chart_path = chart_path
                                print(f"✓ Feature importance chart generated")

                        except Exception as e:
                            print(f"✗ Feature importance generation failed: {e}")
                            import traceback
                            traceback.print_exc()

                        # Save Heart Disease prediction with CORRECT indicators
                        heart_prediction = Prediction.objects.create(
                            blood_report=blood_report,
                            patient=request.user,
                            disease_type='heart_disease',
                            prediction_result='positive' if result['prediction'] == 'heart_disease' else 'negative',
                            confidence_score=result['confidence'],
                            risk_level=result['risk_level'],
                            indicators=heart_indicators,
                            recommendation=result['recommendation'],
                            shap_chart_path=shap_chart_path,
                            shap_values={}
                        )

                        predictions_created.append({
                            'disease': 'heart_disease',
                            'id': str(heart_prediction.id),
                            'result': result['prediction'],
                            'confidence': result['confidence'],
                            'shap': bool(shap_chart_path)
                        })

                        print(f"✓ Heart Disease prediction saved")

                    except Exception as e:
                        print(f"✗ Heart Disease prediction error: {e}")
                        import traceback
                        traceback.print_exc()

                print(f"\n{'=' * 60}")
                print(f"✓ All predictions completed!")
                print(f"  Total predictions: {len(predictions_created)}")
                for pred in predictions_created:
                    print(f"  - {pred['disease']}: {pred['result']} ({pred['confidence']:.1%})")
                print(f"{'=' * 60}\n")

                # Prepare response
                missing = get_missing_indicators(ocr_result['indicators'])

                response_data = {
                    'success': True,
                    'report': BloodReportSerializer(blood_report).data,
                    'ocr_result': {
                        'indicators_found': ocr_result['extracted_count'],
                        'indicators_list': list(ocr_result['indicators'].keys()),
                        'missing_indicators': missing,
                        'warnings': ocr_result.get('warnings', [])
                    },
                    'predictions_created': len(predictions_created),
                    'predictions': predictions_created,
                    'message': (
                        f'✓ Blood report uploaded and processed successfully! '
                        f'Found {len(ocr_result["indicators"])} indicators. '
                        f'{len(predictions_created)} disease predictions completed automatically.'
                    )
                }

                return Response(response_data, status=status.HTTP_201_CREATED)

            else:
                print(f"✗ OCR failed or no indicators found")
                blood_report.status = 'completed'
                blood_report.processed_date = timezone.now()
                blood_report.save()

                return Response({
                    'success': True,
                    'report': BloodReportSerializer(blood_report).data,
                    'message': 'Report uploaded but could not extract indicators automatically.',
                    'ocr_error': ocr_result.get('error', 'Unknown error')
                }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class BloodReportListView(generics.ListAPIView):
    """List all blood reports for the authenticated user."""
    permission_classes = [IsAuthenticated]
    serializer_class = BloodReportSerializer

    def get_queryset(self):
        user = self.request.user
        if user.user_type == 'doctor':
            return BloodReport.objects.all().order_by('-upload_date')
        else:
            return BloodReport.objects.filter(patient=user).order_by('-upload_date')


class BloodReportDetailView(generics.RetrieveAPIView):
    """Get detailed blood report with all indicators."""
    permission_classes = [IsAuthenticated]
    serializer_class = BloodReportSerializer

    def get_queryset(self):
        user = self.request.user
        if user.user_type == 'doctor':
            return BloodReport.objects.all()
        else:
            return BloodReport.objects.filter(patient=user)


class BloodReportDeleteView(generics.DestroyAPIView):
    """Delete a blood report (patients only can delete their own)."""
    permission_classes = [IsAuthenticated]
    serializer_class = BloodReportSerializer

    def get_queryset(self):
        return BloodReport.objects.filter(patient=self.request.user)