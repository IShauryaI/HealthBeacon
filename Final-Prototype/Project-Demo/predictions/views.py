"""
Views for disease predictions
Author: Shaurya Parshad
Description: Handle CKD predictions with SHAP visualizations for doctors
"""

from rest_framework import status, generics
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from django.conf import settings
from django.db import models
from rest_framework.decorators import api_view

from .models import Prediction
from .serializers import PredictionSerializer, SimplePredictionSerializer
from reports.models import BloodReport
from .shap_generator import generate_shap_charts
from patient_blockchain import PatientBlockchain
from django.contrib.auth import get_user_model

User = get_user_model()


import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path to import predictors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ml_models.ckd_predictor import CKDPredictor

    # Load CKD model once when server starts
    ckd_model = CKDPredictor()
    ckd_model.load_model(settings.ML_MODEL_PATH)
    print("✓ CKD model loaded successfully in predictions/views.py")
except Exception as e:
    ckd_model = None
    print(f"Warning: Could not load CKD model in predictions/views.py: {e}")

try:
    from ml_models.diabetes_predictor import DiabetesPredictor

    # Load Diabetes model once when server starts
    diabetes_model = DiabetesPredictor()
    models_dir = os.path.dirname(settings.ML_MODEL_PATH)
    diabetes_model.load_models(models_dir)
    print("✓ Diabetes model loaded successfully in predictions/views.py")
except Exception as e:
    diabetes_model = None
    print(f"Warning: Could not load Diabetes model in predictions/views.py: {e}")

try:
    from ml_models.heart_predictor import HeartDiseasePredictor

    # Load Heart Disease models once when server starts
    heart_model = HeartDiseasePredictor()
    models_dir = os.path.dirname(settings.ML_MODEL_PATH)
    heart_model.load_models(models_dir)
    print("✓ Heart Disease models loaded successfully in predictions/views.py")
except Exception as e:
    heart_model = None
    print(f"Warning: Could not load Heart Disease models in predictions/views.py: {e}")


class CKDPredictionView(APIView):
    """
    Make CKD prediction from blood report with SHAP visualization.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            if not ckd_model:
                return Response(
                    {'error': 'CKD model not loaded. Please contact administrator.'},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )

            blood_report_id = request.data.get('blood_report_id')
            indicators = request.data.get('indicators')

            if not blood_report_id or not indicators:
                return Response(
                    {'error': 'blood_report_id and indicators are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            blood_report = get_object_or_404(BloodReport, id=blood_report_id)

            if blood_report.patient != request.user and request.user.user_type != 'doctor':
                return Response(
                    {'error': 'Permission denied'},
                    status=status.HTTP_403_FORBIDDEN
                )

            print(f"\n{'=' * 60}")
            print(f"Making CKD Prediction")
            print(f"{'=' * 60}")

            result = ckd_model.predict(indicators)

            print(f"✓ Prediction completed")
            print(f"  Result: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.1%}")

            # Generate SHAP charts for doctor view
            shap_data = None

            if request.user.user_type == 'doctor':
                try:
                    print(f"\nGenerating SHAP visualizations...")

                    feature_names = list(indicators.keys())
                    feature_values = list(indicators.values())
                    X_data = np.array([feature_values])

                    rf_model = None
                    if hasattr(ckd_model.model, 'named_estimators_'):
                        rf_model = ckd_model.model.named_estimators_.get('rf')
                    elif hasattr(ckd_model.model, 'feature_importances_'):
                        rf_model = ckd_model.model

                    if rf_model:
                        shap_data = generate_shap_charts(
                            rf_model,
                            X_data,
                            feature_names,
                            str(blood_report.id)
                        )

                        if shap_data:
                            print(f"✓ SHAP charts generated")
                    else:
                        print(f"✗ Could not extract Random Forest model")

                except Exception as e:
                    print(f"✗ SHAP generation error: {e}")

            # Save prediction
            prediction = Prediction.objects.create(
                blood_report=blood_report,
                patient=blood_report.patient,
                disease_type='ckd',
                prediction_result='positive' if result['prediction'] == 'ckd' else 'negative',
                confidence_score=result['confidence'],
                risk_level=result['risk_level'],
                indicators=indicators,
                recommendation=result['recommendation'],
                shap_chart_path=shap_data['waterfall'] if shap_data else '',
                shap_values=shap_data['shap_values'] if shap_data else {}
            )

            print(f"✓ Prediction saved (ID: {prediction.id})")
            print(f"{'=' * 60}\n")

            response_data = {
                'prediction_id': str(prediction.id),
                'result': result
            }

            if request.user.user_type == 'doctor' and shap_data:
                response_data['shap_charts'] = {
                    'waterfall': f'/media/{shap_data["waterfall"]}',
                    'summary': f'/media/{shap_data["summary"]}'
                }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"\n✗ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PredictionListView(generics.ListAPIView):
    """
    List predictions based on user type and blockchain access.
    """

    permission_classes = [IsAuthenticated]
    serializer_class = PredictionSerializer

    def get_queryset(self):
        user = self.request.user

        if user.user_type == 'patient':
            return Prediction.objects.filter(patient=user).order_by('-created_at')

        elif user.user_type == 'doctor':
            from patient_blockchain import PatientBlockchain

            try:
                bc = PatientBlockchain("chain.json")
                doctor_id = str(user.id)
                all_predictions = Prediction.objects.all().order_by('-created_at')
                accessible_predictions = []

                for pred in all_predictions:
                    pred_name = f"{pred.disease_type}_pred_{pred.created_at.strftime('%Y%m%d_%H%M%S')}"
                    access_list = bc.get_current_access(pred_name)

                    if doctor_id in access_list:
                        accessible_predictions.append(pred.id)

                return Prediction.objects.filter(id__in=accessible_predictions).order_by('-created_at')

            except Exception as e:
                print(f"Error checking blockchain access: {e}")
                return Prediction.objects.none()

        else:
            return Prediction.objects.none()


class PredictionDetailView(generics.RetrieveAPIView):
    """
    Get detailed prediction with SHAP charts and analysis.
    """

    permission_classes = [IsAuthenticated]

    def get_serializer_class(self):
        if self.request.user.user_type == 'doctor':
            return PredictionSerializer
        return SimplePredictionSerializer

    def get_queryset(self):
        user = self.request.user

        if user.user_type == 'doctor':
            return Prediction.objects.all().select_related('patient', 'blood_report')
        else:
            return Prediction.objects.filter(patient=user).select_related('blood_report')

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        data = serializer.data

        # Add SHAP chart URLs for doctors
        if request.user.user_type == 'doctor' and instance.shap_chart_path:
            data['shap_charts'] = {
                'waterfall': f'/media/{instance.shap_chart_path}',
                'summary': f'/media/{instance.shap_chart_path.replace("_waterfall", "_summary")}'
            }

        return Response(data)


class DiabetesPredictionView(APIView):
    """
    Make Diabetes prediction from blood report.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            if not diabetes_model:
                return Response(
                    {'error': 'Diabetes model not loaded.'},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )

            blood_report_id = request.data.get('blood_report_id')
            indicators = request.data.get('indicators', {})

            blood_report = None
            if blood_report_id:
                blood_report = get_object_or_404(BloodReport, id=blood_report_id)

                if blood_report.patient != request.user and request.user.user_type != 'doctor':
                    return Response(
                        {'error': 'Permission denied'},
                        status=status.HTTP_403_FORBIDDEN
                    )

            print(f"\n{'=' * 60}")
            print(f"Making Diabetes Prediction")
            print(f"{'=' * 60}")

            # Get user profile data
            user_data = {}
            try:
                if hasattr(request.user, 'age'):
                    user_data['age'] = request.user.age
                if hasattr(request.user, 'bmi'):
                    user_data['bmi'] = request.user.bmi
            except Exception as e:
                print(f"Warning: Could not collect user profile data: {e}")

            result = diabetes_model.predict(indicators, user_data=user_data)

            print(f"✓ Prediction completed")
            print(f"  Result: {result['prediction']}")

            prediction = Prediction.objects.create(
                blood_report=blood_report,
                patient=request.user,
                disease_type='diabetes',
                prediction_result='positive' if result['prediction'] == 'diabetes' else 'negative',
                confidence_score=result['confidence'],
                risk_level=result['risk_level'],
                indicators=indicators,
                recommendation=result['recommendation'],
                shap_chart_path='',
                shap_values={}
            )

            print(f"✓ Prediction saved (ID: {prediction.id})")
            print(f"{'=' * 60}\n")

            return Response({
                'prediction_id': str(prediction.id),
                'result': result
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"\n✗ Diabetes prediction error: {e}")
            import traceback
            traceback.print_exc()
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HeartDiseasePredictionView(APIView):
    """
    Make Heart Disease prediction from user profile.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            if not heart_model:
                return Response(
                    {'error': 'Heart Disease model not loaded.'},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )

            blood_report_id = request.data.get('blood_report_id')
            indicators = request.data.get('indicators', {})
            model_choice = request.data.get('model_choice', 'knn')

            blood_report = None
            if blood_report_id:
                blood_report = get_object_or_404(BloodReport, id=blood_report_id)

                if blood_report.patient != request.user and request.user.user_type != 'doctor':
                    return Response(
                        {'error': 'Permission denied'},
                        status=status.HTTP_403_FORBIDDEN
                    )

            print(f"\n{'=' * 60}")
            print(f"Making Heart Disease Prediction")
            print(f"{'=' * 60}")

            # Get user profile data
            user_data = {}
            try:
                if hasattr(request.user, 'age'):
                    user_data['age'] = request.user.age
                if hasattr(request.user, 'bmi'):
                    user_data['bmi'] = request.user.bmi
                if hasattr(request.user, 'is_smoker'):
                    user_data['is_smoker'] = request.user.is_smoker
            except Exception as e:
                print(f"Warning: Could not collect user profile data: {e}")

            result = heart_model.predict(
                indicators,
                model_choice=model_choice,
                user_data=user_data
            )

            print(f"✓ Prediction completed")

            prediction = Prediction.objects.create(
                blood_report=blood_report,
                patient=request.user,
                disease_type='heart_disease',
                prediction_result='positive' if result['prediction'] == 'heart_disease' else 'negative',
                confidence_score=result['confidence'],
                risk_level=result['risk_level'],
                indicators=indicators,
                recommendation=result['recommendation'],
                shap_chart_path='',
                shap_values={}
            )

            print(f"✓ Prediction saved (ID: {prediction.id})")
            print(f"{'=' * 60}\n")

            return Response({
                'prediction_id': str(prediction.id),
                'result': result
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"\n✗ Heart Disease prediction error: {e}")
            import traceback
            traceback.print_exc()
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class UpdateClinicalNotesView(APIView):
    """
    Update clinical notes for a prediction (doctors only).
    """

    permission_classes = [IsAuthenticated]

    def patch(self, request, pk):
        if request.user.user_type != 'doctor':
            return Response(
                {'error': 'Only doctors can add clinical notes'},
                status=status.HTTP_403_FORBIDDEN
            )

        try:
            prediction = get_object_or_404(Prediction, id=pk)

            doctor_notes = request.data.get('doctor_notes', '')
            prediction.doctor_notes = doctor_notes

            if request.data.get('reviewed', False):
                prediction.reviewed_by_doctor = request.user

            prediction.save()

            print(f"✓ Clinical notes updated for prediction {pk}")

            serializer = PredictionSerializer(prediction)
            return Response({
                'success': True,
                'message': 'Clinical notes saved successfully',
                'prediction': serializer.data
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"✗ Failed to update clinical notes: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GrantAccessView(APIView):
    """
    Grant doctor access to patient predictions via blockchain.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            if request.user.user_type != 'patient':
                return Response(
                    {'error': 'Only patients can grant access'},
                    status=status.HTTP_403_FORBIDDEN
                )

            doctor_email = request.data.get('doctor_email')
            prediction_ids = request.data.get('prediction_ids', [])

            if not doctor_email:
                return Response(
                    {'error': 'doctor_email is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            try:
                doctor = User.objects.get(email=doctor_email, user_type='doctor')
            except User.DoesNotExist:
                return Response(
                    {'error': 'Doctor not found'},
                    status=status.HTTP_404_NOT_FOUND
                )

            bc = PatientBlockchain("chain.json")

            if prediction_ids:
                predictions = Prediction.objects.filter(
                    id__in=prediction_ids,
                    patient=request.user
                )
            else:
                predictions = Prediction.objects.filter(patient=request.user)

            if not predictions.exists():
                return Response(
                    {'error': 'No predictions found'},
                    status=status.HTTP_404_NOT_FOUND
                )

            patient_id = str(request.user.id)
            doctor_id = str(doctor.id)
            granted_count = 0

            for pred in predictions:
                pred_name = f"{pred.disease_type}_pred_{pred.created_at.strftime('%Y%m%d_%H%M%S')}"

                existing = bc._get_latest_block(pred_name)
                if not existing:
                    bc.record_prediction(
                        patient_id=patient_id,
                        prediction_name=pred_name,
                        prediction_data={
                            'disease_type': pred.disease_type,
                            'risk_level': pred.risk_level,
                            'confidence': pred.confidence_score,
                            'timestamp': pred.created_at.isoformat()
                        }
                    )

                bc.grant_access(patient_id, pred_name, doctor_id)
                granted_count += 1

            print(f"✓ Granted access to {granted_count} predictions")

            return Response({
                'success': True,
                'message': f'Access granted to Dr. {doctor.first_name} {doctor.last_name}',
                'doctor_name': f"{doctor.first_name} {doctor.last_name}",
                'doctor_email': doctor.email,
                'predictions_shared': granted_count
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"✗ Grant access error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RevokeAccessView(APIView):
    """
    Revoke doctor access from patient predictions.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            if request.user.user_type != 'patient':
                return Response(
                    {'error': 'Only patients can revoke access'},
                    status=status.HTTP_403_FORBIDDEN
                )

            doctor_email = request.data.get('doctor_email')

            if not doctor_email:
                return Response(
                    {'error': 'doctor_email is required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            try:
                doctor = User.objects.get(email=doctor_email, user_type='doctor')
            except User.DoesNotExist:
                return Response(
                    {'error': 'Doctor not found'},
                    status=status.HTTP_404_NOT_FOUND
                )

            bc = PatientBlockchain("chain.json")
            predictions = Prediction.objects.filter(patient=request.user)

            patient_id = str(request.user.id)
            doctor_id = str(doctor.id)
            revoked_count = 0

            for pred in predictions:
                pred_name = f"{pred.disease_type}_pred_{pred.created_at.strftime('%Y%m%d_%H%M%S')}"

                if bc._get_latest_block(pred_name):
                    bc.revoke_access(patient_id, pred_name, doctor_id)
                    revoked_count += 1

            print(f"✓ Revoked access from {revoked_count} predictions")

            return Response({
                'success': True,
                'message': f'Access revoked from Dr. {doctor.first_name} {doctor.last_name}',
                'predictions_affected': revoked_count
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"✗ Revoke access error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GetAccessListView(APIView):
    """
    Get list of doctors with access to patient's predictions.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            if request.user.user_type != 'patient':
                return Response(
                    {'error': 'Only patients can view access list'},
                    status=status.HTTP_403_FORBIDDEN
                )

            bc = PatientBlockchain("chain.json")
            predictions = Prediction.objects.filter(patient=request.user)

            if not predictions.exists():
                return Response({'doctors': []}, status=status.HTTP_200_OK)

            doctors_with_access = set()
            patient_id = str(request.user.id)

            for pred in predictions:
                pred_name = f"{pred.disease_type}_pred_{pred.created_at.strftime('%Y%m%d_%H%M%S')}"
                access_list = bc.get_current_access(pred_name)

                for account_id in access_list:
                    if account_id != patient_id:
                        doctors_with_access.add(account_id)

            doctors = []
            for doctor_id in doctors_with_access:
                try:
                    doctor = User.objects.get(id=doctor_id, user_type='doctor')
                    doctors.append({
                        'id': doctor.id,
                        'name': f"{doctor.first_name} {doctor.last_name}",
                        'email': doctor.email,
                        'first_name': doctor.first_name,
                        'last_name': doctor.last_name
                    })
                except User.DoesNotExist:
                    continue

            return Response({
                'doctors': doctors,
                'total': len(doctors)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"✗ Get access list error: {e}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )