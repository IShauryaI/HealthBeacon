"""
Serializers for predictions.
"""

from rest_framework import serializers
from .models import Prediction


class PredictionSerializer(serializers.ModelSerializer):
    """Serializer for Prediction model."""

    patient_name = serializers.CharField(source='patient.get_full_name', read_only=True)
    reviewed_by_name = serializers.CharField(source='reviewed_by_doctor.get_full_name', read_only=True)

    class Meta:
        model = Prediction
        fields = '__all__'
        read_only_fields = ['id', 'created_at']


class SimplePredictionSerializer(serializers.ModelSerializer):
    """Simple serializer for patient view."""

    class Meta:
        model = Prediction
        fields = ['id', 'disease_type', 'prediction_result', 'confidence_score',
                  'risk_level', 'recommendation', 'created_at']