"""
Serializers for blood reports.
"""

from rest_framework import serializers
from .models import BloodReport, MedicalIndicator


class MedicalIndicatorSerializer(serializers.ModelSerializer):
    """Serializer for Medical Indicator."""

    class Meta:
        model = MedicalIndicator
        fields = '__all__'


class BloodReportSerializer(serializers.ModelSerializer):
    """Serializer for Blood Report."""

    indicators = MedicalIndicatorSerializer(many=True, read_only=True)
    patient_name = serializers.CharField(source='patient.get_full_name', read_only=True)

    class Meta:
        model = BloodReport
        fields = '__all__'
        read_only_fields = ['id', 'upload_date', 'processed_date', 'status']


class BloodReportUploadSerializer(serializers.ModelSerializer):
    """Serializer for uploading blood reports."""

    class Meta:
        model = BloodReport
        fields = ['report_file', 'report_type']