"""
Models for disease predictions.
"""

from django.db import models
from accounts.models import User
from reports.models import BloodReport
import uuid


class Prediction(models.Model):
    """Disease prediction results."""

    DISEASE_CHOICES = (
        ('ckd', 'Chronic Kidney Disease'),
        ('diabetes', 'Diabetes'),
        ('heart_disease', 'Heart Disease'),
    )

    RESULT_CHOICES = (
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('inconclusive', 'Inconclusive'),
    )

    RISK_CHOICES = (
        ('low', 'Low Risk'),
        ('medium', 'Medium Risk'),
        ('high', 'High Risk'),
        ('critical', 'Critical Risk'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    blood_report = models.ForeignKey(BloodReport, on_delete=models.CASCADE, related_name='predictions')
    patient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    disease_type = models.CharField(max_length=20, choices=DISEASE_CHOICES)
    prediction_result = models.CharField(max_length=20, choices=RESULT_CHOICES)
    confidence_score = models.FloatField()
    risk_level = models.CharField(max_length=20, choices=RISK_CHOICES)
    indicators = models.JSONField(default=dict)
    shap_values = models.JSONField(default=dict, blank=True)
    shap_chart_path = models.CharField(max_length=255, blank=True)
    recommendation = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    reviewed_by_doctor = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reviewed_predictions'
    )
    doctor_notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.disease_type} - {self.patient.email} - {self.prediction_result}"