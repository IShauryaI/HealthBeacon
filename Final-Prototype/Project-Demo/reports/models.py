"""
Models for blood reports and medical indicators.
"""

from django.db import models
from accounts.models import User
import uuid


class BloodReport(models.Model):
    """Blood report uploaded by patients."""

    STATUS_CHOICES = (
        ('pending', 'Pending Processing'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('error', 'Error'),
    )

    REPORT_TYPE_CHOICES = (
        ('comprehensive', 'Comprehensive'),
        ('basic', 'Basic'),
        ('annual', 'Annual Checkup'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='blood_reports')
    report_file = models.FileField(upload_to='blood_reports/%Y/%m/%d/')
    report_type = models.CharField(max_length=20, choices=REPORT_TYPE_CHOICES, default='comprehensive')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    upload_date = models.DateTimeField(auto_now_add=True)
    processed_date = models.DateTimeField(null=True, blank=True)
    ocr_extracted_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-upload_date']

    def __str__(self):
        return f"Report {self.id} - {self.patient.email} - {self.upload_date.strftime('%Y-%m-%d')}"


class MedicalIndicator(models.Model):
    """Individual medical indicators extracted from blood reports."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    blood_report = models.ForeignKey(BloodReport, on_delete=models.CASCADE, related_name='indicators')
    indicator_name = models.CharField(max_length=50)
    indicator_value = models.FloatField()
    unit = models.CharField(max_length=20)
    reference_range_min = models.FloatField()
    reference_range_max = models.FloatField()
    is_abnormal = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.indicator_name}: {self.indicator_value} {self.unit}"

    def check_abnormal(self):
        """Check if indicator is outside normal range."""
        self.is_abnormal = (
                self.indicator_value < self.reference_range_min or
                self.indicator_value > self.reference_range_max
        )
        self.save()