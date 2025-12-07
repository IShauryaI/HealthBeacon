"""
User models for authentication
"""

from django.contrib.auth.models import AbstractUser
from django.db import models
import uuid


class User(AbstractUser):
    """
    Custom User model with additional fields for patient health data.
    Supports CKD, Diabetes, and Heart Disease predictions.
    """

    USER_TYPE_CHOICES = (
        ('patient', 'Patient'),
        ('doctor', 'Doctor'),
    )

    SEX_CHOICES = (
        ('Male', 'Male'),
        ('Female', 'Female'),
    )

    RACE_CHOICES = (
        ('White', 'White'),
        ('Black', 'Black/African American'),
        ('Asian', 'Asian'),
        ('Hispanic', 'Hispanic/Latino'),
        ('Native American', 'Native American'),
        ('Pacific Islander', 'Pacific Islander'),
        ('Other', 'Other'),
    )

    HEALTH_CHOICES = (
        ('Excellent', 'Excellent'),
        ('Very Good', 'Very Good'),
        ('Good', 'Good'),
        ('Fair', 'Fair'),
        ('Poor', 'Poor'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES)

    # Make email unique and required
    email = models.EmailField(unique=True)  # Add this line to override AbstractUser's email

    # Basic Demographics
    age = models.IntegerField(null=True, blank=True)
    sex = models.CharField(max_length=10, choices=SEX_CHOICES, default='Male', null=True, blank=True)
    race = models.CharField(max_length=50, choices=RACE_CHOICES, default='White', null=True, blank=True)

    # Physical Measurements
    bmi = models.FloatField(null=True, blank=True, help_text='Body Mass Index')
    sleep_hours = models.FloatField(null=True, blank=True, help_text='Average sleep hours per night')

    # Health Status
    physical_health_days = models.IntegerField(
        null=True,
        blank=True,
        help_text='Days of poor physical health in past 30 days'
    )
    mental_health_days = models.IntegerField(
        null=True,
        blank=True,
        help_text='Days of poor mental health in past 30 days'
    )
    overall_health = models.CharField(
        max_length=20,
        choices=HEALTH_CHOICES,
        default='Good',
        null=True,
        blank=True
    )

    # Lifestyle & Habits
    is_smoker = models.BooleanField(default=False, help_text='Has smoked 100+ cigarettes')
    alcohol_drinking = models.BooleanField(default=False, help_text='Heavy alcohol consumption')
    physical_activity = models.BooleanField(default=True, help_text='Regular physical activity')
    difficulty_walking = models.BooleanField(default=False, help_text='Difficulty walking/climbing stairs')

    # Medical History
    is_diabetic = models.BooleanField(default=False, help_text='Diagnosed with diabetes')
    previous_stroke = models.BooleanField(default=False, help_text='Ever had a stroke')
    has_asthma = models.BooleanField(default=False, help_text='Diagnosed with asthma')
    has_kidney_disease = models.BooleanField(default=False, help_text='Diagnosed with kidney disease')
    has_skin_cancer = models.BooleanField(default=False, help_text='Diagnosed with skin cancer')

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.email} ({self.user_type})"

    class Meta:
        ordering = ['-created_at']