from django.contrib import admin
from .models import BloodReport, MedicalIndicator

admin.site.register(BloodReport)
admin.site.register(MedicalIndicator)