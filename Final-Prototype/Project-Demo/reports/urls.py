"""
URLs for reports app
"""

from django.urls import path
from .views import (
    BloodReportUploadView,
    BloodReportListView,
    BloodReportDetailView,
    BloodReportDeleteView,
)

urlpatterns = [
    path('upload/', BloodReportUploadView.as_view(), name='report_upload'),
    path('', BloodReportListView.as_view(), name='report_list'),
    path('<uuid:pk>/', BloodReportDetailView.as_view(), name='report_detail'),
    path('<uuid:pk>/delete/', BloodReportDeleteView.as_view(), name='report_delete'),
]