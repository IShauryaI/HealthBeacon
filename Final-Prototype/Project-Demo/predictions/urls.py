"""
URLs for predictions app
"""

from django.urls import path
from .views import (
    PredictionListView,
    PredictionDetailView,
    UpdateClinicalNotesView,
    GrantAccessView,
    RevokeAccessView,
    GetAccessListView,
)

urlpatterns = [
    path('', PredictionListView.as_view(), name='prediction_list'),
    path('<uuid:pk>/', PredictionDetailView.as_view(), name='prediction_detail'),
    path('<uuid:pk>/clinical-notes/', UpdateClinicalNotesView.as_view(), name='update_clinical_notes'),

    # Blockchain access control
    path('access/grant/', GrantAccessView.as_view(), name='grant_access'),
    path('access/revoke/', RevokeAccessView.as_view(), name='revoke_access'),
    path('access/list/', GetAccessListView.as_view(), name='access_list'),
]