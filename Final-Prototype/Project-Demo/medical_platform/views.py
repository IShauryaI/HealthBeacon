"""
Root views for the platform.
"""

from django.http import JsonResponse


def home(request):
    """API home page with available endpoints."""
    return JsonResponse({
        'message': 'Medical Prediction Platform API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            'admin': '/admin/',
            'auth': {
                'register': '/api/auth/register/',
                'login': '/api/token/',
                'refresh': '/api/token/refresh/',
                'profile': '/api/auth/profile/',
            },
            'predictions': {
                'ckd': '/api/predictions/ckd/',
                'diabetes': '/api/predictions/diabetes/',
                'heart_disease': '/api/predictions/heart-disease/',
                'list': '/api/predictions/',
            },
            'reports': {
                'upload': '/api/reports/upload/',
                'list': '/api/reports/',
            }
        },
        'documentation': 'API documentation coming soon'
    })