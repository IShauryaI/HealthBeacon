"""
Admin configuration for accounts
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """
    Admin interface for User model with all health fields.
    """

    list_display = ['email', 'username', 'user_type', 'age', 'sex', 'is_staff', 'created_at']
    list_filter = ['user_type', 'sex', 'is_diabetic', 'is_smoker', 'overall_health']
    search_fields = ['email', 'username', 'first_name', 'last_name']

    fieldsets = (
        ('Account Info', {
            'fields': ('username', 'email', 'password', 'user_type')
        }),
        ('Personal Info', {
            'fields': ('first_name', 'last_name')
        }),
        ('Demographics', {
            'fields': ('age', 'sex', 'race')
        }),
        ('Physical Measurements', {
            'fields': ('bmi', 'sleep_hours')
        }),
        ('Health Status', {
            'fields': ('physical_health_days', 'mental_health_days', 'overall_health')
        }),
        ('Lifestyle & Habits', {
            'fields': ('is_smoker', 'alcohol_drinking', 'physical_activity', 'difficulty_walking')
        }),
        ('Medical History', {
            'fields': ('is_diabetic', 'previous_stroke', 'has_asthma', 'has_kidney_disease', 'has_skin_cancer')
        }),
        ('Permissions', {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')
        }),
        ('Important Dates', {
            'fields': ('last_login', 'date_joined', 'created_at', 'updated_at')
        }),
    )

    readonly_fields = ['created_at', 'updated_at', 'last_login', 'date_joined']

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'email', 'password1', 'password2', 'user_type'),
        }),
    )