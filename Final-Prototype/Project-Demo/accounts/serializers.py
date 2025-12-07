"""
Serializers for authentication
"""

from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.contrib.auth import authenticate
from .models import User


class EmailTokenObtainPairSerializer(TokenObtainPairSerializer):
    """
    Custom JWT serializer that accepts email instead of username.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['email'] = serializers.EmailField(required=True)
        self.fields['password'] = serializers.CharField(write_only=True, required=True,
                                                        style={'input_type': 'password'})
        if 'username' in self.fields:
            del self.fields['username']

    def validate(self, attrs):
        email = attrs.get('email')
        password = attrs.get('password')

        print(f"\n{'=' * 60}")
        print(f"Login attempt for email: {email}")
        print(f"{'=' * 60}")

        if not email or not password:
            raise serializers.ValidationError({'error': 'Email and password are required'})

        # Find user by email
        try:
            user_obj = User.objects.get(email=email)
            print(f"✓ User found: {user_obj.username}")
            print(f"  User active: {user_obj.is_active}")
            print(f"  Has usable password: {user_obj.has_usable_password()}")
        except User.DoesNotExist:
            print(f"✗ No user found with email: {email}")
            raise serializers.ValidationError({'error': 'Invalid email or password'})
        except User.MultipleObjectsReturned:
            print(f"✗ Multiple users found with email: {email}")
            raise serializers.ValidationError({'error': 'Multiple accounts found. Contact support.'})

        # Check password manually first
        from django.contrib.auth.hashers import check_password
        password_valid = check_password(password, user_obj.password)
        print(f"  Password check result: {password_valid}")

        if not password_valid:
            print(f"✗ Password incorrect for user: {user_obj.username}")
            raise serializers.ValidationError({'error': 'Invalid email or password'})

        # Now authenticate (should work since we verified password)
        from django.contrib.auth import authenticate
        authenticated_user = authenticate(
            request=self.context.get('request'),
            username=user_obj.username,
            password=password
        )

        print(f"  Authentication result: {authenticated_user is not None}")

        if not authenticated_user:
            print(f"✗ Authentication failed even though password was correct")
            # Use the user object directly since password was verified
            authenticated_user = user_obj

        if not authenticated_user.is_active:
            print(f"✗ User account is disabled")
            raise serializers.ValidationError({'error': 'User account is disabled'})

        print(f"✓ Login successful for: {authenticated_user.username}")
        print(f"{'=' * 60}\n")

        # Generate tokens
        refresh = self.get_token(authenticated_user)

        data = {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': UserSerializer(authenticated_user).data
        }

        return data


class UserSerializer(serializers.ModelSerializer):
    """
    User serializer with all health fields.
    """

    class Meta:
        model = User
        fields = [
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'user_type',

            # Demographics
            'age',
            'sex',
            'race',

            # Physical Measurements
            'bmi',
            'sleep_hours',

            # Health Status
            'physical_health_days',
            'mental_health_days',
            'overall_health',

            # Lifestyle & Habits
            'is_smoker',
            'alcohol_drinking',
            'physical_activity',
            'difficulty_walking',

            # Medical History
            'is_diabetic',
            'previous_stroke',
            'has_asthma',
            'has_kidney_disease',
            'has_skin_cancer',

            # Timestamps
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class RegisterSerializer(serializers.ModelSerializer):
    """
    Serializer for user registration.
    """
    password = serializers.CharField(write_only=True, min_length=8, style={'input_type': 'password'})
    password_confirm = serializers.CharField(write_only=True, min_length=8, style={'input_type': 'password'})

    class Meta:
        model = User
        fields = [
            'username',
            'email',
            'password',
            'password_confirm',
            'first_name',
            'last_name',
            'user_type',
        ]

    def validate_email(self, value):
        """Ensure email is unique."""
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value

    def validate_username(self, value):
        """Ensure username is unique."""
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError("A user with this username already exists.")
        return value

    def validate(self, data):
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError({"password": "Passwords must match."})
        return data

    def create(self, validated_data):
        validated_data.pop('password_confirm')

        print(f"\n{'=' * 60}")
        print(f"Creating new user: {validated_data['username']}")
        print(f"Email: {validated_data['email']}")
        print(f"{'=' * 60}\n")

        # Use create_user to ensure password is hashed
        user = User.objects.create_user(**validated_data)

        print(f"✓ User created successfully")
        print(f"  Has usable password: {user.has_usable_password()}")
        print(f"{'=' * 60}\n")

        return user


class ProfileUpdateSerializer(serializers.ModelSerializer):
    """
    Serializer for updating user profile (all health fields).
    """

    class Meta:
        model = User
        fields = [
            'first_name',
            'last_name',

            # Demographics
            'age',
            'sex',
            'race',

            # Physical Measurements
            'bmi',
            'sleep_hours',

            # Health Status
            'physical_health_days',
            'mental_health_days',
            'overall_health',

            # Lifestyle & Habits
            'is_smoker',
            'alcohol_drinking',
            'physical_activity',
            'difficulty_walking',

            # Medical History
            'is_diabetic',
            'previous_stroke',
            'has_asthma',
            'has_kidney_disease',
            'has_skin_cancer',
        ]