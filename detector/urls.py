from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("analyze/", views.analyze_video, name="analyze_video"),
    path("signin/", views.signin_view, name="signin"),
    path("signup/", views.signup_view, name="signup"),
    path("logout/", views.logout_view, name="logout"),
]
