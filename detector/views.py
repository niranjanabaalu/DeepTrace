from pathlib import Path
from uuid import uuid4
import re
import subprocess
import sys

from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages
from django.views.decorators.cache import never_cache


PROJECT_ROOT = Path(settings.BASE_DIR)


@login_required(login_url="signin")
@never_cache
def home(request):
    return render(request, "detector/home.html")


@login_required(login_url="signin")
@never_cache
def analyze_video(request):
    if request.method != "POST":
        return redirect("home")

    uploaded_video = request.FILES.get("video")

    if not uploaded_video:
        return render(
            request,
            "detector/home.html",
            {"error": "Please select a video first."},
        )

    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    extension = Path(uploaded_video.name).suffix.lower()

    if extension not in allowed_extensions:
        return render(
            request,
            "detector/home.html",
            {"error": "Unsupported video format."},
        )

    upload_dir = PROJECT_ROOT / "media" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / f"{uuid4().hex}{extension}"

    try:
        with video_path.open("wb") as destination:
            for chunk in uploaded_video.chunks():
                destination.write(chunk)

        script_path = PROJECT_ROOT / "training" / "analyze_video.py"

        result = subprocess.run(
            [sys.executable, str(script_path), str(video_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            print("DeepTrace analysis error:", result.stderr[-4000:])
            return render(
                request,
                "detector/home.html",
                {"error": "Analysis failed. Check the terminal for details."},
            )

        output = result.stdout

        # Extract prediction and statistics from the model report.
        prediction_match = re.search(
            r"Prediction:\s*(REAL|FAKE)\b",
            output,
            re.IGNORECASE,
        )
        fake_match = re.search(
            r"Fake probability:\s*([\d.]+)\s*%",
            output,
            re.IGNORECASE,
        )
        real_match = re.search(
            r"Real probability:\s*([\d.]+)\s*%",
            output,
            re.IGNORECASE,
        )
        frames_match = re.search(
            r"Sampled frames:\s*(\d+)",
            output,
            re.IGNORECASE,
        )
        faces_match = re.search(
            r"Faces analyzed:\s*(\d+)",
            output,
            re.IGNORECASE,
        )

        if not prediction_match:
            print("Could not parse prediction from output:", output)
            return render(
                request,
                "detector/home.html",
                {
                    "error": "Analysis finished, but the prediction could not be read.",
                    "analysis_output": output,
                },
            )

        context = {
            "prediction": prediction_match.group(1).upper(),
            "fake_score": fake_match.group(1) if fake_match else None,
            "real_score": real_match.group(1) if real_match else None,
            "sampled_frames": frames_match.group(1) if frames_match else None,
            "faces_analyzed": faces_match.group(1) if faces_match else None,
            "analysis_output": output,
        }

        # Preserve the existing results dashboard and terminal-output feature.
        return render(request, "detector/home.html", context)

    except subprocess.TimeoutExpired:
        return render(
            request,
            "detector/home.html",
            {"error": "Analysis timed out. Please try a shorter video."},
        )

    except OSError as exc:
        print("DeepTrace error:", exc)
        return render(
            request,
            "detector/home.html",
            {"error": "Could not save the video or start model analysis."},
        )


def signup_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    form = UserCreationForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(
            request,
            "Your account has been created. Please sign in.",
        )
        return redirect("signin")

    return render(request, "detector/signup.html", {"form": form})


def signin_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    form = AuthenticationForm(request, data=request.POST or None)

    if request.method == "POST" and form.is_valid():
        user = form.get_user()
        login(request, user)
        return redirect("home")

    return render(request, "detector/signin.html", {"form": form})


def logout_view(request):
    if request.method == "POST":
        logout(request)
        response = redirect("signin")
        response["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response["Pragma"] = "no-cache"
        response["Expires"] = "0"
        return response

    return redirect("home")
