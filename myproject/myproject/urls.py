from django.contrib import admin
from django.urls import path
from myapp import views  # Assuming your app is named "myapp"

urlpatterns = [
    path('admin/', admin.site.urls),
    path('submit-url/', views.submit_url, name='submit_url'),  # Your URL to handle form submission
    path('', views.home, name='home'),  # Add a view for the root URL
    
]
