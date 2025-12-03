"""cropyield URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from users import views as users
from admins import views as admins
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', users.index, name='index'),
    path('UserLogin/', users.UserLogin, name='UserLogin'),
    path('UserRegister/', users.UserRegister, name='UserRegister'),
    path('UserRegisterAction/', users.UserRegisterAction, name='UserRegisterAction'),
    path('UserLoginCheck/', users.UserLoginCheck, name='UserLoginCheck'),
    path('UserSendCrop/', users.UserSendCrop, name='UserSendCrop'),
    path('UserSendCropanalysis/', users.UserSendCropanalysis, name='UserSendCropanalysis'),
    path('yeilddetails/', users.yeilddetails, name='yeilddetails'),
    path('ML/', users.ML, name='ML'),


    path('AdminLogin/', admins.AdminLogin, name='AdminLogin'),
    path('AdminLoginCheck/', admins.AdminLoginCheck, name='AdminLoginCheck'),
    path('AdminViewUsers/', admins.AdminViewUsers, name='AdminViewUsers'),
    path('AdminActivaUsers/', admins.AdminActivaUsers, name='AdminActivaUsers'),
    path('Sendcropdata/', admins.Sendcropdata, name='Sendcropdata'),
    path('sendcrop/', admins.sendcrop, name='sendcrop'),
    path('storecsvdata/', admins.storecsvdata, name='storecsvdata'),
    path('MLprocess/', admins.MLprocess, name='MLprocess'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
