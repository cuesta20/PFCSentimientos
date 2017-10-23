#-*- coding=utf-8 -*-
from django.conf.urls import include, url  
from django.contrib import admin
from django.conf import settings 
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from AppSentimientos.views import index, entrenamiento, clasificar,conectarServidor,corpus #indice
from django.contrib.auth.models import User
from rest_framework import routers, serializers, viewsets

admin.autodiscover()

# Serializers define the API representation.
class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'is_staff')

# ViewSets define the view behavior.
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

# Routers provide an easy way of automatically determining the URL conf.
router = routers.DefaultRouter()
router.register(r'users', UserViewSet)

urlpatterns = [
	url(r'^admin/', include(admin.site.urls)),
	url(r'^admin/doc/', include('django.contrib.admindocs.urls')),
	url(r'^',index),
	#url(r'^indice$', indice, name='indice'),
	url(r'^entrenamiento$', entrenamiento),
	url(r'^clasificar$', clasificar),
	url(r'^conectarServidor$', conectarServidor),
	url(r'^corpus$', corpus),
	url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),
	#url(r'^(?P<txtTexto>\w+)/$', indice),
	#url(r'^index$', index),
	#url(r'^indice/', indice, name='indice'),
]
