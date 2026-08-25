from django.urls import path

from .views import ask_rag_api


urlpatterns = [
    path("ask/", ask_rag_api, name="ask-rag-api"),
]
