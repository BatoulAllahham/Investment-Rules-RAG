from django.urls import path

from .views import ask_rag_api, rag_sources_api


urlpatterns = [
    path("ask/", ask_rag_api, name="ask-rag-api"),
    path("sources/", rag_sources_api, name="rag-sources-api"),
]
