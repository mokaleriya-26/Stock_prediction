# core/urls.py


from django.urls import path  
from django.contrib import admin
from . import views
from predictor.views import get_stock_prediction,compare_stocks

urlpatterns = [
    path('', views.home, name='home'),      # Path for the main page (http://127.0.0.1:8000/)
    path('market-data/', views.market_data, name='market_data'), # Path for the market data page (http://127.0.0.1:8000/maket-data/)
    path('about/', views.about, name='about'), # Path for the about page (http://127.0.0.1:8000/about/)
    path('signup/', views.signup_view, name='signup'), # Path for signup (http://127.0.0.1:8000/signup/)
    path('signin/', views.signin_view, name='signin'), # Path for signin (http://127.0.0.1:8000/signin/)
    path('signout/', views.signout_view, name='signout'), # Path for signout
    path('privacy-policy/', views.privacy_policy, name='privacy_policy'), # Path for the privacy policy page (http://127.0.0.1:8000/privacy-policy/)
    path('terms-of-service/', views.terms_of_service, name='terms_of_service'), # Path for the terms of service page (http://127.0.0.1:8000/terms-of-service/)
    path('analysis/', views.analysis, name='analysis'),
    path('comparison/', views.comparison, name='comparison_page'),
    path('api/predict/<str:ticker>/', get_stock_prediction, name='predict_stock'),
    path('api/compare-stocks/', compare_stocks, name='compare_stocks'),
    path('api/sector-sentiment/', views.sector_sentiment_api, name='sector_sentiment_api'),
    path('api/generate-verdict-explanation/', views.generate_verdict_explanation, name='generate_verdict_explanation'),
]