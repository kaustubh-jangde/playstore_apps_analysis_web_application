from django.urls import path
from gplay_analysis import views

app_name = 'gplay'

urlpatterns = [
    path('', views.home_page, name='home'),
    path('categorical/graphs', views.cat_graphs, name='category_graphs'),
    path('categorical/tables', views.cat_tables),
    path('charts/', views.charts, name='chart'),
    path('cluster_analysis/', views.cluster, name='cluster')
]