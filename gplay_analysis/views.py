from django.shortcuts import render
import numpy as np
import pandas as pd
from plotly.offline import plot
import plotly.express as px
from sklearn import preprocessing
from sklearn.cluster import KMeans
import os
from datetime import datetime

# Create your views here.
BASE = os.path.dirname(os.path.abspath(__file__))


def home_page(request):
    # Exploratory Analysis and Data Cleaning.
    """
    df = pd.read_csv(os.path.join(BASE, 'data\Google-Playstore.csv'))
    df.loc[737542:737543, 'App Name'] = 'NA'
    df['Category'].fillna(value=df['Category'].mode()[0], inplace=True)
    df['Rating'].fillna(value=round(df['Rating'].mean(), 1), inplace=True)
    df['Rating Count'].fillna(value=df['Rating Count'].median(), inplace=True)
    df['Minimum Installs'].fillna(value=df['Minimum Installs'].median(), inplace=True)
    df['Installs'].fillna(value='1000+', inplace=True)
    df['Currency'].fillna(value=df['Currency'].mode()[0], inplace=True)
    df['Minimum Android'].fillna(value=df['Minimum Android'].mode()[0], inplace=True)
    df.to_csv(os.path.join(BASE, 'data\cleaned.csv'))
    df.drop('App Id', axis=1, inplace=True)
    df.drop('Minimum Installs', axis=1, inplace=True)


    def size(x):
    if x=='Varies with device':
        return 0
    elif x[-1]=='M':
        return float(x[:-1])
    else:
        return float(x[:-1].replace(',','')) / 1024
    df['Modified Size'] = df['Size'].apply(size)

    def date_change(x):
        return datetime.strptime(x, '%b %d, %Y')
    df['Last Updated'] = df['Last Updated'].apply(lambda x: datetime.strptime(x, '%b %d, %Y'))

    df.drop('Last Updated', axis=1, inplace=True)

    from datetime import datetime
    def date_change(x):
        return datetime.strptime(x, '%b %d, %Y')
    """
    df = pd.read_csv(os.path.join(BASE, 'data\cleaned.csv'))

    total_app_count = df['App Name'].count()

    # Category-wise grouping
    grp_by_category = df.groupby(by='Category')
    category_counts = grp_by_category['App Name'].count()
    category_downloads_bar = category_counts.values
    category_downloads = category_counts.values / total_app_count * 100
    categories = category_counts.index

    free_app_count = df[df['Free'] == True]['App Name'].count()
    free_per = free_app_count / total_app_count * 100
    paid_per = 100 - free_per
    per = [free_per, paid_per]

    in_app_per = df[(df['In App Purchases'] == True) & (df['Free'] == True)]['App Name'].count() / free_app_count * 100
    in_app_per = [in_app_per, 100 - in_app_per]

    add_support = df[df['Ad Supported'] == True]['App Name'].count() / df['App Name'].count() * 100
    add_support = [add_support, 100 - add_support]
    context = {'categories': categories.tolist(),
               'category_downloads': category_downloads.tolist(),
               'category_downloads_bar': category_downloads_bar.tolist(),
               'percentage': per,
               'in_app_per': in_app_per,
               'add_support': add_support}
    return render(request, 'homepage.html', context=context)


def cat_graphs(request):
    df = pd.read_csv(os.path.join(BASE, 'data\cleaned.csv'))
    # Category-wise grouping
    grp_by_category = df.groupby(by='Category')

    avg_cat_installs = grp_by_category['Maximum Installs'].agg(np.mean)
    categories = avg_cat_installs.index.tolist()
    avg_installs = [round(i) for i in avg_cat_installs.values]

    avg_cat_rating = grp_by_category['Rating'].agg(np.mean)
    avg_rating = [round(i, 2) for i in avg_cat_rating.values]

    avg_cat_review = grp_by_category['Rating Count'].agg(np.mean)
    avg_review = [round(i) for i in avg_cat_review]

    grp_by_category_priced = df[df['Free'] == False].groupby(by='Category')
    grp_price = grp_by_category_priced['Price'].agg(np.mean)
    grp_price = [round(i, 2) for i in grp_price]

    rel_date = df[df['Released'].notnull()]
    rel_date_count = plot(px.histogram(rel_date, x='Released', color='Category', template='plotly_white', height=600),
                          output_type='div', include_plotlyjs=False)
    context = {'categories': categories,
               'avg_installs': avg_installs,
               'avg_rating': avg_rating,
               'avg_review': avg_review,
               'grp_price': grp_price,
               'rel_date_count': rel_date_count,
               }
    return render(request, 'cat_graphs.html', context=context)


def cat_tables(request):
    df = pd.read_csv(os.path.join(BASE, 'data\cleaned.csv'))
    grp_by_category = df.groupby(by='Category')
    avg_cat_installs = grp_by_category['Maximum Installs'].agg(np.mean)
    categories = avg_cat_installs.index.tolist()
    avg_installs = [(i, round(avg_cat_installs[i])) for i in categories]
    context = {'avg_installs': avg_installs}
    return render(request, 'cat_tables.html', context=context)


def charts(request):
    df = pd.read_csv(os.path.join(BASE, 'data\cleaned.csv'))

    cost = df['Free'].apply(lambda x: 'Free' if x == True else 'Paid')

    installs_counts = plot(
        px.scatter(df, x='Maximum Installs', y='Rating Count', template='plotly_white', color=cost,
                   color_discrete_sequence=["#61b15a", "#ec4646"], height=600, labels={'Maximum Installs': 'Number of Installs',
                                                                                       'color': 'Pricing'}),
        output_type='div', include_plotlyjs=False)

    grp_by_category = df.groupby(by='Category')
    grp_by_category = grp_by_category[['Rating Count', 'Maximum Installs', 'Free']]
    percent_of_installs = grp_by_category.mean()
    percent_of_installs['Rating as % of Installs'] = percent_of_installs['Rating Count'] / percent_of_installs[
        'Maximum Installs'] * 100
    percent_of_installs.reset_index(inplace=True)
    rating_as_install_per = plot(
        px.bar(percent_of_installs, x='Category', y='Rating as % of Installs', template='plotly_white',
               color_discrete_sequence=px.colors.qualitative.D3, labels={'Rating as % of Installs': 'Rating as percentage of Installs'}, height=600),
        output_type='div', include_plotlyjs=False)

    price_apps = df[df['Free'] == False]
    price_dist = plot(
        px.box(price_apps, x='Price', template='plotly_white',
               color_discrete_sequence=px.colors.qualitative.Vivid, labels={'Price': 'Price (INR)'}, height=600),
        output_type='div', include_plotlyjs=False
    )

    price_installs = plot(
        px.histogram(price_apps, x='Price', y='Maximum Installs', template='plotly_white',
                     color_discrete_sequence=px.colors.qualitative.Bold, labels={'Price': 'Price (INR)',
                                                                                 'Maximum Installs': 'Installs'}, height=600),
        output_type='div', include_plotlyjs=False
    )

    size_apps = df[df['Modified Size'] != 0]
    size_by_category = size_apps.groupby(by='Category')
    size_by_category = pd.DataFrame(size_by_category['Modified Size'].agg(np.mean))
    size_by_category.reset_index(inplace=True)
    size_by_category['Average Size'] = size_by_category['Modified Size']
    average_size = plot(
        px.bar(size_by_category, x='Category', y='Average Size', template='plotly_white',
               color_discrete_sequence=["#41aea9"], height=600, labels={'Average Size': 'Average Size (in MB)'}),
        output_type='div', include_plotlyjs=False)

    rating_price = plot(
        px.scatter(price_apps, y='Rating', x='Price', template='plotly_white',
                   color_discrete_sequence=["#00bcd4"], height=600),
        output_type='div', include_plotlyjs=False)

    context = {'installs_counts': installs_counts,
               'rating_as_install_per': rating_as_install_per,
               'price_dist': price_dist,
               'price_installs': price_installs,
               'average_size': average_size,
               'rating_price': rating_price
               }
    return render(request, 'charts.html', context=context)


def cluster(request):
    df = pd.read_csv(os.path.join(BASE, 'data\cleaned.csv'))
    x = df[['Maximum Installs', 'Rating Count']]
    x.columns = ['Installs', 'Reviews']
    scaled_x = preprocessing.scale(x)
    kmeans = KMeans(4)
    kmeans.fit(scaled_x)
    clusters = kmeans.fit_predict(scaled_x)
    fig = px.scatter(x, x='Reviews', y='Installs', color=clusters, template='plotly_dark', labels={'Reviews': 'Number of Reviews',
                                                                                                   'Installs': 'Number of Installs'}, height=600)
    review_installs = plot(
        fig.update_traces(marker_coloraxis=None),
        output_type='div', include_plotlyjs=False)

    x = df[['Rating', 'Maximum Installs']]
    scaled_x = preprocessing.scale(x)
    kmeans = KMeans(3)
    kmeans.fit(scaled_x)
    clusters = kmeans.fit_predict(scaled_x)
    fig = px.scatter(x, x='Rating', y='Maximum Installs', color=clusters, template='plotly_dark', labels={'Maximum Installs': 'Number of Installs'}, height=600)
    rating_installs = plot(
        fig.update_traces(marker_coloraxis=None),
        output_type='div', include_plotlyjs=False)

    priced = df[df['Free'] == False]
    x = priced[['Price', 'Maximum Installs', 'Rating']]
    scaled_x = preprocessing.scale(x)
    kmeans = KMeans(4)
    kmeans.fit(scaled_x)
    x['clusters'] = [str(i) for i in kmeans.fit_predict(scaled_x)]
    plot_3d = plot(
        px.scatter_3d(x, x='Price', y='Rating', z='Maximum Installs', template='plotly_dark',
                      color='clusters', color_discrete_sequence=["#ff577f", "#ff884b", "#ffc764", "#ffe5b9"], labels={'Maximum Installs': 'Number of Installs',
                                                                                                                      'Price': 'Price (INR)'}),
        output_type='div')

    context = {'review_installs': review_installs,
               'rating_installs': rating_installs,
               'plot_3d': plot_3d}
    return render(request, 'cluster_charts.html', context=context)

