import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Dictionary koordinat provinsi Indonesia
lat_lon_data = {
    "Jawa Barat": (-6.9147, 107.6098),
    "Jawa Timur": (-7.2504, 112.7688),
    "Jawa Tengah": (-7.0051, 110.4381),
    "Sumatera Utara": (3.5913, 98.6749),
    "Banten": (-6.4058, 106.0640),
    "Sulawesi Selatan": (-5.1354, 119.4238),
    "Sumatera Selatan": (-3.3194, 104.9147),
    "Lampung": (-5.4500, 105.2667),
    "D.K.I. Jakarta": (-6.2088, 106.8456),
    "Riau": (0.5071, 101.4478),
    "Nusa Tenggara Timur": (-10.1771, 123.6070),
    "Nusa Tenggara Barat": (-8.6529, 116.4194),
    "Sumatera Barat": (-0.9471, 100.4172),
    "Kalimantan Barat": (-0.0263, 109.3425),
    "Aceh": (5.5483, 95.3238),
    "Kalimantan Timur": (0.5387, 116.4194),
    "Bali": (-8.6500, 115.2167),
    "Kalimantan Selatan": (-3.3194, 114.5908),
    "Jambi": (-1.6118, 103.6123),
    "D.I. Yogyakarta": (-7.7956, 110.3695),
    "Sulawesi Tengah": (-0.8998, 119.8707),
    "Sulawesi Tenggara": (-3.9728, 122.5150),
    "Kalimantan Tengah": (-1.6815, 113.3824),
    "Sulawesi Utara": (1.4931, 124.8413),
    "Kepulauan Riau": (0.9167, 104.4500),
    "Maluku": (-3.6577, 128.1902),
    "Bengkulu": (-3.8004, 102.2655),
    "Maluku Utara": (1.4823, 127.8356),
    "Sulawesi Barat": (-2.7569, 119.3666),
    "Kepulauan Bangka Belitung": (-2.7410, 106.4406),
    "Papua Tengah": (-3.9959, 137.7200),
    "Gorontalo": (0.5472, 123.0615),
    "Papua": (-4.2699, 138.0804),
    "Papua Pegunungan": (-4.0735, 138.9325),
    "Papua Selatan": (-6.5167, 140.4000),
    "Kalimantan Utara": (3.0462, 116.1979),
    "Papua Barat": (-0.8629, 131.2801),
    "Papua Barat Daya": (-1.3375, 132.2276),
}

def get_indonesia_geojson():
    """Generate GeoJSON for Indonesian provinces"""
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for province, coords in lat_lon_data.items():
        lat, lon = coords
        delta = 0.5  # Size of the province area
        feature = {
            "type": "Feature",
            "properties": {
                "name": province,
                "center": coords
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon - delta, lat - delta],
                    [lon + delta, lat - delta],
                    [lon + delta, lat + delta],
                    [lon - delta, lat + delta],
                    [lon - delta, lat - delta]
                ]]
            }
        }
        geojson["features"].append(feature)
    
    return geojson

def preprocess_data(df):
    """Preprocess the input data"""
    # Rename columns for consistency
    df.columns = ['no', 'wilayah', 'total', 'total_l', 'total_p', 
                  'tk_total', 'tk_l', 'tk_p',
                  'kb_total', 'kb_l', 'kb_p',
                  'tpa_total', 'tpa_l', 'tpa_p',
                  'sps_total', 'sps_l', 'sps_p',
                  'pkbm_total', 'pkbm_l', 'pkbm_p',
                  'skb_total', 'skb_l', 'skb_p',
                  'sd_total', 'sd_l', 'sd_p',
                  'smp_total', 'smp_l', 'smp_p',
                  'sma_total', 'sma_l', 'sma_p',
                  'smk_total', 'smk_l', 'smk_p',
                  'slb_total', 'slb_l', 'slb_p']
    
    # Remove rows with invalid data
    df = df[df['no'].apply(lambda x: str(x).isdigit())]
    
    # Clean province names
    df['wilayah'] = df['wilayah'].str.replace('Prov. ', '')
    
    # Clean numeric columns
    numeric_cols = df.columns.drop(['no', 'wilayah'])
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', ''), errors='coerce').fillna(0)
    
    # Add coordinates
    df['latitude'] = df['wilayah'].map(lambda x: lat_lon_data.get(x, (None, None))[0])
    df['longitude'] = df['wilayah'].map(lambda x: lat_lon_data.get(x, (None, None))[1])

    df['student_density'] = df['total'] / df['total'].sum()
    df['gender_ratio'] = df['total_l'] / df['total_p']
    df['education_level_ratio'] = (df['sma_total'] + df['smk_total']) / df['sd_total']
    
    return df.dropna(subset=['latitude', 'longitude'])

def perform_classification(df):
    """Perform classification to analyze geospatial datasets of students in Indonesia"""
    # Create more sophisticated features
    coord_scaler = MinMaxScaler()
    student_scaler = StandardScaler()
    ratio_scaler = StandardScaler()
    
    # Enhanced feature engineering
    df['total_per_capita'] = df['total'] / df['total'].mean()
    df['tk_ratio'] = df['tk_total'] / df['total']
    df['sd_ratio'] = df['sd_total'] / df['total']
    df['smp_ratio'] = df['smp_total'] / df['total']
    df['sma_smk_ratio'] = (df['sma_total'] + df['smk_total']) / df['total']
    df['gender_balance'] = abs(df['total_l'] - df['total_p']) / df['total']
    df['early_education_focus'] = (df['tk_total'] + df['kb_total'] + df['tpa_total']) / df['total']
    df['higher_education_focus'] = (df['sma_total'] + df['smk_total']) / df['total']
    
    # Scale coordinates with increased weight
    coords = coord_scaler.fit_transform(df[['latitude', 'longitude']])
    
    # Scale student numbers and density with balanced weight
    students = student_scaler.fit_transform(df[[ 
        'total_per_capita',
        'tk_ratio',
        'sd_ratio',
        'smp_ratio',
        'sma_smk_ratio'
    ]])
    
    # Scale demographic features
    demographics = ratio_scaler.fit_transform(df[[ 
        'gender_balance',
        'early_education_focus',
        'higher_education_focus'
    ]])
    
    # Combine features with optimized weights
    features = np.hstack([
        coords * 0.25,  # Geographic proximity
        students * 0.45,  # Educational distribution
        demographics * 0.3  # Demographic patterns
    ])
    
    # Apply PCA with variance preservation
    pca = PCA(n_components=0.95)  # Adjusted variance retention
    features_pca = pca.fit_transform(features)
    
    # Create target variable (clusters)
    df['cluster'] = df['wilayah'].apply(lambda x: 0 if x in ['Jawa Barat', 'Jawa Timur', 'Jawa Tengah'] else 1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_pca, df['cluster'], test_size=0.3, random_state=42)
    
    # Train a classification model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return clf, accuracy, report, features_pca

def create_choropleth(df):
    """Create choropleth map visualization"""
    # Calculate cluster statistics for hover information
    cluster_stats = df.groupby('cluster').agg({
        'total': 'sum',
        'total_l': 'sum',
        'total_p': 'sum'
    }).to_dict('index')
    
    # Add cluster statistics to the dataframe
    df['cluster_total'] = df['cluster'].map(lambda x: cluster_stats[x]['total'])
    df['cluster_total_l'] = df['cluster'].map(lambda x: cluster_stats[x]['total_l'])
    df['cluster_total_p'] = df['cluster'].map(lambda x: cluster_stats[x]['total_p'])

    fig = px.choropleth_mapbox(
        df,
        geojson=get_indonesia_geojson(),
        locations='wilayah',
        featureidkey="properties.name",
        color='cluster',
        hover_data={
            'wilayah': True,
            'total': True,
            'total_l': True,
            'total_p': True,
            'cluster_total': True,
            'cluster': True
        },
        hover_name='wilayah',
        mapbox_style="carto-positron",
        center={"lat": -2.5, "lon": 118},
        zoom=4,
        opacity=0.7,
        color_continuous_scale="Viridis",
        labels={
            'cluster': 'Cluster',
            'total': 'Total Students',
            'total_l': 'Male Students',
            'total_p': 'Female Students',
            'cluster_total': 'Cluster Total Students'
        }
    )

    fig.update_layout(
        margin={"r":0,"t":30,"l":0,"b":0},
        mapbox=dict(
            zoom=4,
            center=dict(lat=-2.5, lon=118)
        ),
        title=dict(
            text="Education Distribution Clusters in Indonesia (Classification)",
            x=0.5,
            xanchor='center'
        )
    )

    return fig.to_json()
    

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    try:
        # Load and preprocess data
        df = pd.read_csv(file, encoding='utf-8')
        df_processed = preprocess_data(df)
        
        # Perform classification and get features
        clf, accuracy, report, features_pca = perform_classification(df_processed)
        
        # Generate visualization
        choropleth = create_choropleth(df_processed)
        
        # Generate analysis summary
        cluster_summary = df_processed.groupby('cluster')['wilayah'].apply(list).to_dict()
        cluster_stats = df_processed.groupby('cluster').agg({
            'total': 'sum',
            'total_l': 'sum',
            'total_p': 'sum'
        }).to_dict('index')

        largest_cluster = max(cluster_stats.items(), key=lambda x: x[1]['total'])[0]
        
        analysis_summary = {
            "total_provinces": len(df_processed),
            "total_clusters": len(cluster_summary),
            "cluster_summary": {
                str(cluster): {
                    "provinces": provinces,
                    "total_students": int(cluster_stats[cluster]['total']),
                    "male_students": int(cluster_stats[cluster]['total_l']),
                    "female_students": int(cluster_stats[cluster]['total_p'])
                }
                for cluster, provinces in cluster_summary.items()
            },
            "largest_cluster": {
                "cluster_id": int(largest_cluster),
                "provinces": cluster_summary[largest_cluster],
                "total_students": int(cluster_stats[largest_cluster]['total']),
                "male_students": int(cluster_stats[largest_cluster]['total_l']),
                "female_students": int(cluster_stats[largest_cluster]['total_p'])
            }
        }

        return jsonify({
            'heatmap': choropleth,
            'data_preview': df_processed.head().to_html(
                classes='table table-bordered table-striped table-hover',
                index=False
            ),
            'analysis_summary': analysis_summary,
            'evaluation': {
                'accuracy': accuracy,
                'classification_report': report
            },
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)