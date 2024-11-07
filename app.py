import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify
import io
import base64
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def clean_numeric(x):
    if isinstance(x, str):
        x = x.replace(',', '').replace(' ', '').replace('.', '')
        if x == '' or x.lower() == 'nan':
            return 0
    if pd.isna(x) or (isinstance(x, float) and (np.isinf(x) or np.isnan(x))):
        return 0
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0

def preprocess_data(df):
    df = df.drop(df.index[-1])
    
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

    df['wilayah'] = df['wilayah'].str.replace('Prov. ', '')
    
    numeric_cols = df.columns.drop(['no', 'wilayah'])
    
    for col in numeric_cols:
        df[col] = df[col].apply(clean_numeric)
        df[col] = df[col].fillna(0)
        df[col] = df[col].astype(int)
    
    return df

def perform_clustering(df, n_clusters=3):
    features = df.drop(columns=['no', 'wilayah'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    return df, kmeans

def create_visualizations(df):
    # Top 10 Provinces visualization
    fig_top10 = px.bar(
        df.nlargest(10, 'total'),
        x='wilayah',
        y='total',
        title='10 Provinsi dengan Jumlah Siswa Terbanyak',
        labels={'wilayah': 'Provinsi', 'total': 'Jumlah Siswa'},
        text='total'
    )
    fig_top10.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    
    # Education distribution visualization
    education_levels = {
        'TK': df['tk_total'].sum(),
        'KB': df['kb_total'].sum(),
        'TPA': df['tpa_total'].sum(),
        'SPS': df['sps_total'].sum(),
        'PKBM': df['pkbm_total'].sum(),
        'SKB': df['skb_total'].sum(),
        'SD': df['sd_total'].sum(),
        'SMP': df['smp_total'].sum(),
        'SMA': df['sma_total'].sum(),
        'SMK': df['smk_total'].sum(),
        'SLB': df['slb_total'].sum()
    }
    
    fig_edu_dist = px.pie(
        values=list(education_levels.values()),
        names=list(education_levels.keys()),
        title='Distribusi Siswa Berdasarkan Jenjang Pendidikan'
    )
    fig_edu_dist.update_traces(texttemplate='%{value:,.0f}<br>(%{percent:.1f}%)')
    
    # Gender distribution visualization
    gender_data = {
        'Jenis Kelamin': ['Laki-laki', 'Perempuan'],
        'Jumlah': [df['total_l'].sum(), df['total_p'].sum()]
    }
    fig_gender = px.pie(
        gender_data,
        values='Jumlah',
        names='Jenis Kelamin',
        title='Distribusi Siswa Berdasarkan Jenis Kelamin'
    )
    fig_gender.update_traces(texttemplate='%{value:,.0f}<br>(%{percent:.1f}%)')
    
    # Education by gender visualization
    education_gender = pd.DataFrame({
        'Jenjang': ['TK', 'KB', 'TPA', 'SPS', 'PKBM', 'SKB', 'SD', 'SMP', 'SMA', 'SMK', 'SLB'],
        'Laki-laki': [
            df['tk_l'].sum(), df['kb_l'].sum(), df['tpa_l'].sum(),
            df['sps_l'].sum(), df['pkbm_l'].sum(), df['skb_l'].sum(),
            df['sd_l'].sum(), df['smp_l'].sum(), df['sma_l'].sum(),
            df['smk_l'].sum(), df['slb_l'].sum()
        ],
        'Perempuan': [
            df['tk_p'].sum(), df['kb_p'].sum(), df['tpa_p'].sum(),
            df['sps_p'].sum(), df['pkbm_p'].sum(), df['skb_p'].sum(),
            df['sd_p'].sum(), df['smp_p'].sum(), df['sma_p'].sum(),
            df['smk_p'].sum(), df['slb_p'].sum()
        ]
    })
    
    fig_edu_gender = px.bar(
        education_gender,
        x='Jenjang',
        y=['Laki-laki', 'Perempuan'],
        title='Distribusi Gender per Jenjang Pendidikan',
        barmode='group',
        text_auto=True
    )
    fig_edu_gender.update_traces(texttemplate='%{text:,.0f}')

    # Regional Distribution: Replace map with horizontal bar chart
    fig_regional = px.bar(
        df.sort_values('total', ascending=True),
        y='wilayah',
        x='total',
        orientation='h',
        title='Distribusi Siswa per Provinsi',
        labels={'wilayah': 'Provinsi', 'total': 'Jumlah Siswa'},
        text='total'
    )
    fig_regional.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    
    # Education Level Trends
    df_melted = pd.melt(education_gender, 
                        id_vars=['Jenjang'], 
                        value_vars=['Laki-laki', 'Perempuan'])
    fig_trends = px.line(
        df_melted,
        x='Jenjang',
        y='value',
        color='variable',
        title='Tren Partisipasi Siswa per Jenjang Pendidikan',
        labels={'value': 'Jumlah Siswa', 'variable': 'Jenis Kelamin'}
    )
    
    # Gender Ratio by Province
    df['gender_ratio'] = df['total_l'] / df['total_p']
    fig_gender_ratio = px.bar(
        df.sort_values('gender_ratio'),
        x='wilayah',
        y='gender_ratio',
        title='Rasio Gender (L:P) per Provinsi',
        labels={'wilayah': 'Provinsi', 'gender_ratio': 'Rasio L:P'}
    )
    fig_gender_ratio.add_hline(y=1, line_dash="dash", line_color="red")
    
    # Educational Institution Distribution
    institution_data = pd.DataFrame({
        'Institusi': ['TK', 'KB', 'TPA', 'SPS', 'PKBM', 'SKB', 'SD', 'SMP', 'SMA', 'SMK', 'SLB'],
        'Jumlah_Siswa': [
            df['tk_total'].sum(),
            df['kb_total'].sum(),
            df['tpa_total'].sum(),
            df['sps_total'].sum(),
            df['pkbm_total'].sum(),
            df['skb_total'].sum(),
            df['sd_total'].sum(),
            df['smp_total'].sum(),
            df['sma_total'].sum(),
            df['smk_total'].sum(),
            df['slb_total'].sum()
        ]
    })
    fig_inst_dist = px.treemap(
        institution_data,
        path=['Institusi'],
        values='Jumlah_Siswa',
        title='Distribusi Siswa per Jenis Institusi Pendidikan'
    )
    
    # Clustering visualization
    fig_clusters = px.scatter(
        df,
        x='total',
        y='total_l',
        color='cluster',
        title='Klasterisasi Provinsi',
        labels={'total': 'Total Siswa', 'total_l': 'Total Siswa Laki-laki'},
        hover_data={'wilayah': True}  # Menambahkan nama provinsi sebagai hover data
    )
    
    # Konversi semua figur ke JSON
    plots = {
        'top10': fig_top10.to_json(),
        'edu_dist': fig_edu_dist.to_json(),
        'gender': fig_gender.to_json(),
        'edu_gender': fig_edu_gender.to_json(),
        'map': fig_regional.to_json(),
        'trends': fig_trends.to_json(),
        'gender_ratio': fig_gender_ratio.to_json(),
        'inst_dist': fig_inst_dist.to_json(),
        'clusters': fig_clusters.to_json()
    }
    
    # Statistik yang ditingkatkan
    stats = {
        'total_siswa': f"{df['total'].sum():,}",
        'total_provinsi' : len(df.iloc[3:44]),  # Excluding 'Luar Negeri'
        'ratio_gender': round(df['total_l'].sum() / df['total_p'].sum(), 2),
        'sd_participation': f"{(df['sd_total'].sum() / df['total'].sum() * 100):.1f}%",
        'avg_students_province': f"{int(df[df['wilayah'] != 'Luar Negeri']['total'].mean()):,}"
    }
    
    return plots, stats
    
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
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        df = pd.read_csv(file, encoding='utf-8')
        df_processed = preprocess_data(df)
        df_clustered, kmeans = perform_clustering(df_processed)
        plots, stats = create_visualizations(df_clustered)
        
        return jsonify({
            'plots': plots,
            'stats': stats,
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)