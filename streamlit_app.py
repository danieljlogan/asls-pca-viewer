import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import base64

st.set_page_config(page_title="Raman PCA Viewer", layout="wide")

st.title("🔬 Raman PCA Interactive Viewer")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    wavenumbers = df.iloc[:, 0].values
    data = df.iloc[:, 1:].values.T
    rows, cols = 63, 74
    spatial_data = data.reshape(rows, cols, -1)
    
    # PCA
    data_clean = np.nan_to_num(data)
    data_snv = (data_clean - np.mean(data_clean, axis=1, keepdims=True)) / np.std(data_clean, axis=1, keepdims=True)
    
    n_comps = st.sidebar.slider("Number of PCs", 2, 10, 5)
    pca = PCA(n_components=n_comps)
    scores = pca.fit_transform(data_snv)
    score_map = scores.reshape(rows, cols, -1)
    loadings = pca.components_
    var_ratio = pca.explained_variance_ratio_
    
    pc_idx = st.sidebar.selectbox("Select PC", range(n_comps), format_func=lambda x: f"PC{x+1} ({var_ratio[x]:.1%})")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spatial Map")
        fig1 = go.Figure(data=go.Heatmap(z=score_map[:,:,pc_idx], colorscale='Viridis'))
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        x_pixel = st.number_input("X", 0, cols-1, 0)
        y_pixel = st.number_input("Y", 0, rows-1, 0)
        
        if st.button("Get Spectrum"):
            spectrum = spatial_data[y_pixel, x_pixel, :]
            st.session_state.spectrum = spectrum
            st.session_state.pixel = (x_pixel, y_pixel)
    
    with col2:
        st.subheader("Loadings")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=wavenumbers, y=loadings[pc_idx,:], mode='lines', line=dict(color='crimson')))
        fig2.add_hline(y=0, line_dash="dash")
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Raman Spectrum")
    
    if 'spectrum' in st.session_state:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=wavenumbers, y=st.session_state.spectrum, mode='lines'))
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        if st.button("Save Spectrum CSV"):
            spec_df = pd.DataFrame({'Wavenumber': wavenumbers, 'Intensity': st.session_state.spectrum})
            csv = spec_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="spectrum.csv">Download CSV</a>', unsafe_allow_html=True)
    else:
        st.info("Select X/Y and click 'Get Spectrum'")
else:
    st.info("Please upload a CSV file to begin")
