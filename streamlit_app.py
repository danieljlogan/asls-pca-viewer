import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Raman PCA Interactive Viewer",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'current_pc' not in st.session_state:
    st.session_state.current_pc = 0
if 'selected_spectrum' not in st.session_state:
    st.session_state.selected_spectrum = None
if 'selected_pixel' not in st.session_state:
    st.session_state.selected_pixel = None
if 'pca_model' not in st.session_state:
    st.session_state.pca_model = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

st.title("🔬 Raman PCA Interactive Viewer Using AsLS")
st.markdown("---")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Data Loading")
    uploaded_file = st.file_uploader(
        "Load Baselined Data CSV",
        type=['csv'],
        help="Upload your 'Baselined_data again.csv' file"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            wavenumbers = df.iloc[:, 0].values
            data = df.iloc[:, 1:].values.T
            
            # Reshape to spatial grid (assuming 63x74 grid)
            rows, cols = 63, 74
            spatial_data = data.reshape(rows, cols, -1)
            
            # Perform PCA
            with st.spinner("Performing PCA analysis..."):
                # SNV normalization
                data_snv = (data - np.nanmean(data, axis=1, keepdims=True)) / \
                           np.nanstd(data, axis=1, keepdims=True)
                
                # Handle any remaining NaNs
                data_snv = np.nan_to_num(data_snv)
                
                n_comps = st.slider("Number of Principal Components", 2, 10, 5)
                pca = PCA(n_components=n_comps)
                scores = pca.fit_transform(data_snv)
                score_map = scores.reshape(rows, cols, -1)
                loadings = pca.components_
                explained_variance = pca.explained_variance_ratio_
                
                # Store in session state
                st.session_state.wavenumbers = wavenumbers
                st.session_state.spatial_data = spatial_data
                st.session_state.score_map = score_map
                st.session_state.loadings = loadings
                st.session_state.explained_variance = explained_variance
                st.session_state.n_comps = n_comps
                st.session_state.rows = rows
                st.session_state.cols = cols
                st.session_state.data_loaded = True
                st.session_state.current_pc = 0  # Reset to PC1
                
                st.success(f"✅ Data loaded! {rows}x{cols} grid, {n_comps} PCs calculated")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.session_state.data_loaded = False

# Main app - only show if data is loaded
if st.session_state.data_loaded:
    # Control panel
    col_controls, col_export = st.columns([2, 1])
    
    with col_controls:
        st.subheader("🎮 Component Selection")
        # Create buttons for PC selection
        pc_buttons = st.columns(min(st.session_state.n_comps, 5))
        for i in range(st.session_state.n_comps):
            col_idx = i % 5
            with pc_buttons[col_idx]:
                if st.button(
                    f"PC{i+1}\n({st.session_state.explained_variance[i]:.1%})",
                    use_container_width=True,
                    type="primary" if i == st.session_state.current_pc else "secondary"
                ):
                    st.session_state.current_pc = i
                    st.rerun()
    
    with col_export:
        st.subheader("💾 Export Options")
        export_format = st.selectbox("Format", ["SVG", "PNG", "PDF"])
        export_dpi = st.selectbox("DPI", ["150", "300", "600"], index=1)
        
        if st.button("📸 Save All Plots", use_container_width=True):
            st.info("Click save in each plot window below")
    
    # Main plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📊 Spatial Map: PC{st.session_state.current_pc+1}")
        # Spatial map with click interaction
        score_data = st.session_state.score_map[:, :, st.session_state.current_pc]
        
        fig_map = go.Figure(data=go.Heatmap(
            z=score_data,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='X: %{x}<br>Y: %{y}<br>Score: %{z:.3f}<extra></extra>'
        ))
        
        fig_map.update_layout(
            height=500,
            xaxis_title="X Pixel",
            yaxis_title="Y Pixel",
            hovermode='closest'
        )
        
        # Display the map
        st.plotly_chart(fig_map, use_container_width=True, key="spatial_map")
        
        # Manual pixel selection (since Plotly click events need JS)
        st.markdown("**📍 Select Pixel Coordinates:**")
        px_col1, px_col2, px_col3 = st.columns([1,1,2])
        with px_col1:
            x_pixel = st.number_input("X Pixel", 0, st.session_state.cols-1, 0, key="x_pixel")
        with px_col2:
            y_pixel = st.number_input("Y Pixel", 0, st.session_state.rows-1, 0, key="y_pixel")
        with px_col3:
            if st.button("🔍 Load Spectrum at Pixel", use_container_width=True):
                st.session_state.selected_spectrum = st.session_state.spatial_data[y_pixel, x_pixel, :]
                st.session_state.selected_pixel = (x_pixel, y_pixel)
                st.success(f"Spectrum loaded from pixel ({x_pixel}, {y_pixel})")
    
    with col2:
        st.subheader(f"📈 Chemical Loading: PC{st.session_state.current_pc+1}")
        
        fig_load = go.Figure()
        fig_load.add_trace(go.Scatter(
            x=st.session_state.wavenumbers,
            y=st.session_state.loadings[st.session_state.current_pc, :],
            mode='lines',
            name=f'PC{st.session_state.current_pc+1}',
            line=dict(color='crimson', width=2)
        ))
        fig_load.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig_load.update_layout(
            height=500,
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Loading Intensity",
            showlegend=False,
            hovermode='closest'
        )
        fig_load.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig_load.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        st.plotly_chart(fig_load, use_container_width=True)
    
    # Spectrum display
    st.markdown("---")
    st.subheader("🔬 Raman Spectrum")
    
    if st.session_state.selected_spectrum is not None:
        spec_col1, spec_col2, spec_col3 = st.columns([3,1,1])
        
        with spec_col1:
            fig_spec = go.Figure()
            fig_spec.add_trace(go.Scatter(
                x=st.session_state.wavenumbers,
                y=st.session_state.selected_spectrum,
                mode='lines',
                name='Spectrum',
                line=dict(color='black', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,0,0,0.1)'
            ))
            fig_spec.update_layout(
                height=400,
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Intensity",
                hovermode='closest',
                title=f"Spectrum at Pixel {st.session_state.selected_pixel}"
            )
            fig_spec.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig_spec.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            st.plotly_chart(fig_spec, use_container_width=True)
        
        with spec_col2:
            if st.button("💾 Save as CSV", use_container_width=True):
                spec_df = pd.DataFrame({
                    'Wavenumber': st.session_state.wavenumbers,
                    'Intensity': st.session_state.selected_spectrum
                })
                csv = spec_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="raman_spectrum.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with spec_col3:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.info("Select all (Ctrl+A) and copy from the plot")
    else:
        st.info("👆 Click 'Load Spectrum at Pixel' to view a Raman spectrum")
    
    # Variance explanation panel
    with st.expander("📊 Explained Variance Ratio"):
        var_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(st.session_state.n_comps)],
            'Explained Variance': st.session_state.explained_variance,
            'Cumulative': np.cumsum(st.session_state.explained_variance)
        })
        var_df['Explained Variance'] = var_df['Explained Variance'].apply(lambda x: f"{x:.2%}")
        var_df['Cumulative'] = var_df['Cumulative'].apply(lambda x: f"{x:.2%}")
        st.dataframe(var_df, use_container_width=True)
        
        # Bar chart of variance
        fig_var = go.Figure(data=[
            go.Bar(name='Individual', x=[f'PC{i+1}' for i in range(st.session_state.n_comps)], 
                   y=st.session_state.explained_variance)
        ])
        fig_var.update_layout(yaxis_title="Explained Variance Ratio", yaxis_tickformat=".0%")
        st.plotly_chart(fig_var, use_container_width=True)

else:
    # Instructions when no data is loaded
    st.info("👈 Please upload your Baselined Raman data CSV file to begin")
    
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### How to use this PCA Viewer:
        
        1. **Upload your data** using the sidebar on the left
        2. **Select number of PCs** for analysis
        3. **Click PC buttons** to switch between components
        4. **Enter X/Y coordinates** and click 'Load Spectrum' to view Raman spectra
        5. **Export plots** using the format selector and save buttons
        
        ### Expected data format:
        - CSV file with wavenumbers in the first column
        - Each subsequent column represents a Raman spectrum at a specific pixel
        - Data will be reshaped to a 63x74 spatial grid (adjust rows/cols in code if needed)
        
        ### Features:
        - Interactive spatial maps of PC scores
        - Loading plots for chemical interpretation
        - Extract and export individual spectra
        - Variance explanation statistics
        """)

# Footer
st.markdown("---")
st.caption("Raman PCA Interactive Viewer | Built with Streamlit & Plotly")
