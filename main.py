from matplotlib.pyplot import step
import streamlit as st
from video_analysis import VideoAnalysis
from object_detector import ObjectDetector
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import base64


st.title('Análisis de video')


detector = ObjectDetector('config/yolov4.weights',
                          'config/yolov4.cfg', 'config/coco.names')

analysis_opt = st.radio('Selecciona un método para hacer análisis', ['Archivo de video', 'CSV con registros'])


video_progress = st.progress(0)

def process(progress):
    print(progress)
    video_progress.progress(progress)

def process_video(filename):
    return pd.DataFrame(detector.stream_videofile(filename, process, 1))

f = st.file_uploader('Selecciona un archivo')

@st.cache(allow_output_mutation=True)
def get_records():
    if analysis_opt == 'Archivo de video':
        if f is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(f.read())
            return process_video(tfile.name)

    elif analysis_opt == 'CSV con registros':
        if f is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(f.read())
            return pd.read_csv(tfile.name)

    return []

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="records.csv">Descarga los datos en csv</a>'


def compute_analysis(records):
    if (len(records)) == 0:
        return
    video_analysis = VideoAnalysis(records)
    st.markdown(get_table_download_link(records), unsafe_allow_html=True)
    st.header('Registros de detecciones')
    st.dataframe(records)

    df = video_analysis.get_complete_class_counts(as_seconds=True)

    fig = px.bar(df, x='timestamp', y=df.columns, title='Cantidad de instancias detectadas durante el video')

    fig.update_layout(
        yaxis=dict(
            title=dict(text='instancias'),
        ),
        xaxis=dict(
            title=dict(text='segundos'),
            rangeslider=dict(
                visible=True
            ),
            type="linear"
        )
    )

    st.plotly_chart(
        fig)

    st.header('Buscar rangos de tiempo donde aparecen un conjunto de clases')

    s1_col1, s1_col2 = st.beta_columns(2)

    with s1_col1:
        class_selection = st.multiselect('Selecciona las clases que deseas buscar', options=video_analysis.get_unique_classes())

    with s1_col2:    
        ranges = []
        if len(class_selection) > 0:
            ranges = video_analysis.get_timeranges_with_classes(class_selection)
        
        range_table = []
        for rng in ranges:
            range_table.append({
                'Desde': rng[0] / 1000,
                'Hasta': rng[1] / 1000
            })

        st.write('Rangos de tiempo donde aparecen las clases seleccionadas (en segundos)')
        st.table(range_table)

    instance_counts_selection = {}

    st.header('Buscar rangos de tiempo donde aparecen una cantidad especifica de instancias')

    s2_col1, s2_col2 = st.beta_columns(2)

    with s2_col1:
        for classname in video_analysis.get_unique_classes():
            instance_counts_selection[classname] = st.number_input(classname, min_value=0, value=0, step=1)
        
    with s2_col2:
        clean_selections = {}
        for key in instance_counts_selection:
            if instance_counts_selection[key] > 0:
                clean_selections[key] = int(instance_counts_selection[key])
        
        ranges = []
        if len(clean_selections.keys()) > 0:
            ranges = video_analysis.get_timeranges_by_instance_counts(clean_selections)
        
        range_table = []
        for rng in ranges:
            range_table.append({
                'Desde': rng[0] / 1000,
                'Hasta': rng[1] / 1000
            })
        st.write('Rangos de tiempo donde aparecen las clases seleccionadas (en segundos)')
        st.table(range_table)

compute_analysis(get_records())