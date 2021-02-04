import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class VideoAnalysis:
    records = None

    def __init__(self, records) -> None:
        self.records = records
        # self.records['timestamp'] = self.records['timestamp'] / 1000 

    def get_class_occurrences(self, classname, start_time=0, end_time=-1):
        max_timestamp = self.records['timestamp'].max()

        assert start_time >= 0

        # Limita el tiempo final al máximo registro del video
        if end_time > max_timestamp or end_time == -1:
            end_time = max_timestamp

        assert start_time < end_time

        # filtra registros dentro del lapso de tiempo [start_time, end_time]
        filtered_records = self.records[(self.records['timestamp'] >= start_time) & (
            self.records['timestamp'] <= end_time)]

        # Cuenta instancias de cada clase en el lapso especificado
        records_by_class = filtered_records[filtered_records['class'] == classname]
        instances_per_timestamp = records_by_class.groupby(['timestamp']).count()[
            'class']

        # Agrega todos los frames en los que no se detectaron instancias y asigna la cuenta en 0
        # (Evita interpolación de instancias cuando no se detectó ninguna)
        missing = np.setdiff1d(
            filtered_records['timestamp'].unique(), instances_per_timestamp.index)
        missing_series = pd.Series(
            np.zeros(len(missing)), index=missing, dtype=np.int64)

        concatenated = pd.concat([instances_per_timestamp, missing_series])

        return concatenated.sort_index()

    def get_unique_classes(self, start_time=0, end_time=-1, as_seconds=False):
        records = self.records.copy()
        if as_seconds:
            start_time *= 1000
            end_time *= 1000

        filtered_records = records[records['timestamp'] >= start_time]
        if end_time != -1:
            filtered_records = filtered_records[filtered_records['timestamp'] <= end_time]

        return filtered_records['class'].unique()

    def get_complete_class_counts(self, as_seconds=False):
        processed_records = self.records.copy()
        if as_seconds:
            processed_records['timestamp'] = processed_records['timestamp'] / 1000
        ts_class_records = processed_records[['timestamp', 'class']]
        instance_counts = ts_class_records.pivot_table(index='timestamp', columns='class', aggfunc=len)
        instance_counts = instance_counts.fillna(0)

        df = pd.DataFrame(instance_counts.values, columns=list(instance_counts.columns))
        df['timestamp'] = instance_counts.index

        return df

    def get_timeranges(self, filtered_records, time_tolerance = 2000):
        
        min_timestamp = filtered_records['timestamp'].min()
        max_timestamp = filtered_records['timestamp'].max()

        # Dataframe con solo los timestamps
        filtered_timestamps = filtered_records[['timestamp']]

        # calcula diferencia entre frames consecutivos (Si hay mucha diferencia, las detecciones
        # ocurren en diferentes lapsos de tiempo)
        filtered_timestamps['time_diff'] = filtered_timestamps.diff()[
            'timestamp']

        time_breakpoints = filtered_timestamps[filtered_timestamps['time_diff']
                                               > time_tolerance]

        time_ranges = []
        previous_timestamp = min_timestamp

        # Crea los rangos de tiempo con separaciones en donde hay mucha diferencia de tiempo
        if len(filtered_timestamps) > 0 and len(time_breakpoints) == 0:
            time_ranges.append((min_timestamp, max_timestamp))

        max_index = time_breakpoints.index.max()
        for index, row in time_breakpoints.iterrows():
            time_ranges.append(
                (previous_timestamp, row['timestamp'] - row['time_diff']))
            previous_timestamp = row['timestamp']

            if index == max_index:
                time_ranges.append((row['timestamp'], max_timestamp))

        return time_ranges
    
    def get_timeranges_by_instance_counts(self, class_counts, time_tolerance = 2000):
        assert isinstance(class_counts, dict)
        target_classes = list(class_counts.keys())
        records = self.records.copy()

        # Filtra registros que solo estan en ciertas clases
        records_by_class = records[records['class'].isin(target_classes)]

        # Crea una pivot table con la cuenta de instancias en cada timestamp
        ts_class_records = records_by_class[['timestamp', 'class']]
        instance_counts = ts_class_records.pivot_table(index='timestamp', columns='class', aggfunc=len, fill_value=0)

        # obtiene condiciones para filtrar por la cantidad de instancias
        instance_count_conditions = []
        for i, classname in enumerate(class_counts):
            if i > 0:
                instance_count_conditions = np.logical_and(instance_count_conditions, instance_counts[classname] == class_counts[classname])
            else:
                instance_count_conditions = instance_counts[classname] == class_counts[classname]

        timestamps_list = instance_counts[instance_count_conditions].index

        # filtra los registros cuyos timestamps tienen todas las clases
        filtered_timestamp_records = records_by_class[records_by_class['timestamp'].isin(timestamps_list)]

        return self.get_timeranges(filtered_timestamp_records, time_tolerance)

    def get_timeranges_with_classes(self, target_classes, time_tolerance = 2000):
        
        assert isinstance(target_classes, list)
        target_classes = list(set(target_classes))
        records = self.records.copy()

        # Filtra registros que solo estan en ciertas clases
        records_by_class = records[records['class'].isin(target_classes)]

        # Obtiene timestamps en los que se detectaron las n (target_clases) clases
        timestamp_list = records_by_class.groupby('timestamp').nunique()[
            'class'] == len(target_classes)
        timestamp_list = timestamp_list[timestamp_list == True].index

        # filtra los registros cuyos timestamps tienen todas las clases
        filtered_timestamp_records = records_by_class[records_by_class['timestamp'].isin(
            timestamp_list)]

        return self.get_timeranges(filtered_timestamp_records, time_tolerance)
