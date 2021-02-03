import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class VideoAnalysis:
    records = None

    def __init__(self, records) -> None:
        self.records = records

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

    def get_timeranges_with_classes(self, target_classes, second_tolerance=2):

        #  = ['person', 'bicycle']
        assert  isinstance(target_classes, list)
        target_classes = list(set(target_classes))
        records = self.records

        # Filtra registros que solo estan en ciertas clases
        records_by_class = records[records['class'].isin(target_classes)]

        # Obtiene timestamps en los que se detectaron las n (target_clases) clases
        condition = records_by_class.groupby('timestamp').nunique()[
            'class'] == len(target_classes)
        condition = condition[condition == True]

        # filtra los registros cuyos timestamps tienen todas las clases
        filtered_timestamp_records = records_by_class[records_by_class['timestamp'].isin(
            condition.index)]

        min_timestamp = filtered_timestamp_records['timestamp'].min()
        max_timestamp = filtered_timestamp_records['timestamp'].max()

        # Dataframe con solo los timestamps
        filtered_timestamps = filtered_timestamp_records[['timestamp']]

        # calcula diferencia entre frames consecutivos (Si hay mucha diferencia, las detecciones
        # ocurren en diferentes lapsos de tiempo)
        filtered_timestamps['time_diff'] = filtered_timestamps.diff()[
            'timestamp']

        time_breakpoints = filtered_timestamps[filtered_timestamps['time_diff']
                                               > second_tolerance]
        
        timestamp_pairs = []
        previous_timestamp = min_timestamp

        # Crea los rangos de tiempo con separaciones en donde hay mucha diferencia de tiempo
        if len(filtered_timestamps) > 0 and len(time_breakpoints) == 0:
            timestamp_pairs.append((min_timestamp, max_timestamp))

        max_index = time_breakpoints.index.max()
        for index, row in time_breakpoints.iterrows():
            timestamp_pairs.append(
                (previous_timestamp, row['timestamp'] - row['time_diff']))
            previous_timestamp = row['timestamp']

            if index == max_index:
                timestamp_pairs.append((row['timestamp'], max_timestamp))

        return timestamp_pairs

    def plot_occurrences(self, classes, start_time=0, end_time=-1):

        plot_counts = len(classes)

        dims = int(math.ceil(math.sqrt(plot_counts)))
        print(dims)
        fig, axs = plt.subplots(dims, dims, squeeze=False)

        class_index = 0
        for i in range(dims):
            for j in range(dims):
                if class_index >= plot_counts:
                    break
                classname = classes[class_index]
                data = self.get_class_occurrences(
                    classname, start_time, end_time)

                ts = np.array(data.index)
                class_counts = np.array(data)

                # plt.title(classname)
                # plt.ylabel('instancias')
                # plt.xlabel('segundos')
                # plt.step(ts, class_counts)
                # plt.show()
                axs[i, j].set_title(classname)
                axs[i, j].step(ts, class_counts)
                class_index += 1

        for ax in axs.flat:
            ax.set(xlabel='segundos', ylabel='instancias')

        for ax in axs.flat:
            ax.label_outer()