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

    def plot_occurrences(self, classes, start_time = 0, end_time = -1):

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
                data = self.get_class_occurrences(classname, start_time, end_time)

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
