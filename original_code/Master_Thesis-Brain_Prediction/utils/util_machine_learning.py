import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def remove_outliers_zscore(data, threshold=3):
    z_scores = stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    outlayers_index = np.where((abs_z_scores >= threshold).any(axis=1))[0]
    no_outlayers_index = np.delete(np.arange(len(data)), outlayers_index)
    filtered_data = data[no_outlayers_index]
    return filtered_data, no_outlayers_index, outlayers_index

def standarize_columns(data):
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    return data_standardized, scaler

def normalize_columns(data):
    scaler = MinMaxScaler()
    data_standardized = scaler.fit_transform(data)
    return data_standardized, scaler

def correlation_df(data_df, corr_th=0.9):
    corr_df = data_df.corr().abs()
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    tri_df = corr_df.mask(mask)
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > corr_th)]

    return corr_df, to_drop


# class Preprocessing:


def cluster_by_mean_df(data, columns):
    clusters = []
    columns_bool = [f"{column}_bool" for column in columns]
    data_to_cluster = data.copy()[columns]
    means = data_to_cluster.mean().values

    for i, column in enumerate(columns):
        data_to_cluster[columns_bool[i]] = data_to_cluster[column].apply(lambda x: 1 if x > means[i] else 0)

    binaries = data_to_cluster[columns_bool].astype(str).values
    for binary in binaries:
        clusters.append(int(''.join(binary), 2))
    return clusters


def cluster_by_mean_with_center_df(data, columns, center_th):
    clusters = []
    columns_bool = [f"{column}_bool" for column in columns]
    data_to_cluster = data.copy()[columns]
    means = data_to_cluster.mean().values

    for i, column in enumerate(columns):
        add_center = means[i] * center_th
        data_to_cluster[columns_bool[i]] = data_to_cluster[column].apply(lambda x: 0 if x < means[i]-add_center else 2 if x > means[i]+add_center else 1)

    binaries = data_to_cluster[columns_bool].astype(str).values
    classes = []
    for binary in binaries:
        classes.append(''.join(binary))
    classes_uniques = np.unique(classes)
    for c in classes:
        clusters.append(np.where(classes_uniques == c)[0][0])
    return clusters