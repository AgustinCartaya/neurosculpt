
import numpy as np
from . import functions as fc
import matplotlib.pyplot as plt
import seaborn as sns




def show_rg_img(img_r, img_g, name="", current_plane=116):
    img_rg = np.stack((img_r, img_g, np.zeros_like(img_r)), axis=3)
    fc.imgshow_3D(name, img_rg, current_plane=current_plane, transpose=True, flip_ax=0)


def shows_3_images(img_hunt3, img_hunt4, img_gen, current_plane=108, show=True):
    if img_hunt3.ndim == 3:
        img_hunt3 = img_hunt3[..., current_plane]
        img_hunt4 = img_hunt4[..., current_plane]
        img_gen = img_gen[..., current_plane]

    img_hunt3 = np.flip(img_hunt3.T)
    img_hunt4 = np.flip(img_hunt4.T)
    img_gen = np.flip(img_gen.T)

    if show:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        fc.imgshow_matplotlib(f"HUNT4", img_hunt4, subplot=axs[0])
        fc.imgshow_matplotlib(f"img_gen", img_gen, subplot=axs[1])
        fc.imgshow_matplotlib(f"HUNT3", img_hunt3, subplot=axs[2])

        plt.show()
    return img_hunt3, img_hunt4, img_gen


# ------------------ STATISTICS ------------------
def plot_data_voxplot(data, columns_name=None, nb_features=-1, nb_features_from=0, figsize=(20, 10)):
    if columns_name is None:
        columns_name = [str(i) for i in range(data.shape[1])]

    if nb_features > 0:
        data = data[:, nb_features_from:nb_features_from+nb_features]
        columns_name = columns_name[nb_features_from:nb_features_from+nb_features]

    plt.figure(figsize=figsize)
    plt.boxplot(data, labels=columns_name)
    plt.title("Features")
    plt.plot()


def plot_data_precomputed_correlation_np(corr_matrix, figsize=(30, 30)):
    plt.figure(figsize=figsize)
    plt.imshow(corr_matrix, cmap='coolwarm')
    plt.colorbar()
    plt.title('Matriz de correlación')
    plt.show()

def plot_data_correlation_np(data, figsize=(30, 30)):
    corr_matrix = np.corrcoef(data, rowvar=False)
    plot_data_precomputed_correlation_np(corr_matrix, figsize=figsize)


def plot_data_precomputed_correlation_df(corr_matrix_df, figsize=(30, 30)):
    plt.figure(figsize=figsize)
    ax =sns.heatmap(corr_matrix_df, annot=True, cmap='coolwarm')
    ax.xaxis.tick_top()

    plt.title('Correlation Matrix')

    plt.show()



def plot_data_correlation_df(data_df, figsize=(30, 30)):
    corr_matrix_df = data_df.corr()
    plot_data_precomputed_correlation_df(corr_matrix_df, data_df, figsize=figsize)


def plot_2D_points(x, y, x_name, y_name, clusters=None, figsize=(10, 10), ):
    fig, ax = plt.subplots(figsize=figsize)
    if clusters is not None:
        for i, cluster in enumerate(clusters):
            color = 'C{}'.format(cluster)
            ax.scatter(x[i], y[i], color=color)
    else:
        ax.scatter(x, y)
    # Configurar el gráfico
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f'{x_name} vs {y_name}')
    plt.show()


def plot_2D_points_2v2_all_data_np(data, clusters=None, figsize=(10, 10)):
    for i in range(0, data.shape[1]):
        for j in range(i+1, data.shape[1]):
            plot_2D_points(data[:, i], data[:, j], str(i), str(j), clusters=clusters, figsize=figsize)

def plot_2D_points_2v2_all_data_df(data, clusters=None, figsize=(10, 10)):
    columns = data.columns
    for i in range(0, len(columns)):
        for j in range(i+1, len(columns)):
            plot_2D_points(data[columns[i]], data[columns[j]], columns[i], columns[j], clusters=clusters, figsize=figsize) 


def plot_hist(data, name="", bins="auto", kde=True, binrange=None, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    sns.histplot(data, kde=kde, bins=bins, binrange=binrange)
    plt.title(f'{name}')
    plt.show()

def plot_hits_df(data, column, bins="auto", kde=True, binrange=None, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    sns.histplot(data[column], kde=kde, bins=bins, binrange=binrange)
    plt.title(f'{column}')
    plt.show()

def plot_hist_all_data_df(data, bins="auto", kde=True, binrange=None, figsize=(10, 10)):
    columns = data.columns
    for column in columns:
        plot_hits_df(data, column, bins=bins, kde=kde, binrange=binrange, figsize=figsize)


def visualize_clusters(data_df, clusters, cluster_by, figsize=(10,5), hist=False, bins=0):
    if len(cluster_by) == 1:
        if hist:
            # bins = 3 if cluster_with_center else 2
            plot_hits_df(data_df, cluster_by[0], bins=bins, kde=False, binrange=None, figsize=figsize)
        else:
            plot_2D_points(data_df[cluster_by[0]], np.zeros(data_df[cluster_by[0]].shape[0]), cluster_by[0], "", clusters=clusters, figsize=figsize)

    elif len(cluster_by) == 2:
        plot_2D_points(data_df[cluster_by[0]], data_df[cluster_by[1]], cluster_by[0], cluster_by[1], clusters=clusters, figsize=figsize)
