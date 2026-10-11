"""Matplotlib / OpenCV display helpers: interactive 3D viewers, image tables and statistics plots."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from brainpred.utils.intensity import normalize
# Re-exported so that ``display.<name>`` keeps working for code written against the old utils/functions.py.
from brainpred.utils.arrays import (  # noqa: F401
    calc_hist, extract_layer, pad_2D_img, reduce_3D_images_batch, remove_padding,
)

# Keeps references to the interactive sliders alive (otherwise matplotlib garbage-collects them).
_SLIDERS = []


# ---- conversions / helpers
def img_float2unit8(img, color_image=False, max_intensity=255):
    max_val = np.max(img)
    if max_val > 0:
        min_val = np.min(img)
        img_unit8 = max_intensity * ((img - min_val) / (max_val - min_val))
        img_unit8 = img_unit8.astype(np.uint8)
    else:
        img_unit8 = img.astype(np.uint8)

    if color_image:
        import cv2 as cv
        img_unit8 = cv.cvtColor(img_unit8, cv.COLOR_GRAY2RGB)
    return img_unit8


def get_color_copy(img):
    import cv2 as cv
    if len(img.shape) < 3:
        return cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    return img.copy()


def get_resize_dimensions(img, scale_percent=1):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    return (width, height)


# ---- drawing (OpenCV)
def draw_circle(img, circle, color=(0, 0, 255), thickness=-1):
    import cv2 as cv
    img_copy = get_color_copy(img)
    return cv.circle(img_copy, circle[0], circle[1], color, thickness)


def draw_point(img, point, radius=2, color=(0, 0, 255), thickness=-1):
    img_copy = get_color_copy(img)
    return draw_circle(img_copy, (point, radius), color, thickness)


def draw_points(img, points, radius=2, color=(0, 0, 255), thickness=-1):
    img_copy = get_color_copy(img)
    for point in points:
        img_copy = draw_point(img_copy, point, radius=radius, color=color, thickness=thickness)
    return img_copy


# ---- 2D display
def imgshow_matplotlib(name, img, scale=1, subplot=None, bgr2rgb=False, cmap='gray'):
    width, height = get_resize_dimensions(img, scale)
    if bgr2rgb:
        import cv2 as cv
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if subplot is None:
        plt.figure()
        plt.imshow(img, extent=[0, width, 0, height], cmap=cmap)
        plt.title(name)
        plt.axis('off')
    else:
        subplot.imshow(img, extent=[0, width, 0, height], cmap=cmap)
        subplot.set_title(name)
        subplot.axis('off')


def plot_precomputed_hist(hist, bins, title="Histogram", color="blue", alpha=0.5, ax=None):
    if ax is None:
        plt.figure(figsize=(8, 6))
        plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]), align='edge', alpha=alpha, color=color)
        plt.title(title)
        plt.tight_layout()
    else:
        ax.bar(bins[:-1], hist, width=(bins[1] - bins[0]), align='edge', alpha=alpha, color=color)
        ax.set_title(title)


# ---- 3D interactive viewers
def get_3Dimg_plane(img3D, layer, mooving_plane, transpose=False, flip_ax=-1):
    if mooving_plane == 0:
        img = img3D[layer]
    elif mooving_plane == 1:
        img = img3D[:, layer]
    elif mooving_plane == 2:
        img = img3D[:, :, layer]

    if transpose:
        img = np.transpose(img, (1, 0) + tuple(range(2, img.ndim)))
    if flip_ax >= 0:
        img = np.flip(img, axis=flip_ax)
    return img


def imgshow_3D(name, img3D, current_plane=100, cmap='gray', mooving_plane=2, transpose=True, flip_ax=0, show=True,
               range_values=(None, None), figsize=(8, 6)):
    """Show a 3D (optionally RGB) volume with a slider to move through ``mooving_plane``."""
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.25)

    img = ax.imshow(get_3Dimg_plane(img3D, current_plane, mooving_plane, transpose=transpose, flip_ax=flip_ax),
                    cmap=cmap, vmin=range_values[0], vmax=range_values[1])

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, f'layer {mooving_plane}', 0, img3D.shape[mooving_plane] - 1, valinit=current_plane, valstep=1)
    _SLIDERS.append(slider)

    def update(val):
        plane = int(slider.val)
        img.set_data(get_3Dimg_plane(img3D, plane, mooving_plane, transpose=transpose, flip_ax=flip_ax))
        fig.canvas.draw_idle()

    ax.set_title(name)
    slider.on_changed(update)
    if show:
        plt.show()


def create_index_square_table_size_n(nb_elements):
    numbers = list(range(nb_elements))
    columns = int(np.ceil(np.sqrt(nb_elements)))
    rows = int(np.ceil(nb_elements / columns))

    padded_array = np.full(rows * columns, None)
    padded_array[:nb_elements] = numbers
    return padded_array.reshape(rows, columns)


def create_index_rectangular_table_size_n(nb_elements, rows=None, cols=None):
    """Index table with ``rows`` or ``cols`` fixed (pass a negative value for the free dimension)."""
    if rows is not None and rows < 0:
        rows = None
    if cols is not None and cols < 0:
        cols = None
    if rows is not None and cols is not None:
        print("Only specify either rows or cols, not both.")
        return None
    if rows is None and cols is None:
        print("Specify either rows or cols.")
        return None
    if rows:
        cols = (nb_elements + rows - 1) // rows
    else:
        rows = (nb_elements + cols - 1) // cols

    padded_array = np.full(rows * cols, None)
    padded_array[:nb_elements] = list(range(nb_elements))
    return padded_array.reshape(rows, cols)


def concat_2Dimages_into_rectangular_table(images, custom_disposition=None):
    """Tile 2D images; ``custom_disposition`` is None (square), a (rows, cols) tuple or an index table."""
    if custom_disposition is None:
        _table = create_index_square_table_size_n(len(images))
    elif type(custom_disposition) == tuple:
        _table = create_index_rectangular_table_size_n(len(images), rows=custom_disposition[0], cols=custom_disposition[1])
    else:
        _table = custom_disposition
    imgs_dim = images[0].shape
    if len(imgs_dim) == 2:
        full_img = np.zeros((imgs_dim[0] * _table.shape[0], imgs_dim[1] * _table.shape[1]), dtype=np.float32)
    else:
        full_img = np.zeros((imgs_dim[0] * _table.shape[0], imgs_dim[1] * _table.shape[1], imgs_dim[2]), dtype=np.float32)

    for i in range(_table.shape[0]):
        for j in range(_table.shape[1]):
            if _table[i, j] is not None:
                full_img[i * imgs_dim[0]: (i + 1) * imgs_dim[0], j * imgs_dim[1]: (j + 1) * imgs_dim[1]] = images[_table[i, j]]
    return full_img


def get_3Dimg_list_plane(img3D_list, layer, mooving_plane, transpose=False, flip_ax=-1, subtitle_space=20):
    img2D_list = []
    for img3D in img3D_list:
        slice_tuple = [slice(None)] * img3D.ndim
        slice_tuple[mooving_plane] = layer
        img2D = img3D[tuple(slice_tuple)]
        if transpose:
            img2D = np.transpose(img2D, (1, 0) + tuple(range(2, img2D.ndim)))
        if flip_ax >= 0:
            img2D = np.flip(img2D, axis=flip_ax)
        if subtitle_space is not None:
            if img2D.ndim == 2:
                img2D = np.pad(img2D, ((subtitle_space, 0), (0, 0)), mode='constant', constant_values=0)
            else:
                img2D = np.pad(img2D, ((subtitle_space, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        img2D_list.append(img2D)
    return img2D_list


def imgshow_3D_list(name, img3D_list, current_plane=108, cmap='gray', mooving_plane=2, transpose=True, flip_ax=0, show=True,
                    range_values=(None, None), subimg_titles=None, subtitle_space=20, subtitle_help=(20, 10), figsize=(10, 8),
                    custom_disposition=None, sliders=_SLIDERS, save_dict=None):
    """Show a table of 3D (optionally RGB) volumes sharing one slider.

    ``save_dict = {"path_name": ..., "dpi": ...}`` saves the figure at the initial plane.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.15)

    img2D_list = get_3Dimg_list_plane(img3D_list, current_plane, mooving_plane, transpose=transpose, flip_ax=flip_ax, subtitle_space=subtitle_space)
    _full_img = concat_2Dimages_into_rectangular_table(img2D_list, custom_disposition=custom_disposition).astype(img3D_list[0].dtype)
    img = ax.imshow(_full_img, cmap=cmap, vmin=range_values[0], vmax=range_values[1])

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, f'layer {mooving_plane}', 0, img3D_list[0].shape[mooving_plane] - 1, valinit=current_plane, valstep=1)
    sliders.append(slider)

    def update(val):
        plane = int(slider.val)
        planes = get_3Dimg_list_plane(img3D_list, plane, mooving_plane, transpose=transpose, flip_ax=flip_ax, subtitle_space=subtitle_space)
        img.set_data(concat_2Dimages_into_rectangular_table(planes, custom_disposition=custom_disposition).astype(img3D_list[0].dtype))
        fig.canvas.draw_idle()

    ax.set_title(name)
    slider.on_changed(update)
    ax.axis('off')

    plt.rcParams['font.family'] = 'monospace'
    if subimg_titles is not None:
        if custom_disposition is None:
            custom_disposition = create_index_square_table_size_n(len(subimg_titles))
        elif type(custom_disposition) == tuple:
            custom_disposition = create_index_rectangular_table_size_n(len(subimg_titles), rows=custom_disposition[0], cols=custom_disposition[1])
        fontsize = 12
        font_size_inch = fontsize / 72
        font_size_pixels = int(np.ceil(font_size_inch * fig.dpi))

        img_size = img2D_list[0].shape
        for row in range(custom_disposition.shape[0]):
            for col in range(custom_disposition.shape[1]):
                if custom_disposition[row, col] is not None:
                    title = subimg_titles[custom_disposition[row, col]]
                    init_center_x = img_size[1] // 2 - (len(title) * font_size_pixels) // 2
                    pos_x = init_center_x + col * img_size[1] + subtitle_help[0]
                    pos_y = font_size_pixels + row * img_size[0] + subtitle_help[1]
                    ax.text(pos_x, pos_y, title, color='white',
                            fontdict={'family': 'monospace', 'weight': 'normal', 'size': fontsize})
    if save_dict is not None:
        plt.savefig(save_dict["path_name"], dpi=save_dict["dpi"], bbox_inches='tight')

    if show:
        plt.colorbar(img)
        plt.show()


def show_rg_img(img_r, img_g, name="", current_plane=116):
    """Overlay two volumes as the red and green channels."""
    img_rg = np.stack((img_r, img_g, np.zeros_like(img_r)), axis=3)
    imgshow_3D(name, img_rg, current_plane=current_plane, transpose=True, flip_ax=0)


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
        imgshow_matplotlib("HUNT4", img_hunt4, subplot=axs[0])
        imgshow_matplotlib("img_gen", img_gen, subplot=axs[1])
        imgshow_matplotlib("HUNT3", img_hunt3, subplot=axs[2])
        plt.show()
    return img_hunt3, img_hunt4, img_gen


# ---- landmarks (formerly utils/util.py)
def draw_landmarks(img3D, layer, landmarks=None, mooving_plane=2, lm_radius=5, mask3D=None, mask_tranparence=100):
    if mooving_plane == 0:
        img = img3D[layer].T
    elif mooving_plane == 1:
        img = img3D[:, layer, :].T
    elif mooving_plane == 2:
        img = img3D[..., layer].T

    if landmarks is None:
        return img

    if mooving_plane == 0:  # moving in x
        lm = landmarks[landmarks[:, 0] == layer][:, 1:].astype(np.uint32)
    elif mooving_plane == 1:  # moving in y
        lm = landmarks[landmarks[:, 1] == layer][:, [0, 2]].astype(np.uint32)
    elif mooving_plane == 2:  # moving in z
        lm = landmarks[landmarks[:, 2] == layer][:, :2].astype(np.uint32)

    img_wp = draw_points(img, lm, radius=lm_radius, color=(0, 255, 0))

    if mask3D is not None:
        if mooving_plane == 0:
            mask = mask3D[layer].T
        elif mooving_plane == 1:
            mask = mask3D[:, layer, :].T
        elif mooving_plane == 2:
            mask = mask3D[..., layer].T
        img_wp[..., 0] += (mask.astype(np.uint8) * mask_tranparence)
    return img_wp


def imgshow_3D_landmarks(name, img3D, landmarks=None, mask=None, current_plane=100, cmap='gray', mooving_plane=2,
                         lm_radius=5, mask_to_draw=None, mask_tranparence=20):
    """3D viewer drawing landmarks (formerly ``utils.util.imgshow_3D``)."""
    img3D = (normalize(img3D, mask) * 255).astype(np.uint8)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    img = ax.imshow(draw_landmarks(img3D, current_plane, landmarks, mooving_plane=mooving_plane, lm_radius=lm_radius,
                                   mask3D=mask_to_draw, mask_tranparence=mask_tranparence), cmap=cmap)

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, f'layer {mooving_plane}', 0, img3D.shape[mooving_plane] - 1, valinit=current_plane, valstep=1)
    _SLIDERS.append(slider)

    def update(val):
        plane = int(slider.val)
        img.set_data(draw_landmarks(img3D, plane, landmarks, mooving_plane=mooving_plane, lm_radius=lm_radius,
                                    mask3D=mask_to_draw, mask_tranparence=mask_tranparence))
        fig.canvas.draw_idle()

    ax.set_title(name)
    slider.on_changed(update)
    plt.show()


def plot_2Dhistograms(title, hist, x_edges, y_edges, axs=None, fig=None, x_label="", y_label="", cmap="gray"):
    if axs is None:
        plt.imshow(hist.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', cmap=cmap)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.colorbar()
    else:
        axs.imshow(hist.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], origin='lower', cmap=cmap)
        axs.set_xlabel(x_label)
        axs.set_ylabel(y_label)
        axs.set_title(title)


def plot_3d_vector_field(data, spacing=4):
    w, h, d = data.shape[:3]

    X, Y, Z, U, V, W = [], [], [], [], [], []
    for i in range(0, w, spacing):
        for j in range(0, h, spacing):
            for k in range(0, d, spacing // 4):
                X.append(i)
                Y.append(j)
                Z.append(k)
                U.append(data[i, j, k, 0])
                V.append(data[i, j, k, 1])
                W.append(data[i, j, k, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, length=5)
    ax.set_xlim([0, w])
    ax.set_ylim([0, h])
    ax.set_zlim([0, d])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# ---- tabular statistics (formerly utils/visualization.py)
def plot_data_voxplot(data, columns_name=None, nb_features=-1, nb_features_from=0, figsize=(20, 10)):
    if columns_name is None:
        columns_name = [str(i) for i in range(data.shape[1])]

    if nb_features > 0:
        data = data[:, nb_features_from:nb_features_from + nb_features]
        columns_name = columns_name[nb_features_from:nb_features_from + nb_features]

    plt.figure(figsize=figsize)
    plt.boxplot(data, labels=columns_name)
    plt.title("Features")
    plt.plot()


def plot_data_precomputed_correlation_np(corr_matrix, figsize=(30, 30)):
    plt.figure(figsize=figsize)
    plt.imshow(corr_matrix, cmap='coolwarm')
    plt.colorbar()
    plt.title('Correlation matrix')
    plt.show()


def plot_data_correlation_np(data, figsize=(30, 30)):
    corr_matrix = np.corrcoef(data, rowvar=False)
    plot_data_precomputed_correlation_np(corr_matrix, figsize=figsize)


def plot_data_precomputed_correlation_df(corr_matrix_df, figsize=(30, 30)):
    import seaborn as sns
    plt.figure(figsize=figsize)
    ax = sns.heatmap(corr_matrix_df, annot=True, cmap='coolwarm')
    ax.xaxis.tick_top()
    plt.title('Correlation Matrix')
    plt.show()


def plot_data_correlation_df(data_df, figsize=(30, 30)):
    plot_data_precomputed_correlation_df(data_df.corr(), figsize=figsize)


def plot_2D_points(x, y, x_name, y_name, clusters=None, figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    if clusters is not None:
        for i, cluster in enumerate(clusters):
            ax.scatter(x[i], y[i], color='C{}'.format(cluster))
    else:
        ax.scatter(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f'{x_name} vs {y_name}')
    plt.show()


def plot_2D_points_2v2_all_data_np(data, clusters=None, figsize=(10, 10)):
    for i in range(0, data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            plot_2D_points(data[:, i], data[:, j], str(i), str(j), clusters=clusters, figsize=figsize)


def plot_2D_points_2v2_all_data_df(data, clusters=None, figsize=(10, 10)):
    columns = data.columns
    for i in range(0, len(columns)):
        for j in range(i + 1, len(columns)):
            plot_2D_points(data[columns[i]], data[columns[j]], columns[i], columns[j], clusters=clusters, figsize=figsize)


def plot_hist(data, name="", bins="auto", kde=True, binrange=None, figsize=(10, 10)):
    import seaborn as sns
    plt.figure(figsize=figsize)
    sns.histplot(data, kde=kde, bins=bins, binrange=binrange)
    plt.title(f'{name}')
    plt.show()


def plot_hits_df(data, column, bins="auto", kde=True, binrange=None, figsize=(10, 10)):
    import seaborn as sns
    plt.figure(figsize=figsize)
    sns.histplot(data[column], kde=kde, bins=bins, binrange=binrange)
    plt.title(f'{column}')
    plt.show()


def plot_hist_all_data_df(data, bins="auto", kde=True, binrange=None, figsize=(10, 10)):
    for column in data.columns:
        plot_hits_df(data, column, bins=bins, kde=kde, binrange=binrange, figsize=figsize)


def visualize_clusters(data_df, clusters, cluster_by, figsize=(10, 5), hist=False, bins=0):
    if len(cluster_by) == 1:
        if hist:
            plot_hits_df(data_df, cluster_by[0], bins=bins, kde=False, binrange=None, figsize=figsize)
        else:
            plot_2D_points(data_df[cluster_by[0]], np.zeros(data_df[cluster_by[0]].shape[0]), cluster_by[0], "",
                           clusters=clusters, figsize=figsize)
    elif len(cluster_by) == 2:
        plot_2D_points(data_df[cluster_by[0]], data_df[cluster_by[1]], cluster_by[0], cluster_by[1], clusters=clusters, figsize=figsize)
