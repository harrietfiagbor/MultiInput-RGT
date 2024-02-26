import logging
import os
import numpy as np
from tsai.all import *
from PIL import Image
from pytorch_hooks import *
from matplotlib.collections import LineCollection
from scipy.signal import resample
from matplotlib.colors import BoundaryNorm

logging.getLogger().setLevel(logging.DEBUG)
from neptune.new.types import File


def multi_color(
    x,
    y,
    diff,
    fig_name,
    cmap="Spectral_r",
    levels=254,
    linestyles="solid",
    linewidth=1,
    alpha=1.0,
):

    cmap = plt.get_cmap(cmap, lut=levels)
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    colors = cmap(np.linspace(diff.min(), diff.max(), levels))
    norm = BoundaryNorm(np.linspace(diff.min(), diff.max(), levels + 1), len(colors))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        linestyles=linestyles,
        alpha=alpha,
    )
    lc.set_array(diff)
    lc.set_linewidth(1)
    line = plt.gca().add_collection(lc)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())

    fig.colorbar(line, ax=axs)
    plt.savefig(fig_name)


def get_gradcam_map(config, learner, targetclass, threshold, signals):
    target_layer = config["target_layer"]
    target_layer = learner.model[int(target_layer)]
    with HookGradient(target_layer) as hookg:
        with HookActivation(target_layer) as hook:
            output = learner.model.cpu().eval()(tensor(signals))
            act = hook.stored
        output[0, targetclass].backward()
        grad = hookg.stored
        p0, p1 = output.cpu().detach()[0]
    w = grad[0].mean(dim=(1), keepdim=True)
    gradcam_map = (w * act[0]).sum(0).detach().cpu()
    if threshold:
        gradcam_map = torch.clamp(gradcam_map, min=0)
    return gradcam_map


def color_and_stack(
    config,
    learner,
    signal,
    channels,
    foldername,
    filename,
    targetclass,
    threshold,
    neptune_run,
):
    images = []
    x = TSTensor(signal).unsqueeze(0)
    gradcam_map = get_gradcam_map(config, learner, targetclass, threshold, x)
    for j in range(len(config["channels"])):
        ax = TSTensor(signal[j]).show()
        line = ax.lines[0]
        x = line.get_xdata()
        y = line.get_ydata()
        fig_name = channels[j] + str(j) + ".png"
        multi_color(x, y, gradcam_map, fig_name)
        images.append(fig_name)
    v_stack = []
    for k in images:
        image = Image.open(k)
        v_stack.append(np.array(image))
    vstack_array = np.vstack(v_stack)
    final_img = Image.fromarray(vstack_array)
    image_name = (
        foldername
        + "/"
        + filename[:-4]
        + "_class_"
        + str(targetclass)
        + "_"
        + "_".join(channels)
        + "_channel.png"
    )
    final_img.save(image_name)
    if neptune_run:
        neptune_run[foldername].log(File(image_name))


def color_signals(config, learner, loaded_data, neptune_run=None):

    plt.style.use("dark_background")
    plot_width_px = 1800
    plot_height_px = 800
    pixels_per_inch = 80
    plt.rcParams["figure.dpi"] = pixels_per_inch
    plt.rcParams["font.family"] = [
        "Liberation Mono",
        "DejaVu Sans Mono",
        "mono",
        "sans-serif",
    ]
    plt.rcParams["font.size"] = 18
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.figsize"] = [
        plot_width_px / pixels_per_inch,
        plot_height_px / pixels_per_inch,
    ]
    plt.rcParams["lines.linewidth"] = 1
    plt.rcParams["lines.antialiased"] = True
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["axes.grid"] = True

    filenames = list(loaded_data.keys())
    filenames.sort()

    validation_patients = config["validation_patients"]
    channels = config["channels"]
    resample_points = config["resample_points"]
    target_classes = config["target_classes"]
    thresholds = [True, False]
    combinations = list(itertools.product(thresholds, target_classes))

    def get_subject_id(filename):
        return filename[:-6]

    for threshold, target_class in combinations:
        if threshold:
            foldername = "threshold class " + str(target_class)
        else:
            foldername = "class " + str(target_class)
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        used_filenames = []
        for filename in filenames:
            if get_subject_id(filename) in validation_patients:
                logging.debug(f"filename = {filename}")
                try:
                    orig_df_columns = channels
                    orig_df = loaded_data[filename]
                    data = orig_df[orig_df_columns].to_numpy()
                    logging.debug(f"data.shape = {data.shape}")
                    whistle_binary = (orig_df["Category"] == "Whistle").to_numpy()
                    if not whistle_binary.size:
                        raise IndexError(
                            f"orig_df.Category is empty for filename={filename}."
                        )
                    entire_signal = resample(data, num=resample_points)
                    logging.debug(f"entire_signal.shape = {entire_signal.shape}")
                    logging.debug(f"entire_signal.shape = {entire_signal.shape}")
                    X = np.swapaxes(np.array(entire_signal), 0, 1)
                    color_and_stack(
                        config,
                        learner,
                        X,
                        channels,
                        foldername,
                        filename,
                        target_class,
                        threshold,
                        neptune_run,
                    )
                    used_filenames.append(filename)
                    subject_id = get_subject_id(filename)
                    logging.debug(f"subject_id = {subject_id}")
                except (IndexError, KeyError) as error:
                    logging.warning(f"error = {error}. Skipping")

        logging.info(f"used filenames = {used_filenames}")
