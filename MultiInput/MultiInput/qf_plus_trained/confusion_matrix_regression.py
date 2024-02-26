import matplotlib.pyplot as plt


def plot_confusion_matrix_regression(
    labels, prediction, healthy_threshold, mae, rmse, ba
):
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
    plt.rcParams["lines.linewidth"] = 3
    plt.rcParams["lines.antialiased"] = True
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["axes.grid"] = True

    plt.scatter(labels, prediction, c="green")
    plt.axhline(y=healthy_threshold, color="r", linestyle="-")
    plt.axvline(x=healthy_threshold, color="y", linestyle="-")
    plt.xlim([0, healthy_threshold * 2])
    plt.ylim([0, healthy_threshold * 2])
    plt.plot([i for i in range(0, healthy_threshold * 2)])
    plt.axis("square")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title("RMSE: " + str(rmse) + " MAE: " + str(mae) + " B.A: " + str(ba))
    return plt
