import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


def setup_style():
    sns.set(style="ticks")
    sns.set_context("paper", rc={"lines.linewidth": 2})


def postprocess_style():
    sns.despine()


def setup_latex():
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            # Adjust to your LaTex-Engine
            "pgf.texsystem": "lualatex",
            "pgf.preamble": "\n".join([
                r"\usepackage{amsmath}",
                r"\usepackage{amssymb}"
            ]),
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "axes.unicode_minus": False,
        }
    )


def get_sizes(factor=1):
    """Get textwidth and textheight in inches scaled by factor."""
    textwidth = 4.9823
    textheight = 8.2457
    return factor * textwidth, factor * textheight


def get_fig_size(ratio=8 / 6, factor=1):
    fullwidth = get_sizes(factor)[0]
    return fullwidth, fullwidth / ratio


def get_fig_ax(ratio=8 / 6, factor=1):
    setup_style()
    return plt.subplots(1, 1, figsize=get_fig_size(ratio=ratio, factor=factor))


def get_fig_axs(rows, cols, ratio=8 / 6, factor=1, **kwargs):
    setup_style()
    w, h = get_fig_size(ratio=ratio, factor=factor)
    return plt.subplots(rows, cols, figsize=(w, h * rows), **kwargs)


class Colors:
    def __init__(self):
        self.colors = sns.color_palette("Paired")

    def get(self, i: int, variation=False):
        if i > len(self.colors) - 1:
            raise RuntimeWarning(f"Index {i} too big for color palette. Reusing colors.")
        return self.colors[i * 2 + 1 - variation]

    def __getitem__(self, item):
        return self.get(item)
