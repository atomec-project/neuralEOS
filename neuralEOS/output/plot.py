import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.font_manager
import matplotlib as mpl
import scipy.stats as stats
import pandas as pd

# from models import nn_pipeline


def fig_initialize(
    latex=False,
    setsize=False,
    size="preprint",
    fraction=1,
    subplots=(1, 1),
    plotstyle=1,
):

    mpl.rcParams["lines.marker"] = ""
    mpl.rcParams["lines.markersize"] = 1.2

    if latex == True:
        # Set up tex rendering
        plt.rc("text", usetex=True)
        plt.rc(
            "text.latex", preamble=r"\usepackage{amsmath, amsthm, amssymb, amsfonts}"
        )
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = "STIX"
        mpl.rcParams["mathtext.fontset"] = "stix"
    if setsize == True:
        if size == "reprint":
            mpl.rcParams["font.size"] = 10
            mpl.rcParams["axes.linewidth"] = 0.5
            mpl.rcParams["xtick.major.width"] = 0.5
            mpl.rcParams["ytick.major.width"] = 0.5
            mpl.rcParams["lines.linewidth"] = 1
            mpl.rcParams["axes.labelsize"] = 10
            mpl.rcParams["xtick.labelsize"] = 8
            mpl.rcParams["ytick.labelsize"] = 8
            plt.rc("legend", **{"fontsize": 8})
            plt.rc("legend", **{"frameon": False})
            mpl.rcParams["legend.labelspacing"] = 0.25
        elif size == "preprint":
            mpl.rcParams["font.size"] = 12
            mpl.rcParams["axes.linewidth"] = 0.75
            mpl.rcParams["xtick.major.width"] = 0.75
            mpl.rcParams["ytick.major.width"] = 0.75
            mpl.rcParams["lines.linewidth"] = 1
            mpl.rcParams["axes.labelsize"] = 12
            mpl.rcParams["xtick.labelsize"] = 10
            mpl.rcParams["ytick.labelsize"] = 10
            plt.rc("legend", **{"fontsize": 10})
            plt.rc("legend", **{"frameon": False})
            mpl.rcParams["legend.labelspacing"] = 0.35
        elif size == "beamer":
            mpl.rcParams["font.size"] = 10
            mpl.rcParams["lines.linewidth"] = 0.75
            mpl.rcParams["axes.labelsize"] = 10
            mpl.rcParams["xtick.labelsize"] = 8
            mpl.rcParams["ytick.labelsize"] = 8
            mpl.rcParams["legend.labelspacing"] = 0.25
            plt.rc("legend", **{"fontsize": 8})
            plt.rc("legend", **{"frameon": False})

    # Define a custom cycler
    if plotstyle == 1:
        custom_cycler = (
            cycler(color=["orange", "steelblue", "violet", "midnightblue", "maroon"])
            + cycler(linestyle=["-", "--", ":", "-.", "-"])
            + cycler(lw=[1.0, 1.1, 1.5, 1.2, 1.0])
            + cycler(alpha=[1.0, 1.0, 1.0, 1.0, 1.0])  # cycler(lw=[1,0.8,1.33,1,1]) + \
            + cycler(
                markerfacecolor=[
                    "orange",
                    "steelblue",
                    "violet",
                    "midnightblue",
                    "maroon",
                ]
            )
        )

    elif plotstyle == 2:
        custom_cycler = (
            cycler(color=["red", "steelblue", "orange", "midnightblue", "maroon"])
            + cycler(linestyle=["-", "-.", ":", "--", "-"])
            + cycler(lw=[1.0, 1.2, 1.3, 1.1, 1.0])
            + cycler(alpha=[1.0, 1.0, 1.0, 1.0, 1.0])  # cycler(lw=[1,0.8,1.33,1,1]) + \
            + cycler(
                markerfacecolor=[
                    "orange",
                    "steelblue",
                    "violet",
                    "midnightblue",
                    "maroon",
                ]
            )
        )

    elif plotstyle == 3:
        custom_cycler = (
            cycler(
                color=[
                    "orange",
                    "steelblue",
                    "steelblue",
                    "violet",
                    "violet",
                    "midnightblue",
                    "maroon",
                ]
            )
            + cycler(linestyle=["--", "-", ":", "-", ":", "-.", "--"])
            + cycler(lw=[1.1, 1.0, 1.5, 1.0, 1.5, 1.2, 1.1])
            + cycler(
                alpha=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )  # cycler(lw=[1,0.8,1.33,1,1]) + \
            + cycler(
                markerfacecolor=[
                    "orange",
                    "steelblue",
                    "steelblue",
                    "violet",
                    "violet",
                    "midnightblue",
                    "maroon",
                ]
            )
        )

    plt.rc("axes", prop_cycle=custom_cycler)

    # Define the x-axis label
    xlab = r"$\tau\ (\textrm{eV}),\ X_{nl}(R_\textrm{VS})=0$"

    # determine fig height and width
    if size == "reprint":
        width_pt = 243
    elif size == "preprint":
        width_pt = 468.0
    elif size == "Hirschegg":
        width_pt = 650
    elif size == "beamer":
        width_pt = 307.0

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    figdims = (fig_width_in, fig_height_in)

    return xlab, figdims


def add_subplot_axes(ax, rect, axisbg="w"):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height], facecolor=axisbg)
    # x_labelsize = subax.get_xticklabels()[0].get_size()
    # y_labelsize = subax.get_yticklabels()[0].get_size()
    # x_labelsize *= rect[2]**0.5
    # y_labelsize *= rect[3]**0.5
    # subax.xaxis.set_tick_params(labelsize=x_labelsize)
    # subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def plot_error_x_temp(
    Y_val,
    Y_pred,
    temp,
    rho,
    pretty=False,
    size="preprint",
    save=False,
    filename="Error_x_temp",
):

    if pretty:
        xlab, figdims = fig_initialize(latex=True, setsize=True, size=size)
        fig, ax = plt.subplots(figsize=figdims)
    else:
        xlab, figdims = fig_initialize(latex=False, setsize=False)
        fig, ax = plt.subplots()

    error = calc_MAPE(Y_val, Y_pred)[1]

    N = 21
    cmap = plt.get_cmap("seismic", N)

    vmin = min(np.log10(rho))
    vmax = max(np.log10(rho))

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(
        sm, label=r"$\log_{10}(\rho_\textrm{m})\ (\textrm{g cm}^{-3})$", ax=ax
    )  # , ticks=np.linspace(0, 2, N), boundaries=np.arange(-0.05, 2.1, 0.1))

    ax.scatter(temp, error, s=2, color=cmap(np.log10(rho)))

    ax.set_yscale("symlog", linthresh=20)

    ax.set_xscale("log")

    ax.fill_between([0, 1e6], [5, 5], [0, 0], color="k", alpha=0.3)
    ax.fill_between([0, 1e6], [20, 20], [0, 0], color="k", alpha=0.3)
    ax.plot([0, 1e6], [5, 5], ls="--", color="k", lw=1)
    ax.plot([0, 1e6], [20, 20], ls="--", color="k", lw=1)
    ax.set_xlim(0.05, 1e4)
    ax.set_ylim(0, 2e2)

    ax.set_xlabel(r"Temperature (eV)")
    # ax.set_ylabel(r"MAPE (\%)")
    ax.set_ylabel(
        r"$|P_\textrm{ref} - P_\textrm{pred}|/P_\textrm{ref}\ (\%)$", labelpad=30
    )

    ax.text(
        6e-3,
        12,
        "Linear",
        ha="center",
        va="center",
        rotation=90,
        size=10,
        bbox=dict(boxstyle="larrow,pad=0.3", fc="lightblue", ec="steelblue", lw=2),
    )

    ax.text(
        6e-3,
        100,
        "Logarithmic",
        ha="center",
        va="center",
        rotation=90,
        size=10,
        bbox=dict(boxstyle="rarrow,pad=0.3", fc="lightblue", ec="steelblue", lw=2),
    )

    ax.set_yticks([5, 20, 100, 500])
    ax.set_yticklabels([5, 20, 100, 500])

    ax.set_ylim(0, 1e3)

    plt.minorticks_off()

    if save:
        plt.savefig(filename + ".pdf", bbox_inches="tight")

    plt.show()


def plot_error_x_rho(
    Y_val,
    Y_pred,
    temp,
    rho,
    pretty=False,
    size="preprint",
    save=False,
    filename="Error_x_temp",
):

    if pretty:
        xlab, figdims = fig_initialize(latex=True, setsize=True, size=size)
        fig, ax = plt.subplots(figsize=figdims)
    else:
        xlab, figdims = fig_initialize(latex=False, setsize=False)
        fig, ax = plt.subplots()

    error = calc_SMAPE(Y_val, Y_pred)[1]

    N = 21
    cmap = plt.get_cmap("seismic", N)

    vmin = min(np.log10(temp))
    vmax = max(np.log10(temp))

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(
        sm, label=r"$\log_{10}(T)\ (\textrm{eV})$", ax=ax
    )  # , ticks=np.linspace(0, 2, N), boundaries=np.arange(-0.05, 2.1, 0.1))

    ax.scatter(rho, error, s=2, color=cmap(np.log10(temp)))

    ax.set_yscale("symlog", linthresh=20)

    ax.set_xscale("log")

    ax.fill_between([0, 1e6], [5, 5], [0, 0], color="k", alpha=0.3)
    ax.fill_between([0, 1e6], [20, 20], [0, 0], color="k", alpha=0.3)
    ax.plot([0, 1e6], [5, 5], ls="--", color="k", lw=1)
    ax.plot([0, 1e6], [20, 20], ls="--", color="k", lw=1)
    ax.set_xlim(0.05, 1e3)

    ax.set_xlabel(r"Density (g cm 3)")
    # ax.set_ylabel(r"MAPE (\%)")
    ax.set_ylabel(
        r"$|P_\textrm{ref} - P_\textrm{pred}|/P_\textrm{ref}\ (\%)$", labelpad=30
    )

    ax.text(
        6e-3,
        12,
        "Linear",
        ha="center",
        va="center",
        rotation=90,
        size=10,
        bbox=dict(boxstyle="larrow,pad=0.3", fc="lightblue", ec="steelblue", lw=2),
    )

    ax.text(
        6e-3,
        100,
        "Logarithmic",
        ha="center",
        va="center",
        rotation=90,
        size=10,
        bbox=dict(boxstyle="rarrow,pad=0.3", fc="lightblue", ec="steelblue", lw=2),
    )

    ax.set_yticks([5, 20, 100, 500])
    ax.set_yticklabels([5, 20, 100, 500])

    ax.set_ylim(0, 1e3)

    plt.minorticks_off()
    if save:
        plt.savefig(filename + ".pdf", bbox_inches="tight")

    plt.show()


def plot_error_x_temp_aa(
    df, pretty=False, size="preprint", save=False, filename="Error_x_temp_aa.pdf"
):

    if pretty:
        xlab, figdims = fig_initialize(
            latex=True, setsize=True, size=size, subplots=(3, 2)
        )
        fig, axes = plt.subplots(
            3, 2, figsize=figdims, gridspec_kw={"width_ratios": [1, 1.15]}
        )
    else:
        xlab, figdims = fig_initialize(latex=False, setsize=False)
        fig, axes = plt.subplots(2, 2)

    error_fd = calc_MAPE(df.P_ref, df.P_aa_fd + df.P_ion)[1]
    error_st_tr = calc_MAPE(df.P_ref, df.P_aa_st_tr + df.P_ion)[1]
    error_st_rr = calc_MAPE(df.P_ref, df.P_aa_st_rr + df.P_ion)[1]
    error_vir_corr = calc_MAPE(df.P_ref, df.P_aa_vir_corr + df.P_ion)[1]
    error_vir_nocorr = calc_MAPE(df.P_ref, df.P_aa_vir_nocorr + df.P_ion)[1]
    error_ideal = calc_MAPE(df.P_ref, df.P_aa_ideal + df.P_ion)[1]

    N = 21
    cmap = plt.get_cmap("seismic", N)

    vmin = min(np.log10(df.rho))
    vmax = max(np.log10(df.rho))

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    for j in range(3):
        plt.colorbar(
            sm,
            label=r"$\log_{10}(\rho_\textrm{m})\ (\textrm{g cm}^{-3})$",
            ax=axes[j, 1],
        )  # , ticks=np.linspace(0, 2, N), boundaries=np.arange(-0.05, 2.1, 0.1))
        # axes[1, j].set_xlabel(r"Temperature (eV)")
        # axes[j, 0].set_ylabel(
        #     r"$|P_\textrm{ref} - P_\textrm{AA}|/P_\textrm{ref}\ (\%)$"
        # )

    axes[0, 0].scatter(df.temp, error_fd, s=2, color=cmap(np.log10(df.rho)))
    axes[0, 1].scatter(df.temp, error_ideal, s=2, color=cmap(np.log10(df.rho)))
    axes[1, 0].scatter(df.temp, error_st_rr, s=2, color=cmap(np.log10(df.rho)))
    axes[1, 1].scatter(df.temp, error_st_tr, s=2, color=cmap(np.log10(df.rho)))
    axes[2, 0].scatter(df.temp, error_vir_nocorr, s=2, color=cmap(np.log10(df.rho)))
    axes[2, 1].scatter(df.temp, error_vir_corr, s=2, color=cmap(np.log10(df.rho)))

    axes[0, 0].text(500, 300, r"$P^\textrm{fd}$")
    axes[0, 1].text(500, 300, r"$P_\textrm{e}^\textrm{id}$")
    axes[1, 0].text(500, 300, r"$P_{rr}^\textrm{st}$")
    axes[1, 1].text(500, 300, r"$P_\textrm{tr}^\textrm{st}$")
    axes[2, 0].text(500, 300, r"$P_{T}^\textrm{vir}$")
    axes[2, 1].text(500, 300, r"$P_{K_{12}}^\textrm{vir}$")

    for i in range(3):
        for j in range(2):
            axes[i, j].fill_between([0, 1e6], [5, 5], [0, 0], color="k", alpha=0.3)
            axes[i, j].fill_between([0, 1e6], [20, 20], [0, 0], color="k", alpha=0.3)
            axes[i, j].plot([0, 1e6], [5, 5], ls="--", color="k", lw=1)
            axes[i, j].plot([0, 1e6], [20, 20], ls="--", color="k", lw=1)
            axes[i, j].set_xlim(0.05, 1e4)
            axes[i, j].set_ylim(0, 1.2e3)
            axes[i, j].set_yscale("symlog", linthresh=20)
            axes[i, j].set_xscale("log")

        axes[0, j].set_xticklabels([])
        axes[i, 1].set_yticklabels([])

    fig.text(
        0.01,
        0.5,
        r"$|P_\textrm{ref} - P_\textrm{AA}|/P_\textrm{ref}\ (\%)$",
        va="center",
        rotation="vertical",
        fontsize=14,
    )

    fig.text(0.5, 0.01, r"Temperature (eV)", ha="center", fontsize=14)

    if save:
        plt.savefig(filename, bbox_inches="tight")

    plt.show()


def plot_error_x_temp_nn(
    df_aa,
    df_no_aa,
    pretty=False,
    size="preprint",
    save=False,
    filename="Error_x_temp_nn.pdf",
):

    if pretty:
        xlab, figdims = fig_initialize(
            latex=True, setsize=True, size=size, subplots=(1, 2)
        )
        fig, axes = plt.subplots(
            1, 2, figsize=figdims, gridspec_kw={"width_ratios": [1, 1.15]}
        )
    else:
        xlab, figdims = fig_initialize(latex=False, setsize=False)
        fig, axes = plt.subplots(1, 2)

    error_aa = calc_MAPE(df_aa.P_ref, df_aa.P_pred)[1]
    error_no_aa = calc_MAPE(df_no_aa.P_ref, df_no_aa.P_pred)[1]

    N = 21
    cmap = plt.get_cmap("seismic", N)

    vmin = min(np.log10(df_aa.rho))
    vmax = max(np.log10(df_aa.rho))

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(
        sm,
        label=r"$\log_{10}(\rho_\textrm{m})\ (\textrm{g cm}^{-3})$",
        ax=axes[1],
    )  # , ticks=np.linspace(0, 2, N), boundaries=np.arange(-0.05, 2.1, 0.1))
    axes[0].set_xlabel(r"Temperature (eV)")
    axes[1].set_xlabel(r"Temperature (eV)")
    axes[0].set_ylabel(r"$|P_\textrm{ref} - P_\textrm{pred}|/P_\textrm{ref}\ (\%)$")

    axes[0].scatter(df_aa.temp, error_aa, s=2, color=cmap(np.log10(df_aa.rho)))
    axes[1].scatter(df_no_aa.temp, error_no_aa, s=2, color=cmap(np.log10(df_no_aa.rho)))

    axes[0].text(500, 300, r"$P_\textrm{nn}^\textrm{AA}$")
    axes[1].text(400, 300, r"$P_\textrm{nn}^\textrm{no-AA}$")

    for i in range(2):
        axes[i].fill_between([0, 1e6], [5, 5], [0, 0], color="k", alpha=0.3)
        axes[i].fill_between([0, 1e6], [20, 20], [0, 0], color="k", alpha=0.3)
        axes[i].plot([0, 1e6], [5, 5], ls="--", color="k", lw=1)
        axes[i].plot([0, 1e6], [20, 20], ls="--", color="k", lw=1)
        axes[i].set_xlim(0.05, 1e4)
        axes[i].set_ylim(0, 1.2e3)
        axes[i].set_yscale("symlog", linthresh=20)
        axes[i].set_xscale("log")

    axes[1].set_yticklabels([])

    if save:
        plt.savefig(filename, bbox_inches="tight")

    plt.show()


def plot_log_error_aa(
    df, pretty=False, size="preprint", save=False, filename="log_aa.pdf"
):

    if pretty:
        xlab, figdims = fig_initialize(
            latex=True, setsize=True, size=size, subplots=(4, 2)
        )
        fig, axes = plt.subplots(
            6, 2, figsize=figdims, gridspec_kw={"height_ratios": [2, 1, 2, 1, 2, 1]}
        )
    else:
        xlab, figdims = fig_initialize(latex=False, setsize=False)
        fig, axes = plt.subplots(
            6, 2, figsize=figdims, gridspec_kw={"height_ratios": [2, 1, 2, 1, 2, 1]}
        )

    alpha = 0.7
    s1 = 5

    df = df[
        (df.P_aa_fd_i > 0)
        & (df.P_aa_st_tr_i > 0)
        & (df.P_aa_st_rr_i > 0)
        & (df.P_aa_vir_corr_i > 0)
        & (df.P_aa_vir_nocorr_i > 0)
        & (df.P_aa_ideal_i > 0)
    ]

    P_ref = df.P_ref
    P_aa_fd = df.P_aa_fd_i
    P_aa_st_tr = df.P_aa_st_tr_i
    P_aa_st_rr = df.P_aa_st_rr_i
    P_aa_vir_corr = df.P_aa_vir_corr_i
    P_aa_vir_nocorr = df.P_aa_vir_nocorr_i
    P_aa_ideal = df.P_aa_ideal_i

    for i in range(6):
        for j in range(2):
            axes[i, j].set_xscale("log")
            axes[i, j].set_ylim(1e-1, 1e9)
            axes[i, j].set_xlim(1e-1, 1e9)

    for i in range(0, 6, 2):
        for j in range(2):
            axes[i, j].plot(P_ref, P_ref, color="darkorange")
            axes[i, j].set_yscale("log")
            axes[i + 1, j].set_ylim(0, 2)

    axes[0, 0].scatter(
        P_ref, P_aa_fd, color="navy", alpha=alpha, edgecolors="none", s=s1
    )
    axes[1, 0].scatter(
        P_ref,
        np.abs(np.log10(P_ref) - np.log10(P_aa_fd)),
        color="darkolivegreen",
        alpha=alpha,
        edgecolors="none",
        s=s1,
    )

    axes[0, 1].scatter(
        P_ref, P_aa_ideal, color="navy", alpha=alpha, edgecolors="none", s=s1
    )

    axes[1, 1].scatter(
        P_ref,
        np.abs(np.log10(P_ref) - np.log10(np.abs(P_aa_ideal))),
        color="darkolivegreen",
        alpha=alpha,
        edgecolors="none",
        s=s1,
    )

    axes[2, 0].scatter(
        P_ref, P_aa_st_tr, color="navy", alpha=alpha, edgecolors="none", s=s1
    )
    axes[2, 1].scatter(
        P_ref, P_aa_st_rr, color="navy", alpha=alpha, edgecolors="none", s=s1
    )
    axes[3, 0].scatter(
        P_ref,
        np.abs(np.log10(P_ref) - np.log10(np.abs(P_aa_st_tr))),
        color="darkolivegreen",
        alpha=alpha,
        edgecolors="none",
        s=s1,
    )
    axes[3, 1].scatter(
        P_ref,
        np.abs(np.log10(P_ref) - np.log10(np.abs(P_aa_st_rr))),
        color="darkolivegreen",
        alpha=alpha,
        edgecolors="none",
        s=s1,
    )

    axes[4, 0].scatter(
        P_ref, P_aa_vir_nocorr, color="navy", alpha=alpha, edgecolors="none", s=s1
    )
    axes[4, 1].scatter(
        P_ref, P_aa_vir_corr, color="navy", alpha=alpha, edgecolors="none", s=s1
    )

    axes[5, 0].scatter(
        P_ref,
        np.abs(np.log10(P_ref) - np.log10(np.abs(P_aa_vir_nocorr))),
        color="darkolivegreen",
        alpha=alpha,
        edgecolors="none",
        s=s1,
    )

    axes[5, 1].scatter(
        P_ref,
        np.abs(np.log10(P_ref) - np.log10(np.abs(P_aa_vir_corr))),
        color="darkolivegreen",
        alpha=alpha,
        edgecolors="none",
        s=s1,
    )

    for i in range(5):
        axes[i, 0].set_xticklabels([])
        axes[i, 1].set_xticklabels([])
    for j in range(6):
        axes[j, 1].set_yticklabels([])

    for i in range(0, 6, 2):
        axes[i, 0].set_ylabel(r"$P_\textrm{AA}$ (GPa)")
        axes[i + 1, 0].set_ylabel(r"$|\Delta \log P|$", labelpad=18)

    axes[5, 0].set_xlabel(r"$P_\textrm{ref}$ (GPa)")
    axes[5, 1].set_xlabel(r"$P_\textrm{ref}$ (GPa)")

    axes[0, 0].text(1, 1e7, r"$P^\textrm{fd}$")
    axes[0, 1].text(1, 1e7, r"$P^\textrm{id}$")
    axes[2, 0].text(1, 1e7, r"$P_{tr}^\textrm{st}$")
    axes[2, 1].text(1, 1e7, r"$P_\textrm{rr}^\textrm{st}$")
    axes[4, 0].text(1, 1e7, r"$P_T^\textrm{vir}$")
    axes[4, 1].text(1, 1e7, r"$P_{K_{12}}^\textrm{vir}$")

    if save:
        plt.savefig(filename, bbox_inches="tight")


def plot_log_error_nn(
    df_aa, df_no_aa, pretty=False, size="preprint", save=False, filename="log_aa.pdf"
):

    if pretty:
        xlab, figdims = fig_initialize(
            latex=True, setsize=True, size=size, subplots=(1, 2)
        )
        fig, axes = plt.subplots(
            2, 2, figsize=figdims, gridspec_kw={"height_ratios": [2, 1]}
        )
    else:
        xlab, figdims = fig_initialize(latex=False, setsize=False)
        fig, axes = plt.subplots(
            2, 2, figsize=figdims, gridspec_kw={"height_ratios": [2, 1]}
        )

    alpha = 0.7
    s1 = 5

    P_ref = df_aa.P_ref
    P_aa = df_aa.P_pred
    P_no_aa = df_no_aa.P_pred

    for i in range(2):
        for j in range(2):
            axes[i, j].set_xscale("log")
            axes[i, j].set_ylim(1e-1, 1e9)
            axes[i, j].set_xlim(1e-1, 1e9)

    for i in range(0, 2, 2):
        for j in range(2):
            axes[i, j].plot(P_ref, P_ref, color="darkorange")
            axes[i, j].set_yscale("log")
            axes[i + 1, j].set_ylim(0, 0.6)

    axes[0, 0].scatter(P_ref, P_aa, color="navy", alpha=alpha, edgecolors="none", s=s1)
    axes[1, 0].scatter(
        P_ref,
        np.abs(np.log10(P_ref) - np.log10(np.abs(P_aa))),
        color="darkolivegreen",
        alpha=alpha,
        edgecolors="none",
        s=s1,
    )

    axes[0, 1].scatter(
        df_no_aa.P_ref, P_no_aa, color="navy", alpha=alpha, edgecolors="none", s=s1
    )
    axes[1, 1].scatter(
        df_no_aa.P_ref,
        np.abs(np.log10(df_no_aa.P_ref) - np.log10(np.abs(P_no_aa))),
        color="darkolivegreen",
        alpha=alpha,
        edgecolors="none",
        s=s1,
    )

    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])
    axes[0, 1].set_yticklabels([])
    axes[1, 1].set_yticklabels([])

    axes[0, 0].set_ylabel(r"$P_\textrm{pred}$ (GPa)")
    axes[1, 0].set_ylabel(r"$|\Delta \log P|$", labelpad=18)

    axes[1, 0].set_xlabel(r"$P_\textrm{ref}$ (GPa)")
    axes[1, 1].set_xlabel(r"$P_\textrm{ref}$ (GPa)")

    axes[0, 0].text(1, 1e7, r"$P_\textrm{nn}^\textrm{AA}$")
    axes[0, 1].text(1, 1e7, r"$P_\textrm{nn}^\textrm{no-AA}$")

    if save:
        plt.savefig(filename, bbox_inches="tight")


def plot_nn_aa_errs(
    df_aa, df_no_aa, pretty=False, size="preprint", save=False, filename="log_aa.pdf"
):

    if pretty:
        xlab, figdims = fig_initialize(
            latex=True, setsize=True, size=size, subplots=(1, 2)
        )
        fig, axes = plt.subplots(1, 2, figsize=figdims)
    else:
        xlab, figdims = fig_initialize(latex=False, setsize=False)
        fig, axes = plt.subplots(1, 2, figsize=figdims)

    P_aa_fd = df_aa.P_aa_fd + df_aa.P_ion
    P_nn_aa = df_aa.P_pred
    P_nn_no_aa = df_no_aa.P_pred

    MAPE_aa = calc_MAPE(df_aa.P_ref, P_aa_fd)[1]
    MAPE_nn_aa = calc_MAPE(df_aa.P_ref, P_nn_aa)[1]
    MAPE_nn_no_aa = calc_MAPE(df_no_aa.P_ref, df_no_aa.P_pred)[1]

    alpha = 0.5

    axes[0].scatter(
        df_aa.temp,
        MAPE_aa,
        color="navy",
        alpha=alpha,
        label=r"$P^\textrm{fd}$",
    )
    axes[0].scatter(
        df_no_aa.temp,
        MAPE_nn_no_aa,
        color="darkorange",
        alpha=alpha,
        label=r"$P_\textrm{nn}^\textrm{no-AA}$",
    )
    axes[0].scatter(
        df_aa.temp,
        MAPE_nn_aa,
        color="darkolivegreen",
        alpha=alpha,
        label=r"$P_\textrm{nn}^\textrm{AA}$",
    )
    axes[0].set_yscale("symlog", linthresh=20)
    axes[0].set_xscale("log")

    alpha = 0.6
    s1 = 5
    axes[1].scatter(
        df_aa.P_ref,
        np.abs(np.log10(df_aa.P_ref) - np.log10(np.abs(P_aa_fd))),
        color="navy",
        alpha=alpha,
        edgecolors="none",
        s=s1,
    )

    axes[1].scatter(
        df_no_aa.P_ref,
        np.abs(np.log10(df_no_aa.P_ref) - np.log10(np.abs(P_nn_no_aa))),
        color="darkorange",
        alpha=alpha,
        edgecolors="none",
        s=s1,
    )

    axes[1].scatter(
        df_aa.P_ref,
        np.abs(np.log10(df_aa.P_ref) - np.log10(np.abs(P_nn_aa))),
        color="darkolivegreen",
        alpha=alpha,
        edgecolors="none",
        s=s1,
    )

    axes[1].set_xscale("log")

    axes[0].set_xlim(0.05, 1e4)
    axes[0].set_ylim(0, 1.2e3)
    axes[1].set_xlim(1e-1, 1e9)
    axes[1].set_ylim(0, 1)

    axes[1].set_xlabel(r"$P_\textrm{pred}$ (GPa)")
    axes[1].set_ylabel(r"$|\Delta \log P|$")

    axes[0].set_xlabel(r"Temperature (eV)")
    axes[0].set_ylabel(r"$|P_\textrm{ref} - P_\textrm{pred}|/P_\textrm{ref}\ (\%)$")

    axes[0].legend()

    axes[0].fill_between([0, 1e6], [5, 5], [0, 0], color="k", alpha=0.2)
    axes[0].fill_between([0, 1e6], [20, 20], [0, 0], color="k", alpha=0.2)
    axes[0].plot([0, 1e6], [5, 5], ls="--", color="k", lw=1)
    axes[0].plot([0, 1e6], [20, 20], ls="--", color="k", lw=1)

    if save:
        plt.subplots_adjust(wspace=0.35)
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.tight_layout()


def plot_correlations(
    Y_val,
    Y_pred_1,
    Y_pred_2,
    setsize=False,
    size="preprint",
    latex=False,
    savefig=False,
    figname="corr_plot.pdf",
):

    xlab, figdims = fig_initialize(
        latex=latex, setsize=setsize, size=size, subplots=(2, 2)
    )
    fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, figsize=figdims)

    axes[0, 0].scatter(Y_pred_1, Y_val, color="navy")
    axes[1, 0].scatter(Y_pred_1, Y_val, color="navy")
    axes[0, 1].scatter(Y_pred_2, Y_val, color="navy")
    axes[1, 1].scatter(Y_pred_2, Y_val, color="navy")

    axes[1, 0].set_xscale("log")
    axes[1, 1].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")

    # axes[0, 0].set_xlim(1e-1, 1e8)
    # axes[0, 0].set_ylim(1e-1, 1e8)
    axes[0, 1].set_yticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xlim(0.5, 14)
    axes[1, 1].set_ylim(1, 1e9)
    axes[1, 0].set_ylim(1, 1e9)
    axes[1, 0].set_xlim(1e-2, 70)
    axes[0, 1].set_xlim(0.5, 14)
    axes[1, 0].set_xlim(1e-2, 70)

    axes[0, 0].set_ylabel(r"$P_\textrm{ref}$ (GPa)")
    axes[1, 0].set_ylabel(r"$P_\textrm{ref}$ (GPa)")
    axes[1, 0].set_xlabel(
        r"$\textrm{d}v_\textrm{s}(r)/\textrm{d}r\ |_{R_\textrm{VS}}\ (\textrm{Ha})$"
    )
    axes[1, 1].set_xlabel(r"$Z^*$")

    if savefig:
        plt.savefig(figname, bbox_inches="tight")


def plot_feature_errs(
    df,
    latex=False,
    setsize=False,
    size="preprint",
    savefig=False,
    figname="features_err.pdf",
):

    xlab, figdims = fig_initialize(
        latex=latex, setsize=setsize, size=size, subplots=(5, 3), fraction=0.75
    )

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figdims)

    axes[0].scatter(df.n_features, df.MAPE, color="navy", s=5)
    axes[1].scatter(df.n_features, 100 * df.MAE, color="navy", s=5)

    min_features = int(df.n_features.min())
    max_features = int(df.n_features.max() + 1)
    feature_range = max_features - min_features
    x_features = [i for i in range(min_features, max_features)]
    avg_scores = np.zeros((2, feature_range))

    for i in range(min_features, max_features):
        df_tmp = df[df.n_features == i]
        avg_scores[0, i - min_features] = df_tmp.MAPE.mean()
        avg_scores[1, i - min_features] = df_tmp.MAE.mean()

    axes[0].plot(
        x_features, avg_scores[0], marker="x", markersize=10, color="darkorange"
    )
    axes[1].plot(
        x_features, 100 * avg_scores[1], marker="x", markersize=10, color="darkorange"
    )

    axes[0].set_ylabel(r"$|P_\textrm{ref} - P_\textrm{pred}|/P_\textrm{ref}\ (\%)$")
    axes[1].set_ylabel(
        r"$100\times|\log_{10}(P_\textrm{ref} - \log_{10}(P_\textrm{pred})|$"
    )

    axes[1].set_xlabel("Number of features")

    if savefig:
        plt.savefig(figname, bbox_inches="tight")


def plot_feature_errs_comp(
    df_list,
    latex=False,
    setsize=False,
    size="preprint",
    savefig=False,
    figname="features_err_comp.pdf",
):

    xlab, figdims = fig_initialize(
        latex=latex, setsize=setsize, size=size, subplots=(5, 3), fraction=0.75
    )

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figdims)

    # axes[0].scatter(df.n_features, df.MAPE, color="navy", s=5)
    # axes[1].scatter(df.n_features, 100 * df.MAE, color="navy", s=5)

    cols = ["darkorange", "navy", "darkolivegreen"]
    labels = ["With ion", "No ion", "Ion separate"]

    for j, df in enumerate(df_list):
        min_features = int(df.n_features.min())
        max_features = int(df.n_features.max() + 1)
        feature_range = max_features - min_features
        x_features = [i for i in range(min_features, max_features)]
        avg_scores = np.zeros((2, feature_range))

        for i in range(min_features, max_features):
            df_tmp = df[df.n_features == i]
            avg_scores[0, i - min_features] = df_tmp.MAPE.mean()
            avg_scores[1, i - min_features] = df_tmp.MAE.mean()

        axes[0].plot(
            x_features,
            avg_scores[0],
            marker="x",
            markersize=10,
            color=cols[j],
            label=labels[j],
        )
        axes[1].plot(
            x_features, 100 * avg_scores[1], marker="x", markersize=10, color=cols[j]
        )

    axes[0].set_ylabel(r"$|P_\textrm{ref} - P_\textrm{pred}|/P_\textrm{ref}\ (\%)$")
    axes[1].set_ylabel(
        r"$100\times|\log_{10}(P_\textrm{ref} - \log_{10}(P_\textrm{pred})|$"
    )

    axes[1].set_xlabel("Number of features")

    axes[0].legend()

    if savefig:
        plt.savefig(figname, bbox_inches="tight")


def plot_log_pressure(
    Y_val,
    Y_pred,
    Y_AA=None,
    flatten=False,
    pretty=True,
    size="preprint",
    save=False,
    filename="Pressure_log",
    xlabel="(DFT-MD / PIMC)",
    ylabel="(AA / pred)",
):

    s1 = 5
    s2 = 5
    alpha = 0.6

    if pretty:
        xlab, figdims = fig_initialize(
            latex=True, setsize=True, size=size, subplots=(4, 3), fraction=0.85
        )
        fig, [ax, ax2] = plt.subplots(
            figsize=figdims,
            nrows=2,
            ncols=1,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
    else:
        xlab, figdims = fig_initialize(latex=False, setsize=False)
        fig, ax = plt.subplots()

    if flatten:
        ax.plot(Y_val.flatten(), Y_val.flatten(), color="r", lw=1.5, zorder=0)
    else:
        ax.plot(Y_val, Y_val, color="r", lw=1.5)
    ax.scatter(
        Y_val,
        Y_pred,
        s=s1,
        color="b",
        zorder=10,
        alpha=alpha,
        edgecolors="none",
    )
    if Y_AA is not None:
        ax.scatter(
            Y_val,
            Y_AA,
            s=s1,
            color="green",
            zorder=5,
            alpha=alpha,
            label="Raw AA",
            edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax2.set_xscale("log")
    # ax2.set_yscale("log")

    ax2.scatter(
        np.abs(Y_val),
        np.abs(np.log10(np.abs(Y_val)) - np.log10(np.abs(Y_pred))),
        s=s2,
        color="blue",
        alpha=alpha,
        zorder=5,
        edgecolors="none",
    )

    if Y_AA is not None:
        ax2.scatter(
            np.abs(Y_val),
            np.abs(np.log10(np.abs(Y_val)) - np.log10(np.abs(Y_AA))),
            s=s2,
            color="green",
            alpha=alpha,
            zorder=0,
            edgecolors="none",
        )

    ax.set_ylabel(r"$P$ " + ylabel + " (GPa)")
    ax2.set_ylabel(r"$|\Delta \log(P)|$")
    ax2.set_xlabel(r"$P$ " + xlabel + " (GPa)")

    ax2.set_xlim(min(np.abs(Y_val)), max(Y_val))
    ax2.set_ylim(0, 1.0)
    ax.set_ylim(min(np.abs(Y_val)), max(Y_val))

    plt.subplots_adjust(hspace=0.1)

    if save:
        plt.savefig(filename + ".pdf", bbox_inches="tight")
        # plt.show()
    else:
        plt.show()


def plot_log_pressure_nn_aa(
    Y_val,
    Y_pred_AA,
    Y_pred_no_AA,
    Y_AA,
    flatten=False,
    pretty=True,
    size="preprint",
    save=False,
    filename="Pressure_log",
    xlabel=r"$P_\textrm{ref}\ (\textrm{GPa})$",
    ylabel=r"$P_\textrm{nn/AA}\ (\textrm{GPa})$",
):

    s1 = 5
    s2 = 5
    alpha = 0.6

    if pretty:
        xlab, figdims = fig_initialize(
            latex=True, setsize=True, size=size, subplots=(3, 4), fraction=1
        )
        fig, axes = plt.subplots(
            figsize=figdims,
            nrows=2,
            ncols=2,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
    else:
        xlab, figdims = fig_initialize(latex=False, setsize=False)
        fig, ax = plt.subplots()

    if flatten:
        axes[0, 0].plot(Y_val.flatten(), Y_val.flatten(), color="r", lw=1.5, zorder=0)
        axes[0, 1].plot(Y_val.flatten(), Y_val.flatten(), color="r", lw=1.5, zorder=0)
    else:
        axes[0, 0].plot(Y_val, Y_val, color="r", lw=1.5)
        axes[0, 1].plot(Y_val, Y_val, color="r", lw=1.5)

    axes[0, 0].scatter(
        Y_val,
        Y_pred_AA,
        s=s1,
        color="olive",
        zorder=10,
        alpha=alpha,
        label=r"$P_\textrm{nn}^\textrm{AA}$",
        edgecolors="none",
    )
    axes[0, 0].scatter(
        Y_val,
        Y_AA,
        s=s1,
        color="blue",
        zorder=5,
        alpha=alpha,
        label=r"$P_{T}^\textrm{vir}$",
        edgecolors="none",
    )

    axes[0, 1].scatter(
        Y_val,
        Y_pred_no_AA,
        s=s1,
        color="forestgreen",
        zorder=10,
        alpha=alpha,
        label=r"$P_\textrm{nn}^\textrm{no-AA}$",
        edgecolors="none",
    )
    axes[0, 1].scatter(
        Y_val,
        Y_AA,
        s=s1,
        color="blue",
        zorder=5,
        alpha=alpha,
        label=r"$P_{T}^\textrm{vir}$",
        edgecolors="none",
    )

    for i in range(2):
        axes[0, i].set_xscale("log")
        axes[0, i].set_yscale("log")
        axes[0, i].legend()
        axes[1, i].set_xscale("log")
        axes[1, i].set_xlabel(xlabel)
        axes[1, i].set_xlim(min(np.abs(Y_val)), max(Y_val))
        axes[1, i].set_ylim(0, 1.0)
        axes[0, i].set_ylim(min(np.abs(Y_val)), max(Y_val))

    # ax2.set_yscale("log")

    axes[1, 0].scatter(
        np.abs(Y_val),
        np.abs(np.log10(np.abs(Y_val)) - np.log10(np.abs(Y_pred_AA))),
        s=s2,
        color="olive",
        alpha=alpha,
        zorder=5,
        edgecolors="none",
    )

    axes[1, 0].scatter(
        np.abs(Y_val),
        np.abs(np.log10(np.abs(Y_val)) - np.log10(np.abs(Y_AA))),
        s=s2,
        color="blue",
        alpha=alpha,
        zorder=0,
        edgecolors="none",
    )

    axes[1, 1].scatter(
        np.abs(Y_val),
        np.abs(np.log10(np.abs(Y_val)) - np.log10(np.abs(Y_pred_no_AA))),
        s=s2,
        color="forestgreen",
        alpha=alpha,
        zorder=5,
        edgecolors="none",
    )

    axes[1, 1].scatter(
        np.abs(Y_val),
        np.abs(np.log10(np.abs(Y_val)) - np.log10(np.abs(Y_AA))),
        s=s2,
        color="blue",
        alpha=alpha,
        zorder=0,
        edgecolors="none",
    )

    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(r"$|\Delta \log(P)|$")

    plt.subplots_adjust(hspace=0.1)

    if save:
        plt.savefig(filename + ".pdf", bbox_inches="tight")
        # plt.show()
    else:
        plt.show()


def plot_low_T_errs(
    df_aa, pretty=True, size="preprint", save=False, filename="low_T_errs.pdf"
):

    if pretty:
        xlab, figdims = fig_initialize(
            latex=True, setsize=True, size=size, fraction=0.7, plotstyle=3
        )
        fig, ax = plt.subplots(figsize=figdims, nrows=1, ncols=1)
    else:
        xlab, figdims = fig_initialize(latex=False, setsize=False)
        fig, ax = plt.subplots()

    ax.plot(
        df_aa.rho[:5],
        df_aa.P_ref[:5],
        marker="o",
        markersize=5,
        label=r"$P^\textrm{ref}$",
    )

    ax.plot(
        df_aa.rho[:5],
        df_aa.P_aa_st_rr[:5] + df_aa.P_ion[:5],
        marker="o",
        markersize=5,
        label=r"$P_{rr}^\textrm{st}$",
    )

    ax.plot(
        df_aa.rho[:5],
        df_aa.P_aa_st_tr[:5] + df_aa.P_ion[:5],
        marker="o",
        markersize=5,
        label=r"$P_\textrm{tr}^\textrm{st}$",
    )

    ax.plot(
        df_aa.rho[:5],
        df_aa.P_aa_vir_nocorr[:5] + df_aa.P_ion[:5],
        marker="o",
        markersize=5,
        label=r"$P_{T}^\textrm{vir}$",
    )
    ax.plot(
        df_aa.rho[:5],
        df_aa.P_aa_vir_corr[:5] + df_aa.P_ion[:5],
        marker="o",
        markersize=5,
        label=r"$P_{K_{12}}^\textrm{vir}$",
    )

    ax.plot(
        df_aa.rho[:5],
        df_aa.P_aa_fd[:5] + df_aa.P_ion[:5],
        marker="o",
        markersize=5,
        label=r"$P^\textrm{fd}$",
    )
    ax.plot(
        df_aa.rho[:5],
        df_aa.P_aa_ideal[:5] + df_aa.P_ion[:5],
        marker="o",
        markersize=5,
        label=r"$P^\textrm{id}$",
    )

    ax.plot([0, 3], [0, 0], ls="-", color="k", lw=0.8)

    ax.set_xlim(0.35, 2.8)
    ax.set_ylim(-100, 480)
    ax.legend(ncols=2)
    ax.set_xlabel(r"$\rho_\textrm{m}\ (\textrm{g cm}^{-3})$")
    ax.set_ylabel(r"$P$ (GPa)")

    if save:
        plt.savefig(filename, bbox_inches="tight")


def plot_dist_comps(
    df_FP,
    df_Be,
    pretty=True,
    size="preprint",
    save=False,
    filename="rho_temp_hist.pdf",
    nbins=8,
):

    if pretty:
        xlab, figdims = fig_initialize(
            latex=True, setsize=True, size=size, fraction=1.0, subplots=(1, 3)
        )
        fig, axes = plt.subplots(figsize=figdims, nrows=1, ncols=3, sharey=False)
    else:
        xlab, figdims = fig_initialize(latex=False, setsize=False)
        fig, ax = plt.subplots()

    temp_FP = np.log10(df_FP["temp"])
    temp_Be = np.log10(df_Be["temp"])

    rho_FP = np.log10(df_FP["rho"])
    rho_Be = np.log10(df_Be["rho"])

    P_ref_FP = np.log10(df_FP["P_ref"])
    P_ref_Be = np.log10(df_Be["P_ref"])

    list_FP_vars = [temp_FP, rho_FP, P_ref_FP]
    list_Be_vars = [temp_Be, rho_Be, P_ref_Be]

    for i in range(3):
        axes[i].hist(
            list_FP_vars[i],
            density=True,
            bins=nbins,
            color="steelblue",
            alpha=0.3,
            label="FPEOS",
        )
        axes[i].hist(
            list_Be_vars[i],
            density=True,
            bins=nbins,
            color="orange",
            alpha=0.3,
            label="FP-Be",
        )
        mu, sigma = stats.norm.fit(list_FP_vars[i])
        x = np.linspace(list_FP_vars[i].min(), list_FP_vars[i].max(), 100)
        pdf = stats.norm.pdf(x, mu, sigma)
        axes[i].plot(x, pdf, color="steelblue", ls="-")

        mu, sigma = stats.norm.fit(list_Be_vars[i])
        x = np.linspace(list_Be_vars[i].min(), list_Be_vars[i].max(), 100)
        pdf = stats.norm.pdf(x, mu, sigma)
        axes[i].plot(x, pdf, color="orange", ls="-", lw=1.5)

        axes[i].set_yticks([])

    axes[1].legend()
    plt.subplots_adjust(wspace=0.03)

    axes[0].set_xlabel(r"$\log_{10} T$")
    axes[1].set_xlabel(r"$\log_{10} \rho_\textrm{m}$")
    axes[2].set_xlabel(r"$\log_{10} P_\textrm{ref}$")
    axes[0].set_ylabel(r"Frequency")

    if save:
        plt.savefig(filename, bbox_inches="tight")


def plot_missing_aa(
    full_csv,
    aa_csv,
    pretty=True,
    size="preprint",
    save=False,
    filename="aa_missing.pdf",
):

    # Read the CSV files into DataFrames
    df_full = pd.read_csv(full_csv)
    df_aa = pd.read_csv(aa_csv)

    # Find the common pairs of rho, temp
    common_df = pd.merge(df_full, df_aa, on=["rho", "temp"])

    # Find the pairs in df_full that are not in df_aa
    uncommon_df = df_full.merge(df_aa, on=["rho", "temp"], how="left", indicator=True)
    uncommon_df = uncommon_df[uncommon_df["_merge"] == "left_only"]
    uncommon_df.drop("_merge", axis=1, inplace=True)

    # Convert the DataFrames to NumPy arrays
    aa_complete = common_df[["rho", "temp"]].to_numpy()
    aa_missing = uncommon_df[["rho", "temp"]].to_numpy()

    # scatter plot the common points
    if pretty:
        xlab, figdims = fig_initialize(
            latex=True, setsize=True, size=size, fraction=0.8, subplots=(1, 1)
        )
        fig, ax = plt.subplots(figsize=figdims, nrows=1, ncols=1, sharey=False)
    else:
        xlab, figdims = fig_initialize(latex=False, setsize=False)
        fig, ax = plt.subplots()

    ax.scatter(
        aa_complete[:, 0],
        aa_complete[:, 1],
        color="green",
        s=5,
        label="present",
        alpha=0.5,
    )
    ax.scatter(
        aa_missing[:, 0], aa_missing[:, 1], color="red", s=5, label="missing", alpha=0.5
    )

    ax.set_xlabel(r"$\rho_\textrm{m}\ (\textrm{g cm}^{-3})$")
    ax.set_ylabel(r"$T$ (eV)")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylim(5e-2, 1e6)

    ax.legend()

    if save:
        plt.savefig(filename, bbox_inches="tight")


def calc_MAPE(Y_val, Y_pred, epsilon=1e-4):

    array_errors = 100 * np.abs((Y_val - Y_pred) / (Y_val + epsilon))

    MAPE = np.average(array_errors)

    return MAPE, array_errors


def calc_SMAPE(Y_val, Y_pred, epsilon=1e-4):

    array_errors = 200 * np.abs(
        (Y_val - Y_pred) / (np.abs(Y_val) + np.abs(Y_pred) + epsilon)
    )

    MAPE = np.average(array_errors)

    return MAPE, array_errors
