import os

import correlation
import dash
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import utils
from dash import dcc, html
from dash.dependencies import Input, Output, State
from matplotlib.ticker import MaxNLocator
from PIL import Image
from scipy import signal

# plt.rcParams["font.family"] = "Times new Roman"
plt.rcParams["text.usetex"] = False
higher_vars = [
    "abs(z)",
    "abs(vz)",
    "ke",
    "lt",
    "speed",
    "dz",
    "displacement",
    "pe",
]  # highter indicates gas
lower_vars = [
    "ld",
    "coordination",
]  # lower indicates gas0


def plot_var_dist(
    df,
    var,
    t=None,
    color_phase=True,
    mode="mol",
    nbins=None,
    title="",
    xlabel="",
    norm=True,
):
    start = df["actime"]
    lag = utils.get_lag(df)
    if str(lag) not in var:
        start += lag
    if mode == "mol":
        x = utils.parse(df, var)

        if color_phase:
            phase = df["molecule"]["phase"]
            if not nbins:
                nbins = [10, 10]
        elif nbins is None:
            nbins = 10

    elif mode == "atom":
        x = utils.parse(df, var, mode="atom")
        if color_phase:
            phase = df["atom"]["phase"]
            if not nbins:
                nbins = [100, 100]

        elif nbins is None:
            nbins = 100
    if t and t < start:
        t = None
        print("t out of bounds, plotting all t")

    if t is not None:
        x = x[t]
        if color_phase:
            phase = phase[t]
    else:
        x = x[start:].flatten()
        if color_phase:
            phase = phase[start:]
            print(phase.shape)
            phase = phase.flatten()
        print(x.shape, phase.shape)

    if norm:
        x = utils.normalize_arr(x)

    fig = go.Figure()

    if color_phase:
        mask1 = phase == 0
        mask2 = phase == 1

        x1 = x[mask1]
        x2 = x[mask2]

        if isinstance(nbins, int):
            n1 = nbins
            n2 = int(np.round(n1 * (x2.max() - x2.min()) / (x1.max() - x1.min())))
            nbins = [n1, n2]

        counts1, bins1 = np.histogram(x1, bins=nbins[0])
        bins1 = 0.5 * (bins1[:-1] + bins1[1:])

        counts2, bins2 = np.histogram(x2, bins=nbins[1])
        bins2 = 0.5 * (bins2[:-1] + bins2[1:])
        norm = mask1.sum() + mask2.sum()

        fig.add_trace(
            go.Bar(x=bins1, y=counts1 / norm, name="liquid", marker=dict(color="navy"))
        )
        fig.add_trace(
            go.Bar(x=bins2, y=counts2 / norm, name="gas", marker=dict(color="crimson"))
        )

        fig.update_traces(opacity=0.7, marker_line_width=0)
        fig.update_layout(barmode="overlay", bargap=0.0, bargroupgap=0.0)
        fig.update_layout(
            legend=dict(font=dict(size=16)), title={"text": title, "x": 0.5}
        )

    else:
        counts, bins = np.histogram(x, bins=nbins)
        bins = 0.5 * (bins[:-1] + bins[1:])
        norm = x.shape[0]

        fig.add_trace(
            go.Bar(x=bins, y=counts / norm, name=var, marker=dict(color="navy"))
        )
        fig.update_traces(opacity=0.7, marker_line_width=0)
        fig.update_layout(barmode="overlay", bargap=0.0, bargroupgap=0.0)

    if not xlabel:
        xlabel = utils.parse_label(var)
        xlabel = xlabel.strip("$")
        xlabel = r"$\large{" + xlabel + r"}$"

    if not title:
        title = utils.parse_label(var)[:-1]
        title += r" \ \text{Distribution}$"
        title = title.strip("$")
        title = r"$\large{" + title + r"}$"

    fig.update_layout(
        margin=dict(l=60, r=0, t=50, b=60),
        template="plotly_dark",
        title={"text": title, "x": 0.5, "font.size": 24},
        xaxis_title={"text": xlabel, "standoff": 10, "font.size": 20},
        yaxis_title={
            "text": r"$\large{\text{Percent}}$",
            "standoff": 10,
            "font.size": 20,
        },
        width=800,
        height=600,
    )

    return fig


def plot_dist(
    x,
    phase=None,
    t=None,
    nbins=None,
    title="",
    xlabel="",
):
    if t:
        x = x[t]
        if phase is not None:
            phase = phase[t]
    else:
        x = x.flatten()
        if phase is not None:
            phase = phase.flatten()

    fig = go.Figure()

    if phase is not None:
        mask1 = phase == 0
        mask2 = phase == 1
        x1 = x[mask1]
        x2 = x[mask2]

        if isinstance(nbins, int):
            n1 = nbins
            n2 = int(np.round(n1 * (x2.max() - x2.min()) / (x1.max() - x1.min())))
            nbins = [n1, n2]

        counts1, bins1 = np.histogram(x1, bins=nbins[0])
        bins1 = 0.5 * (bins1[:-1] + bins1[1:])

        counts2, bins2 = np.histogram(x2, bins=nbins[1])
        bins2 = 0.5 * (bins2[:-1] + bins2[1:])
        norm = mask1.sum() + mask2.sum()

        fig.add_trace(
            go.Bar(x=bins1, y=counts1 / norm, name="liquid", marker=dict(color="navy"))
        )
        fig.add_trace(
            go.Bar(x=bins2, y=counts2 / norm, name="gas", marker=dict(color="crimson"))
        )

    else:
        if not isinstance(nbins, int):
            nbins = 20

        counts, bins = np.histogram(x, bins=nbins)
        bins = 0.5 * (bins[:-1] + bins[1:])
        norm = x.shape[0]
        fig.add_trace(go.Bar(x=bins, y=counts / norm, marker=dict(color="navy")))

    fig.update_traces(opacity=0.7, marker_line_width=0)
    fig.update_layout(barmode="overlay", bargap=0.0, bargroupgap=0.0)
    fig.update_layout(legend=dict(font=dict(size=16)), title={"text": title, "x": 0.5})

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_dark",
        xaxis_title={"text": xlabel, "standoff": 30, "font.size": 18},
        yaxis_title={"text": "Percent", "standoff": 20, "font.size": 18},
        width=800,
        height=600,
    )

    fig.show()

    return fig


def plot_metal(df, t, dynamics=False):
    if dynamics:
        x = df["atom"]["x"][t]
        y = df["atom"]["y"][t]
        z = df["atom"]["z"][t]
        metal = df["metal_mask"]
        metal_scatter = go.Scatter3d(
            x=x[metal],
            y=y[metal],
            z=z[metal],
            mode="markers",
            marker=dict(size=3, color="gold"),
            showlegend=False,
        )

    else:
        x0 = df["bounds"][0, 0]
        x1 = df["bounds"][0, 1]
        y0 = df["bounds"][1, 0]
        y1 = df["bounds"][1, 1]
        z0 = df["lower_surface"]
        z1 = df["upper_surface"]
        metal_scatter = go.Mesh3d(
            x=[x0, x0, x0, x0, x1, x1, x1, x1],
            y=[y0, y0, y1, y1, y0, y0, y1, y1],
            z=[z0, z1, z0, z1, z0, z1, z0, z1],
            alphahull=0,
            opacity=1,
            color="gold",
            name="metal",
            showlegend=True,
            flatshading=False,
        )

    return metal_scatter


def plot_energyac(E, ylabel="", title="", filter_size=40, fs=16, figsize=(12, 3.5)):
    smoothed = signal.savgol_filter(E, filter_size, 3)
    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=figsize)
    ax[0].plot(np.arange(len(E)), E)
    ax[0].plot(np.arange(len(smoothed)), smoothed, color="tab:red")
    ax[0].set_xlabel("Timestep", fontsize=fs)
    ax[0].set_ylabel(ylabel, fontsize=fs)
    ax[0].set_title(title, fontsize=fs)
    ax[0].tick_params(axis="both", which="major", labelsize=fs)
    ax[0].grid()
    formatter = matplotlib.ticker.ScalarFormatter()
    formatter.set_powerlimits((2, 4))
    ax[0].get_yaxis().set_major_formatter(formatter)
    kappa, tcutoff, ac = correlation.actime(E)
    ax[1].plot(np.arange(len(ac)), ac)
    ax[1].axvline(tcutoff, color="tab:red", linestyle="--")
    ax[1].set_xlabel("Time Lag", fontsize=fs)
    ax[1].set_ylabel("Autocorrelation", fontsize=fs, labelpad=-5)
    ax[1].tick_params(axis="both", which="major", labelsize=fs)
    ax[1].set_title(title, fontsize=fs)

    plt.tight_layout()
    return fig, ax


def plot_molecule_trajectory(
    df,
    mol_idx,
    kind="both",
    ms=5,
    fs=14,
    norm="count",
    xlim=None,
    axes=None,
    absval=True,
    center=True,
    auto_range=True,
    figsize=(12, 5),
    density_title="",
):
    axs = plot_density(
        df,
        kind=kind,
        fs=fs,
        norm=norm,
        xlim=xlim,
        axes=axes,
        absval=absval,
        center=center,
        auto_range=auto_range,
        figsize=figsize,
        title=density_title,
    )

    fig = plt.gcf()
    z = df["molecule"]["z"][:, mol_idx].copy()
    lower_mask = df["molecule"]["lower_mask"][:, mol_idx]
    z[lower_mask] -= df["offset"]
    if center:
        non_metal = df["non_metal"]
        com = np.mean(df["atom"]["z"][:, non_metal], axis=1)
        z = utils.pbc((z - com).reshape(1, -1), utils.get_L(df)).flatten()

    lag = utils.get_lag(df)
    start = lag + df["actime"]

    molecule_phase = df["molecule"]["phase"][:, mol_idx][start:]
    gas_mask = molecule_phase == 1
    z = z[start:]
    t = np.arange(df["nt"])
    t = t[start:]
    z1 = np.abs(z[gas_mask])
    t1 = t[gas_mask]
    z2 = np.abs(z[~gas_mask])
    t2 = t[~gas_mask]

    def plot(z1, t1, z2, t2, ax):
        ax2 = ax.twinx()
        ax2.set_ylabel("t", size=fs)
        ax2.scatter(z1, t1, color="tab:red", s=ms, label="gas")
        ax2.scatter(z2, t2, color="tab:blue", s=ms, label="liquid")
        xlim = ax.get_xlim()
        if (z1 > xlim[-1]).all() and (z2 > xlim[-1]).all():
            ax.set_xlim(xlim[0], df["bounds"][-1, -1])

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax2.legend(lines + lines2, labels + labels2, loc=0, prop={"size": 12})

        for handle in legend.legend_handles:
            if isinstance(handle, matplotlib.collections.PathCollection):
                handle.set_sizes([30])
            elif isinstance(handle, matplotlib.lines.Line2D):
                handle.set_linewidth(2.0)

    if not isinstance(axs, np.ndarray):
        plot(z1, t1, z2, t2, axs)

    else:
        for ax in axs:
            plot(z1, t1, z2, t2, ax)

    title = r"$\text{Classification on }"
    title += (
        utils.concat_labels(df["cluster_vars"])
        + r"\ \text{for Molecule } "
        + rf"{mol_idx}$"
    )
    # cluster_vars = df["cluster_vars"]
    # nvars = len(cluster_vars)
    # for i, var in enumerate(df["cluster_vars"]):
    #     if i == 0:
    #         title += "[" + var
    #     else:
    #         title += " " + var
    #     if nvars > 1 and i < nvars - 1:
    #         title += ","

    # title += f"] for Molecule {mol_idx}"
    fig.suptitle(title, fontsize=fs)
    fig.tight_layout()
    return axs


def plot_mse(x, y, mse, title, fs=14):
    ax = sns.heatmap(
        mse,
        robust=True,
        vmin=0,
        annot=True,
        cbar_kws={"label": r"$SSE_{normalized}$"},
    )

    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    for t in ax.texts:
        t.set_text(format(float(t.get_text())))

    ax.invert_yaxis()
    ax.set_xlabel("dt", fontsize=fs)
    ax.set_ylabel("Radius (Å)", fontsize=fs)
    ax.set_yticklabels(y)
    ax.set_xticklabels(x)
    ax.set_title(title)
    cbar = ax.collections[0].colorbar
    cbar.set_label(r"$MSE_{normalized}$", fontsize=fs)

    return ax


def plot_class_mse(x, y, sse, n, title, fs=14):
    if sse.ndim == 3:
        e = []
        norm = np.zeros_like(sse[0])
        for i in range(sse.shape[0]):
            none = np.where(n[i] == 0)
            some = np.where(n[i] > 0)
            n[i][None] = 1
            e.append(sse[i] / n[i])
            norm[some] += 1

        e = np.array(e)
        e = np.sum(e, axis=0)
        none = np.where(norm == 0)
        some = np.where(norm > 0)
        e[some] /= norm[some]
        e[none] = -1

    elif sse.ndim == 2:
        none = np.where(n == 0)
        some = np.where(n > 0)
        n[none] = 1
        e = sse / n
        e[none] = -1

    ax = plot_mse(x, y, e, title, fs)
    return ax


def plot_traces(df, t, ntraces, top=False, bottom=False):
    assert t >= ntraces, "t must be greater than ntraces"

    x = df["molecule"]["x"][t - ntraces : t + 1]
    y = df["molecule"]["y"][t - ntraces : t + 1]
    z = df["molecule"]["z"][t - ntraces : t + 1]
    phase = df["molecule"]["phase"][t - ntraces : t + 1]

    if top:
        mask = np.min(z, axis=0) > 0
        x = x[:, mask]
        y = y[:, mask]
        z = z[:, mask]
        phase = phase[:, mask]
    elif bottom:
        mask = np.min(z, axis=0) < 0
        x = x[:, mask]
        y = y[:, mask]
        z = z[:, mask]
        phase = phase[:, mask]

    paths = []
    xchange = (np.min(x, axis=0) < df["bounds"][0, 0] + 20) & (
        np.max(x, axis=0) > df["bounds"][0, 1] - 20
    )
    ychange = (np.min(y, axis=0) < df["bounds"][1, 0] + 20) & (
        np.max(y, axis=0) > df["bounds"][1, 1] - 20
    )
    zchange = (np.min(z, axis=0) < 0) & (np.max(z, axis=0) > 0)
    skip = xchange | ychange | zchange
    x = x[:, ~skip]
    y = y[:, ~skip]
    z = z[:, ~skip]
    phase = phase[:, ~skip]
    x = np.vstack((x, np.nan * np.ones(x.shape[-1])))
    y = np.vstack((y, np.nan * np.ones(y.shape[-1])))
    z = np.vstack((z, np.nan * np.ones(z.shape[-1])))
    phase = np.vstack((phase, np.nan * np.ones(phase.shape[-1])))
    trace = go.Scatter3d(
        x=x.T.flatten(),
        y=y.T.flatten(),
        z=z.T.flatten(),
        mode="lines",
        line=dict(
            width=6,
            color=phase.T.flatten(),
            colorscale=[[0, "green"], [0.5, "navy"], [1, "crimson"]],
            cmin=-1,
            cmax=1,
        ),
        opacity=1,
        showlegend=False,
        hoverinfo="None",
        connectgaps=False,
    )
    paths.append(trace)

    return paths


def scatter_var(
    df,
    var,
    t,
    mode="atom",
    phase=None,
    title="",
    traces=False,
    ntraces=5,
    reversescale=None,
    metal=True,
    metal_dynamics=False,
    top=False,
    bottom=False,
    absval=False,
    step=False,
    z0=None,
    center=True,
    width=600,
    height=800,
    fs=16,
    size=None,
    mask=None,
):
    paths = []
    if var not in list(df[mode].keys()):
        raise KeyError(f"{var} is not in the {mode} dataset")

    x = df[mode]["x"][t]
    y = df[mode]["y"][t]
    z = df[mode]["z"][t]
    color = utils.parse(df, var, mode=mode)[t]
    if mask is not None and phase is None and not traces:
        x = x[mask]
        y = y[mask]
        z = z[mask]
        color = color[mask]
    if size is None:
        size = 3 if mode == "atom" else 7

    if center:
        com = np.mean(z)
        z = utils.pbc((z - com).reshape(1, -1), utils.get_L(df)).flatten()

    if traces:

        if mode == "atom":
            print("traces for atomic data too expensive. Continuing without...")
        elif phase is not None:
            print(
                "Traces for a single phase have the issue that molecules are changing. Too annoying to implement."
            )
        else:
            paths = plot_traces(df, t, ntraces, top=top, bottom=bottom)  # TODO phase

    if phase is not None:
        phase_mask = df[mode]["phase"][t] == phase
        x = x[phase_mask]
        y = y[phase_mask]
        z = z[phase_mask]
        color = color[phase_mask]

    if top:
        mask = z > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        color = color[mask]
    elif bottom:
        mask = z < 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        color = color[mask]
    elif absval:
        z = np.abs(z)

    data = []
    if var == "phase":
        gas = color == 1
        liquid = color == 0

        if gas.any() or liquid.any():
            gas_scatter = go.Scatter3d(
                x=x[gas],
                y=y[gas],
                z=z[gas],
                mode="markers",
                marker=dict(size=size, color="crimson"),
                name="gas",
                showlegend=True,
                opacity=1,
            )
            liquid_scatter = go.Scatter3d(
                x=x[liquid],
                y=y[liquid],
                z=z[liquid],
                mode="markers",
                marker=dict(size=size, color="navy"),
                name="liquid",
                showlegend=True,
                opacity=1,
            )
            data.append(gas_scatter)
            data.append(liquid_scatter)
        else:
            scatter = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=size, color="green"),
                name="",
                showlegend=False,
                opacity=1,
            )
            data.append(scatter)

    else:
        global lower_vars
        if var.split("_")[0] in lower_vars and reversescale is None:
            reversescale = True
        elif reversescale is None:
            reversescale = False

        scatter = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=size,
                color=color,
                cmin=color.min(),
                cmax=color.max(),
                colorscale="Jet",
                colorbar=dict(title=var),
                reversescale=reversescale,
            ),
            showlegend=False,
        )
        data.append(scatter)
    if metal:
        metal_scatter = plot_metal(df, t, metal_dynamics)
        data.append(metal_scatter)

    data.extend(paths)
    if z0 is not None:
        light_yellow = [[0, "#FFDB58"], [1, "#FFDB58"]]
        L = utils.get_L(df)
        Lx = L[0]
        Ly = L[1]
        x = np.arange(-Lx, Lx)
        y = np.arange(-Ly, Ly)
        z_plane = z0 * np.ones((x.size, y.size))
        upper_z0 = go.Surface(
            x=x,
            y=y,
            z=z_plane,
            colorscale=light_yellow,
            showscale=False,
            opacity=1,
            name="z0",
            showlegend=True,
        )
        lower_z0 = go.Surface(
            x=x,
            y=y,
            z=-z_plane,
            colorscale=light_yellow,
            showscale=False,
            opacity=1,
        )
        data.extend([upper_z0, lower_z0])

    bounds = df["bounds"] * 1.05
    layout = go.Layout(
        width=width,
        height=height,
        title={"text": title, "x": 0.5, "font": {"size": fs}},
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            itemsizing="constant", yanchor="top", y=0.99, xanchor="left", x=0.01
        ),
        uirevision=True,
        template="plotly_dark",
        scene=dict(
            xaxis=dict(range=[bounds[0, 0], bounds[0, 1]]),
            yaxis=dict(range=[bounds[1, 0], bounds[1, 1]]),
            zaxis=dict(range=[bounds[2, 0], bounds[2, 1]]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        scene_camera=dict(eye=dict(x=2, y=2, z=0.25)),
    )

    if step:
        fig = dict(data=data, layout=layout)

    else:
        fig = go.Figure(data=data, layout=layout)

    return fig


def get_marks(timesteps):
    timestep_indices = np.arange(timesteps.size)
    if timestep_indices.size <= 50:
        step = 1
    elif timestep_indices.size <= 201:
        step = 10

    elif timestep_indices.size <= 1001:
        step = 50

    else:
        step = 100

    marks = {t: t for t in timestep_indices[::step].astype(str)}
    slider_timesteps = np.array(list(marks.keys()), dtype=int)
    t_min = slider_timesteps.min()
    t_max = slider_timesteps.max()
    return marks, step, t_min, t_max


def get_title(df, var, t):
    t_ns = utils.get_t_ns(df, t)

    molecule = str(df["molecule_name"])
    temp = str(df["temp"])
    if var == "phase":
        title = rf"\text{{{molecule} {temp}K}}"
        title += utils.concat_labels(df["cluster_vars"])
        title += r"\ \text{at} \ " + rf"t = {t_ns}" + r"\ \text{ns}"
        title = r"$" + title + r"$"

    else:
        title = (
            rf"\text{{{molecule} {temp}K}}"
            + r"\ \text{at} \ "
            + rf"t = {t_ns}"
            + r"\ \text{ns}"
        )
        title = r"$" + title + r"$"
    return title


def scatter_var_step(
    df,
    var,
    t,
    mode="atom",
    phase=None,
    traces=False,
    ntraces=5,
    reversescale=None,
    metal=True,
    metal_dynamics=False,
    top=False,
    bottom=False,
    port="63854",
):
    nt = df["nt"]
    if t > nt:
        print("t greater than last timestep. Setting t = 0")
        t = 0

    timesteps = df["timesteps"]
    marks, step, t_min, t_max = get_marks(timesteps)

    app = dash.Dash(__name__)

    app.layout = html.Center(
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="graph", mathjax=True)],
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button("←", id="back"),
                                html.Button("→ ", id="forward"),
                            ],
                        ),
                        html.Div(
                            [
                                dcc.Slider(
                                    min=t_min.min(),
                                    max=t_max.max(),
                                    step=step,
                                    value=t,
                                    id="timestep",
                                    marks=marks,
                                ),
                            ],
                            style={
                                "width": "90%",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "width": "600px",
                    },
                ),
                html.Div(
                    [html.Button("Save", id="save")],
                    style={
                        "width": "50px",
                    },
                ),
            ],
        )
    )

    @app.callback(
        Output("graph", "figure", allow_duplicate=True),
        Input("timestep", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def update_slider(t):
        title = get_title(df, var, t)

        fig = scatter_var(
            df,
            var,
            t,
            phase=phase,
            mode=mode,
            title=title,
            traces=traces,
            ntraces=ntraces,
            reversescale=reversescale,
            metal=metal,
            metal_dynamics=metal_dynamics,
            top=top,
            bottom=bottom,
            step=True,
        )

        return fig

    @app.callback(
        Output("graph", "figure", allow_duplicate=True),
        Output("timestep", "value", allow_duplicate=True),
        Input("forward", "n_clicks"),
        State("timestep", "value"),
        prevent_initial_call=True,
    )
    def step_forward(forward, t):
        if forward:
            t = min(t + 1, nt - 1)

        title = get_title(df, var, t)

        fig = scatter_var(
            df,
            var,
            t,
            phase=phase,
            mode=mode,
            title=title,
            traces=traces,
            ntraces=ntraces,
            reversescale=reversescale,
            metal=metal,
            metal_dynamics=metal_dynamics,
            top=top,
            bottom=bottom,
            step=True,
        )

        return fig, t

    @app.callback(
        Output("graph", "figure", allow_duplicate=True),
        Output("timestep", "value", allow_duplicate=True),
        Input("back", "n_clicks"),
        State("timestep", "value"),
        prevent_initial_call=True,
    )
    def step_backward(back, t):
        if back:
            if traces:
                t = max(t - 1, 0, ntraces)
            else:
                t = max(t - 1, 0)
        title = get_title(df, var, t)

        fig = scatter_var(
            df,
            var,
            t,
            phase=phase,
            mode=mode,
            title=title,
            traces=traces,
            ntraces=ntraces,
            reversescale=reversescale,
            metal=metal,
            metal_dynamics=metal_dynamics,
            top=top,
            bottom=bottom,
            step=True,
        )

        return fig, t

    @app.callback(
        Input("save", "n_clicks"),
        State("graph", "relayoutData"),
        State("timestep", "value"),
    )
    def save_fig(save, relayout_data, t):
        if save:
            t = (t - df["timesteps"][0]) // df["dt"]
            title = get_title(df, var, t)
            fig = scatter_var(
                df,
                var,
                t,
                phase=phase,
                mode=mode,
                title=title,
                traces=traces,
                ntraces=ntraces,
                reversescale=reversescale,
                metal=metal,
                metal_dynamics=metal_dynamics,
                top=top,
                bottom=bottom,
                step=False,
            )
            if relayout_data is not None and "scene.camera" in relayout_data:
                camera = relayout_data["scene.camera"]
                fig.update_layout(scene_camera=camera)

            fig.write_image(f"{var}_{t}.png", scale=3)
        return None

    app.run_server(
        debug=True,
        port=port,
        jupyter_height=900,
    )

    return app


def dz_hist(df, bins=20, fs=14):
    keys = list(df["molecule"].keys())
    dts = [int(key.split("_")[-1]) for key in keys if "dz" in key]

    def plot(ax, df, dt, fs):
        ax.hist(df["molecule"][f"dz_{dt}"].flatten(), bins=bins, zorder=10)
        ax.set_xlabel("dz", fontsize=fs)
        ax.set_ylabel("Molecule Count (Thousands)", fontsize=fs - 2)
        ax.set_title(f"dt = {dt}", fontsize=fs)
        ax.grid()
        ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ndz = len(dts)

    nrows = ndz // 2

    try:
        ncols = ndz // nrows
    except:
        ncols = 1

    if nrows * ncols < ndz:
        nrows += 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

    if not isinstance(axs, np.ndarray):
        plot(axs, df, dts[0], fs, abs)

    else:
        for i, ax in enumerate(axs.flatten()):
            if i >= ndz:
                ax.axis("off")
                continue

            plot(ax, df, dts[i], fs)

    fig.tight_layout()
    plt.show()
    return None


def coord_hist(df, bins=20, fs=14):
    keys = list(df["molecule"].keys())
    radii = [int(key.split("_")[-1]) for key in keys if "coordination" in key]
    radii = np.sort(radii)

    def plot(ax, df, rad, fs):
        ax.hist(df["molecule"][f"coordination_{rad}"].flatten(), bins=bins, zorder=10)
        ax.set_xlabel("coordination", fontsize=fs)
        ax.set_ylabel("Molecule Count (Thousands)", fontsize=fs - 2)
        ax.set_title(f"Cutoff Radius = {rad} Å", fontsize=fs)
        ax.grid()
        ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    nrad = len(radii)

    nrows = nrad // 2

    try:
        ncols = nrad // nrows
    except:
        ncols = 1

    if nrows * ncols < nrad:
        nrows += 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

    if not isinstance(axs, np.ndarray):
        plot(axs, df, radii[0], fs)

    else:
        for i, ax in enumerate(axs.flatten()):
            if i >= nrad:
                ax.axis("off")
                continue

            plot(ax, df, radii[i], fs)

    fig.tight_layout()
    plt.show()

def displacement_hist(df, bins=20, fs=14):
    keys = list(df["molecule"].keys())
    radii = [int(key.split("_")[-1]) for key in keys if "displacement" in key]
    radii = np.sort(radii)

    def plot(ax, df, rad, fs):
        ax.hist(df["molecule"][f"displacement_{rad}"].flatten(), bins=bins, zorder=10)
        ax.set_xlabel("displacement ( Å)", fontsize=fs-2)
        ax.set_ylabel("Molecule Count (Thousands)", fontsize=fs - 4)
        ax.set_title(f"Time Lag = {rad*50} ps", fontsize=fs)
        ax.grid()
        ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis = 'x', labelsize = 8)

    nrad = len(radii)
    

    nrows = nrad // 2

    try:
        ncols = nrad // nrows
    except:
        ncols = 1

    if nrows * ncols < nrad:
        nrows += 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

    if not isinstance(axs, np.ndarray):
        plot(axs, df, radii[0], fs)

    else:
        for i, ax in enumerate(axs.flatten()):
            if i >= nrad:
                ax.axis("off")
                continue

            plot(ax, df, radii[i], fs)

    fig.tight_layout()
    plt.show()


def scatter_coord(df, abs=True):
    fs = 14
    keys = list(df["molecule"].keys())
    radii = [int(key.split("_")[-1]) for key in keys if "coordination" in key]

    def plot(ax, df, rad, fs, abs):
        x = df["molecule"]["z"]
        if abs:
            x = np.abs(x)
        y = df["molecule"][f"coordination_{rad}"]

        ax.scatter(x, y, s=1.5)
        ax.set_xlabel("Z (Å)", fontsize=fs)
        rad = rf"{str(rad)}"
        ylabel = r"$coordination_{" + rad + r"}$"
        ax.set_ylabel(ylabel, fontsize=fs)

    nrad = len(radii)

    nrows = nrad // 2

    try:
        ncols = nrad // nrows
    except:
        ncols = 1

    if nrows * ncols < nrad:
        nrows += 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

    if not isinstance(axs, np.ndarray):
        plot(axs, df, radii[0], fs, abs)

    else:
        for i, ax in enumerate(axs.flatten()):
            if i >= nrad:
                ax.axis("off")
                continue

            plot(ax, df, radii[i], fs, abs)

    fig.tight_layout()
    plt.show()
    return None


def plot_molecule_dz(df, lag, mol_idx):
    fig, ax = plt.subplots(figsize=(7, 5))

    fs = 14
    x = np.arange(df["nt"])
    y = df["molecule"]["z"][:, mol_idx]
    ax.plot(x, y, label="Z")
    ax2 = ax.twinx()
    y = df["molecule"][f"dz_{lag}"][:, mol_idx]
    x = np.arange(y.shape[0]) + lag
    ax2.plot(x, y, color="tab:orange", label="dz")
    ax.set_xlabel("t", fontsize=fs)
    ax.set_ylabel("Z (Å)", fontsize=fs)
    ax2.set_ylabel("dZ", fontsize=fs)
    ax.set_title(f"Molecule {mol_idx}, dt = {lag}", fontsize=fs)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax2.legend(lines + lines2, labels + labels2, loc=0, prop={"size": 12})

    for handle in legend.legend_handles:
        if isinstance(handle, matplotlib.collections.PathCollection):
            handle.set_sizes([30])
        elif isinstance(handle, matplotlib.lines.Line2D):
            handle.set_linewidth(2.0)

    return ax, ax2


def anim_phase(
    df,
    start,
    end,
    mode="atom",
    duration=10,
    traces=False,
    ntraces=10,
    title="",
    metal_dynamics=False,
):
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "t:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50, "r": 30},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    frames = []
    titles = []
    for t in range(start, end + 1):
        if title:
            frame_title = title + f", t = {t}"
        else:
            frame_title = f"t = {t}"
        scatter_fig = scatter_var(
            df,
            "phase",
            t,
            mode=mode,
            traces=traces,
            ntraces=ntraces,
            title=frame_title,
            metal_dynamics=metal_dynamics,
        )

        frame_title = scatter_fig["layout"]["title"]
        scatter = scatter_fig.data
        if t > start:
            scatter = go.Frame(data=scatter)

        frames.append(scatter)
        titles.append(frame_title)
        slider_step = {
            "args": [
                [t],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": t,
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)
    # print(frames[0])
    # print()
    # print(frames[1])
    scatter = frames[0]
    bounds = df["bounds"]
    fig = go.Figure(data=scatter, frames=frames[1:])
    for i in range(1, len(fig.frames) + 1):
        fig.frames[i - 1]["layout"].update(title=titles[i])
    fig.update_layout(
        uirevision=True,
        title=titles[0],
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(range=[bounds[0, 0], bounds[0, 1]]),
            yaxis=dict(range=[bounds[1, 0], bounds[1, 1]]),
            zaxis=dict(range=[0, bounds[2, 1]]),
        ),
        width=600,
        height=800,
        legend=dict(
            itemsizing="constant", yanchor="top", y=0.99, xanchor="left", x=0.01
        ),
        template="plotly_dark",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(
                                    redraw=True,
                                    fromcurrent=True,
                                    mode="immediate",
                                    transistion=dict(duration=0),
                                    duration=duration * 1000 / (end - start - 1),
                                )
                            ),
                        ],
                    ),
                    dict(
                        label="←",
                        method="update",
                        args=[frames[0], {"title": ""}, np.array(range(100))],
                    ),  # data, layout
                ],
            )
        ],
    )
    fig.update_layout(sliders=[sliders_dict])
    # fig["layout"]["sliders"] = [sliders_dict]
    # print(titles)
    return fig


def plot_density(
    df,
    nbins=200,
    kind="both",
    norm="count",
    title="",
    fs=14,
    xlim=None,
    axes=None,
    hist_range=None,
    time_avg=True,
    phase_mask=None,
    absval=False,
    center=True,
    actime=True,
    auto_range=True,
    std=False,
    t=None,
    figsize=(12, 5),
):
    if axes is None:
        if kind == "both":
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=200)

        else:
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=200)
    else:
        axs = axes[1:]

    def plot(kind, df, ax):
        if kind == "both":
            return plot("atom", df, ax[0]), plot("molecule", df, ax[1])
        # if kind == 'atom':
        #     plot_title = "Atomic " + title
        # else:
        #     plot_title = "Molecular " + title

        density, zbin, err = utils.density(
            df,
            nbins,
            kind=kind,
            norm=norm,
            hist_range=hist_range,
            time_avg=time_avg,
            phase_mask=phase_mask,
            absval=absval,
            center=center,
            actime=actime,
            auto_range=auto_range,
            std=std,
            t=t,
        )

        if norm == "prob" or norm == "percent":

            density *= 100
            ylabel = "%"

        elif norm == "mass":
            ylabel = r"$\rho \ (kg \cdot m^{-3})$"

        else:
            ylabel = "Count"

        ax.plot(zbin, density, label=r"$\rho$", color="black")

        if std:
            ax.fill_between(zbin, density - err, density + err, alpha=0.4)

        ax.set_xlabel("Z (Å)", size=fs)
        ax.set_ylabel(ylabel, size=fs)
        ax.set_xlim(xlim)
        ax.set_title(title, size=fs)

        return None

    plot(kind, df, axs)
    fig.tight_layout()
    return axs


def plot_pair_correlation(df, title=None, filter_size=None, fs=16, figsize=(6, 4)):
    """
    Plot the pair correlation function g(r) for all t

    Args:
        g: pair correlation function
        r: bin centers
    Returns:
        fig, ax
    """
    g = df["molecule"]["g"]
    r = df["molecule"]["r"]
    L = utils.get_L(df)
    dr = r[1] - r[0]
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
    if filter_size is not None:
        smoothed = signal.savgol_filter(g, filter_size, 3)[1:]
        ax.plot(r[1:], smoothed + 1, color="tab:red", linestyle="--", linewidth=1.5)

    ax.plot(r, g + 1)
    ax.set_xlabel(r"$|r| \ (Å)$", fontsize=fs)
    ax.set_ylabel("g(r)", fontsize=fs)
    if title is None:
        title = f"dr = {dr}Å"
    else:
        title += f" dr = {dr}Å"
    ax.set_title(title, fontsize=fs)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.minorticks_on()
    ax.grid()
    ax.set_xlim(0, L.min() / 2)

    return fig, ax


def plot_sk():
    pass


def anim_scatter(
    df,
    mode,
    var,
    tmin,
    tmax,
    tstep,
    path,
    interval=300,
    phase=None,
    metal=False,
    title="",
    traces=False,
    ntraces=5,
    reversescale=None,
    metal_dynamics=False,
    top=False,
    bottom=False,
    absval=False,
    step=False,
    z0=None,
    center=True,
    width=600,
    height=800,
    fs=16,
    size=None,
):

    for t in range(tmin, tmax, tstep):
        fig_title = f"t={utils.get_t_ns(df, t):.2f}ns"
        fig = scatter_var(
            df,
            var,
            t,
            phase=phase,
            mode=mode,
            traces=traces,
            title=fig_title,
            ntraces=ntraces,
            reversescale=reversescale,
            metal=metal,
            metal_dynamics=metal_dynamics,
            top=top,
            bottom=bottom,
            absval=absval,
            step=step,
            z0=z0,
            center=center,
            width=width,
            height=height,
            fs=fs,
            size=size,
        )
        camera = dict(eye=dict(x=2, y=2, z=0.25))

        fig.update_layout(scene_camera=camera)
        fig.write_image(f"{path}fig{t}.png", scale=2)

    image_paths = [im for im in sorted(os.listdir(path)) if im.endswith(".png")]
    images = [np.array(Image.open(path + image)) for image in image_paths]

    fig, ax = plt.subplots(dpi=200)
    ax.set_frame_on(False)
    ax.axis("off")
    im = ax.imshow(images[0])
    ax.set_title(title)

    def animate(i):
        im.set_array(images[i])
        ax.set_title(title)
        return (im,)

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(images),
        interval=interval,
        blit=True,
        repeat_delay=1000,
    )
    ani.save(path + "animation.gif")

    return ani


def heatmap(mat, features, title="", cbar_title="", fs=10, cmap="coolwarm"):


    ax = sns.heatmap(
        mat,
        robust=True,
        annot=True,
        cbar_kws={"label": cbar_title},
        fmt=".2f",
        cmap=cmap,
    )

    fig = plt.gcf()
    fig.set_size_inches(6, 5)
    fig.set_dpi(200)

    # ax.invert_yaxis()
    labels = [utils.parse_label(feat) for feat in features]
    ax.set_yticklabels(labels, fontsize = fs)
    ax.set_xticklabels(labels, fontsize = fs)
    ax.set_title(title, fontsize = fs + 4)
    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_title, fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    return fig, ax
    


# def anim_scatter_plotly():

#     frame0 = plotting.scatter_var(df['lvc'][400], 'phase', 500, phase = 0, mode = 'molecule', metal = False, step = True)
#     frames = []
#     for t in range(501, 700, 10):
#         frame = plotting.scatter_var(df['lvc'][400], 'phase', t, phase = 0, mode = 'molecule', metal = False, step = True)
#         frames.append(go.Frame(data = frame['data']))

#     fig = go.Figure(data = frame0['data'], layout = frame0['layout'], frames = frames)
#     def frame_args(duration):
#         return {
#                 "frame": {"duration": duration},
#                 "mode": "immediate",
#                 "fromcurrent": True,
#                 "transition": {"duration": duration, "easing": "linear"},
#             }

#     sliders = [
#                 {
#                     "pad": {"b": 10, "t": 60},
#                     "len": 0.9,
#                     "x": 0.1,
#                     "y": 0,
#                     "steps": [
#                         {
#                             "args": [[f.name], frame_args(0)],
#                             "label": str(k),
#                             "method": "animate",
#                         }
#                         for k, f in enumerate(fig.frames)
#                     ],
#                 }
#             ]

#     fig.update_layout(sliders=sliders)

# import plotly.io as pio

# ii = 1
# pio.write_html(fig, file="test.html", auto_open=True)

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# Create a figure
# fig = plt.figure(figsize=(10, 10))

# # Create a 3x3 GridSpec
# gs = gridspec.GridSpec(3, 3)

# # Create subplots in the first two rows
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[0, 2])
# ax4 = fig.add_subplot(gs[1, 0])
# ax5 = fig.add_subplot(gs[1, 1])
# ax6 = fig.add_subplot(gs[1, 2])

# # Create subplots in the last row, centered horizontally
# gs_bottom = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols = 4, subplot_spec = gs[2,:], width_ratios = [1, 2, 2, 1])
# ax7 = fig.add_subplot(gs_bottom[0, 1])
# ax8 = fig.add_subplot(gs_bottom[0, 2])

# # Optionally, you can add titles or labels to the subplots
# ax1.set_title('Subplot 1')
# ax2.set_title('Subplot 2')
# ax3.set_title('Subplot 3')
# ax4.set_title('Subplot 4')
# ax5.set_title('Subplot 5')
# ax6.set_title('Subplot 6')
# ax7.set_title('Subplot 7')
# ax8.set_title('Subplot 8')

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Show the plot
# plt.show()
