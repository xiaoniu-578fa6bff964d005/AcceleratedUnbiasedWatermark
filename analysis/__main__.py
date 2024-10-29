import polars as pl
import numpy as np
import os
import itertools

from . import utils

#  data_path = "data"
#  data_path = "test_data.2024.05.01.16.46"
#  data_path = "test_data.2024.05.01.17.37"
#  data_path = "incomplete_data.2024.05.02.00.57"
#  data_path = "data.2024.05.02.07.57"
#  data_path = "incomplete_data.2024.05.02.17.50"
#  data_path = "verify_data.2024.05.02.19.36"
#  data_path = "verify_data.2024.05.02.20.03"
#  data_path = "verify_data.2024.05.03.08.01"
#  data_path = "verify_data.2024.05.03.08.47"
#  data_path = "verify_data.2024.05.03.08.53"
#  data_path = "verify_data.2024.05.03.08.58"
#  data_path = "verify_data.2024.05.03.09.09"
#  data_path = "verify_data.2024.05.03.09.44"
#  data_path = "verify_data.2024.05.03.10.11"
#  data_path = "verify_data.2024.05.03.11.49"
#  data_path = "verify_data.2024.05.03.14.36"
#  data_path = "verify_data.2024.05.03.15.15"
#  data_path = "verify_data.2024.05.03.15.26"
#  data_path = "verify_data.2024.05.03.15.32"
#  data_path = "verify_data.2024.05.03.16.20"
#  data_path = "verify_data.2024.05.03.18.11"
#  data_path = "verify_data.2024.05.04.18.47"
#  data_path = "verify_data.2024.05.04.23.23"
#  data_path = "verify_data.2024.05.04.23.32"
#  data_path = "verify_data.2024.05.04.23.58"
#  data_path = "data.2024.05.05.00.20"
#  data_path = "data.2024.05.05.00.37"
#  data_path = "data.2024.05.05.08.06"
#  data_path = "data.2024.05.05.10.32"
#  data_path = "data.2024.05.07.13.46"
#  data_path = "data.2024.05.10.23.38"
data_path = "final_data"


def read_ds(paths: list):
    #  root is parent folder of analysis
    # e.g. ["data_root", data_path, "summarization_scan_n", "llama-7b_llama-160m"]
    path = os.path.join(os.path.dirname(__file__), "..", *paths, "*")
    ds = pl.read_parquet(path)
    return ds


from experiments.worker import score_strs


def compute_atn(ds, group_params):
    atn_ds = (
        ds.select(
            [
                *group_params,
                "gen_seq_lens",
            ]
        )
        .group_by(group_params)
        .agg(
            nums=pl.col("gen_seq_lens").flatten(),
        )
        .with_columns(
            tobeunnest=pl.struct(["nums"]).map_elements(
                lambda x: {
                    k: v
                    for k, v in zip(
                        ["atn_mu", "atn_sigma", "atn_std"],
                        utils.mu_sigma_std(**x),
                    )
                }
            )
        )
        .unnest("tobeunnest")
    ).drop(["nums"])
    return atn_ds


def compute_ptt(ds, group_params):
    ptt_ds = (
        ds.select(
            [
                *group_params,
                "gen_seq_lens",
                "t_got_first_output",
                "t_got_last_output",
            ]
        )
        .group_by(group_params)
        .agg(
            nums=pl.col("gen_seq_lens").list.slice(1, None).list.sum(),
            sums=pl.col("t_got_last_output") - pl.col("t_got_first_output"),
        )
        .with_columns(
            tobeunnest=pl.struct(["nums", "sums"]).map_elements(
                lambda x: {
                    k: v
                    for k, v in zip(
                        ["ptt_mu", "ptt_sigma", "ptt_std"],
                        utils.mu_sigma_std_from_sums(**x),
                    )
                }
            )
        )
        .unnest("tobeunnest")
    ).drop(["nums", "sums"])
    return ptt_ds


def compute_ppl(ds, group_params):
    ppl_ds = (
        ds.select(
            [
                *group_params,
                "gen_seq_lens",
                "logperplexity",
            ]
        )
        .group_by(group_params)
        .agg(
            nums=pl.col("gen_seq_lens").list.sum(),
            means=pl.col("logperplexity"),
        )
        .with_columns(
            tobeunnest=pl.struct(["nums", "means"]).map_elements(
                lambda x: {
                    k: v
                    for k, v in zip(
                        ["log_ppl_mu", "log_ppl_sigma", "log_ppl_std"],
                        #  ["log_ppl_mu"],
                        utils.mu_sigma_std_from_sums(
                            nums=x["nums"],
                            sums=np.array(x["means"]) * np.array(x["nums"]),
                        ),
                    )
                }
            )
        )
        .unnest("tobeunnest")
    ).drop(["nums", "means"])
    return ppl_ds


def compute_ptlpv(ds, group_params):
    ds = ds.select(
        [
            *group_params,
            "gen_seq_lens",
            "log_p_values",
        ]
    )

    def do_statistic(ds):
        local_group_params = [*group_params, "score_type"]
        ptlpv_ds = (
            ds.group_by(local_group_params)
            .agg(
                nums=pl.col("gen_seq_lens").list.sum(),
                #  sums=pl.col("log_p_value") * pl.col("gen_seq_lens").list.sum(),
                sums=pl.col("log_p_value"),
            )
            .with_columns(
                tobeunnest=pl.struct(["nums", "sums"]).map_elements(
                    lambda x: {
                        k: v
                        for k, v in zip(
                            ["ptlpv_mu", "ptlpv_sigma", "ptlpv_std"],
                            utils.mu_sigma_std_from_sums(**x),
                        )
                    }
                )
            )
            .unnest("tobeunnest")
        ).drop(["nums", "sums"])
        return ptlpv_ds

    dss = []
    non_ds = do_statistic(
        ds.filter(pl.col("reweight") == "none")
        .drop(["log_p_values"])
        .with_columns(log_p_value=pl.lit(0.0), score_type=pl.lit("none"))
    )
    dss.append(non_ds)
    for reweight in score_strs:
        for i, score_type in enumerate(score_strs[reweight]):
            ptlpv_ds = do_statistic(
                ds.filter(pl.col("reweight") == reweight)
                .with_columns(
                    log_p_value=pl.col("log_p_values").list.get(i),
                    score_type=pl.lit(score_type),
                )
                .drop(["log_p_values"])
            )
            dss.append(ptlpv_ds)
    return pl.concat(dss)


def filter_null(ds):
    ds = ds.filter(
        pl.col("log_p_values").list.eval(pl.element().is_null()).list.any() == False
    )
    return ds


def summarize_ds(paths):
    ds = read_ds(paths)
    ds = filter_null(ds)
    group_params = ["n", "method", "reweight"]
    atn_ds = compute_atn(ds, group_params).sort(by=group_params)
    ptt_ds = compute_ptt(ds, group_params).sort(by=group_params)
    ppl_ds = compute_ppl(ds, group_params).sort(by=group_params)
    ptlpv_ds = compute_ptlpv(ds, group_params).sort(by=group_params)
    return atn_ds, ptt_ds, ppl_ds, ptlpv_ds


def test1():
    atn_ds, ptt_ds, ppl_ds, ptlpv_ds = summarize_ds(
        ["data_root", data_path, "summarization_scan_n", "llama-7b_llama-68m"]
        #  ["data_root", data_path, "oeg_scan_n", "llama-7b_llama-68m"]
    )
    utils.large_print(atn_ds)
    utils.large_print(ptt_ds)
    utils.large_print(ppl_ds)
    utils.large_print(ptlpv_ds)


def test2():
    # find null in log_p_values
    ds = read_ds(["data_root", data_path, "oeg_scan_n", "llama-7b_llama-68m"])

    tds = ds.select(["method", "n", "reweight", "gen_seq_lens", "log_p_values"]).filter(
        pl.col("log_p_values").list.eval(pl.element().is_null()).list.any()
    )
    with pl.Config(
        fmt_str_lengths=6000, tbl_width_chars=6000, fmt_table_cell_list_len=1000
    ):
        print(tds.drop(["gen_seq_lens"]))


def test_plot():
    ds = read_ds(
        #  ["data_root", data_path, "summarization_scan_n", "llama-7b_llama-160m"]
        ["data_root", data_path, "summarization_scan_n", "llama-7b_llama-68m"]
    )
    atn_ds, ptt_ds, ppl_ds, ptlpv_ds = summarize_ds(
        ["data_root", data_path, "summarization_scan_n", "llama-7b_llama-68m"]
    )
    atn_ds = atn_ds.drop(["atn_sigma"])
    ptlpv_ds = ptlpv_ds.drop(["ptlpv_sigma"])
    show_reweights = ["none", "deltagumbel"]
    show_ns = [1, 2]
    show_score_types = ["DeltaGumbel_U", "Gamma_U", "none"]
    atn_ds = atn_ds.filter(
        pl.col("reweight").is_in(show_reweights) & pl.col("n").is_in(show_ns)
    )
    ptlpv_ds = ptlpv_ds.filter(
        pl.col("reweight").is_in(show_reweights)
        & pl.col("n").is_in(show_ns)
        & pl.col("score_type").is_in(show_score_types)
    )

    def format_label(x):
        reweight_map = {
            "none": "No Reweight",
            "deltagumbel": "DeltaGumbel",
            "gamma": "Gamma",
        }
        method_map = {
            "mc": "SpS",
            "mc_uwm_speed": "SpS_WM_Speed",
            "mc_uwm_strength": "SpS_WM_Strength",
            "basic": "Basic",
            "basic_uwm": "Basic_WM",
        }

        if "mc" in x["method"]:
            return (
                f"{method_map[x['method']]}(n={x['n']}),{reweight_map[x['reweight']]}"
            )
        else:
            return f"{method_map[x['method']]},{reweight_map[x['reweight']]}"

    atn_ds = atn_ds.with_columns(
        label=pl.struct(["n", "method", "reweight"]).map_elements(
            lambda x: format_label(x)
        )
    )
    utils.large_print(atn_ds)
    utils.large_print(ptlpv_ds)

    import matplotlib.pyplot as plt

    plt.errorbar(
        atn_ds.get_column("atn_mu"),
        -np.array(ptlpv_ds.get_column("ptlpv_mu")),
        xerr=atn_ds.get_column("atn_std"),
        yerr=ptlpv_ds.get_column("ptlpv_std"),
        #  label=list(atn_ds.get_column("label")),
        fmt="o",
    )
    for i, label in enumerate(atn_ds.get_column("label")):
        x = atn_ds.get_column("atn_mu")[i]
        y = -ptlpv_ds.get_column("ptlpv_mu")[i]
        plt.text(x, y, label)
        #  plt.annotate(
        #      label,
        #      (x, y),
        #      textcoords="offset points",
        #      xytext=(5, 5),
        #      ha="right",
        #      va="bottom",
        #  )

    #  plt.legend()
    plt.xlabel("ATN")
    plt.ylabel("PTS")
    plt.title("Different methods")
    plt.savefig("figures/test.png", bbox_inches="tight")
    #
    #
    #  #  group_params = ["method", "reweight", "n"]
    #  group_params = ["n", "method", "reweight"]
    #
    #  ds = ds.sort(by=group_params)
    #  print(ds)
    #  print(ds.schema)
    #  atn_ds = compute_atn(ds, group_params).sort(by=group_params)
    #  utils.large_print(atn_ds)
    #  ptt_ds = compute_ptt(ds, group_params).sort(by=group_params)
    #  utils.large_print(ptt_ds)
    #  ppl_ds = compute_ppl(ds, group_params).sort(by=group_params)
    #  utils.large_print(ppl_ds)
    #  ptlpv_ds = compute_ptlpv(ds, group_params).sort(by=group_params)
    #  utils.large_print(ptlpv_ds)


def plot1():
    tasks = ["summarization_scan_n", "oeg_scan_n"]
    models = ["llama-7b_llama-68m", "llama-13b_llama-68m"]
    for task, model in itertools.product(tasks, models):
        plot1_(task, model)


def plot1_(task, model):
    atn_ds, ptt_ds, ppl_ds, ptlpv_ds = summarize_ds(
        #  ["data_root", data_path, "summarization_scan_n", "llama-7b_llama-68m"]
        #  ["data_root", data_path, "oeg_scan_n", "llama-7b_llama-68m"]
        ["data_root", data_path, task, model]
    )

    for selected_reweight in ["deltagumbel", "gamma"]:
        if selected_reweight == "deltagumbel":
            all_score_types = ["RobustLLR", "DeltaGumbel_U"]
        else:
            all_score_types = ["RobustLLR", "Gamma_U"]
        for selected_score_type in all_score_types:

            def local_filter(ds):
                ds = ds.filter(
                    pl.col("reweight").is_in(["none", selected_reweight])
                ).drop("reweight")
                if "score_type" in ds.columns:
                    ds = ds.filter(
                        pl.col("score_type").is_in(["none", selected_score_type])
                    ).drop("score_type")
                ds = ds.drop([c for c in ds.columns if "sigma" in c])
                return ds

            ds = local_filter(atn_ds).join(
                local_filter(ptlpv_ds),
                on=["n", "method"],
                how="inner",
            )
            #  utils.large_print(ds)

            import matplotlib.pyplot as plt

            plt.rcParams.update(
                {
                    "font.size": 12,  # Default font size
                    "axes.titlesize": 14,  # Title font size
                    "axes.labelsize": 12,  # X and Y label font size
                    "xtick.labelsize": 10,  # X tick label font size
                    "ytick.labelsize": 10,  # Y tick label font size
                    "legend.fontsize": 10,  # Legend font size
                    "figure.figsize": [6, 4],  # Figure size (inches)
                    "lines.linewidth": 2,  # Line width
                    "lines.markersize": 6,  # Marker size
                    "axes.linewidth": 1,  # Axis line width
                    "grid.linestyle": "--",  # Grid line style
                    "grid.linewidth": 0.5,  # Grid line width
                    "savefig.dpi": 300,  # Figure DPI for saving
                    "savefig.bbox": "tight",  # Tight bounding box when saving
                    "axes.grid": True,  # Enable grid
                }
            )
            fig = plt.figure(figsize=(3.2, 3.2))
            ax = fig.add_subplot(111)

            #  \definecolor{basic}{RGB}{0,0,0}
            #  \definecolor{basic_uwm}{RGB}{255,165,0}
            #  \definecolor{mc}{RGB}{50,205,50}
            #  \definecolor{mc_uwm_strength}{RGB}{0,0,255}
            #  \definecolor{mc_uwm_speed}{RGB}{148,0,211}
            color_map = {
                "basic": "black",
                "basic_uwm": "orange",
                "mc": "green",
                "mc_uwm_strength": "blue",
                "mc_uwm_speed": "purple",
            }
            marker_map = {
                1: "o",
                2: "s",
                3: "D",
                4: "X",
            }

            for i, row in enumerate(ds.to_pandas().itertuples()):
                plt.errorbar(
                    row.atn_mu,
                    -row.ptlpv_mu,
                    xerr=3 * row.atn_std,
                    yerr=3 * row.ptlpv_std,
                    label=row.method,
                    fmt=marker_map[row.n],
                    color=color_map[row.method],
                    clip_on=False,
                    #  markerfacecolor="none",
                    markerfacecolor="white",
                    markeredgecolor=color_map[row.method],
                    zorder=5,
                )

            # Create custom legend handles
            legend_elements_n = []
            legend_elements_method = []
            from matplotlib.lines import Line2D

            for n in marker_map:
                legend_elements_n.append(
                    Line2D(
                        [0],
                        [0],
                        marker=marker_map[n],
                        markersize=5,
                        markerfacecolor="white",
                        markeredgecolor="black",
                        label=f"K={n}",
                        linestyle="None",
                    )
                )
            name_map = {
                "basic": "Basic",
                "mc": "VSpS",
                "basic_uwm": "VUW",
                "mc_uwm_speed": "MSE",
                "mc_uwm_strength": "MWS",
            }
            for method in color_map:
                legend_elements_method.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        markersize=5,
                        markerfacecolor="white",
                        markeredgecolor=color_map[method],
                        label=name_map[method],
                        linestyle="None",
                    )
                )

            # Add the legend
            legend1 = ax.legend(handles=legend_elements_n, loc="lower left")
            legend2 = ax.legend(
                handles=legend_elements_method,
                loc="lower left",
                bbox_to_anchor=(0.0, 0.4),
            )
            ax.add_artist(legend1)

            plt.ylim(0, None)
            plt.xlim(1, None)
            #  plt.xlabel("Average Accepted Tokens Per Step")
            #  plt.ylabel("Average Negative Log P-value Per Token")
            plt.xlabel("AATPS")
            score_name_map = {
                "RobustLLR": "maximin-LLR",
                "DeltaGumbel_U": "U Score",
                "Gamma_U": "U Score",
            }
            plt.ylabel(f"ANLPPT({score_name_map[selected_score_type]})")
            plt.savefig(
                f"figures/plot1_{task}_{model}_{selected_reweight}_{selected_score_type}.pdf",
                bbox_inches="tight",
            )
            plt.show()


def table1():
    tasks = ["summarization_scan_n", "oeg_scan_n"]
    models = ["llama-7b_llama-68m", "llama-13b_llama-68m"]
    for task, model in itertools.product(tasks, models):
        table1_(task, model)


def table1_(task, model):
    atn_ds, ptt_ds, ppl_ds, ptlpv_ds = summarize_ds(
        #  ["data_root", data_path, "summarization_scan_n", "llama-7b_llama-68m"]
        #  ["data_root", data_path, "oeg_scan_n", "llama-7b_llama-68m"]
        ["data_root", data_path, task, model]
    )
    summary_ds = atn_ds.join(
        ptt_ds,
        on=["n", "method", "reweight"],
        how="inner",
    ).join(
        ppl_ds,
        on=["n", "method", "reweight"],
        how="inner",
    )
    new_ptlpv_ds = (
        ptlpv_ds.filter(
            pl.col("score_type").is_in(["DeltaGumbel_U", "Gamma_U", "RobustLLR"])
        )
        .with_columns(
            score_type=pl.when(
                pl.col("score_type").is_in(["DeltaGumbel_U", "Gamma_U"]),
            )
            .then(pl.lit("U_Score"))
            .otherwise(pl.col("score_type"))
        )
        .pivot(
            index=["n", "method", "reweight"],
            columns="score_type",
            values=["ptlpv_mu", "ptlpv_sigma", "ptlpv_std"],
        )
    )
    summary_ds = summary_ds.join(
        new_ptlpv_ds,
        on=["n", "method", "reweight"],
        how="left",
    ).fill_null(0)

    summary_ds = summary_ds.drop([c for c in summary_ds.columns if "sigma" in c])

    name_map = {
        "basic": "Basic",
        "mc": "VSpS",
        "basic_uwm": "VUW",
        "mc_uwm_speed": "MSE",
        "mc_uwm_strength": "MWS",
    }
    reweight_map = {
        "none": "No Reweight",
        "deltagumbel": "DeltaGumbel",
        "gamma": "Gamma",
    }
    summary_ds = summary_ds.with_columns(
        method=pl.col("method").replace(list(name_map.keys()), list(name_map.values())),
        reweight=pl.col("reweight").replace(
            list(reweight_map.keys()), list(reweight_map.values())
        ),
    )

    n_sigma_multiplier = 3
    summary_ds = summary_ds.with_columns(
        **{c: pl.col(c) * n_sigma_multiplier for c in summary_ds.columns if "std" in c}
    )
    #  ms PTT
    summary_ds = summary_ds.with_columns(
        ptt_mu=1e3 * pl.col("ptt_mu"),
        ptt_std=1e3 * pl.col("ptt_std"),
    )

    show_ds = summary_ds.with_columns(
        AATPS=pl.struct(
            mu=pl.col("atn_mu"),
            std=pl.col("atn_std"),
        ).map_elements(lambda x: utils.format_mu_std(**x)),
        PTT=pl.struct(
            mu=pl.col("ptt_mu"),
            std=pl.col("ptt_std"),
        ).map_elements(lambda x: utils.format_mu_std(**x)),
        LOGPPL=pl.struct(
            mu=pl.col("log_ppl_mu"),
            std=pl.col("log_ppl_std"),
        ).map_elements(lambda x: utils.format_mu_std(**x)),
        U_Score=pl.struct(
            mu=-pl.col("ptlpv_mu_score_type_U_Score"),
            std=pl.col("ptlpv_std_score_type_U_Score"),
        ).map_elements(lambda x: utils.format_mu_std(**x)),
        RobustLLR=pl.struct(
            mu=-pl.col("ptlpv_mu_score_type_RobustLLR"),
            std=pl.col("ptlpv_std_score_type_RobustLLR"),
        ).map_elements(lambda x: utils.format_mu_std(**x)),
    ).drop(
        [
            "atn_mu",
            "atn_std",
            "ptt_mu",
            "ptt_std",
            "log_ppl_mu",
            "log_ppl_std",
            "ptlpv_mu_score_type_U_Score",
            "ptlpv_std_score_type_U_Score",
            "ptlpv_mu_score_type_RobustLLR",
            "ptlpv_std_score_type_RobustLLR",
        ]
    )
    #  utils.large_print(show_ds)
    latex_ds = summary_ds.with_columns(
        AATPS=pl.struct(
            mu=pl.col("atn_mu"),
            std=pl.col("atn_std"),
        ).map_elements(lambda x: utils.format_mu_std(**x, latex=True)),
        PTT=pl.struct(
            mu=pl.col("ptt_mu"),
            std=pl.col("ptt_std"),
        ).map_elements(lambda x: utils.format_mu_std(**x, latex=True)),
        LOGPPL=pl.struct(
            mu=pl.col("log_ppl_mu"),
            std=pl.col("log_ppl_std"),
        ).map_elements(lambda x: utils.format_mu_std(**x, latex=True)),
        U_Score=pl.struct(
            mu=-pl.col("ptlpv_mu_score_type_U_Score"),
            std=pl.col("ptlpv_std_score_type_U_Score"),
        ).map_elements(lambda x: utils.format_mu_std(**x, latex=True)),
        RobustLLR=pl.struct(
            mu=-pl.col("ptlpv_mu_score_type_RobustLLR"),
            std=pl.col("ptlpv_std_score_type_RobustLLR"),
        ).map_elements(lambda x: utils.format_mu_std(**x, latex=True)),
    ).drop(
        [
            "atn_mu",
            "atn_std",
            "ptt_mu",
            "ptt_std",
            "log_ppl_mu",
            "log_ppl_std",
            "ptlpv_mu_score_type_U_Score",
            "ptlpv_std_score_type_U_Score",
            "ptlpv_mu_score_type_RobustLLR",
            "ptlpv_std_score_type_RobustLLR",
        ]
    )

    def find_highest(xs, r_tol=1e-2):
        xs = np.array(xs)
        return xs >= (1 - r_tol) * xs.max()

    def bm_latex_code(latex_code):
        # if fomular
        if latex_code.startswith("$"):
            return f"$\\bm{{{latex_code[1:-1]}}}$"
        return f"\\textbf{{{latex_code}}}"

    def _do_bm(ds, v_col, bm_col):
        return ds.with_columns(
            **{
                v_col: pl.when(pl.col(bm_col) == True)
                .then(pl.lit("\\bm{") + pl.col(v_col) + pl.lit("}"))
                .otherwise(pl.col(v_col))
            }
        )

    for c in ["U_Score", "RobustLLR"]:
        highest_foreach_reweight = (
            summary_ds.select(["reweight", f"ptlpv_mu_score_type_{c}"])
            .group_by("reweight")
            .agg(highest=-pl.col(f"ptlpv_mu_score_type_{c}").min())
        )
        latex_ds = _do_bm(
            latex_ds.join(
                highest_foreach_reweight,
                on="reweight",
                how="inner",
            )
            .join(
                summary_ds.select(
                    ["n", "method", "reweight", f"ptlpv_mu_score_type_{c}"]
                ),
                on=["n", "method", "reweight"],
                how="inner",
            )
            .with_columns(
                is_highest=(
                    -pl.col(f"ptlpv_mu_score_type_{c}") >= 0.97 * pl.col("highest")
                )
                & (pl.col("reweight") != "No Reweight")
            ),
            c,
            "is_highest",
        ).drop(["highest", "is_highest", f"ptlpv_mu_score_type_{c}"])
    for c in ["AATPS"]:
        highest_foreach_n = (
            summary_ds.select(["n", f"atn_mu"])
            .group_by("n")
            .agg(highest=pl.col(f"atn_mu").max())
        )
        latex_ds = _do_bm(
            latex_ds.join(
                highest_foreach_n,
                on="n",
                how="inner",
            )
            .join(
                summary_ds.select(["n", "method", "reweight", f"atn_mu"]),
                on=["n", "method", "reweight"],
                how="inner",
            )
            .with_columns(is_highest=(pl.col(f"atn_mu") >= 0.99 * pl.col("highest"))),
            c,
            "is_highest",
        ).drop(["highest", "is_highest", f"atn_mu"])

    latex_ds = latex_ds.rename(
        {
            "U_Score": "ANLPPT(U Score)",
            "RobustLLR": "ANLPPT(maximin-LLR)",
            "n": "$K$",
        }
    )

    def add_n_sep(latex_code):
        lines = latex_code.split("\n")
        heads = [l.split("&")[0] for l in lines]
        out_lines = []
        prev_n = None
        for i, l in enumerate(lines):
            #  if is integer
            if heads[i].strip().isdigit() and heads[i] != prev_n:
                out_lines.append(r"\midrule")
                prev_n = heads[i]
            out_lines.append(l)
        return "\n".join(out_lines)

    #  utils.large_print(latex_ds)

    latex_code = ""
    latex_code += r"""\begin{table}[H]
\centering
"""

    latex_code += add_n_sep(
        latex_ds.drop(["ANLPPT(U Score)", "ANLPPT(maximin-LLR)"])
        .to_pandas()
        .to_latex(index=False)
    )
    latex_code += "\n\n"
    latex_code += add_n_sep(
        latex_ds.drop(["AATPS", "PTT", "LOGPPL"]).to_pandas().to_latex(index=False)
    )

    def make_caption(task, model):
        task_map = {
            "summarization_scan_n": "Text summarization",
            "oeg_scan_n": "Open-ended text generation",
        }
        model_map = {
            "llama-7b_llama-68m": "LLaMa-7b",
            "llama-13b_llama-68m": "LLaMa-13b",
        }
        return rf"{task_map[task]} task with {model_map[model]} model \cite{{touvron2023llama}} as target model and LLaMa-68m model \cite{{miao2023specinfer}} as reference model."

    latex_code += rf"""\caption{{{make_caption(task, model)}}}\label{{tab:table1_{task}_{model}}}
\end{{table}}
"""

    with open(f"tables/table1_{task}_{model}.tex", "w") as f:
        f.write(latex_code)


if __name__ == "__main__":
    #  test1()
    #  test2()
    #  test_plot()
    plot1()
    table1()
