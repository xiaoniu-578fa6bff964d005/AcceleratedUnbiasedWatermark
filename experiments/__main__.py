import ray

import os


def to_display_model_name(s):
    if "/" in s:
        s = s.split("/")[-1]
    s = s.replace("-chat-hf", "")
    return s


def estimate_fpass(config):
    n_fm1 = (
        1  # basic
        + len(config["ns"])  # mc
        + len(config["reweights"])  # basic_uwm
        + len(config["ns"]) * len(config["reweights"])  # mc_uwm_strength
        + len(config["ns"]) * len(config["reweights"])  # mc_uwm_speed
    )
    n_fm2 = len(config["seeds"])
    return config["ds_cut_len"] * n_fm1 * n_fm2


def estimate_time(config):
    fpass = estimate_fpass(config)
    # for max_length=128, a6000, 4.6s per fpass
    t = fpass * 4.6
    return t


def run(config):
    from . import tasks
    from .worker import Worker

    if config["ds_name"] == "summarization":
        ds = tasks.get_summarization_ds(config.get("ds_cut_len", None))
    elif config["ds_name"] == "oeg":
        ds = tasks.get_oeg_ds(config.get("ds_cut_len", None))
    worker_param = {
        k: config[k]
        for k in [
            "model_str",
            "ref_model_str",
            "task",
            "device",
            "print_output",
            "assert_cch",
            "assert_log_p_values",
        ]
    }
    save_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data_root",
        config["data_folder"],
        config["task"],
        f"{to_display_model_name(config['model_str'])}_{to_display_model_name(config['ref_model_str'])}",
    )

    def enrich(d):
        ds = []
        if "basic" in config["methods"]:
            ds.append({**d, "method": "basic", "n": 1, "reweight": "none"})
        if "basic_uwm" in config["methods"]:
            ds.extend(
                [
                    {**d, "method": "basic_uwm", "n": 1, "reweight": reweight}
                    for reweight in config["reweights"]
                ]
            )
        if "mc" in config["methods"]:
            ds.extend(
                [
                    {**d, "method": "mc", "n": n, "reweight": "none"}
                    for n in config["ns"]
                ]
            )
        if "mc_uwm_strength" in config["methods"]:
            ds.extend(
                [
                    {**d, "method": "mc_uwm_strength", "n": n, "reweight": reweight}
                    for n in config["ns"]
                    for reweight in config["reweights"]
                ]
            )
        if "mc_uwm_speed" in config["methods"]:
            ds.extend(
                [
                    {**d, "method": "mc_uwm_speed", "n": n, "reweight": reweight}
                    for n in config["ns"]
                    for reweight in config["reweights"]
                ]
            )
        return ds

    outds = (
        ray.data.from_huggingface(ds)
        .map(
            lambda d: {
                **d,
                "max_length": config["max_length"],
                "private_key": bytes(config["private_key"], "utf-8"),
            }
        )
        .flat_map(lambda d: enrich(d))
        .flat_map(lambda d: [{**d, "seed": seed} for seed in config["seeds"]])
        .repartition(config["repartition_size"])
        .map_batches(
            Worker,
            batch_size=config["batch_size"],
            compute=ray.data.ActorPoolStrategy(),
            fn_constructor_kwargs={"param": worker_param},
            num_gpus=1,
            max_restarts=0,
            concurrency_groups={"model": 1},
        )
    )

    outds.write_parquet(save_path)


def main():
    ray.init()

    ray.data.DatasetContext.get_current().execution_options.preserve_order = True
    ray.data.DataContext.get_current().execution_options.verbose_progress = True

    import json, _jsonnet

    configs = json.loads(
        _jsonnet.evaluate_file(
            os.path.join(os.path.dirname(__file__), "config.jsonnet")
        )
    )
    for config in configs:
        seconds = estimate_time(config)
        print(
            f"Running {config['task']}. Estimated time: {seconds / 3600:.2f} GPU hours"
        )
        run(config)

    ray.shutdown()


def test_worker():
    model_str = "huggyllama/llama-7b"
    ref_model_str = "JackFram/llama-68m"
    import numpy as np

    np.seterr(all="raise")
    worker_param = {
        "model_str": model_str,
        "ref_model_str": ref_model_str,
        "task": "oeg_scan_n",
        "device": "cuda:0",
    }
    from .worker import Worker

    worker = Worker(param=worker_param)
    #  for method in ["basic", "mc", "basic_uwm", "mc_uwm_strength", "mc_uwm_speed"]:
    for seed in range(10):
        for method in ["mc", "mc_uwm_strength"]:
            if "uwm" in method:
                #  _rws = ["deltagumbel", "gamma"]
                _rws = ["deltagumbel"]
            else:
                _rws = ["none"]
            if "mc" in method:
                #  _ns = [1, 4]
                _ns = [1]
            else:
                _ns = [1]
            for reweight in _rws:
                for n in _ns:
                    r = worker.process(
                        {
                            "prompt": "Hello. This is a test",
                            "seed": seed,
                            "method": method,
                            "reweight": reweight,
                            "private_key": b"1234",
                            "n": n,
                            "max_length": 128,
                        }
                    )
                    #  print(r)


if __name__ == "__main__":
    if os.environ.get("EXP_DEBUG", None) == "0":
        test_worker()
        exit()
    main()
