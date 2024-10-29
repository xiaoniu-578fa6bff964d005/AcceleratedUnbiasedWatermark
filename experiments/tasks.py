from datasets import load_dataset
import os


#  def get_translation_ds():
#      wmt16 = load_dataset("wmt16", "ro-en")
#      ds = wmt16["test"]
#      ds = ds.select(range(0, 1000))
#      ds = exp_debug_cut(ds)
#      ds = ds.flatten()
#
#      def create_prompt(d, idx):
#          s = {}
#          s["idx"] = idx
#          #  s["prompt"] = "Translate from English to Romanian: " + d["translation.en"]
#          s[
#              "prompt"
#          ] = f"System:Translate from English to Romanian.\nINPUT:{d['translation.en']}\nOUTPUT:"
#          return s
#
#      ds = ds.map(create_prompt, with_indices=True, remove_columns=ds.column_names)
#      return ds


def get_summarization_ds(ds_cut_len=None):
    cnn_daily = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)
    ds = cnn_daily["test"]
    ds = ds.filter(lambda x: len(x["article"]) < 3000)
    ds = ds.select(range(0, 1000))
    if ds_cut_len is not None:
        ds = ds.select(range(0, ds_cut_len))

    def create_prompt(d, idx):
        s = {}
        s["idx"] = idx
        #  s["prompt"] = d["article"] + "\nTL;DR:"
        #  s["prompt"] = d["article"][:1000] + "\nTL;DR:\n"
        s[
            "prompt"
        ] = f"System:Summarize the following article.\nINPUT:{d['article'][:1000]}\nOUTPUT:"
        #  s["prompt"] = d["article"][:1000] + "\nRe-type (copy the above word by word):\n"
        return s

    ds = ds.map(create_prompt, with_indices=True, remove_columns=ds.column_names)
    return ds


def get_oeg_ds(ds_cut_len=None):
    cnn_daily = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)
    ds = cnn_daily["test"]
    ds = ds.select(range(0, 1000))
    if ds_cut_len is not None:
        ds = ds.select(range(0, ds_cut_len))

    def smart_truncate(content, length):
        if len(content) <= length:
            return content
        else:
            return " ".join(content[: length + 1].split(" ")[0:-1])

    def create_prompt(d, idx):
        s = {}
        s["idx"] = idx
        s["prompt"] = smart_truncate(d["article"], length=100)
        return s

    ds = ds.map(create_prompt, with_indices=True, remove_columns=ds.column_names)
    return ds
