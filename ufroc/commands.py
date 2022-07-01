import os
from collections import defaultdict

from tqdm import tqdm
from dpipe.commands import load_from_folder
from dpipe.io import save_json

from ufroc.utils import np_sigmoid


def metric_probably_with_extra_attr(true, pred, _attr, metric):
    try:
        return metric(true, pred, _attr)
    except TypeError:
        return metric(true, pred)


def evaluate_individual_metrics_probably_with_froc(load_y, metrics, logits_path, results_path, activation=np_sigmoid):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=False)

    results = defaultdict(dict)
    for _id, logit in tqdm(load_from_folder(logits_path)):
        true = load_y(_id)
        pred = activation(logit)

        for metric_name, metric in metrics.items():
            results[metric_name][_id] = metric_probably_with_extra_attr(true, logit, pred, metric)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)
