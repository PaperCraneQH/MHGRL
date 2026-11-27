"""Logging, seeding, and regression metrics adapted from GraphDTA."""
import os
import random
import fcntl
from datetime import datetime, timedelta
from math import sqrt

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LinearRegression


class Logger:
    """Write messages to the terminal and an optional log file."""

    def __init__(self, log_file=None):
        self.log_file = log_file
        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self._fh = open(log_file, 'a', encoding='utf-8')
        else:
            self._fh = None

    def _ts(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def log(self, msg='', with_ts=True, to_terminal=True):
        """Write one message; set to_terminal=False for file-only logging."""
        line = f'[{self._ts()}] {msg}' if with_ts else msg
        if to_terminal:
            print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + '\n')
            self._fh.flush()

    def section(self, title, char='=', width=80, to_terminal=True):
        bar = char * width
        self.log('', with_ts=False, to_terminal=to_terminal)
        self.log(bar, with_ts=False, to_terminal=to_terminal)
        self.log(f'{title:^{width}}', with_ts=False, to_terminal=to_terminal)
        self.log(bar, with_ts=False, to_terminal=to_terminal)

    def subsection(self, title, char='-', width=80, to_terminal=True):
        bar = char * width
        self.log('', with_ts=False, to_terminal=to_terminal)
        self.log(bar, with_ts=False, to_terminal=to_terminal)
        self.log(title, with_ts=False, to_terminal=to_terminal)
        self.log(bar, with_ts=False, to_terminal=to_terminal)

    def close(self):
        if self._fh is not None:
            self._fh.close()


def fmt_secs(s):
    return str(timedelta(seconds=int(s)))


def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))


def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    return stats.spearmanr(y, f)[0]


class Evaluate:
    """Equivalent to the original GraphDTA Evaluate: RMSE / MAE / SD / Pearson r."""

    def _rmse(self, y, p):
        return sqrt(((y - p) ** 2).mean(axis=0))

    def _mae(self, y, p):
        return (np.abs(y - p)).mean()

    def _sd(self, y, p):
        p, y = p.reshape(-1, 1), y.reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(p, y)
        y_ = lr.predict(p)
        return (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5

    def _pearson(self, y, p):
        return np.corrcoef(y, p)[0, 1]

    def evaluate(self, y_true, pred):
        return self._rmse(y_true, pred), self._mae(y_true, pred), \
               self._pearson(y_true, pred), self._sd(y_true, pred)


def cal_final_results(result_file, n_repeat, logger=None, *, std_ddof=0):
    """Atomically add one ``avg`` and ``std`` row to a repeat-result CSV.

    Lock the results file across workers and replace existing summary rows
    so repeated aggregation does not duplicate them.
    """
    metrics = ['rmse', 'mae', 'sd', 'r']
    lock_path = result_file + '.lock'
    with open(lock_path, 'a', encoding='utf-8') as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            res = pd.read_csv(result_file)
            if 'repeat' not in res.columns:
                raise ValueError(f'missing repeat column: {result_file}')
            repeat_as_text = res['repeat'].astype(str).str.strip()
            repeats = res[~repeat_as_text.isin(['avg', 'std'])].copy()
            repeats['_repeat_id'] = pd.to_numeric(
                repeats['repeat'], errors='coerce')
            repeats = repeats.dropna(subset=['_repeat_id'])
            repeats['_repeat_id'] = repeats['_repeat_id'].astype(int)
            repeats = repeats.drop_duplicates('_repeat_id', keep='last')
            repeats = repeats.set_index('_repeat_id')
            expected = list(range(int(n_repeat)))
            missing = [i for i in expected if i not in repeats.index]
            if missing:
                raise ValueError(
                    f'cannot aggregate incomplete result {result_file}; '
                    f'missing repeats={missing}')

            ordered = repeats.loc[expected].reset_index(drop=True)
            res_ls = {
                metric: [float(value) for value in ordered[metric].tolist()]
                for metric in metrics
            }
            avg_res = [round(float(np.mean(res_ls[m])), 4) for m in metrics]
            std_res = [round(float(np.std(res_ls[m], ddof=std_ddof)), 4)
                       if len(res_ls[m]) > std_ddof else float('nan')
                       for m in metrics]
            output = ordered[['repeat'] + metrics].copy()
            output.loc[len(output)] = ['avg'] + avg_res
            output.loc[len(output)] = ['std'] + std_res
            temporary = result_file + '.tmp'
            output.to_csv(temporary, index=False)
            os.replace(temporary, result_file)
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)

    if logger is not None:
        logger.subsection(f'Aggregated over {n_repeat} repeats  ->  {os.path.basename(result_file)}')
        logger.log(f'  {"metric":<8}{"mean":>12}{"std":>12}', with_ts=False)
        for m, a, s in zip(metrics, avg_res, std_res):
            logger.log(f'  {m:<8}{a:>12.4f}{s:>12.4f}', with_ts=False)
