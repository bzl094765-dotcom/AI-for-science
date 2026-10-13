"""R0 per-fold AP/HL/RL/AUC and mean±sd. Rγ is not run here."""
import csv
import os

BASE = os.path.dirname(os.path.abspath(__file__))
R0_CSV = os.path.join(BASE, 'metrics_R0.csv')
OUT = os.path.join(BASE, 'P7_summary.md')
# RANK's test log prints evaluation_results[3] as AUC. That value is stored as AUCme.
COLUMNS = [('AP', 'AP'), ('HL', 'HL'), ('RL', 'RL'), ('AUC', 'AUCme')]


def _read(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    rows = []
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row or row.get('fold_idx') in (None, ''):
                continue
            rows.append(row)
    rows.sort(key=lambda row: int(row['fold_idx']))
    return rows


def _mean_sd(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, var ** 0.5


def main():
    rows = _read(R0_CSV)
    lines = [
        '# P7 summary',
        '',
        'dataset: corel5k',
        'mask_view_ratio: 0.5',
        'mask_label_ratio: 0.5',
        'training_sample_ratio: 0.7',
        'epochs: 250',
        'folds_requested: 10',
        'arm: R0',
        'n_completed_R0: %d' % len(rows),
        'Rgamma: run locally, pending',
        'AUC: RANK test-log AUC (records column AUCme, evaluation_results[3])',
        'sd: population std, ddof=0, same divisor as RANK AverageMeter',
        '',
    ]
    if not rows:
        lines.append('status: no_completed_R0_folds')
        lines.append('')
    else:
        lines.append('| fold_idx | AP | HL | RL | AUC |')
        lines.append('| --- | ---: | ---: | ---: | ---: |')
        for row in rows:
            lines.append('| %s | %.6f | %.6f | %.6f | %.6f |' % (
                row['fold_idx'],
                float(row['AP']),
                float(row['HL']),
                float(row['RL']),
                float(row['AUCme']),
            ))
        means = []
        for label, key in COLUMNS:
            mean, sd = _mean_sd([float(row[key]) for row in rows])
            means.append((label, mean, sd))
        lines.append('| mean±sd | %s | %s | %s | %s |' % tuple(
            '%.6f±%.6f' % (mean, sd) for _, mean, sd in means
        ))
        lines.append('')
    text = '\n'.join(lines)
    with open(OUT, 'w') as handle:
        handle.write(text)
    print(text)


if __name__ == '__main__':
    main()
