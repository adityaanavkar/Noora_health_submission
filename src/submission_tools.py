"""Shared data loading, optional training, and plotting for SUBMISSION.ipynb.

All paths resolve within this src directory. Existing experiment implementations
are retained under experiments so their relative predecessor paths still work.
"""
from pathlib import Path
import importlib.util
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.calibration import calibration_curve

SRC = Path(__file__).resolve().parent
DATA = SRC / 'intent-classification-assignment' / 'intent-classification-assignment'
MODEL_DIRS = {
    'LR': SRC / 'experiments' / 'version 1' / 'part2',
    'RF': SRC / 'experiments' / 'version 2',
    'MLP': SRC / 'experiments' / 'version 3',
}
SYSTEMS = {'LR': 'tuned_lr', 'RF': 'tuned_rf', 'MLP': 'tuned_mlp'}
POLICIES = ['default_0_5', 'balanced_macro_f1', 'recall_first']


def table(model, filename):
    return pd.read_csv(MODEL_DIRS[model] / 'outputs' / filename)


def summary(model):
    return json.loads((MODEL_DIRS[model] / 'outputs' / 'summary.json').read_text(encoding='utf-8'))


def training_data():
    train = pd.read_csv(DATA / 'train_queries.csv')
    test = pd.read_csv(DATA / 'test_queries.csv')
    embeddings = {name.upper(): np.load(DATA / f'train_embeddings_{name}.npy', mmap_mode='r') for name in 'abc'}
    return train, test, embeddings


def train_models():
    """Explicit full retraining; never called by default. LR -> RF -> MLP."""
    for name, directory in MODEL_DIRS.items():
        print(f'Starting {name}; this runs the complete nested search.', flush=True)
        module_name = f'submission_training_{name.lower()}'
        spec = importlib.util.spec_from_file_location(module_name, directory / 'src' / 'experiment.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if name == 'MLP':
            # The legacy callback expects the old repository's reporting layout.
            # This combined notebook performs reporting directly from outputs.
            module._call_hubble_reporting = lambda root: {'status': 'handled_by_combined_notebook'}
            module.run_experiment(directory, pilot=False)
        else:
            module.run_experiment(directory, force_recompute=True)
        print(f'{name} finished; results saved under src/experiments.', flush=True)


def comparison():
    frames = []
    for name in MODEL_DIRS:
        frame = table(name, 'metrics.csv')
        frame = frame[(frame.system == SYSTEMS[name]) & (frame.evaluation == 'outer_oof_aggregate') & frame.policy.isin(POLICIES)].copy()
        frame.insert(0, 'model', name)
        frames.append(frame)
    llm = table('RF', 'metrics.csv')
    llm = llm[(llm.system == 'llm_train') & (llm.evaluation == 'outer_oof_aggregate')].copy()
    llm.insert(0, 'model', 'LLM')
    frames.append(llm)
    return pd.concat(frames, ignore_index=True)


def pca_plot(train, embeddings):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, matrix) in zip(axes, embeddings.items()):
        reducer = PCA(n_components=2, random_state=42)
        points = reducer.fit_transform(matrix)
        for label, color in [('irrelevant', '#c77b35'), ('relevant', '#197a87')]:
            mask = train.label.eq(label).to_numpy()
            ax.scatter(points[mask, 0], points[mask, 1], s=12, alpha=.5, label=label, color=color)
        ax.set(title=f'{name}: {reducer.explained_variance_ratio_.sum():.1%} variance shown', xlabel='PCA component 1', ylabel='PCA component 2')
    axes[0].legend()
    fig.tight_layout()
    return fig


def model_section(name):
    """One consistent compact results section per local model."""
    from IPython.display import display, Markdown
    result = summary(name)
    display(Markdown(f'**{name}: final settings selected using development data**'))
    display(pd.Series(result['selection'], name='Saved value').to_frame())
    rows = comparison()
    rows = rows[rows.model.eq(name)]
    display(rows[['policy', 'accuracy', 'precision', 'recall', 'macro_f1', 'fn', 'fp', 'medical_fn', 'referrals']].style.format({key: '{:.2%}' for key in ['accuracy', 'precision', 'recall', 'macro_f1']}))
    sweep = table(name, 'threshold_sweep.csv').sort_values('threshold')
    if 'selection_scope' in sweep:
        sweep = sweep[sweep.selection_scope.eq('final_all_training')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for column, label in [('precision', 'Precision'), ('relevant_recall', 'Relevant recall'), ('medical_recall', 'Medical recall')]:
        axes[0].plot(sweep.threshold, sweep[column], label=label)
    axes[0].set(title=f'{name}: inner-development threshold trade-offs', xlabel='Threshold', ylabel='Rate', ylim=(0, 1.04))
    axes[0].legend()
    referrals = sweep.referrals if 'referrals' in sweep else sweep.tp + sweep.fp
    axes[1].plot(sweep.threshold, referrals, label='Total referrals')
    axes[1].plot(sweep.threshold, sweep.fn, label='Relevant missed')
    axes[1].plot(sweep.threshold, sweep.fp, label='Unnecessary referrals')
    axes[1].set(title='Counts in development data', xlabel='Threshold', ylabel='Queries')
    axes[1].legend()
    fig.tight_layout()
    plt.show()
    baseline = rows[rows.policy.eq('default_0_5')].iloc[0]
    cautious = rows[rows.policy.eq('recall_first')].iloc[0]
    display(Markdown(f"Compared with default 0.50, recall-first changes referrals by **{int(cautious.referrals - baseline.referrals):+d}**, relevant misses by **{int(cautious.fn - baseline.fn):+d}**, and Medical misses by **{int(cautious.medical_fn - baseline.medical_fn):+d}**, across the 903 held-out development predictions."))


def ranking_and_calibration():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    scores = []
    for name in MODEL_DIRS:
        outer = table(name, 'outer_predictions.csv')
        y, p = outer.truth.to_numpy(), outer.p_chosen.to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        observed, predicted = calibration_curve(y, p, n_bins=8, strategy='quantile')
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
        axes[2].plot(predicted, observed, 'o-', label=name)
        scores.append({'model': name, 'ROC AUC': roc_auc_score(y, p), 'Average precision': average_precision_score(y, p), 'Brier score': brier_score_loss(y, p)})
    axes[0].plot([0, 1], [0, 1], '--', color='grey')
    axes[2].plot([0, 1], [0, 1], '--', color='grey')
    for ax, title, x, y in zip(axes, ['ROC', 'Precision–recall', 'Calibration'], ['False-positive rate', 'Recall', 'Predicted probability'], ['Recall', 'Precision', 'Observed relevant fraction']):
        ax.set(title=title, xlabel=x, ylabel=y)
        ax.legend()
    fig.tight_layout()
    plt.show()
    return pd.DataFrame(scores)


def best_local(rows):
    return rows[rows.model.isin(MODEL_DIRS)].sort_values(['medical_fn', 'fn', 'fp', 'macro_f1'], ascending=[True, True, True, False], kind='stable').iloc[0]


def intent_results(choice):
    outer = table(choice.model, 'outer_predictions.csv')
    pred = outer['label_' + choice.policy].astype(int)
    frame = outer[['intent', 'truth']].copy()
    frame['missed_relevant'] = ((outer.truth == 1) & (pred == 0)).astype(int)
    frame['unnecessary_referrals'] = ((outer.truth == 0) & (pred == 1)).astype(int)
    frame['referred'] = pred
    result = frame.groupby('intent').agg(rows=('truth', 'size'), relevant=('truth', 'sum'), missed_relevant=('missed_relevant', 'sum'), unnecessary_referrals=('unnecessary_referrals', 'sum'), referrals=('referred', 'sum'))
    result['relevant_recall'] = 1 - result.missed_relevant / result.relevant.replace(0, np.nan)
    return result


def export_predictions(choice):
    source = table(choice.model, 'test_predictions.csv')
    selected_summary = summary(choice.model)['selection']
    threshold = float(selected_summary['final_thresholds'][choice.policy])
    probabilities = source['p_chosen'] if 'p_chosen' in source else source['p_relevant']
    exported = pd.DataFrame({'query_id': source.query_id, 'p_relevant': probabilities, 'predicted_label': np.where(probabilities >= threshold, 'relevant', 'irrelevant'), 'model': choice.model, 'policy': choice.policy, 'threshold': threshold})
    destination = SRC / 'results'
    destination.mkdir(exist_ok=True)
    exported.to_csv(destination / 'recommended_test_predictions.csv', index=False)
    (destination / 'recommendation.json').write_text(json.dumps({'model': choice.model, 'policy': choice.policy, 'threshold': threshold, 'selection_rule': 'Medical FN, relevant FN, FP, then descending macro-F1', 'test_truth_available': False, 'model_artifact': str((MODEL_DIRS[choice.model] / 'outputs' / 'final_model.joblib').relative_to(SRC))}, indent=2), encoding='utf-8')
    return exported, destination / 'recommended_test_predictions.csv'
