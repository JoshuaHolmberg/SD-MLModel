# Serial Dependence in Dermatological Judgment

**ML analysis from the Whitney Lab for Perception and Action, UC Berkeley**

Investigates whether observers exhibit *serial dependence* when classifying skin lesions — specifically, whether they systematically fail to update their response when the stimulus category changes from one trial to the next. Features are drawn from three data modalities: eye tracking (pupil dilation), mouse tracking, and behavioral responses.

---

## Research Context

Serial dependence is a well-documented perceptual phenomenon: prior stimuli bias current judgments, even when participants are unaware of the effect. This project asks whether serial dependence persists in a *high-stakes* classification task — distinguishing malignant from benign lesions — and what signals best predict when it occurs.

The analysis restricts to **different-category trials** (where the current stimulus belongs to a different category than the previous one), isolating cases where serial dependence constitutes a classification error.

**Target variable:** `sd_event` — binary indicator that a participant gave the same response as their previous trial despite the stimulus category having changed (i.e., failed to switch).

---

## Data

Multi-modal experimental data collected from 50+ participants across multiple sessions:

| Modality | Features |
|---|---|
| **Eye tracking** | Pupil diameter time-series during inter-trial interval (ITI) and stimulus period |
| **Mouse tracking** | Response time from stimulus onset |
| **Behavioral** | Response (malignant/benign), confidence (high/low), correctness, trial and block position |
| **Stimulus** | Cosine similarity of current vs. previous lesion image |

---

## Feature Engineering

### Pupil-derived features (computed separately for ITI and trial epochs)

| Feature | Description |
|---|---|
| `_mean` | Mean pupil diameter |
| `_std` | Standard deviation |
| `_range` | Max minus min |
| `_slope` | Linear trend over epoch |
| `_early_mean` / `_late_mean` | Mean of first / last third of epoch |
| `_dilation` | Late mean minus early mean (net dilation) |
| `pupil_task_vs_iti` | Trial mean minus ITI mean (task-evoked change) |

### Behavioral and stimulus features

| Feature | Description |
|---|---|
| `cosine_similarity` | Visual similarity between consecutive stimuli |
| `prev_confidence` | Participant's confidence on the previous trial |
| `prev_correct` | Whether the previous response was correct |
| `prev_malignant` | Previous stimulus category |
| `rt_s` | Response time in seconds |
| `iti_duration_s` | Inter-trial interval length |
| `trial_num` / `block_num` | Position within session |

---

## Models

Three classifiers evaluated with **StratifiedKFold (5-fold) cross-validation**, scored by ROC-AUC and accuracy:

| Model | Configuration |
|---|---|
| Logistic Regression | L2 penalty, C=0.1, class_weight='balanced' |
| Random Forest | 300 trees, max_depth=6, class_weight='balanced' |
| Gradient Boosting | 300 estimators, max_depth=3, lr=0.05, subsample=0.8 |

Feature importance estimated via **permutation importance** (20 repeats) on the best-performing model.

---

## Supplementary Analysis

- **Cosine similarity vs. SD rate:** Mann-Whitney U test comparing cosine similarity distributions between SD and non-SD trials; Spearman correlation within each transition direction
- **Transition direction effects:** SD rates compared between Malignant→Benign and Benign→Malignant transitions
- **Participant-level SD rates:** Individual variation in serial dependence frequency
- **SD rate by cosine quartile:** Whether visual similarity between consecutive stimuli predicts the probability of serial dependence

---

## Visualizations

The notebook produces a 4×3 figure panel including:
- (A) SD rate per participant
- (B) Cosine similarity distributions by SD outcome
- (C) SD rate by cosine similarity quartile
- (D) Transition direction comparison
- (E) ROC curves for all three models
- (F) Model AUC comparison
- (G–L) Top feature importances and pupil dilation analysis

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
```

```bash
pip install -r requirements.txt
```

The notebook expects `all_data_et.pkl` — the preprocessed experimental dataset including eye tracking time-series. Run `MLM_SerialDependence.ipynb` from top to bottom.

---

*Research conducted at the Whitney Lab for Perception and Action, UC Berkeley, 2025–2026.*
