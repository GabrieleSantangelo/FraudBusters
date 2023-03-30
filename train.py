from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import make_scorer, fbeta_score
from autogluon.core.metrics import make_scorer as ag_make_scorer

train_data = TabularDataset("https://datafraudhacathon.s3.eu-central-1.amazonaws.com/fraudTrain.csv")

save_path="fraud"
label = "is_fraud"

def fbeta_score_custom(y_true,y_pred):
    return fbeta_score(y_true,y_pred, beta=0.5)

ag_f05_scorer = ag_make_scorer(name="f0.5", score_func=fbeta_score_custom, optimum=1, greater_is_better=True)

predictor = TabularPredictor(label=label, problem_type='binary', eval_metric=ag_f05_scorer).fit(train_data,time_limit=3600)

result = predictor.fit_summary(show_plot=False)


