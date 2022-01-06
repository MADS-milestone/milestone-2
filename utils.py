import pandas as pd
import altair as alt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, \
    precision_recall_curve


def plot_metrics(fpr, tpr, prec, rec):
    roc_df = pd.DataFrame(data={'fpr': fpr, 'tpr': tpr})
    prec_rec_df = pd.DataFrame(data={'prec': prec, 'rec': rec})
    line_df = pd.DataFrame({'fpr': [0, 1], 'tpr': [0, 1]})

    roc_ch = alt.Chart(data=roc_df, title='ROC curve').mark_line().encode(
        x=alt.X('fpr:Q', title='False Positive Rate'),
        y=alt.Y('tpr:Q', title='True Positive Rate')
    ).properties(width=250, height=300)

    line_ch = alt.Chart(data=line_df).mark_line(strokeWidth=0.5, strokeDash=[5, 5]).encode(
        x=alt.X('fpr:Q'),
        y=alt.Y('tpr:Q')
    )

    prec_rec_ch = alt.Chart(data=prec_rec_df, title='Precision-Recall curve').mark_line().encode(
        x=alt.X('prec:Q', title='Precision'),
        y=alt.Y('rec:Q', title='Recall')
    ).properties(width=250, height=300)

    ((roc_ch + line_ch) | prec_rec_ch).display()


def plot_feature_importance(clsf_type, cols, feat_imp):
    feat_imp_df = pd.DataFrame(data={'columns': cols, 'Feature importance': feat_imp})
    ch = alt.Chart(feat_imp_df, title=f'{clsf_type} feature importance').mark_bar().encode(
        x=alt.X('Feature importance:Q'),
        y=alt.Y('columns:N', title=None)
    )
    ch.display()


def get_classifier_summary(clsf_type, clsf, X_test, y_test):

    plot_feature_importance(clsf_type=clsf_type, cols=X_test.columns, feat_imp=clsf.feature_importances_)

    y_predicted = clsf.predict(X_test)
    confusion = confusion_matrix(y_test, y_predicted)
    confusion_df = pd.DataFrame(data=confusion, columns=['Positive (predicted)', 'Negative (predicted)'],
                                index=['Positive (actual)', 'Negative (actual)'])
    print('Confusion matrix')
    display(confusion_df)

    y_pred_proba = clsf.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    metrics_df = pd.DataFrame(data={'Accuracy': [accuracy_score(y_test, y_predicted)],
                                    'Precision': [precision_score(y_test, y_predicted)],
                                    'Recall': [recall_score(y_test, y_predicted)],
                                    'F1 score': [f1_score(y_test, y_predicted)],
                                    'ROC AUC': [roc_auc]})

    print('Classifier metrics')
    display(metrics_df.round(2))

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    plot_metrics(fpr=fpr, tpr=tpr, prec=precision, rec=recall)