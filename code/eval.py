import numpy as np
from docopt import docopt
from scipy.stats import spearmanr
import os


def get_ys(model_answers, true_answers):
    """
    :param model_answers: path to tab-separated answer file (lemma + "\t" + score)
    :param true_answers: path to tab-separated gold answer file (lemma + "\t" + score)
    :return: a numpy array for the model scores, and one for the true scores
    """
    y_hat_tmp = {}
    errors = 0
    with open(model_answers, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            lemma, score = line.strip().split('\t')
            if score == 'nan':
                errors += 1
            y_hat_tmp[lemma] = score
    if errors:
        print('Found %d NaN predictions' % errors)
    y_hat, y = [], []
    with open(true_answers, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            lemma, score = line.strip().split('\t')
            y.append(float(score))
            y_hat.append(float(y_hat_tmp[lemma]))

    return np.array(y_hat), np.array(y)


def eval_task2(model_answers, true_answers):
    """
    Computes the Spearman's correlation coefficient against the true rank as annotated by humans
    :param model_answers: path to tab-separated answer file (lemma + "\t" + score)
    :param true_answers: path to tab-separated gold answer file (lemma + "\t" + score)
    :return: (Spearman's correlation coefficient, p-value)
    """
    y_hat, y = get_ys(model_answers, true_answers)
    r, p = spearmanr(y_hat, y, nan_policy='omit')

    return r, p


def main():
    """
    Evaluate lexical semantic change detection results.
    """

    # Get the arguments
    args = docopt("""Evaluate lexical semantic change detection results.

    Usage:
        eval.py <modelAnsPath> <trueAnsPath>

    Arguments:
        <modelAnsPath> = path to tab-separated answer file for Task (lemma + "\t" + change score)
        <trueAnsPath> = path to tab-separated gold answer file for Task (lemma + "\t" + change score)

    """)

    modelAnsPath = args['<modelAnsPath>']
    trueAnsPath = args['<trueAnsPath>']

    if os.path.isfile(modelAnsPath):
        r, p = eval_task2(modelAnsPath, trueAnsPath)
        print('Spearman rho score: {:.3f}  p: {:.3f}'.format(r, p))
    else:
        print('Task predictions not found!')


if __name__ == '__main__':
    main()
