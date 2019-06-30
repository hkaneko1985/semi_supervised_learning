# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import math
import warnings

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, svm, tree
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RBF, ConstantKernel
from sklearn.linear_model import Ridge, Lasso, ElasticNet, ElasticNetCV
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

regression_method = 'pls'  # 'pls' or 'rr' or 'lasso' or 'en' or 'lsvr' or 'nsvr' or 'dt' or 'rf' or 'gp'

max_pca_component_number = 150
threshold_of_rate_of_same_value = 1
fold_number = 2
max_pls_component_number = 30
ridge_lambdas = 2 ** np.arange(-5, 10, dtype=float)  # L2 weight in ridge regression
lasso_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)  # L1 weight in LASSO
elastic_net_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)  # Lambda in elastic net
elastic_net_alphas = np.arange(0.01, 1.00, 0.01, dtype=float)  # Alpha in elastic net
linear_svr_cs = 2 ** np.arange(-5, 5, dtype=float)  # C for linear svr
linear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Epsilon for linear svr
nonlinear_svr_cs = 2 ** np.arange(-5, 10, dtype=float)  # C for nonlinear svr
nonlinear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Epsilon for nonlinear svr
nonlinear_svr_gammas = 2 ** np.arange(-20, 10, dtype=float)  # Gamma for nonlinear svr
dt_max_max_depth = 30  # 木の深さの最大値、の最大値
dt_min_samples_leaf = 3  # 葉ごとのサンプル数の最小値
random_forest_number_of_trees = 300  # Number of decision trees for random forest
random_forest_x_variables_rates = np.arange(1, 10,
                                            dtype=float) / 10  # Ratio of the number of X-variables for random forest

# load data set
supervised_dataset = pd.read_csv('descriptors_with_logS.csv', encoding='SHIFT-JIS', index_col=0)
unsupervised_dataset = pd.read_csv('descriptors_for_prediction.csv', encoding='SHIFT-JIS', index_col=0)
number_of_supervised_samples = supervised_dataset.shape[0]
x_all_dataset = pd.concat([supervised_dataset.iloc[:, 1:], unsupervised_dataset], axis=0)
x_all_dataset = x_all_dataset.loc[:, x_all_dataset.mean().index]  # 平均を計算できる変数だけ選択
x_all_dataset = x_all_dataset.replace(np.inf, np.nan).fillna(np.nan)  # infをnanに置き換えておく
x_all_dataset = x_all_dataset.dropna(axis=1)  # nanのある変数を削除

y_train = supervised_dataset.iloc[:, 0]

rate_of_same_value = list()
num = 0
for X_variable_name in x_all_dataset.columns:
    num += 1
    #    print('{0} / {1}'.format(num, x_all_dataset.shape[1]))
    same_value_number = x_all_dataset[X_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x_all_dataset.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)

"""
# delete descriptors with zero variance
deleting_variable_numbers = np.where( raw_Xtrain.var() == 0 )
"""

if len(deleting_variable_numbers[0]) == 0:
    x_all = x_all_dataset.copy()
else:
    x_all = x_all_dataset.drop(x_all_dataset.columns[deleting_variable_numbers], axis=1)
    print('Variable numbers zero variance: {0}'.format(deleting_variable_numbers[0] + 1))
print('# of X-variables: {0}'.format(x_all.shape[1]))

# autoscaling      
autoscaled_x_all = (x_all - x_all.mean(axis=0)) / x_all.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0, ddof=1)
# PCA
pca = PCA()  # PCA を行ったり PCA の結果を格納したりするための変数を、pca として宣言
pca.fit(autoscaled_x_all)  # PCA を実行
# score
score_all = pd.DataFrame(pca.transform(autoscaled_x_all), index=x_all.index)  # 主成分スコアの計算した後、pandas の DataFrame 型に変換
score_train = score_all.iloc[:number_of_supervised_samples, :]
score_test = score_all.iloc[number_of_supervised_samples:, :]
# scaling      
autoscaled_score_train = score_train / score_train.std(axis=0, ddof=1)
autoscaled_score_test = score_test / score_train.std(axis=0, ddof=1)

# optimization of number of PCs
set_max_pca_component_number = min(np.linalg.matrix_rank(autoscaled_score_train), max_pca_component_number)
r2cvs = []
for number_of_pcs in range(set_max_pca_component_number):
    print('PC:', number_of_pcs + 1, '/', set_max_pca_component_number)
    autoscaled_x_train = autoscaled_score_train.iloc[:, :number_of_pcs + 1]
    if regression_method == 'pls':  # Partial Least Squares
        pls_components = np.arange(1, min(np.linalg.matrix_rank(autoscaled_x_train) + 1, max_pls_component_number + 1),
                                   1)
        r2cvall = []
        for pls_component in pls_components:
            pls_model_in_cv = PLSRegression(n_components=pls_component)
            estimated_y_in_cv = np.ndarray.flatten(
                model_selection.cross_val_predict(pls_model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                  cv=fold_number))
            estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
            r2cvall.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
        optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall))[0][0] + 1
        regression_model = PLSRegression(n_components=optimal_pls_component_number)
    elif regression_method == 'rr':  # ridge regression
        r2cvall = list()
        for ridge_lambda in ridge_lambdas:
            rr_model_in_cv = Ridge(alpha=ridge_lambda)
            estimated_y_in_cv = model_selection.cross_val_predict(rr_model_in_cv, autoscaled_x_train,
                                                                  autoscaled_y_train,
                                                                  cv=fold_number)
            estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
            r2cvall.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
        optimal_ridge_lambda = ridge_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
        regression_model = Ridge(alpha=optimal_ridge_lambda)
    elif regression_method == 'lasso':  # LASSO
        r2cvall = list()
        for lasso_lambda in lasso_lambdas:
            lasso_model_in_cv = Lasso(alpha=lasso_lambda)
            estimated_y_in_cv = model_selection.cross_val_predict(lasso_model_in_cv, autoscaled_x_train,
                                                                  autoscaled_y_train,
                                                                  cv=fold_number)
            estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
            r2cvall.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
        optimal_lasso_lambda = lasso_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
        regression_model = Lasso(alpha=optimal_lasso_lambda)
    elif regression_method == 'en':  # Elastic net
        elastic_net_in_cv = ElasticNetCV(cv=fold_number, l1_ratio=elastic_net_lambdas, alphas=elastic_net_alphas)
        elastic_net_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_elastic_net_alpha = elastic_net_in_cv.alpha_
        optimal_elastic_net_lambda = elastic_net_in_cv.l1_ratio_
        regression_model = ElasticNet(l1_ratio=optimal_elastic_net_lambda, alpha=optimal_elastic_net_alpha)
    elif regression_method == 'lsvr':  # Linear SVR
        linear_svr_in_cv = GridSearchCV(svm.SVR(kernel='linear'), {'C': linear_svr_cs, 'epsilon': linear_svr_epsilons},
                                        cv=fold_number)
        linear_svr_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_linear_svr_c = linear_svr_in_cv.best_params_['C']
        optimal_linear_svr_epsilon = linear_svr_in_cv.best_params_['epsilon']
        regression_model = svm.SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon)
    elif regression_method == 'nsvr':  # Nonlinear SVR
        variance_of_gram_matrix = list()
        numpy_autoscaled_Xtrain = np.array(autoscaled_x_train)
        for nonlinear_svr_gamma in nonlinear_svr_gammas:
            gram_matrix = np.exp(
                -nonlinear_svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain) ** 2).sum(
                    axis=2))
            variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
        optimal_nonlinear_gamma = nonlinear_svr_gammas[
            np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
        # CV による ε の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_nonlinear_gamma),
                                   {'epsilon': nonlinear_svr_epsilons},
                                   cv=fold_number, iid=False, verbose=0)
        model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_nonlinear_epsilon = model_in_cv.best_params_['epsilon']
        # CV による C の最適化
        model_in_cv = GridSearchCV(
            svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma),
            {'C': nonlinear_svr_cs}, cv=fold_number, iid=False, verbose=0)
        model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_nonlinear_c = model_in_cv.best_params_['C']
        # CV による γ の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, C=optimal_nonlinear_c),
                                   {'gamma': nonlinear_svr_gammas}, cv=fold_number, iid=False, verbose=0)
        model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_nonlinear_gamma = model_in_cv.best_params_['gamma']
        regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                                   gamma=optimal_nonlinear_gamma)
    elif regression_method == 'dt':  # Decision tree
        # クロスバリデーションによる木の深さの最適化
        r2cv_all = []
        for max_depth in range(2, dt_max_max_depth):
            model_in_cv = tree.DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=dt_min_samples_leaf)
            estimated_y_in_cv = model_selection.cross_val_predict(model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                                  cv=fold_number) * y_train.std(ddof=1) + y_train.mean()
            r2cv_all.append(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2))
        optimal_max_depth = np.where(r2cv_all == np.max(r2cv_all))[0][0] + 2  # r2cvが最も大きい木の深さ
        regression_model = tree.DecisionTreeRegressor(max_depth=optimal_max_depth,
                                                      min_samples_leaf=dt_min_samples_leaf)  # DTモデルの宣言
    elif regression_method == 'rf':  # Random forest
        rmse_oob_all = list()
        for random_forest_x_variables_rate in random_forest_x_variables_rates:
            RandomForestResult = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
                max(math.ceil(autoscaled_x_train.shape[1] * random_forest_x_variables_rate), 1)), oob_score=True)
            RandomForestResult.fit(autoscaled_x_train, autoscaled_y_train)
            estimated_y_in_cv = RandomForestResult.oob_prediction_
            estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
            rmse_oob_all.append((sum((y_train - estimated_y_in_cv) ** 2) / len(y_train)) ** 0.5)
        optimal_random_forest_x_variables_rate = random_forest_x_variables_rates[
            np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]]
        regression_model = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
            max(math.ceil(autoscaled_x_train.shape[1] * optimal_random_forest_x_variables_rate), 1)), oob_score=True)
    elif regression_method == 'gp':  # Gaussian process
        regression_model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel())

    estimated_y_in_cv = np.ndarray.flatten(
        model_selection.cross_val_predict(regression_model, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
    estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
    r2cvs.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))

plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
plt.plot(np.arange(set_max_pca_component_number) + 1, r2cvs, 'b.-')
plt.ylim(0, 1)
plt.xlabel('Number of PCA components')
plt.ylabel('r2cv')
plt.show()

optimal_pca_component_number = np.where(r2cvs == np.max(r2cvs))[0][0] + 1
print('Optimal PCA component number : {0}'.format(optimal_pca_component_number))

autoscaled_x_train = autoscaled_score_train.iloc[:, :optimal_pca_component_number]
autoscaled_x_test = autoscaled_score_test.iloc[:, :optimal_pca_component_number]
if regression_method == 'pls':  # Partial Least Squares
    pls_components = np.arange(1, min(np.linalg.matrix_rank(autoscaled_x_train) + 1, max_pls_component_number + 1), 1)
    r2cvall = []
    for pls_component in pls_components:
        pls_model_in_cv = PLSRegression(n_components=pls_component)
        estimated_y_in_cv = np.ndarray.flatten(
            model_selection.cross_val_predict(pls_model_in_cv, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
        estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
        r2cvall.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
    optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall))[0][0] + 1
    regression_model = PLSRegression(n_components=optimal_pls_component_number)
elif regression_method == 'rr':  # ridge regression
    r2cvall = list()
    for ridge_lambda in ridge_lambdas:
        rr_model_in_cv = Ridge(alpha=ridge_lambda)
        estimated_y_in_cv = model_selection.cross_val_predict(rr_model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                              cv=fold_number)
        estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
        r2cvall.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
    optimal_ridge_lambda = ridge_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
    regression_model = Ridge(alpha=optimal_ridge_lambda)
elif regression_method == 'lasso':  # LASSO
    r2cvall = list()
    for lasso_lambda in lasso_lambdas:
        lasso_model_in_cv = Lasso(alpha=lasso_lambda)
        estimated_y_in_cv = model_selection.cross_val_predict(lasso_model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                              cv=fold_number)
        estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
        r2cvall.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
    optimal_lasso_lambda = lasso_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
    regression_model = Lasso(alpha=optimal_lasso_lambda)
elif regression_method == 'en':  # Elastic net
    elastic_net_in_cv = ElasticNetCV(cv=fold_number, l1_ratio=elastic_net_lambdas, alphas=elastic_net_alphas)
    elastic_net_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_elastic_net_alpha = elastic_net_in_cv.alpha_
    optimal_elastic_net_lambda = elastic_net_in_cv.l1_ratio_
    regression_model = ElasticNet(l1_ratio=optimal_elastic_net_lambda, alpha=optimal_elastic_net_alpha)
elif regression_method == 'lsvr':  # Linear SVR
    linear_svr_in_cv = GridSearchCV(svm.SVR(kernel='linear'), {'C': linear_svr_cs, 'epsilon': linear_svr_epsilons},
                                    cv=fold_number)
    linear_svr_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_linear_svr_c = linear_svr_in_cv.best_params_['C']
    optimal_linear_svr_epsilon = linear_svr_in_cv.best_params_['epsilon']
    regression_model = svm.SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon)
elif regression_method == 'nsvr':  # Nonlinear SVR
    variance_of_gram_matrix = list()
    numpy_autoscaled_Xtrain = np.array(autoscaled_x_train)
    for nonlinear_svr_gamma in nonlinear_svr_gammas:
        gram_matrix = np.exp(
            -nonlinear_svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain) ** 2).sum(
                axis=2))
        variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
    optimal_nonlinear_gamma = nonlinear_svr_gammas[
        np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
    # CV による ε の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_nonlinear_gamma),
                               {'epsilon': nonlinear_svr_epsilons},
                               cv=fold_number, iid=False, verbose=0)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_nonlinear_epsilon = model_in_cv.best_params_['epsilon']
    # CV による C の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma),
                               {'C': nonlinear_svr_cs}, cv=fold_number, iid=False, verbose=0)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_nonlinear_c = model_in_cv.best_params_['C']
    # CV による γ の最適化
    model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, C=optimal_nonlinear_c),
                               {'gamma': nonlinear_svr_gammas}, cv=fold_number, iid=False, verbose=0)
    model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
    optimal_nonlinear_gamma = model_in_cv.best_params_['gamma']
    regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                               gamma=optimal_nonlinear_gamma)
elif regression_method == 'dt':  # Decision tree
    # クロスバリデーションによる木の深さの最適化
    r2cv_all = []
    for max_depth in range(2, dt_max_max_depth):
        model_in_cv = tree.DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=dt_min_samples_leaf)
        estimated_y_in_cv = model_selection.cross_val_predict(model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                              cv=fold_number) * y_train.std(ddof=1) + y_train.mean()
        r2cv_all.append(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2))
    optimal_max_depth = np.where(r2cv_all == np.max(r2cv_all))[0][0] + 2  # r2cvが最も大きい木の深さ
    regression_model = tree.DecisionTreeRegressor(max_depth=optimal_max_depth,
                                                  min_samples_leaf=dt_min_samples_leaf)  # DTモデルの宣言
elif regression_method == 'rf':  # Random forest
    rmse_oob_all = list()
    for random_forest_x_variables_rate in random_forest_x_variables_rates:
        RandomForestResult = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
            max(math.ceil(autoscaled_x_train.shape[1] * random_forest_x_variables_rate), 1)), oob_score=True)
        RandomForestResult.fit(autoscaled_x_train, autoscaled_y_train)
        estimated_y_in_cv = RandomForestResult.oob_prediction_
        estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
        rmse_oob_all.append((sum((y_train - estimated_y_in_cv) ** 2) / len(y_train)) ** 0.5)
    optimal_random_forest_x_variables_rate = random_forest_x_variables_rates[
        np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]]
    regression_model = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
        max(math.ceil(autoscaled_x_train.shape[1] * optimal_random_forest_x_variables_rate), 1)), oob_score=True)
elif regression_method == 'gp':  # Gaussian process
    regression_model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel())

regression_model.fit(autoscaled_x_train, autoscaled_y_train)

# calculate y for training data
calculated_ytrain = np.ndarray.flatten(regression_model.predict(autoscaled_x_train))
calculated_ytrain = calculated_ytrain * y_train.std(ddof=1) + y_train.mean()
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train, calculated_ytrain)
y_max = np.max(np.array([np.array(y_train), calculated_ytrain]))
y_min = np.min(np.array([np.array(y_train), calculated_ytrain]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Calculated Y')
plt.show()
# r2, RMSE, MAE
print('r2: {0}'.format(float(1 - sum((y_train - calculated_ytrain) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSE: {0}'.format(float((sum((y_train - calculated_ytrain) ** 2) / len(y_train)) ** 0.5)))
print('MAE: {0}'.format(float(sum(abs(y_train - calculated_ytrain)) / len(y_train))))

# estimated_y in cross-validation
estimated_y_in_cv = np.ndarray.flatten(
    model_selection.cross_val_predict(regression_model, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y_train, estimated_y_in_cv)
y_max = np.max(np.array([np.array(y_train), estimated_y_in_cv]))
y_min = np.min(np.array([np.array(y_train), estimated_y_in_cv]))
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y in CV')
plt.show()
# r2cv, RMSEcv, MAEcv
print('r2cv: {0}'.format(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2))))
print('RMSEcv: {0}'.format(float((sum((y_train - estimated_y_in_cv) ** 2) / len(y_train)) ** 0.5)))
print('MAEcv: {0}'.format(float(sum(abs(y_train - estimated_y_in_cv)) / len(y_train))))

# estimate y for test data
autoscaled_x_test = np.ndarray.flatten(regression_model.predict(autoscaled_x_test))
autoscaled_x_test = autoscaled_x_test * y_train.std(ddof=1) + y_train.mean()
autoscaled_x_test = pd.DataFrame(autoscaled_x_test, index=unsupervised_dataset.index, columns=['estimated y'])
autoscaled_x_test.to_csv('estimated_y.csv')
