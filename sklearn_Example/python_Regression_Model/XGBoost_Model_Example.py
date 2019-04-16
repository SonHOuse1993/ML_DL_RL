# -*- coding: utf-8 -*-

"""
XGBoost近些年在学术界取得的成果连连捷报，
基本所有的机器学习比赛的冠军方案都使用了XGBoost算法，
对于XGBoost的算法接口有两种，这里我仅介绍XGBoost的sklearn接口。

更多请参考：

https://xgboost.readthedocs.io/en/latest/python/index.html

"""

import xgboost as xgb
xgb_model = xgb.XGBRegressor(max_depth = 3,
                             learning_rate = 0.1,
                             n_estimators = 100,
                             objective = 'reg:linear',
                             n_jobs = -1)

xgb_model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              eval_metric='logloss',
              verbose=100)
y_pred = xgb_model.predict(X_test)
print(mean_squared_error(y_test, y_pred))
