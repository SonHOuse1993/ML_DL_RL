# -*- coding: utf-8 -*-

import lightgbm as lgb
gbm = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(X_train, y_train,
        eval_set=[(X_train, y_train)],
        eval_metric='logloss',
        verbose=100)
y_pred = gbm.predict(X_test)
print(mean_squared_error(y_test, y_pred))
