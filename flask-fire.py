import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from pandas.plotting import scatter_matrix
import math


# 1-1
fires = pd.read_csv("sanbul2district-divby100.csv", sep=",")
fires['burned_area'] = np.log(fires['burned_area'] + 1)

# 1-2
# 상위 5개 행 출력
#print(fires.head())

# 요약 정보
#print(fires.info())

# 통계 요약
#print(fires.describe())

# 카테고리형 변수 value_counts
#print("Month counts:\n", fires["month"].value_counts())
#print("\nDay counts:\n", fires["day"].value_counts())

# 히스토그램 시각화
#fires.hist(bins=30, figsize=(12, 10))
#plt.tight_layout()
#plt.show()

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pandas as pd

# 1-3, 1-4
# 데이터 불러오기

# 일반적 분할
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)

# 계층적 분할 (month 특성 기준)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_idx]
    strat_test_set = fires.loc[test_idx]

# #month 비율 확인
# print("\n[테스트셋 내 month 분포 비율]")
# print(strat_test_set["month"].value_counts() / len(strat_test_set))

# print("\n[전체 데이터의 month 분포 비율]")
# print(fires["month"].value_counts() / len(fires))

# # #1‑6
# from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt

# attrs = ["burned_area",          
#          "max_temp",
#          "avg_temp",
#          "max_wind_speed"]


# scatter_matrix(strat_train_set[attrs],
#                figsize=(8, 6),    
#                alpha=0.2,           
#                diagonal='hist')     

# plt.suptitle("1‑6 Pandas scatter_matrix() for 4+ features",
#              fontsize=16, y=1.0)   
# plt.tight_layout()
# plt.show()

# 1-7
# fires.plot(
#     kind="scatter",
#     x="longitude",
#     y="latitude",
#     alpha=0.6,
#     s=fires["max_temp"] * 5,                         
#     c=fires["burned_area"],                     
#     cmap=plt.get_cmap("YlOrRd"),                    
#     colorbar=True,
#     figsize=(10, 7),
#     label="max_temp"
# )

# plt.title("1-7", fontsize=14)
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.grid(True)
# plt.legend()
# plt.show()

# # 1-9
from sklearn.preprocessing import OneHotEncoder

# 1. 레이블 분리
fires = strat_train_set.drop("burned_area", axis=1)
fires_labels = strat_train_set["burned_area"].copy()

# 2. 수치형 데이터만 분리
fires_num = fires.drop(["month", "day"], axis=1)

# 3. 범주형 데이터 추출
fires_cat = fires[["month", "day"]]

# 4. OneHotEncoder 객체 생성
cat_encoder = OneHotEncoder()

# 5. 범주형 데이터 인코딩 (fit_transform으로 학습 + 변환)
fires_cat_1hot = cat_encoder.fit_transform(fires_cat)

# 6. 희소 행렬(sparse matrix) → 배열로 변환 (원하면)
fires_cat_1hot_array = fires_cat_1hot.toarray()

# 7. 카테고리 정보 출력
print("cat_day_encoder.categories_:", cat_encoder.categories_[1])
print("cat_month_encoder.categories_:", cat_encoder.categories_[0])

# 1-9
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. 수치형 특성 목록 정의
num_attribs = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']

# 2. 범주형 특성 목록 정의
cat_attribs = ['month', 'day']

# 3. 수치형 파이프라인: 표준화(StandardScaler)
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

# 4. 전체 파이프라인: 수치형 + 범주형을 통합 처리
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

# 5. training set에 파이프라인 적용
# `fires`는 burned_area를 제외한 strat_train_set임
fires = strat_train_set.drop("burned_area", axis=1)
fires_labels = strat_train_set["burned_area"].copy()

fires_prepared = full_pipeline.fit_transform(fires)


# 2
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 데이터 분할 (train/valid/test)
X_train, X_valid, y_train, y_valid = train_test_split(fires_prepared, fires_labels, test_size=0.2, random_state=42)

# 테스트셋도 준비 (strat_test_set으로부터)
fires_test = strat_test_set.drop("burned_area", axis=1)
fires_test_labels = strat_test_set["burned_area"].copy()
fires_test_prepared = full_pipeline.transform(fires_test)

X_test = fires_test_prepared
y_test = fires_test_labels

# 2. 시드 고정 (재현성 확보)
np.random.seed(42)
tf.random.set_seed(42)

# 3. 모델 정의
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

# 4. 모델 구조 확인
model.summary()

# 5. 모델 컴파일 및 학습
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.SGD(learning_rate=1e-3))

history = model.fit(X_train, y_train, epochs=200,
                    validation_data=(X_valid, y_valid))

# 6. 모델 저장
model.save("fires_model.keras")

# 7. 모델 예측 테스트
X_new = X_test[:3]
print("\n\nnp.round(model.predict(X_new), 2): \n",
      np.round(model.predict(X_new), 2))

import joblib
joblib.dump(full_pipeline, "fires_pipeline.pkl")  # 파이프라인 저장