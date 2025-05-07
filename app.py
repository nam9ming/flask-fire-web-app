from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap5
import math

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib  # 전처리 파이프라인 저장용

# Flask 기본 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap5(app)

# 🔧 입력 폼 정의
class FireForm(FlaskForm):
    longitude = StringField('longitude(1-7)', validators=[DataRequired()])
    latitude = StringField('latitude(1-7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()])
    day = StringField('day(00~06, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

# 🔍 모델 및 전처리 파이프라인 로딩
model = keras.models.load_model("fires_model.keras")
full_pipeline = joblib.load("fires_pipeline.pkl")  # 사전 저장된 전처리기

# 🎯 수치형 / 범주형 특성 이름
num_attribs = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
cat_attribs = ['month', 'day']

# 🏠 index 라우트
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# 🔥 예측 라우트
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    form = FireForm()
    if form.validate_on_submit():
        # 사용자 입력 수집
        input_data = pd.DataFrame([{
            'longitude': float(form.longitude.data),
            'latitude': float(form.latitude.data),
            'month': form.month.data,
            'day': form.day.data,
            'avg_temp': float(form.avg_temp.data),
            'max_temp': float(form.max_temp.data),
            'max_wind_speed': float(form.max_wind_speed.data),
            'avg_wind': float(form.avg_wind.data),
        }])

        # 전처리
        data_prepared = full_pipeline.transform(input_data)

        # ✅ 예측 (로그 스케일 → ha 단위)
        burned_area_log = model.predict(data_prepared)[0][0]
        burned_area = np.round(np.exp(burned_area_log) - 1, 2)  # ha
        result = np.round(burned_area, 2)       # m²

        return render_template("result.html", result=result)

    return render_template("prediction.html", form=form)

# 실행
if __name__ == '__main__':
    app.run(debug=True)
