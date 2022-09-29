# ライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel

# 学習用データ読み込み：891行
df_train = pd.read_csv('..//input/train.csv')
#df_train.shape

#評価用データ読み込み：382行
df_test = pd.read_csv('..//input/test.csv')
#df_test.shape

#学習用と評価用が分別できるようフラグを立てる
df_train["flag"] = True
df_test["flag"] = False

#一旦、学習用データと評価用データを結合
df = pd.concat([df_train, df_test], axis=0, sort =True)

#病気区分抽出（目的変数）
df_disease = df.loc[:, ["disease", "flag"]]
y  = df_disease

#AG_ratio計算
df["AG_ratio"].fillna(df["Alb"] / (df["TP"] - df["Alb"]), inplace=True)

#欠損値ドロップ
df.dropna()
df.reset_index(drop=True, inplace=True)

#未定義削除
df = df.drop("Unnamed: 0", axis=1)

#性別　男性1女性0
df["Gender"] = df["Gender"].apply(lambda x: 1 if x=="Male" else 0)

#肝機能の異常値　異常
df["ALT_GPT"] = df["ALT_GPT"].apply(lambda x: 1 if x>=60 else 0)
df["AST_GOT"] = df["AST_GOT"].apply(lambda x: 1 if x>=84  else 0)

#年齢　ダミー変数化する
df_age = df.loc[:,["Age"]]
dummy_df = df_age
dummy_df["Age[:30]"] = df_age.apply(lambda row: int(row.Age < 30), axis=1)
dummy_df["Age[31:60]"] = df_age.apply(lambda row: int(row.Age >= 31 and row.Age < 60), axis=1)
dummy_df["Age[61:]"] = df_age.apply(lambda row: int(row.Age >= 60), axis=1)

df_age_dummy = dummy_df.drop(["Age"], axis=1)


#血液データ.性別抽出(年齢・病気区分+ALB除外）
col_categoric = ['Age', 'disease', 'Alb']
df_numeric = df.drop(col_categoric, axis=1)
X_target = pd.concat([df_age_dummy, df_numeric], axis=1)

#訓練データのみ抽出
X_target1 = X_target[X_target["flag"] == True]
X_target1 = X_target1.drop(["flag"], axis = 1)

#目的変数を抽出
y = y[y["flag"] == True]
y = y.drop(["flag"], axis = 1)


# 多項式・交互作用特徴量
polynomial = PolynomialFeatures(degree=2, include_bias=False)
polynomial_arr = polynomial.fit_transform(X_target1)
X_polynomial = pd.DataFrame(polynomial_arr, columns=["poly" + str(x) for x in range(polynomial_arr.shape[1])])

# 生成した多項式・交互作用特徴量の表示
#print(X_polynomial.shape)
#print(X_polynomial.head())

# 組み込み法のモデル、閾値の指定
fs_model = LogisticRegression(penalty='l1', random_state=0)
fs_threshold = "mean"

# 組み込み法モデルの初期化
selector = SelectFromModel(fs_model, threshold=fs_threshold)

# 特徴量選択の実行
selector.fit(X_polynomial, y)
mask = selector.get_support()

# 選択された特徴量だけのサンプル取得
X_polynomial_masked = X_polynomial.loc[:, mask]

# 学習用・評価用データの分割
#X_train, X_test, y_train, y_test = train_test_split(X_target1, y, test_size=0.3, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X_polynomial_masked, y, test_size=0.3, random_state=0)
# モデルの学習・予測

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict_proba(X_test)[:, 1]

#print(X_target)

#提出ファイル
X_test2 = X_target[X_target["flag"] == False]
X_test2 = X_test2.drop(["flag"], axis = 1)

#print(X_test2.shape)
#print(X_polynomial_masked.shape)


# 多項式・交互作用特徴量
polynomial2 = PolynomialFeatures(degree=2, include_bias=False)
polynomial_arr2 = polynomial.fit_transform(X_test2)
X_polynomial2 = pd.DataFrame(polynomial_arr2, columns=["poly" + str(x) for x in range(polynomial_arr2.shape[1])])

# 生成した多項式・交互作用特徴量の表示
#print(X_polynomial.shape)
#print(X_polynomial.head())

# 組み込み法のモデル、閾値の指定
fs_model = LogisticRegression(penalty='l1', random_state=0)
fs_threshold = "mean"

# 組み込み法モデルの初期化
#selector = SelectFromModel(fs_model, threshold=fs_threshold)

# 特徴量選択の実行
#selector.fit(X_polynomial2, y)
#selector.fit(X_polynomial2)
#mask = selector.get_support()

# 選択された特徴量だけのサンプル取得
X_polynomial_masked2 = X_polynomial2.loc[:, mask]

y_pred2 = model.predict_proba(X_polynomial_masked2)[:, 1]

print(y_pred2)

# ROC曲線の描画（偽陽性率、真陽性率、閾値の算出）
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
plt.plot(fpr, tpr, label='roc curve')
plt.plot([0, 1], [0, 1], linestyle=':', label='random')
plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', label='ideal')
plt.legend()
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()

# AUCスコアの算出
auc_score = roc_auc_score(y_true=y_test, y_score=y_pred)
print("AUC:", auc_score)