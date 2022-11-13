import imputena
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Dropout
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def count_nulls(df):
    temp1 = []
    temp2 = []
    columns = df.columns
    for column in columns:
        if df[column].isna().sum() != 0:
            temp1.append(column)
            temp2.append(df[column].isna().sum())
    temp1 = pd.DataFrame(temp1, columns=['Column'])
    temp2 = pd.DataFrame(temp2, columns=['Number of Nulls'])
    nulls = pd.concat([temp1, temp2], axis=1)
    return nulls


def plot_nulls(df):
    nulls = count_nulls(df)
    plt.bar(nulls['Column'], nulls['Number of Nulls'])
    plt.xticks(rotation=90)
    plt.show()


def plot_correlation_graph(df, save=False, filename='untitled.png'):
    pattern = np.triu(np.ones_like(df.corr(), dtype='bool'))
    fig, ax = plt.subplots(figsize=(30, 15))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', mask=pattern, ax=ax)
    if save:
        plt.savefig(filename)
    plt.show()


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

train = train.drop(columns=['id'])
train['product_code'].replace(to_replace=['A', 'B', 'C', 'D', 'E'], value=[0, 1, 2, 3, 4], inplace=True)
train['attribute_0'].replace(to_replace=['material_7', 'material_5'], value=[2, 0], inplace=True)
train['attribute_1'].replace(to_replace=['material_8', 'material_5', 'material_6'], value=[3, 0, 1], inplace=True)
test = test.drop(columns=['id'])
test['product_code'].replace(to_replace=['F', 'G', 'H', 'I'], value=[5, 6, 7, 8], inplace=True)
test['attribute_0'].replace(to_replace=['material_7', 'material_5'], value=[2, 0], inplace=True)
test['attribute_1'].replace(to_replace=['material_6', 'material_7', 'material_5'], value=[1, 2, 0], inplace=True)

print(train.head())
print(train.describe())
print(train.info())
print(train.shape)

plot_correlation_graph(train, save=True, filename='Correlation.png')

columns_to_keep = ['product_code', 'loading', 'attribute_0', 'attribute_1', 'attribute_2', 'attribute_3',
                   'measurement_0', 'measurement_1', 'measurement_2', 'measurement_3', 'measurement_5',
                   'measurement_10', 'measurement_16', 'measurement_17', 'failure']
columns_with_nulls = ['loading', 'measurement_3', 'measurement_5', 'measurement_10', 'measurement_16', 'measurement_17']

train = train[columns_to_keep]
test = test[columns_to_keep[:-1]]

plot_nulls(train)
plot_nulls(test)

train['m_3_missing'] = train['measurement_3'].isna()
train['m_5_missing'] = train['measurement_5'].isna()
test['m_3_missing'] = test['measurement_3'].isna()
test['m_5_missing'] = test['measurement_5'].isna()

for i in range(5):
    print(i)
    mask = train['product_code'] == i
    piece = train[mask]
    print(imputena.recommend_method(piece))
    piece = imputena.impute_by_recommended(piece)
    train[mask] = piece

for i in range(5, 9):
    print(i)
    mask = test['product_code'] == i
    piece = test[mask]
    print(imputena.recommend_method(piece))
    piece = imputena.impute_by_recommended(piece)
    test[mask] = piece

train.drop(columns=['product_code'], inplace=True)
test.drop(columns=['product_code'], inplace=True)
test_columns = test.columns

y = pd.DataFrame(train['failure'], columns=['failure'])
train.drop(columns=['failure'], inplace=True)
train_columns = train.columns

ss = StandardScaler()
train = ss.fit_transform(train)
test = ss.fit_transform(test)

train = pd.DataFrame(train, columns=train_columns)
test = pd.DataFrame(test, columns=test_columns)

X_train, X_test, y_train, y_test = train_test_split(train, y, random_state=28, shuffle=True, test_size=0.2)

tf.random.set_seed(42)
model = Sequential()
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch / 10))

model.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=0.00085), metrics=['AUC'])
history = model.fit(X_train, y_train, epochs=50, callbacks=[early_stopping, lr_scheduler],
                    validation_data=(X_test, y_test))

pd.DataFrame(history.history).plot()
plt.show()

preds = model.predict(test)

submission['failure'] = preds
submission.to_csv('submission.csv', index=False)
