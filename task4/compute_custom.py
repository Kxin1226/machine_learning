import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


def poly_kernel(X, Y, degree=3, coef0=1):
    return (np.dot(X, Y.T) + coef0) ** degree


def gaussian_kernel(X, Y, gamma=0.5):
    X_norm = np.sum(X ** 2, axis=1)[:, None]
    Y_norm = np.sum(Y ** 2, axis=1)[None, :]
    K = X_norm + Y_norm - 2 * np.dot(X, Y.T)
    return np.exp(-gamma * K)


def main():
    df = pd.read_csv('winequality-white.csv', sep=None, engine='python')
    if 'quality' in df.columns:
        y = df['quality'].values
        X = df.drop(columns=['quality']).values
    else:
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].values

    _, working = train_test_split(np.hstack([X, y.reshape(-1, 1)]), test_size=0.2, random_state=42, shuffle=True)
    Xw = working[:, :-1]
    yw = working[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(Xw, yw, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    custom = {
        'poly3': lambda X, Y: poly_kernel(X, Y, degree=3, coef0=1),
        'gaussian': lambda X, Y: gaussian_kernel(X, Y, gamma=0.5)
    }
    for name, ker in custom.items():
        m = SVR(kernel=ker).fit(X_train_s, y_train)
        ypred = m.predict(X_test_s)
        print(f"{name}: {mean_squared_error(y_test, ypred):.6f}")


if __name__ == '__main__':
    main()
