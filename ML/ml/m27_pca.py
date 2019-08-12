from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
x_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA

pca = PCA(n_components=13)
pca.fit(x_scaled)

x_pca = pca.transform(x_scaled)
print("원본 데이터 형태", x_scaled.shape)
print("축소된 데이터 형태", x_pca.shape)
