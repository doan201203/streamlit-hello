import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
import pickle
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.metrics import pairwise_distances_argmin

st.title('Trực quan hóa thuật toán K-Means Clustering')

# Tải dữ liệu MNIST
@st.cache_data()
def load_data():
    # with open('./datasets/mnist/data_mnist.pkl', 'rb') as f:
        # X, y, X_scaled, X_pca = pickle.load(f)
    mnist = fetch_openml('mnist_784', version=1)
    X = np.array(mnist.data)
    y = np.array(mnist.target, dtype=int)
    return X, y

def initialize_centroids(X, k):
    rng = np.random.RandomState(42)
    i = rng.permutation(X.shape[0])[:k]
    return X[i]

# Hàm cập nhật centroid
def update_centroids(X, labels, k):
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids

# Hàm gán nhãn (label) cho mỗi điểm
def assign_labels(X, centroids):
    labels = pairwise_distances_argmin(X, centroids)
    return labels


X, y = load_data()

# Tiền xử lý: chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Áp dụng PCA để giảm chiều dữ liệu cho trực quan hóa
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Mô tả dataset
st.write('## 1. Tập dữ liệu sử dụng')
st.write('Ứng dụng sử dụng tập dữ liệu MNIST gồm 70,000 ảnh chữ số viết tay với kích thước 28x28 pixel. Mỗi ảnh được biểu diễn dưới dạng một vector 784 chiều.')
st.write('Dữ liệu được giảm chiều xuống còn 2 chiều bằng phương pháp PCA để dễ dàng trực quan hóa.')
st.write('Một số ảnh trong tập dữ liệu MNIST:')

# Hiển thị một số ảnh từ tập dữ liệu
fig, ax = plt.subplots(1, 5, figsize=(10, 5))
for i in range(5):
    ax[i].imshow(X[i].reshape(28, 28), cmap='gray')
    ax[i].axis('off')
st.pyplot(fig)

# Display pipeline
st.write('## 2. Luồng xử lý')
st.image('./datasets/mnist/pipeline.png', use_column_width=True)
def init():
    st.session_state.centroids = initialize_centroids(X_pca, st.session_state.k)
    st.session_state.labels = assign_labels(X_pca, st.session_state.centroids)
    st.session_state.iteration = 0
    st.session_state.k = st.session_state.k

# Hiển thị dữ liệu gốc trước khi phân cụm
st.subheader('Dữ liệu MNIST sau khi giảm chiều với PCA:')
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=10)
ax.set_title('MNIST Dữ liệu sau PCA')
st.pyplot(fig)

with st.form(key='my_form_1'):
    k = st.slider('Chọn số cụm (K) để phân cụm dữ liệu', min_value=1, max_value=20, value=10, step=1)
    st.session_state.k = k
    st.form_submit_button("Xác nhận", icon=":material/check:", on_click=init())

# Nút Reset
if st.button('Reset'):
    st.session_state.centroids = initialize_centroids(X_pca, k)
    st.session_state.labels = assign_labels(X_pca, st.session_state.centroids)
    st.session_state.iteration = 0
    st.session_state.k = k

@st.fragment()
def visualize_kmeans():
    with st.form(key='my_form'):
        st.write('## 3. Trực quan hóa quá trình hội tụ của K-Means với K = ', st.session_state.k)
        st.write('Nhấn nút Next để xem quá trình hội tụ của thuật toán K-Means.')
        st.write(
            """
                - ***Lưu ý:***
                    - Nếu muốn chọn số cụm khác.
                        - Bấm nút Reset để khởi tạo lại các centroids.
                        - Di chuyển thanh trượt để chọn số cụm.
            """
            )

        if st.form_submit_button("Next"):
            # Cập nhật centroid và gán lại nhãn
            st.session_state.iteration += 1
            st.session_state.centroids = update_centroids(X_pca, st.session_state.labels, k)
            st.session_state.labels = assign_labels(X_pca, st.session_state.centroids)
        
        st.subheader(f"Kết quả phân cụm (Vòng lặp: {st.session_state.iteration})")
        centroids = st.session_state.centroids
        labels = st.session_state.labels
        with st.spinner('Đang cập nhật...'):
            # Tạo Voronoi diagram
            vor = Voronoi(centroids)
            fig, ax = plt.subplots(figsize=(10, 6))
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=3.5, line_alpha=0.6, point_size=2)

            # Vẽ dữ liệu và centroids
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=10, alpha=0.8, edgecolor='k')
            ax.scatter(centroids[:, 0], centroids[:, 1], c='yellow', s=200, marker='x', label='Centroids')

            # Thay đổi ký hiệu centroids thành số cụm
            for i, (x, y) in enumerate(centroids):
                ax.text(x, y, str(i + 1), color='red', fontsize=12, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            # Thêm chú thích
            ax.set_title(f'K-Means Clustering Progress (Iteration {st.session_state.iteration})')
            ax.legend()
            st.pyplot(fig)

            # Hiển thị hình ảnh từ mỗi cụm
            st.subheader("Một số hình ảnh từ các cụm")
            for cluster in range(k):
                st.write(f"**Cụm {cluster + 1}:**")
                cluster_indices = np.where(labels == cluster)[0]  # Lấy chỉ số của các điểm trong cụm
                sample_indices = np.random.choice(cluster_indices, size=min(10, len(cluster_indices)), replace=False)  # Chọn ngẫu nhiên 10 ảnh

                # Hiển thị hình ảnh từ cụm
                fig, axes = plt.subplots(1, len(sample_indices), figsize=(10, 2))
                for img_idx, ax in zip(sample_indices, axes):
                    ax.imshow(X[img_idx].reshape(28, 28), cmap='gray')  # Dữ liệu MNIST có kích thước 28x28
                    ax.axis('off')
                st.pyplot(fig)
            st.toast("Cập nhật thành công", icon=":material/check:")

visualize_kmeans()