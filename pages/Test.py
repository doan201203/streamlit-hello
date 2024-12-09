import streamlit as st
from streamlit_drawable_canvas import st_canvas

def p1():
   st.header("1. Dataset")
   st.write("Sử dụng tập dữ liệu MNIST Hình ảnh chứa 70.000 hình ảnh chữ số viết tay từ 0 đến 9. Mỗi hình ảnh có kích thước 28x28 pixel. Tập dữ liệu được chia thành 60.000 hình ảnh huấn luyện và 10.000 hình ảnh kiểm tra.")

def p2():
   st.header("2. Phương Pháp")
   st.image('./images/EfficientNet-Architecture.png', use_container_width=True, caption='Kiến trúc mạng EfficientNet')
   st.markdown("""
        - EfficientNet là một họ các mô hình mạng nơ-ron tích chập (CNN) được phát triển bởi Google AI, nổi bật với hiệu suất cao và hiệu quả tính toán vượt trội so với các kiến trúc CNN trước đó. EfficientNet được giới thiệu trong bài báo ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" vào năm 2019](https://arxiv.org/pdf/1905.11946).
            - Ý tưởng cốt lõi:
                - EfficientNet dựa trên ý tưởng cân bằng đồng đều (compound scaling) ba chiều của mạng nơ-ron:
                - Chiều sâu (Depth): Số lượng các lớp (layers) trong mạng.
                - Chiều rộng (Width): Số lượng kênh (channels) trong mỗi lớp.
                - Độ phân giải (Resolution): Kích thước của hình ảnh đầu vào.
            - Các phương pháp trước đây thường chỉ tập trung vào việc mở rộng một trong ba chiều này, dẫn đến hiệu suất bị giới hạn hoặc chi phí tính toán tăng cao. EfficientNet thay vào đó đề xuất một phương pháp cân cân bằng đồng đều cả ba chiều
    """)
   st.image('./images/results_efficient.png', use_container_width=True, caption='So sánh kết quả dựa trên tập dữ liệu ImageNet')
   
def p3():
   st.header("2. Ứng dụng")
   st.write("Huớng dẫn sử dụng")
   st.write('1. Vẽ 1 chữ số từ 0-9')
   st.write('2. Sau khi vẽ xong bấm nút Submit để tiến hành nhận diện')

   with st.form(key="form"):
    col = st.columns(2, gap='large')
    with col[0]:
        canvas_rs = st_canvas(
            display_toolbar=True,
            stroke_width=6,
            fill_color='black',
            drawing_mode='freedraw',
            stroke_color='black',
            height=400,
            width=400,
            update_streamlit=True,
            key="img.name",
        )
        submit = st.form_submit_button('Submit')
p1()
p2()