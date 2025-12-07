
import tensorflow as tf
import numpy as np
from AI.src import facenet
import math
import pickle
from sklearn.svm import SVC


image_size = 160
batch_size = 50

class Classifier:
    def __init__(self, face_processed_path, model_path, output_path):
        main(face_processed_path, model_path, output_path)

def main(face_processed_path, model_path, output_path):
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            np.random.seed(seed=666)
            #lấy ảnh khuôn mặt đã cắt
            dataset = facenet.get_dataset(face_processed_path)
            #lấy đường dẫn ảnh và nhãn
            paths, labels = facenet.get_image_paths_labels(dataset)
            print('Tổng số lớp: %d' % len(dataset))
            print('Tổng số ảnh: %d' % len(paths))
            #Load model
            print("Load mô hình trích xuất đặc trung")
            facenet.load_model(model_path)
            #lấy input và output
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            #trích xuất đặc trưng khuôn mặt
            so_anh = len(paths)
            batch_per_epoch = int(math.ceil(1.0 * so_anh / batch_size))
            #mảng chứa đặc trưng
            emb_arr = np.zeros((so_anh, embedding_size))
            for i in range(batch_per_epoch):
                start_index = i * batch_size
                end_index = min( (i + 1) * batch_size, so_anh )
                paths_batch = paths[start_index:end_index] #lấy ảnh từ start đến end
                images = facenet.load_data(paths_batch, image_size) #load ảnh
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_arr[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict) #trích xuất đặc trưng và lưu vào mảng
            #Huấn luyện bộ phân loại dựa trên nhãn và đặc  trưng đã trích xuất
            print("Bắt đầu huấn luyện bộ phân loại")
            classifier_model = SVC(kernel='linear', probability=True) #khởi tạo mô hình SVM dùng hàm nhân tuyến tính
            classifier_model.fit(emb_arr, labels) #huấn luyện bộ phân loại dựa trên vector đặc trưng và nhãn
            class_name = [cls.name.replace('_', ' ') for cls in dataset]
            #lưu mô hình phân loại
            with open(output_path, 'wb') as outfile:
                pickle.dump((classifier_model, class_name), outfile)
            print("Lưu mô hình phân loại thành công tại : " + output_path)


if __name__ == "__main__":
    face_processed_path = "E:/PythonProjectMain/AI/DataSet/FaceData/processed"
    model_path = "E:/PythonProjectMain/AI/Models/20180402-114759.pb"
    output_path = "E:/PythonProjectMain/AI/Models/classifier.pkl"
    main(face_processed_path, model_path, output_path)

