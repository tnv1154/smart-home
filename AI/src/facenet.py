import os
import tensorflow as tf
import numpy as np
import imageio.v3 as iio

def chuan_hoa_anh(x):
    """chuẩn hóa độ sáng ảnh"""
    #Tính giá trị trung bình độ sáng các pixel
    mean = np.mean(x)
    #Tính độ lệch chuẩn của ảnh
    std = np.std(x)
    #Điều chỉnh độ lệch chuẩn để tránh chia cho 0
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def crop(image, image_size):
    """Cắt ảnh"""
    if image.shape[1] > image_size:
        #Tính tọa độ trung tâm
        sz1 = int(image.shape[1] // 2)
        #Tính nửa kích thước ảnh
        sz2 = int(image_size // 2)
        #Cắt theo chiều dọc, ngang, giữ nguyên kênh màu
        image = image[(sz1 - sz2):(sz1 + sz2), (sz1 - sz2):(sz1 + sz2), :]
    return image

def flip(image):
    """Lật ngang ảnh random"""
    if np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(image):
    """Chuyển ảnh sang màu RGB"""
    cao, rong = image.shape
    ret = np.empty((cao, rong, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = image
    return ret

def load_data(image_paths, image_size):
    """tải dữ liệu ảnh và đưa vào mạng neural"""
    so_luong_anh = len(image_paths)
    images = np.zeros((so_luong_anh, image_size, image_size, 3))
    for i in range(so_luong_anh):
        img = iio.imread(image_paths[i])
        #Nếu ảnh chỉ có 2 kênh màu thì chuyển sang 3 kênh
        if img.ndim == 2:
            img = to_rgb(img)
        img = chuan_hoa_anh(img)
        img = crop(img, image_size)
        img = flip(img)
        #gán ảnh vào mảng kết quả
        images[i,:,:,:] = img
    return images

def get_image_paths_labels(dataset):
    """lấy ảnh và nhãn của các ảnh trong dataset"""
    image_paths_arr = []
    labels_arr = []
    for i in range(len(dataset)):
        #list các ảnh của người thứ i
        image_paths_arr += dataset[i].image_paths
        #Gắn nhãn i cho từng ảnh của lớp đó
        labels_arr += [i] * len(dataset[i].image_paths)
    return image_paths_arr, labels_arr

class ImageClass():
    """Lưu đường dẫn ảnh của người cụ thể"""
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_dataset(paths, has_class_directories=True):
    """lấy tất cả các lớp trong dataset"""
    dataset = []
    path_exp = os.path.expanduser(paths)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset

def get_image_paths(facedir):
    """lấy tất cả đường dẫn các ảnh trong thư mục"""
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def load_model(model, input_map=None):
    """load mô hình"""
    """mở roọng đường dẫn"""
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print(f"Model filename: {model_exp}")
        with tf.io.gfile.GFile(model_exp, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')

def delete_classifier_model():
    """Xóa mô hình classifier"""
    file_path = "E:/PythonProjectMain/AI/Models/classifier.pkl"
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Đã xóa file: {file_path}")
    else:
        print(f"File không tồn tại: {file_path}")