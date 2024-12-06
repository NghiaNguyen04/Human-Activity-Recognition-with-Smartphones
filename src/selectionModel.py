import pickle

class SelectionModel:
    def __init__(self, model_paths):
        """
        Khởi tạo SelectionModel với danh sách các đường dẫn tới file model.pkl.
        """
        self.models = []
        self.load_models(model_paths)

    def load_models(self, model_paths):
        """
        Nạp các mô hình từ danh sách các đường dẫn file.
        """
        self.models = []
        for path in model_paths:
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                    self.models.append(model)
            except Exception as e:
                print(f"Không thể tải model từ {path}: {e}")

    def predict(self, input_data):
        """
        Dự đoán đầu ra cho input_data bằng tất cả các mô hình.
        
        Args:
            input_data: Dữ liệu đầu vào (phụ thuộc vào yêu cầu của các mô hình).
        
        Returns:
            List chứa kết quả từ từng mô hình.
        """
        if not self.models:
            raise ValueError("Chưa nạp bất kỳ mô hình nào.")

        predictions = []
        for i, model in enumerate(self.models):
            try:
                prediction = model.predict(input_data)
                predictions.append(prediction)
            except Exception as e:
                print(f"Lỗi khi dự đoán với mô hình {i + 1}: {e}")
                predictions.append(None)

        return predictions

    def get_model(self, index):
        """
        Lấy một mô hình cụ thể dựa trên chỉ số.
        
        Args:
            index: Chỉ số của mô hình (bắt đầu từ 0).
        
        Returns:
            Mô hình tương ứng hoặc None nếu chỉ số không hợp lệ.
        """
        if index < 0 or index >= len(self.models):
            print("Chỉ số mô hình không hợp lệ.")
            return None
        return self.models[index]


