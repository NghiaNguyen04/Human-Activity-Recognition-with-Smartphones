# # uvicorn src.FastAPI.main:app --reload
#
# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np
# import os
#
# model_path = os.path.join("models", "SVM_model.pkl")
#
# # Tải mô hình
# with open(model_path, "rb") as f:
#     model = joblib.load(f)
#
# print("Model loaded successfully!")
#
#
# # Khởi tạo ứng dụng FastAPI
# app = FastAPI()
#
# # Định nghĩa kiểu dữ liệu cho đầu vào
# class PredictionRequest(BaseModel):
#     text: str
#
# # Hàm xử lý chuỗi đầu vào
# def preprocess_input(input_str: str):
#     elements = input_str.split()  # Tách chuỗi theo khoảng trắng/tab
#     features = [float(e) for e in elements]  # Chuyển sang số thực
#     return np.array(features).reshape(1, -1)  # Chuyển đổi thành mảng 2D
#
# # Endpoint POST để nhận dữ liệu và trả kết quả dự đoán
# @app.post("/predict/")
# async def predict(data: PredictionRequest):
#     # Tiền xử lý dữ liệu đầu vào
#     try:
#         processed_data = preprocess_input(data.text)
#     except ValueError:
#         return {"error": "Invalid input format. Please ensure all values are numeric and separated by spaces or tabs."}
#
#     # Thực hiện dự đoán
#     prediction = model.predict(processed_data)
#     print(prediction)
#     # Trả kết quả dự đoán
#     return {"input": data.text, "prediction": prediction.tolist()}


from fastapi import FastAPI, HTTPException
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import RedirectResponse
from enum import Enum
import joblib
import os
import numpy as np
import subprocess
from tensorflow.keras.models import load_model as load_model_keras


app = FastAPI()

@app.middleware("http")
async def redirect_to_docs(request, call_next):
    if request.url.path == "/":
        return RedirectResponse(url='/docs')
    response = await call_next(request)
    return response

# Định nghĩa các phép toán dưới dạng Enum
class Operation(str, Enum):
    logisticRegression = "Logistic Regression Model"
    SVM = "Support Vector Classifier Model"
    randomForest = "Random Forest Model"
    LSTM = "Long Short-Term Memory Model"


# Hàm xử lý chuỗi đầu vào
def preprocess_input(input_str: str):
    elements = input_str.split()  # Tách chuỗi theo khoảng trắng/tab
    try:
        features = [float(e) for e in elements]  # Chuyển sang số thực
        return np.array(features).reshape(1, -1)  # Chuyển đổi thành mảng 2D
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {input_str}. Error: {e}")

def preprocess_input_LSTM(input_str: str)-> np.ndarray:
    elements = input_str.split()  # Tách chuỗi theo khoảng trắng/tab
    try:
        features = [float(e) for e in elements]  # Chuyển sang số thực
        return np.array(features)  # Chuyển đổi thành mảng 2D
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {input_str}. Error: {e}")

# Tìm và tải model
def load_model(model_paths):
    for path in model_paths:
        if os.path.exists(path):
            print(f"Found model at: {os.path.abspath(path)}")
            with open(path, "rb") as f:
                return joblib.load(f)
    # Nếu không tìm thấy bất kỳ đường dẫn nào
    raise HTTPException(status_code=404, detail="model not found.")

def load_model_LSTM(model_paths):
    for path in model_paths:
        if os.path.exists(path):
            print(f"Found model at: {os.path.abspath(path)}")
            return load_model_keras(path)
    # Nếu không tìm thấy bất kỳ đường dẫn nào
    raise HTTPException(status_code=404, detail="model not found.")

# Hàm dự đoán với Logistic Regression
def logisticRegresstionPredict(x):

    model_paths = [
        "models/LogisticRegression_model.pkl",
        "../../models/LogisticRegression_model.pkl"
    ]
    model = load_model(model_paths)

    try:
        predictions = model.predict(x)  # Trả về numpy array
        return predictions.tolist()  # Chuyển đổi numpy array thành danh sách
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

def SVMPredict(x):

    model_paths = [
        "models/SVM_model.pkl",
        "../../models/SVM_model.pkl"
    ]
    model = load_model(model_paths)

    try:
        predictions = model.predict(x)  # Trả về numpy array
        return predictions.tolist()  # Chuyển đổi numpy array thành danh sách
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

def RandomForestPredict(x):
    model_paths = [
        "models/RandomForest_model.pkl",
        "../../models/RandomForest_model.pkl"
    ]
    model = load_model(model_paths)

    try:
        predictions = model.predict(x)  # Trả về numpy array
        return predictions.tolist()  # Chuyển đổi numpy array thành danh sách
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

def LSTMPredict(x):
    model_paths = [
        "models/model_Exp2_LSTM_number_Units_128.h5",
        "../../models/model_Exp2_LSTM_number_Units_128.h5"
    ]
    model = load_model_LSTM(model_paths)
    try:
        output = model.predict(x)
        index_of_max = np.argmax(output, axis=-1)
        predictions = index_of_max[0, 0]
        return predictions.tolist()  # Chuyển đổi numpy array thành danh sách
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# Endpoint chính
@app.post("/Predicting human activity using sensors/")
def calculate(operation: Operation, input_data: str):
    try:
        processed_input = preprocess_input(input_data)
        if operation == Operation.logisticRegression:
            result = logisticRegresstionPredict(processed_input)
        elif operation == Operation.SVM:
            result = SVMPredict(processed_input)
        elif operation == Operation.randomForest:
            result = RandomForestPredict(processed_input)
        elif operation == Operation.LSTM:
            processed_input_LSTM = preprocess_input_LSTM(input_data)
            processed_input_LSTM = np.expand_dims(processed_input_LSTM, axis=0)  # Thêm chiều cho batch_size (1 mẫu)
            processed_input_LSTM = np.expand_dims(processed_input_LSTM, axis=1)  # Thêm chiều cho features (số đặc trưng)
            result = LSTMPredict(processed_input_LSTM)
        else:
            raise HTTPException(status_code=400, detail="Unsupported operation.")

        activity = "none"
        if isinstance(result, (list, np.ndarray)):
            result = result[0]
            activity = "none"

        if result == 1:
            activity = "Sitting"
        elif result == 2:
            activity = "Standing"
        elif result == 3:
            activity = "Walking"
        elif result == 4:
            activity = "WALKING_DOWNSTAIRS"
        elif result == 5:
            activity = "WALKING_UPSTAIRS"
        elif result == 0:
            activity = "LAYING"

        return {"operation": operation.value, "result": result, "activity": activity}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

if __name__ == "__main__":
    try:
        subprocess.run(["uvicorn", "src.FastAPI.main:app", "--host", "127.0.0.1", "--port", "8080"], check=True)
    except KeyboardInterrupt:
        print("Ứng dụng đã được dừng lại!")
    except subprocess.CalledProcessError as e:
        print(f"Đã xảy ra lỗi khi chạy ứng dụng: {e}")
#%%
