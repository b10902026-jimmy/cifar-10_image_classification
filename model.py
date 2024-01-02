from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from keras.models import load_model

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def save_model(model, path):
    model.save(path)

def convert_model_to_trt(input_saved_model_dir, output_saved_model_dir):
    conversion_params = trt.TrtConversionParams(precision_mode=trt.TrtPrecisionMode.FP16)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir, conversion_params=conversion_params)

    converter.convert()
    converter.save(output_saved_model_dir)

def load_trt_model(path):
    return load_model(path)


