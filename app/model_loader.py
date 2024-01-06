from tensorflow.keras.models import load_model

model = load_model('complete_model.h5')
model.load_weights('model_weights.h5')
