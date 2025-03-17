import tensorflow as tf


#replace path to model
model = tf.keras.models.load_model("ela_model4_xception_32epochs.keras")
# Initialize the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Optional) Set optimizations for smaller size or better performance
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
# Convert the model
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("xception20.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted and saved as model.tflite")
