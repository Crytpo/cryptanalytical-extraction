import tensorflow as tf

model1 = tf.keras.models.load_model("models/mnist784_15x2_1v2.keras")
model2 = tf.keras.models.load_model("models/mnist784_15x3_1v2.keras")
model3 = tf.keras.models.load_model("models/mnist784_15x2_softmax1_1v2.keras")
model4 = tf.keras.models.load_model("models/mnist784_15x2_softmax2_1v2.keras")