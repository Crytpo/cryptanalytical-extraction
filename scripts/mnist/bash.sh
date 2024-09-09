
# dense
python models/train_new_models.py --load_json scripts/mnist/dense/8x1_softmax.json
python models/train_new_models.py --load_json scripts/mnist/dense/64x1_softmax.json
python models/train_new_models.py --load_json scripts/mnist/dense/8x2_softmax.json
python models/train_new_models.py --load_json scripts/mnist/dense/64x2_softmax.json

# cnn
python models/train_new_models.py --load_json scripts/mnist/cnn/8x1_linear.json
python models/train_new_models.py --load_json scripts/mnist/cnn/64x1_linear.json
python models/train_new_models.py --load_json scripts/mnist/cnn/8x2_linear.json
python models/train_new_models.py --load_json scripts/mnist/cnn/64x2_linear.json


python models/train_new_models.py --load_json scripts/mnist/cnn/8x1_fc_linear.json
python models/train_new_models.py --load_json scripts/mnist/cnn/64x1_fc_linear.json
python models/train_new_models.py --load_json scripts/mnist/cnn/8x2_fc_linear.json
python models/train_new_models.py --load_json scripts/mnist/cnn/64x2_fc_linear.json


python models/train_new_models.py --load_json scripts/mnist/cnn/8x1_softmax.json
python models/train_new_models.py --load_json scripts/mnist/cnn/64x1_softmax.json
python models/train_new_models.py --load_json scripts/mnist/cnn/8x2_softmax.json
python models/train_new_models.py --load_json scripts/mnist/cnn/64x2_softmax.json


python models/train_new_models.py --load_json scripts/mnist/cnn/8x1_fc_softmax.json
python models/train_new_models.py --load_json scripts/mnist/cnn/64x1_fc_softmax.json
python models/train_new_models.py --load_json scripts/mnist/cnn/8x2_fc_softmax.json
python models/train_new_models.py --load_json scripts/mnist/cnn/64x2_fc_softmax.json



