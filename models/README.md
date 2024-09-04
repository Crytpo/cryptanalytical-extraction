# Create new models

## Script 
```
python  make_new_models.py
```

## Created Models

 - models/mnist784_15x2_1v2.keras
   ```
   input (InputLayer)          [(None, 784)]             0         
   layer0 (Dense)              (None, 15)                11775     
   layer1 (Dense)              (None, 15)                240                  
   output (Dense)              (None, 1)                 16    
   ```
 - models/mnist784_15x3_1v2.keras
   ```
   input (InputLayer)          [(None, 784)]             0         
   layer0 (Dense)              (None, 15)                11775     
   layer1 (Dense)              (None, 15)                240       
   layer2 (Dense)              (None, 15)                240       
   output (Dense)              (None, 1)                 16 
   ```
 - models/mnist784_15x2_softmax1_1v2.keras
   ```
   input (InputLayer)          [(None, 784)]             0         
   layer0 (Dense)              (None, 15)                11775     
   layersoftmax (Dense)        (None, 15)                240       
   layer1 (Dense)              (None, 15)                240       
   output (Dense)              (None, 1)                 16  
   ```
 - models/mnist784_15x2_softmax2_1v2.keras
   ```
   input (InputLayer)          [(None, 784)]             0  
   layer0 (Dense)              (None, 15)                11775     
   layer1 (Dense)              (None, 15)                240       
   layersoftmax (Dense)        (None, 15)                240       
   output (Dense)              (None, 1)                 16    
   ```

## Tests

Can I reconstruct Softmax layer as immediate layer? 

Take the model: models/mnist784_15x2_softmax2_1v2.keras

```
python -m neuronWiggle --model models/mnist784_15x2_softmax2_1v2.keras --layerID 1 --seed 20 --dataset 'mnist' --quantized 2

python -m neuronWiggle --model models/mnist784_15x2_softmax2_1v2.keras --layerID 2 --seed 20 --dataset 'mnist' --quantized 2

python -m neuronWiggle --model models/mnist784_15x2_softmax2_1v2.keras --layerID 3 --seed 20 --dataset 'mnist' --quantized 2

---> after one day the computation has not finished.
```

```
model.get_layer(index=0).name
'input'
model.get_layer(index=1).name
'layer0'
model.get_layer(index=2).name
'layer1'
model.get_layer(index=3).name
'layersoftmax'
model.get_layer(index=4).name
'output'
```

Question: 
Can the layer afer the softmax layer extracted? --layerID 3

```
model.get_layer(index=0).name
'input'
model.get_layer(index=1).name
'layer0'
model.get_layer(index=2).name
'layersoftmax'
model.get_layer(index=3).name
'layer1'
model.get_layer(index=4).name
'output'
```


```
python -m neuronWiggle --model models/mnist784_15x2_softmax1_1v2.keras --layerID 1 --seed 20 --dataset 'mnist' --quantized 2

python -m neuronWiggle --model models/mnist784_15x2_softmax1_1v2.keras --layerID 2 --seed 20 --dataset 'mnist' --quantized 2

python -m neuronWiggle --model models/mnist784_15x2_softmax1_1v2.keras --layerID 3 --seed 20 --dataset 'mnist' --quantized 2
```

