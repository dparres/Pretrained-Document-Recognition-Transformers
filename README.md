# Pretrained-Document-Recognition-Transformers


Official implementation of *'Handwritten Document Recognition Using Pre-trained Vision Transformers'*.

  ![training_workflow](https://github.com/dparres/Pretrained-Document-Recognition-Transformers/assets/114649578/ad47cb0b-36f9-4f9f-bb7c-79c2ab949233)

## Training workflow

The script to train the models requires 5 parameters in the following order:

* Number of epochs: integer number (Example: ```50```)

* Learning rate: real number (Example: ```1e-4```)

* Frozen strategy:

  For Swin-CTC:

    - ```CTC-fclayer```: train only the last layer of the Swin+CTC model
    - ```CTC-Swin```: train only the encoder of the Swin+CTC model
    - ```all```: train the entire model
      
  For VED:

    - ```VED-encoder```: train only the encoder component of the model
    - ```VED-decoder```: train only the decoder component of the model
    - ```all```: train the entire model
 
* Batch size: integer number (Example: ```4```)

* Model:
  - ```Swin_CTC``` for the Swin+CTC model
  - ```VED``` for the VED model

* Num. of batchs for gradient accumulation: integer number (Example: ```4```)

To train the Swin-CTC model for 50 epochs with a learning rate of 1e-4, a batch size of 4, gradient accumulation of 4 batches, and employing the frozen strategy for the last layers, use the following command:
```
python train.py 50 1e-4 CTC-last 4 Swin_CTC 4
```



