## PaliGemma finetuning for Image Captioning Task
This repository contains the code for finetuning the PaliGemma model for the image captioning task.
Use first 5GB dataset from [here](https://huggingface.co/datasets/jackyhate/text-to-image-2M) to train the model.


## Getting Started - Training in Colab
In Colab, please run the following notebook. The code assumes you are using L4 GPU in Colab Pro/Pro+.
If your using different GPU, please change the batch_size in the `src/config.py` file.

```jupyter
# Cell 1
%%bash
git clone https://github.com/jinho-choi123/PaliGemma-ImageCaptioning.git

# Cell 2
%cd PaliGemma-ImageCaptioning


# Cell 3
%%capture
%%bash
source ./download-data.sh

# Cell 4
!pip install -q transformers bitsandbytes lightning peft datasets evaluate rouge_score

# Cell 5
from huggingface_hub import notebook_login
import wandb
notebook_login()
wandb.login(key="<your_wandb_key>") # add your wandb key here
```

After running all the cells, your environment is ready to train the model. Now open the Colab Terminal(for Colab Pro/Pro+) and run the following command to start training the model.
```bash
$ cd PaliGemma-ImageCaptioning
$ python -m src.train
```

## Getting Started - Inference in Colab
In Colab, please run the following notebook. For inference, T4 GPU is enough.
```jupyter
# Cell 1
%%bash
git clone https://github.com/jinho-choi123/PaliGemma-ImageCaptioning.git

# Cell 2
!pip install -q transformers peft
```

Next, run the following command in the Colab Terminal(or in notebook cell with `%%bash`)
```bash
$ cd PaliGemma-ImageCaptioning
$ python -m src.inference
```

If you want to change the image that is used in image captioning, put the new image in the `examples/` directory and change the `IMG_PATH` variable in the `src/inference.py` file.

## Before/After Training

To compare the performance of the model before and after training, we used a example image in `examples/example.jpg`.
It is a picture of Lebron James wearing Los Angeles Lakers jersey. 

![alt text](examples/example.jpg)

The model before training generated the following caption:
```
bron james
```

The model after training generated the following caption:
```
basketball player in a yellow jersey with the number 6 is holding a basketball on a court. The player has tattoos on his arms and legs, and is wearing a bracelet on his left wrist. The background shows a blurred view of other players and a referee, with a focus on the player in the foreground
```

The training made the model generate more detailed explanation of the image. 


