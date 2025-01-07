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


