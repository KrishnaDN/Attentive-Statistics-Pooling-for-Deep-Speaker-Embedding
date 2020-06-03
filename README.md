# Attentive-Statistics-Pooling-for-Deep-Speaker-Embedding
This repo contains the implementation of the paper "Attentive Statistics Pooling for Deep Speaker Embedding" in Pytorch
The paper is published in Interspeech 2018
Paper: https://www.isca-speech.org/archive/Interspeech_2018/pdfs/0993.pdf

## Installation

I suggest you to install Anaconda3 in your system. First download Anancoda3 from https://docs.anaconda.com/anaconda/install/hashes/lin-3-64/
```bash
bash Anaconda2-2019.03-Linux-x86_64.sh
```
## Clone the repo
```bash
https://github.com/KrishnaDN/x-vector-pytorch.git
```
Once you install anaconda3 successfully, install required packges using requirements.txt
```bash
pip iinstall -r requirements.txt
```

## Data preperation
This step creates training and testing files. Currently we support TIMIT dataset and we plan to add other datasets in future.

```
python datasets.py --processed_data  /media/newhd/TIMIT --meta_store_path meta/ 
```
If you want to add your dataset, take a look at datasets.py code and modify the code accordingly


## Training
This steps starts training the X-vector model with Statistics attenive pooling.  
```
python training_xvector_SAP.py --training_filepath meta/training.txt --evaluation_path meta/evaluation.txt
                             --input_dim 257 --num_classes 462 --batch_size 64 --use_gpu True --num_epochs 100
                             
```

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
For any queries contact : krishnadn94@gmail.com
## License
[MIT](https://choosealicense.com/licenses/mit/)