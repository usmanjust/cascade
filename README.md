CASCADE: Next-Basket Recommendation

Requirements

Install the required dependencies:

pip install -r requirements.txt


The code requires:

Python ≥ 3.9

PyTorch ≥ 1.12

NumPy

SciPy

tqdm

Install PyTorch with CUDA support if GPU training is desired.

Data Preparation

Prepare each dataset in the following format:

basket_1 | basket_2 | ... | basket_t |


Each basket is a space-separated list of item IDs.

Place the files in:

data/<dataset_name>/
├── train.txt
├── val.txt
└── test.txt

Training

To train the model:

python main.py --dataset <dataset_name> --mode train --model cascade --use_adaptive --use_causal


Evaluation

To evaluate the trained model:

python main.py --dataset <dataset_name> --mode test --model cascade


The results will be printed to the console and saved in the results/ directory.

Output

Trained models are saved in saved_models/

Evaluation results are saved in results/