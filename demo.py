import warnings
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from src.utils import set_seed
from src.model import TEMPO, TEMPOConfig
from src.trainer import Trainer
from src.dataset import Dataset_ETT_minute, Dataset_ETT_hour

warnings.filterwarnings("ignore")

set_seed(3407)

trainset = Dataset_ETT_hour(root_path='./data', flag='train',)
valset = Dataset_ETT_hour(root_path='./data', flag='val')
testset = Dataset_ETT_hour(root_path='./data', flag='test')

config = TEMPOConfig(
    num_series=3,
    input_len=trainset.seq_len,
    pred_len=trainset.pred_len,
    n_layer=6,
    model_type='openai-community/gpt2',
    patch_size=16,
    patch_stride=8,
    lora=True,
    lora_config={
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1,
        'enable_lora': [True, True, False],
        'fan_in_fan_out': False,
        'merge_weights': False,
    },
    prompt_config={
        'embed_dim': 768,
        'top_k': 3,
        'prompt_length': 3,
        'pool_size': 30,
    },
    interpret=False,
)

model = TEMPO.from_pretrained(
    config=config,
)

print(f"num of total parameters: {model.num_params['total']/1e6: .2f}M")
print(f"num of trainable parameters: {model.num_params['grad']/1e6: .2f}M")

tra = Trainer(model, use_amp=True, features="S", num_workers=3)

tra.train(trainset, valset, batch_size=200, max_epochs=10, lr=0.001)