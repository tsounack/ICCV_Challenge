import torch
import torch.nn as nn
from vilmedic.blocks.vision import *
from vilmedic.blocks.classifier import *
from vilmedic.blocks.classifier.evaluation import evaluation
from vilmedic.blocks.losses import *
from vilmedic.models.iccv.model import *
from vilmedic.models.iccv import clip

from vilmedic.models.utils import get_n_params

from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from transformers.models.bert_generation import BertGenerationConfig
from vilmedic.blocks.losses.mvqa.LabelSmoothingCrossEntropyLoss import WeightedBCEWithLogitsLoss


class ICCV(nn.Module):
    def __init__(self, classifier, adapter, transformer, loss, **kwargs):
        super(ICCV, self).__init__()

        params = {
            'embed_dim':768,
            'image_resolution': 320,
            'vision_layers': 12,
            'vision_width': 768,
            'vision_patch_size': 16,
            'context_length': 77,
            'vocab_size': 49408,
            'transformer_width': 512,
            'transformer_heads': 8,
            'transformer_layers': 12
        }
        
        # set device 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # load clip pre-trained model
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded in pretrained model.")

        model_path = "/scratch/users/tsounack/ICCV/best_64_5e-05_original_22000_0.864.pt"
        
        # if a model_path is provided, load in weights to backbone
        if model_path != None: 
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.transformer_in = model.visual

        input_dim = 512
        output_dim = 27

        # hidden_units = [400, 300, 200, 100, 50]

        self.dense = model = nn.Sequential(
            nn.Linear(input_dim, 250),
            nn.ReLU(),
            nn.Linear(250, output_dim)
        )

        loss_func = loss.pop('proto')

        self.loss_func = eval(loss_func)(**loss).cuda()

        print(self.loss_func)
        # Evaluation
        self.eval_func = evaluation

    def forward(self, images, labels=None, from_training=True, iteration=None, epoch=None, **kwargs):
        out = self.transformer_in(images.cuda())
        out = self.dense(out)

        loss = torch.tensor(0.)
        if from_training:
            loss = self.loss_func(out, labels.cuda(), **kwargs)

        return {'loss': loss, 'output': out, 'answer': torch.argmax(out, dim=-1), 'attentions': None}

    def __repr__(self):
        s = super().__repr__() + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
