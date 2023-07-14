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

        model_path = "/home/users/tsounack/vilmedic/ICCV_Challenge/ckpt_chexzero/best_128_5e-05_original_22000_0.855.pt"
        
        # if a model_path is provided, load in weights to backbone
        if model_path != None: 
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.transformer_in = model.visual

        last_layer_size = 512

        self.adapter = nn.Sequential(
            nn.Linear(last_layer_size, adapter.pop('output_size')),
            torch.nn.LayerNorm(transformer.hidden_size, eps=transformer.layer_norm_eps)
        )

        bert_conf = BertGenerationConfig(**transformer)
        self.transformer = BertEncoder(bert_conf)
        self.pooler = BertPooler(bert_conf)

        classifier_func = classifier.pop('proto')
        self.classifier = eval(classifier_func)(**classifier)

        loss_func = loss.pop('proto')

        self.loss_func = eval(loss_func)(**loss).cuda()

        print(self.loss_func)
        # Evaluation
        self.eval_func = evaluation

    def forward(self, images, labels=None, from_training=True, iteration=None, epoch=None, **kwargs):
        out = self.transformer_in(images.cuda())
        print("passed transformer_in")
        out = self.adapter(out)
        print("passed adapter")
        out = self.transformer(out, output_attentions=True)
        print("passed transformer")

        attentions = out.attentions  # num_layers, batch_size, num_heads, sequence_length, sequence_length
        
        out = self.pooler(out.last_hidden_state)
        print("passed pooler")
        out = self.classifier(out)

        loss = torch.tensor(0.)
        if from_training:
            loss = self.loss_func(out, labels.cuda(), **kwargs)

        return {'loss': loss, 'output': out, 'answer': torch.argmax(out, dim=-1), 'attentions': attentions}

    def __repr__(self):
        s = super().__repr__() + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
