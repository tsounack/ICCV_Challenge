import torch
import itertools
import numpy as np
from tqdm import tqdm


def evaluation(models, config, dl, **kwargs):
    logits_list = []
    labels_list = []
    losses_list = []
    cumulative_index = 0
    post_processing = {}

    for num_batch, batch in enumerate(tqdm(dl, total=len(dl))):
        label = batch['labels']
        batch_size = label.shape[0]
        num_classes = label.shape[1]

        batch = {k: v.cuda() if (isinstance(v, torch.Tensor) and torch.cuda.is_available()) else v for k, v in batch.items()}
        results = [model(**batch) for model in models]

        # Pre-allocate memory
        if num_batch == 0:
            num_samples = len(dl.dataset)
            logits_list = [np.zeros((num_samples, len(models), num_classes)) for _ in range(batch_size)]
            labels_list = [np.zeros((num_samples, num_classes)) for _ in range(batch_size)]
            losses_list = [[] for _ in range(len(dl))]

        # iterating over the batch, stacking refs and hyps
        for i in range(batch_size):
            for j, r in enumerate(results):
                logits_list[i][cumulative_index: cumulative_index + batch_size, j] = r['output'][i].data.cpu().numpy()
            labels_list[i][cumulative_index: cumulative_index + batch_size] = label[i].data.cpu().numpy()

        # Loss
        for j, r in enumerate(results):
            loss_value = r['loss'].cpu().item() if isinstance(r['loss'], torch.Tensor) and r['loss'].numel() == 1 else r['loss'].cpu().numpy()
            losses_list[num_batch].append(loss_value)

        cumulative_index += batch_size

        break

    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    losses = np.concatenate(losses_list, axis=0)

    preds = np.mean(logits, axis=1)
    loss = np.mean(losses)

    return {'loss': loss, 'refs': labels, 'hyps': logits}
