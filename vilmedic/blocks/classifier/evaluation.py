import torch
import itertools
import numpy as np
from tqdm import tqdm


def evaluation(models, config, dl, **kwargs):
    logits = np.array([])
    labels = np.array([])
    losses = np.array([])
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
            logits = np.zeros((len(dl.dataset), len(models), num_classes))
            labels = np.zeros((len(dl.dataset), num_classes))
            losses = np.zeros((len(dl), len(models)))

        # iterating over the batch, stacking refs and hyps
        for i in range(batch_size):
            for j, r in enumerate(results):
                logits[cumulative_index + i][j] = r['output'][i].data.cpu().numpy()
            labels[cumulative_index + i] = label[i].data.cpu().numpy()

        # Loss
        for j, r in enumerate(results):
            losses[num_batch][j] = r['loss'].cpu().item()

        cumulative_index += batch_size

        break

    preds = np.mean(logits, axis=1)
    loss = np.mean(losses)
    logits = np.squeeze(logits)
    labels = np.squeeze(labels)

    return {'loss': loss, 'refs': labels, 'hyps': logits}
