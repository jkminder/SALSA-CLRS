import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from loguru import logger
import time    

def to_one_hot(preds):
    """
    Converts probabilities to one-hot encoding
    """

    one_hot = torch.zeros_like(preds)
    one_hot[preds.argmax(dim=-1)] = 1

    return one_hot

def calc_metrics(key, preds, batch, type_):
    truth = batch[key]
    preds = preds[key]
    # count ones
    graph_sizes = [(batch.batch == i).sum() for i in range(0, max(batch.batch)+1)]

    if type_ == "pointer":
        """Node Acc. and Graph Acc."""
        node_level = [[] for _ in range(batch.num_graphs)]
        
        for n in range(batch.num_nodes):
            idx = (batch.edge_index[0] == n)
            graph_idx = batch.batch[n]
            
            correct = (preds[idx].argmax(dim=-1) == truth[idx].argmax(dim=-1)).float()
            
            # Calculate node metrics
            node_level[graph_idx].append(correct)

        # Collapse to per graph metrics
        node_level = [torch.tensor(x) for x in node_level]

        # Graph Metrics
        graph_result = torch.tensor([x.all() for x in node_level])

        # Node Metrics
        node_acc = torch.tensor([x.mean() for x in node_level])

        return {
            "node_accuracy": node_acc,
            "graph_result": graph_result,
        }

    elif type_ == "mask":
        """Node Acc., Node F1, Graph Acc."""
        if truth.sum() < 0.05 * truth.numel():
            logger.warning(f"MASK METRIC: Truth has less than 5% ones: {truth.sum()} / {truth.numel()}")

        preds = preds.sigmoid()
        node_f1 = []
        node_acc = []
        graph_result = []


        for n in range(batch.num_graphs):
            gpred = (preds[batch.batch == n]>0.5).bool().cpu().numpy()
            gtruth = truth[batch.batch == n].cpu().numpy()

            node_f1.append(f1_score(gtruth, gpred, average='binary'))
            node_acc.append(accuracy_score(gtruth, gpred))
            graph_result.append((gpred == gtruth).all())

        graph_result = torch.tensor(graph_result)

        return {
                "node_accuracy": node_acc,
                "node_f1": node_f1,
                "graph_result": graph_result,
            }

    elif type_ == "scalar":
        """MSE, Graph Acc."""

        mse = []
        graph_result = []

        for n in range(batch.num_graphs):
            gpred = preds[n]
            gtruth = truth[n]

            mse.append(((gpred - gtruth)**2).mean())
            graph_result.append((gpred.round() == gtruth).float())

        mse = torch.tensor(mse)
        graph_result = torch.tensor(graph_result)

        return {
                "mse": mse,
                "graph_result": graph_result,
            }
    else:
        raise NotImplementedError(f"Unknown metric type {type_}")      