import os
import os.path as osp
import time
import warnings
from typing import Callable, Dict, Iterable

import torch
import torch_geometric.utils as tgu
from einops import einsum, rearrange, reduce
from sklearn.metrics import f1_score, mean_absolute_error
from torch import Tensor, nn
from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from torchmetrics.functional.classification import accuracy, average_precision
from tqdm import tqdm

from graphgps.logger import create_logger
from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy

warnings.filterwarnings("ignore")


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    os.makedirs(cfg.run_dir, exist_ok=True)


def get_ckpt(result_root, model_folder):
    def get_final_pretrained_ckpt(ckpt_dir):
        if osp.exists(ckpt_dir):
            names = os.listdir(ckpt_dir)
            epochs = [int(name.split(".")[0]) for name in names]
            final_epoch = max(epochs)
        else:
            raise FileNotFoundError(f"Pretrained model dir not found: {ckpt_dir}")
        return osp.join(ckpt_dir, f"{final_epoch}.ckpt"), final_epoch

    ckpt_dir = osp.join(result_root, model_folder, "0/ckpt")
    ckpt_path, final_epoch = get_final_pretrained_ckpt(ckpt_dir)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    return ckpt, final_epoch


def init_model(model_cfg, loaders):
    # Load config file
    args = parse_args()
    args.cfg_file = model_cfg
    set_cfg(cfg)
    cfg.run_dir = ""
    load_cfg(cfg, args)
    cfg.share.dim_out = 10  # ! 10 for func | 11 for struct | LP should commont out | 21 for vocsuperpixels
    cfg.train.batch_size = 1
    # init model
    model = create_model()

    if not loaders:  # if loader is None
        loaders = create_loader()
    return model, loaders


def load_models(model_names):
    result_root = "results"
    models = []
    loaders = None
    for m_name in model_names:
        model_folder = "-".join(
            ["peptides-func", m_name]
        )  # + "+LapPE"  # ! pcqm-contact ｜ peptides-struct ｜ peptides-func | vocsuperpixels
        model_cfg_path = osp.join(result_root, model_folder, "config.yaml")
        model, loaders = init_model(model_cfg_path, loaders)
        ckpt, final_epoch = get_ckpt(result_root, model_folder)

        model.load_state_dict(ckpt["model_state"])
        models.append(model)

    return models, loaders, final_epoch


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        self.model.eval()
        _ = self.model(x)
        return self._features


def get_dirchlect_energy_score(X, edge_index, edge_weight=None):
    X = X.float()
    if edge_weight and edge_weight.shape[1] > 1:
        edge_weight = edge_weight.float().mean(1)
    L_edge_index, L_edge_weight = tgu.get_laplacian(
        edge_index=edge_index, edge_weight=edge_weight, normalization="sym"
    )
    L = tgu.to_dense_adj(edge_index=L_edge_index, edge_attr=L_edge_weight)[0]

    try:
        return torch.trace(X.T.mm(L.mm(X))).item()
    except:
        return -1


def f1_score_gps(pred_score, true):
    def _get_pred_int(pred_score):
        if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
            return (pred_score > cfg.model.thresh).long()
        else:
            return pred_score.max(dim=1)[1]

    return f1_score(true, _get_pred_int(pred_score), average="macro", zero_division=0)


@torch.no_grad()
def eval_epoch(logger, loader, model, split="val"):
    model.eval()

    layers_name = [
        f"gnn_layers.{layer}", # vn layer 0
    ]  # post_mp.layer_post_mp ; mp.layer4
    gnn_hooker = FeatureExtractor(model, layers=layers_name)

    time_start = time.time()
    for batch in tqdm(loader, desc=f"{split}..."):
        batch.split = split
        batch.to(torch.device(cfg.device))
        if cfg.gnn.head == "inductive_edge":
            pred, true, extra_stats = model(batch)
            score = extra_stats["mrr"]
        else:
            hooker_batch = batch.clone()
            batch_info = gnn_hooker(hooker_batch)[layers_name[0]]
            batch_x, ori_mask = to_dense_batch(batch_info.x, batch_info.batch)
            mask = (~ori_mask).unsqueeze(1).to(dtype=batch_info.x.dtype) * -1e9
            # * S for node cluster allocation matrix
            vns_emb, S = model.multi_vn_layers[layer](batch_x, None, mask)
            
            graphs.append(batch_info.edge_index.detach().cpu())
            attention_M.append(reduce(S, "h n d -> n d", "mean",).detach().cpu())

            pred, true = model(batch)
            loss, pred_score = compute_loss(pred, true)

            # dirchlect_score = get_dirchlect_energy_score(
            #     X=node_embedding[layers_name[0]].x,
            #     edge_index=node_embedding[layers_name[0]].edge_index,
            #     # X=node_embedding[layers_name[0]],
            #     # edge_index=hooker_batch.edge_index,
            #     # edge_weight=node_embedding["mp.layer4"].edge_attr,
            # )
            #  ! change metric
            # score = average_precision(pred_score, true).item()  # * func
            # score = mean_absolute_error(
            #     true.cpu().numpy(), pred_score.cpu().numpy()
            # )  # * struct
            # score = f1_score_gps(pred_score.cpu(), true.cpu()).item()  # * voc
            extra_stats = {}

        # embeddings.append(graph_embedding)
        # label.append(true)

        if cfg.dataset.name == "ogbg-code2":
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to("cpu", non_blocking=True)
            _pred = pred_score.detach().to("cpu", non_blocking=True)
        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=0,
            time_used=time.time() - time_start,
            params={"Optuna": 666},
            dataset_name=cfg.dataset.name,
            **extra_stats,
        )
        time_start = time.time()


def set_up(model_names):
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    custom_set_run_dir(cfg, 0)
    run_dir = cfg.run_dir
    out_dir = cfg.out_dir
    seed_everything(cfg.seed)

    models, loaders, _ = load_models(model_names)

    cfg.run_dir = run_dir
    cfg.out_dir = out_dir
    cfg.share.num_splits = 3
    loggers = create_logger()

    return models, loaders, loggers


model_names = [
    "GatedGCN",
]  # 'GINE', 'GCN'
models, loaders, loggers = set_up(model_names)
graphs = []
attention_M = []
layer = 1

for idx, model in enumerate(models):
    print(f"Now {model_names[idx]}...")
    eval_epoch(loggers[1], loaders[1], model, split="val")
    # eval_epoch(loggers[2], loaders[2], model, split="test")
    # loggers[2].write_epoch(idx)

torch.save(graphs, 'graphs.pt')
torch.save(attention_M, f'attention_M_l{layer}.pt')