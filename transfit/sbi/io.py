# transfit/sbi/io.py
"""Save/load trained SBI posteriors."""
from __future__ import annotations

import io
import pickle
from typing import Any, Dict

import torch
import torch.nn as nn

from .embedding import SetSummaryNet, MLPEmbeddingNet
from .posterior import SBIPosterior


def save_posterior(posterior: SBIPosterior, path: str) -> None:
    """Save trained posterior to disk.

    Saves the full SBIPosterior as a single .pt file.

    Parameters
    ----------
    posterior : SBIPosterior
        Trained posterior to save.
    path : str
        Output path (e.g. "my_posterior.pt").
    """
    sbi_posterior_blob = _serialize_sbi_posterior(posterior.posterior)
    meta = dict(posterior.meta)
    meta["posterior_serialized"] = sbi_posterior_blob is not None

    state = {
        "model": posterior.model,
        "param_names": posterior.param_names,
        "mode": posterior.mode,
        "band_vocabulary": posterior.band_vocabulary,
        "t_range": list(posterior.t_range),
        "meta": meta,
        "sbi_posterior_blob": sbi_posterior_blob,
        # Also save embedding net separately for reconstruction flexibility
        "embedding_net_state": posterior.embedding_net.state_dict(),
        "embedding_net_class": type(posterior.embedding_net).__name__,
        "embedding_net_config": _get_net_config(posterior.embedding_net),
    }
    torch.save(state, path)


def load_posterior(path: str, *, trusted: bool = False) -> SBIPosterior:
    """Load a saved SBI posterior.

    Parameters
    ----------
    path : str
        Path to saved .pt file.
    trusted : bool
        Must be True to load (mirrors FitResult.load security pattern).

    Returns
    -------
    SBIPosterior
    """
    if not trusted:
        raise ValueError(
            "Loading an SBI posterior requires trusted=True. "
            "Only load posteriors from sources you trust, as they contain "
            "serialized PyTorch model weights."
        )

    state = torch.load(path, map_location="cpu", weights_only=False)

    # Reconstruct embedding network
    embedding_net = _reconstruct_embedding_net(
        state["embedding_net_class"],
        state["embedding_net_config"],
        state["embedding_net_state"],
    )

    if state.get("sbi_posterior_blob") is not None:
        sbi_posterior = _deserialize_sbi_posterior(state["sbi_posterior_blob"])
    else:
        # Backward compatibility for older .pt files that stored the object directly.
        sbi_posterior = state.get("sbi_posterior")

    return SBIPosterior(
        model=state["model"],
        param_names=state["param_names"],
        posterior=sbi_posterior,
        embedding_net=embedding_net,
        meta=state.get("meta", {}),
        band_vocabulary=state.get("band_vocabulary"),
        t_range=tuple(state.get("t_range", (0.0, 150.0))),
        mode=state.get("mode", "multiband"),
    )


def _serialize_sbi_posterior(posterior_obj: Any) -> bytes | None:
    """Serialize the internal sbi posterior when it is pickleable.

    Test doubles may be defined as local classes and cannot be pickled. In that
    case we still save the metadata and embedding network so round-trip IO stays
    usable for lightweight tests.
    """
    if posterior_obj is None:
        return None

    buffer = io.BytesIO()
    try:
        torch.save(posterior_obj, buffer)
    except (AttributeError, TypeError, pickle.PicklingError):
        return None
    return buffer.getvalue()


def _deserialize_sbi_posterior(blob: bytes) -> Any:
    buffer = io.BytesIO(blob)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def _get_net_config(net: nn.Module) -> Dict[str, Any]:
    """Extract config needed to reconstruct an embedding network."""
    if isinstance(net, SetSummaryNet):
        # phi[0].in_features = feature_dim - 1 (mask channel is stripped in forward)
        phi_lin = net.phi[0]
        rho_lin = net.rho[-1]
        return {
            "feature_dim": phi_lin.in_features + 1,
            "hidden_features": phi_lin.out_features,
            "output_dim": rho_lin.out_features,
        }
    elif isinstance(net, MLPEmbeddingNet):
        first = net.net[0]
        last = net.net[-1]
        return {
            "input_dim": first.in_features,
            "hidden_features": first.out_features,
            "output_dim": last.out_features,
        }
    return {}


def _reconstruct_embedding_net(
    class_name: str, config: Dict[str, Any], state_dict: dict
) -> nn.Module:
    """Reconstruct an embedding network from saved config and state dict."""
    if class_name == "SetSummaryNet":
        net = SetSummaryNet(**config)
    elif class_name == "MLPEmbeddingNet":
        net = MLPEmbeddingNet(**config)
    else:
        raise ValueError(f"Unknown embedding net class: {class_name}")

    net.load_state_dict(state_dict)
    net.eval()
    return net
