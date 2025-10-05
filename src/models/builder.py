from __future__ import annotations

import torch
from transformers import ViTConfig

from .embedding import apply_patch_embed_pca
from .preprocessor import build_preprocessor
from .vit import GlobalAttnViT, MyViT, PreconditionedViT

__all__ = [
    "get_model",
    "get_vit_pretrain_model",
    "get_pca_config",
    "get_vit_config",
]


def _load_pca_stats(pca_path: str):
    try:
        loaded = torch.load(pca_path, weights_only=True)
        tname = type(loaded).__name__
        print(f"[warmup] Loaded PCA from '{pca_path}' (weights_only=True) type={tname}")
    except Exception as exc:
        loaded = torch.load(pca_path)
        tname = type(loaded).__name__
        print(
            f"[warmup] Loaded PCA from '{pca_path}' (fallback weights_only=False) type={tname}; reason: {exc}"
        )
    if isinstance(loaded, dict):
        try:
            keys = list(loaded.keys())
            has_mean = "mean" in loaded
            print(f"[warmup] PCA keys: {keys}; has_mean={has_mean}")
        except Exception:
            pass
        return loaded
    try:
        print(f"[warmup] PCA loaded as raw tensor with shape={tuple(loaded.shape)} (stored under key 'V')")
    except Exception:
        pass
    return {"V": loaded}


def get_model(config):
    warmup_cfg = get_pca_config(config)
    vit_config = get_vit_config(config)
    loss_name = config.get("loss", {}).get("name", None)
    freeze_epochs_unified = int(warmup_cfg.get("freeze_qk_epochs", 0) or 0)
    try:
        print(
            f"[warmup] Settings: global={bool(warmup_cfg.get('global', False))}, embed={bool(warmup_cfg.get('embed', False))}, freeze_epochs={freeze_epochs_unified}"
        )
    except Exception:
        pass

    preproc_kind = warmup_cfg.get("preprocessor", None)
    if isinstance(preproc_kind, str):
        lowered = preproc_kind.lower()
        if lowered in {"linear", "linear_affine", "feature_linear"}:
            preproc_kind = "zca_linear"
        else:
            preproc_kind = lowered
    if preproc_kind is None and bool(warmup_cfg.get("global", False)):
        preproc_kind = "global_attention"

    model = None

    if preproc_kind in {"global_attention", "zca_linear"}:
        UV = warmup_cfg.get("UV", "V")
        r = warmup_cfg.get("r", None)
        pca_path = (
            warmup_cfg.get("stats_path")
            or warmup_cfg.get("feature_stats_path")
            or warmup_cfg.get("linear_stats_path")
            or warmup_cfg.get("global_pca_path", None)
            or warmup_cfg.get("pca_path", None)
        )
        use_input_bias = bool(warmup_cfg.get("bias", False))
        use_lora = bool(warmup_cfg.get("lora", False))
        try:
            print(
                f"[warmup] Preprocessor='{preproc_kind}', UV={UV}, r={r}, lora={use_lora}, bias={use_input_bias}, pca_path={pca_path}"
            )
        except Exception:
            pass
        pca_stats = None
        if pca_path is not None:
            pca_stats = _load_pca_stats(pca_path)

        if preproc_kind == "global_attention":
            if (r is not None) and int(r) == 0:
                pca_stats = None
            qk_freeze_epochs = int(freeze_epochs_unified)
            if pca_stats is not None:
                model = GlobalAttnViT(
                    vit_config,
                    pca_stats=pca_stats,
                    loss_name=loss_name,
                    r=r,
                    use_lora=use_lora,
                    qk_freeze_epochs=qk_freeze_epochs,
                    UV=UV,
                    use_input_bias=use_input_bias,
                )
                try:
                    print(f"[global-warmup] Freeze schedule: qk_freeze_epochs={qk_freeze_epochs}")
                except Exception:
                    pass
            else:
                try:
                    print(
                        "[global-warmup] No PCA stats loaded (either no path provided or r==0); using default init for Q/K/V"
                    )
                except Exception:
                    pass
        elif preproc_kind == "zca_linear":
            if pca_stats is None:
                raise ValueError("`zca_linear` preprocessor selected but no PCA/ZCA statistics were provided")
            allow_rect = bool(warmup_cfg.get("zca_allow_rectangular", False))
            preprocessor = build_preprocessor(
                "zca_linear",
                input_dim=int(vit_config.image_size),
                pca_stats=pca_stats,
                uv_key=UV,
                use_input_bias=use_input_bias,
                freeze=False,
                allow_rectangular=allow_rect,
            )
            model = PreconditionedViT(
                vit_config,
                preprocessor=preprocessor,
                loss_name=loss_name,
                model_name="ZCA_ViT",
                freeze_epochs=int(freeze_epochs_unified),
            )
            try:
                shape = tuple(preprocessor.linear.lin.weight.shape)
                print(f"[zca-warmup] ZCA preprocessor initialised with weight shape={shape}")
            except Exception:
                pass

    if model is None:
        model = MyViT(vit_config, loss_name=loss_name)

    try:
        if bool(warmup_cfg.get("embed", False)):
            pth = warmup_cfg.get("embed_pca_path", "pca_patch.pt")
            basis_key = str(warmup_cfg.get("UV", "V"))
            use_mean = bool(warmup_cfg.get("use_pca_mean", False))
            print(
                f"[embed-warmup] Config: path='{pth}', basis={basis_key}, use_pca_mean={use_mean}"
            )
            apply_patch_embed_pca(model, warmup_cfg)
            try:
                model._model_name = f"e{basis_key}{model._model_name}"
            except Exception:
                pass
    except Exception as exc:
        print(f"[embed-warmup] Skipped due to error: {exc}")

    try:
        if hasattr(model, "embed_freeze_epochs"):
            model.embed_freeze_epochs = int(freeze_epochs_unified)
    except Exception:
        pass

    try:
        fz_qk = int(getattr(model, "qk_freeze_epochs", 0) or 0)
        fz_emb = int(getattr(model, "embed_freeze_epochs", 0) or 0)
        fz_cfg = int(freeze_epochs_unified or 0)
        fz_val = max(fz_qk, fz_emb, fz_cfg)
        if isinstance(getattr(model, "_model_name", None), str):
            name = model._model_name
            if "_fz" in name:
                if fz_val > 0 and "_fz0_" in name:
                    model._model_name = name.replace("_fz0_", f"_fz{fz_val}_")
            else:
                if fz_val > 0:
                    model._model_name = f"{name}_fz{fz_val}"
    except Exception:
        pass

    return model


def get_vit_pretrain_model(config):
    vit_config = get_vit_config(config)
    loss_name = config.get("loss", {}).get("name", None)
    return MyViT(vit_config, loss_name=loss_name)


def get_pca_config(config):
    return config.get("warmup", {})


def get_vit_config(config):
    m = config["model"]
    d = config.get("data", {})
    num_labels = int(m.get("num_labels", 1) or 1)
    task = (m.get("task_type") or m.get("task") or "cls").lower()
    if task in ("reg", "regression"):
        p = d.get("param", None)
        if isinstance(p, str) and len(p) > 0:
            plist = [x.strip() for x in p.split(",") if x.strip()]
            if len(plist) >= 1:
                num_labels = len(plist)
        elif isinstance(p, (list, tuple)) and len(p) > 0:
            num_labels = len(p)
        try:
            m["num_labels"] = num_labels
        except Exception:
            pass

    return ViTConfig(
        task_type=m["task_type"],
        image_size=m["image_size"],
        patch_size=m["patch_size"],
        num_channels=1,
        hidden_size=m["hidden_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        intermediate_size=4 * m["hidden_size"],
        stride_ratio=m.get("stride_ratio", 1),
        stride_size=m.get("stride_size", None),
        proj_fn=m["proj_fn"],
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        use_mask_token=False,
        qkv_bias=True,
        num_labels=num_labels,
    )
