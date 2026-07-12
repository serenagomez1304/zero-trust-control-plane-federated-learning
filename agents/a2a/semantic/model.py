"""
Learned trust_eval + the real semantic model
=============================================
``TrustEvalModel`` is the frozen-encoder + trainable-head classifier that scores
purpose/intent/context consistency in [0, 1]. ``RealSemanticModel`` plugs it into
the ``SemanticModel`` interface, with deterministic ``transform`` / ``integrate``
/ ``synthesize`` (see ``docs/m2_design.md``).

Only the head is trained, so federated learning (M4) aggregates a small weight
vector. The encoder is frozen and reconstructed at load time from its spec.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from agents.a2a.trust import (
    Attestation,
    Content,
    ContextEntry,
    DeclaredPurpose,
    Kappa,
    SemanticModel,
)
from agents.a2a.semantic.data import TrustExample
from agents.a2a.semantic.encoders import Encoder, get_encoder

# Longest accumulated context kept verbatim by ``integrate`` before trimming.
MAX_CONTEXT_ENTRIES = 8


# -----------------------------------------------------------------------------
# Shared text construction (identical for training and inference)
# -----------------------------------------------------------------------------

def purpose_text(label: str, description: str) -> str:
    return f"{label}: {description}".strip().rstrip(":").strip() if description else label


def context_text_from_kappa(kappa: Kappa) -> str:
    if not kappa.entries:
        return "(no prior context)"
    return " | ".join(e.response for e in kappa.entries)


def _triples(examples: Sequence[TrustExample]) -> List[Tuple[str, str, str]]:
    return [(e.intent, purpose_text(e.purpose_label, e.purpose_description), e.context) for e in examples]


# -----------------------------------------------------------------------------
# The trainable head
# -----------------------------------------------------------------------------

class _TrustHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits


def build_trust_head(in_dim: int, hidden: int = 128, dropout: float = 0.1) -> nn.Module:
    """Construct a fresh trust head — used by the federated simulation (M4)."""
    return _TrustHead(in_dim, hidden=hidden, dropout=dropout)


class TrustEvalModel:
    """Frozen-encoder + trainable-head consistency classifier."""

    def __init__(self, encoder: Encoder, hidden: int = 128, dropout: float = 0.1):
        self.encoder = encoder
        self.hidden = hidden
        self.dropout = dropout
        # Features: concat(e_intent, e_purpose, e_context, e_intent ⊙ e_purpose).
        self.in_dim = encoder.dim * 4
        self.head = _TrustHead(self.in_dim, hidden=hidden, dropout=dropout)

    # --- feature construction -------------------------------------------------

    def _features(self, triples: Sequence[Tuple[str, str, str]]) -> np.ndarray:
        intents = [t[0] for t in triples]
        purposes = [t[1] for t in triples]
        contexts = [t[2] for t in triples]
        e_i = self.encoder.encode(intents)
        e_p = self.encoder.encode(purposes)
        e_c = self.encoder.encode(contexts)
        return np.concatenate([e_i, e_p, e_c, e_i * e_p], axis=1).astype(np.float32)

    def features_for(self, examples: Sequence[TrustExample]) -> np.ndarray:
        """Encode examples into the head's input features (frozen encoder).

        Exposed for the federated simulation (M4), which precomputes each
        client's feature matrix once and then trains the head over many rounds.
        """
        return self._features(_triples(examples))

    # --- training -------------------------------------------------------------

    def fit(
        self,
        examples: Sequence[TrustExample],
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed)

        X = torch.from_numpy(self._features(_triples(examples)))
        y = torch.tensor([float(e.label) for e in examples])

        opt = torch.optim.Adam(self.head.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        self.head.train()
        n = len(examples)
        last_loss = float("nan")
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                opt.zero_grad()
                loss = loss_fn(self.head(X[idx]), y[idx])
                loss.backward()
                opt.step()
                last_loss = float(loss.item())
        return {"final_batch_loss": last_loss, "n_train": n}

    # --- inference ------------------------------------------------------------

    def predict_proba(self, triples: Sequence[Tuple[str, str, str]]) -> np.ndarray:
        self.head.eval()
        with torch.no_grad():
            X = torch.from_numpy(self._features(triples))
            return torch.sigmoid(self.head(X)).cpu().numpy()

    def predict_one(self, intent: str, purpose: str, context: str) -> float:
        return float(self.predict_proba([(intent, purpose, context)])[0])

    def evaluate(self, examples: Sequence[TrustExample], threshold: float = 0.5) -> dict:
        probs = self.predict_proba(_triples(examples))
        preds = (probs >= threshold).astype(int)
        labels = np.array([e.label for e in examples])
        acc = float((preds == labels).mean())
        pos = labels == 1
        neg = labels == 0
        return {
            "accuracy": acc,
            "n": len(examples),
            "recall_consistent": float((preds[pos] == 1).mean()) if pos.any() else float("nan"),
            "recall_rogue": float((preds[neg] == 0).mean()) if neg.any() else float("nan"),
        }

    # --- persistence ----------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "encoder_spec": self.encoder.spec,
                "in_dim": self.in_dim,
                "hidden": self.hidden,
                "dropout": self.dropout,
                "head_state": self.head.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, encoder: Encoder | None = None) -> "TrustEvalModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        enc = encoder if encoder is not None else get_encoder(ckpt["encoder_spec"])
        model = cls(enc, hidden=ckpt["hidden"], dropout=ckpt["dropout"])
        model.head.load_state_dict(ckpt["head_state"])
        return model


# -----------------------------------------------------------------------------
# The real semantic model (implements the SemanticModel interface)
# -----------------------------------------------------------------------------

_WORKER_PREFIXES = {
    "airline": ("flight", "airport"),
    "hotel": ("hotel", "city"),
    "car_rental": ("vehicle", "rental"),
}
_ROUTING_PREFIXES = ("route", "intent")


def _domain_of_purpose(label: str) -> str:
    low = label.lower()
    for domain, prefixes in _WORKER_PREFIXES.items():
        if low.startswith(prefixes):
            return domain
    if low.startswith(_ROUTING_PREFIXES):
        return "routing"
    return "other"


class RealSemanticModel(SemanticModel):
    """M2 semantic model: learned ``trust_eval``, deterministic everything else."""

    def __init__(self, trust_model: TrustEvalModel):
        self.trust_model = trust_model

    def trust_eval(self, purpose: DeclaredPurpose, attestation: Attestation, kappa: Kappa) -> float:
        return self.trust_model.predict_one(
            intent=kappa.intent,
            purpose=purpose_text(purpose.label, purpose.description),
            context=context_text_from_kappa(kappa),
        )

    def transform(self, content: Content, purpose: DeclaredPurpose, kappa: Kappa) -> Content:
        """Rule-based purpose-scoped projection.

        The view restricts structured data to fields relevant to the purpose's
        domain and annotates it with the declared purpose, so an agent only sees
        a view justified by its purpose — not the raw structured payload.
        """
        domain = _domain_of_purpose(purpose.label)
        if domain in _WORKER_PREFIXES:
            allowed = _WORKER_PREFIXES[domain]
            scoped = {k: v for k, v in content.data.items() if k.lower().startswith(allowed)}
        else:
            scoped = dict(content.data)
        scoped = {**scoped, "_declared_purpose": purpose.label}
        return Content(payload=content.payload, data=scoped)

    def integrate(self, response: str, kappa: Kappa, purpose: DeclaredPurpose) -> Kappa:
        """Append the response, trimming the oldest entries when context grows long."""
        updated = kappa.model_copy(deep=True)
        updated.entries.append(ContextEntry(purpose=purpose.label, response=response))
        if len(updated.entries) > MAX_CONTEXT_ENTRIES:
            dropped = len(updated.entries) - MAX_CONTEXT_ENTRIES
            kept = updated.entries[-MAX_CONTEXT_ENTRIES:]
            updated.entries = [
                ContextEntry(purpose="context-summary", response=f"(earlier context: {dropped} step(s) elided)"),
                *kept,
            ]
        return updated

    def synthesize(self, kappa: Kappa) -> str:
        """Extractive composition over accumulated context.

        Compose the substantive worker responses (dropping pure routing/summary
        steps) rather than echoing a single agent's output.
        """
        if not kappa.entries:
            return ""
        substantive = [
            e for e in kappa.entries
            if _domain_of_purpose(e.purpose) not in ("routing",) and e.purpose != "context-summary"
        ]
        chosen = substantive or kappa.entries
        if len(chosen) == 1:
            return chosen[0].response
        return " ".join(e.response for e in chosen)
