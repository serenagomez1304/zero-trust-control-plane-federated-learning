"""
Semantic Model (Milestone 2)
============================
The real implementation of the message-resident semantic model ``mu``.

Of the four capabilities, only ``trust_eval`` is a learned model (frozen
encoder + trainable head); ``transform`` / ``integrate`` / ``synthesize`` are
real-but-deterministic. See ``docs/m2_design.md``.

Submodules:
- encoders.py: text encoder interface + sentence-transformer / hashing impls
- data.py:     synthetic (intent, purpose, context) -> consistent? dataset
- model.py:    TrustEvalModel (torch head) + RealSemanticModel (SemanticModel)
- train.py:    CLI to generate data, train the head, and save an artifact

Heavy ML deps (torch, sentence-transformers) are imported lazily inside the
submodules so that importing ``agents.a2a`` stays light.
"""
