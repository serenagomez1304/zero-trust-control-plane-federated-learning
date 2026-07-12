"""
Federated Learning of the trust_eval head (Milestone 4)
=======================================================
Cross-silo FL simulation: K deployments each train the trainable head locally on
their own (non-IID) slice of the synthetic trust data; a server aggregates the
head weights. The encoder is frozen, so only a small weight vector is federated.

Submodules:
- partition.py:  split the dataset across K clients with controllable non-IID skew
- strategies.py: FedAvg / FedProx / SCAFFOLD / FedNova aggregation
- simulate.py:   the FL training loop + a centralized baseline
"""
