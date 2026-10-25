"""Second-stage ranking over the hybrid candidate set.

``types`` and ``features`` are dependency-light so the model server can
share the feature definitions; ``candidates``, ``train``, and ``system``
need the database and the models. LightGBM is loaded only through
``ml.ranking.lightgbm_runtime.load_lightgbm`` (see there for why).
"""
