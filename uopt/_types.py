import os
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np

try:
    from numpy.typing import DTypeLike, NDArray

    ArrayLike = NDArray[np.float64]
except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]
    DTypeLike = np.dtype  # type: ignore[misc]

ProblemKind_t = Literal["linear", "quadratic", "unknown"]
Numeric_t = Union[int, float]
Filter_t = Optional[Union[str, Mapping[str, Sequence[Any]]]]
Str_Dict_t = Optional[Union[str, Mapping[str, Sequence[Any]]]]
SinkFullRankInit = Literal["default", "gaussian", "sorting"]
LRInitializer_t = Literal["random", "rank2", "k-means", "generalized-k-means"]

SinkhornInitializer_t = Optional[Union[SinkFullRankInit, LRInitializer_t]]
QuadInitializer_t = Optional[LRInitializer_t]

Initializer_t = Union[SinkhornInitializer_t, LRInitializer_t]
ProblemStage_t = Literal["prepared", "solved"]
Device_t = Union[Literal["cpu", "gpu", "tpu"], str]

ScaleCost_t = Union[float, Literal["mean", "max_cost", "max_bound", "max_norm", "median"]]
OttCostFn_t = Literal[
    "euclidean",
    "sq_euclidean",
    "cosine",
    "pnorm_p",
    "sq_pnorm",
    "cosine",
    "elastic_l1",
    "elastic_l2",
    "elastic_stvs",
    "elastic_sqk_overlap",
    "geodesic",
]
OttCostFnMap_t = Union[OttCostFn_t, Mapping[Literal["xy", "x", "y"], OttCostFn_t]]
GenericCostFn_t = Literal["barcode_distance", "leaf_distance", "custom"]
CostFn_t = Union[str, GenericCostFn_t, OttCostFn_t]
CostFnMap_t = Union[Union[OttCostFn_t, GenericCostFn_t], Mapping[str, Union[OttCostFn_t, GenericCostFn_t]]]
PathLike = Union[os.PathLike, str]  # type: ignore[type-arg]
Policy_t = Literal[
    "sequential",
    "star",
    "external_star",
    "explicit",
    "triu",
    "tril",
]
CostKwargs_t = Union[Mapping[str, Any], Mapping[Literal["x", "y", "xy"], Mapping[str, Any]]]
