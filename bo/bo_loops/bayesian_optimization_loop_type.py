from enum import auto, Enum


class BayesianOptimizationLoopType(Enum):
    CEI = auto()
    CKG_V2 = auto()
    DCKG_INDEPENDENT = auto()       # Each source optimised independently
    # Ablations of DCKG_INDEPENDENT with the coupled cKG candidate removed
    DCKG_NO_COUPLED = auto()        # No coupled candidate; coupled all-zero fallback kept
    DCKG_PURE = auto()              # No coupled candidate; single-source all-zero fallback
