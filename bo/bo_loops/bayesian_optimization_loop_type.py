from enum import auto, Enum


class BayesianOptimizationLoopType(Enum):
    OPTIMISTIC_UCB = auto()
    DCKG_CKG = auto()
    DCKG = auto()
    EIKG = auto()
    DEI = auto()
    CEI = auto()
    CKG = auto()
    # Refactored GPU-aware variants
    DCKG_CKG_V2 = auto()
    DCKG_V2 = auto()
    CKG_V2 = auto()
    DCKG_ALL_SOURCES = auto()       # All sources in one optimize_acqf call
    DCKG_INDEPENDENT = auto()       # Each source optimised independently
    # Ablations of DCKG_INDEPENDENT with the coupled cKG candidate removed
    DCKG_NO_COUPLED = auto()        # No coupled candidate; coupled all-zero fallback kept
    DCKG_PURE = auto()              # No coupled candidate; single-source all-zero fallback