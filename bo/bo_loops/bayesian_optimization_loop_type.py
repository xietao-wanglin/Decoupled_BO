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
    # DCKG_INDEPENDENT with a consistent value/cost rule for the coupled candidate
    DCKG_COST_SUBSET = auto()       # cKG_S / C_S, sample S
    DCKG_COST_FULL = auto()         # cKG_all / C_all, sample every source
    DCKG_COST_BEST = auto()         # better of the two competes with the decoupled sources