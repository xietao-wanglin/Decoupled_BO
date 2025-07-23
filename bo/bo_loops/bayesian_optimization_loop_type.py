from enum import auto, Enum


class BayesianOptimizationLoopType(Enum):
    OPTIMISTIC_UCB = auto()
    DCKG_CKG = auto()
    DCKG = auto()
    EIKG = auto()
    DEI = auto()
    CEI = auto()
    CKG = auto()