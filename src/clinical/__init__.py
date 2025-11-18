"""
Clinical Data Integration Module
Handles UK Biobank phenotypes, ICD-10 outcomes, and CAPTURE-24 annotations
"""

from .ukb_parser import UKBiobankParser
from .icd10_outcomes import ICD10OutcomeProcessor
from .capture24_parser import CAPTURE24Parser
from .alignment import ClinicalAccelerometryAligner

__all__ = [
    'UKBiobankParser',
    'ICD10OutcomeProcessor',
    'CAPTURE24Parser',
    'ClinicalAccelerometryAligner',
]
