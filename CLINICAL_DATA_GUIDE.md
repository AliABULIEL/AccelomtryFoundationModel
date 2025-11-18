# Clinical Data Integration Guide

Complete guide for integrating UK Biobank clinical data with accelerometry for TTM training.

---

## 📋 Overview

This system handles:
- **500,000+ participants** from UK Biobank
- **40,000+ phenotype fields** with sparse data
- **ICD-10 diagnoses** with temporal alignment
- **CAPTURE-24 annotations** for ground truth labels
- **Quality control** for accelerometry data

---

## 🏗️ System Architecture

```
UK Biobank Data
├─ Phenotype CSV (50GB+)
├─ ICD-10 Diagnoses
└─ Death Registry
    ↓
Parse & Filter
├─ Demographics
├─ Accelerometry QC
└─ Clinical Outcomes
    ↓
Temporal Alignment
├─ Incident vs Prevalent
├─ Days to Event
└─ Competing Risks
    ↓
Quality Control
├─ Wear Time ≥66%
├─ Calibration <10mg
└─ ≥3 Valid Days
    ↓
ML-Ready Dataset
├─ Train/Val/Test Split
├─ Geographic Diversity
└─ Proper Temporal Alignment
```

---

## 📦 Components

### 1. UK Biobank Parser (`src/clinical/ukb_parser.py`)

**Purpose:** Extract relevant fields from massive phenotype files

**Key Features:**
- Chunked processing (10,000 rows at a time)
- Memory-efficient (fits in 12GB RAM)
- GDPR compliance (withdrawn participants)
- Field instance parsing (0.0, 0.1, 1.0)

**Example Usage:**
```python
from src.clinical.ukb_parser import UKBiobankParser

# Initialize
parser = UKBiobankParser(
    phenotype_path='ukb12345.csv',
    withdrawn_path='withdrawn.csv',
    chunk_size=10000
)

# Parse demographics
df_demo = parser.parse_demographics()
# Returns: eid, sex, age, bmi, assessment_center,
#          acc_wear_time, acc_valid_days, acc_calibration_error

# Parse ICD-10 diagnoses
df_diagnoses = parser.parse_icd10_diagnoses()
# Returns: eid, instance, icd10_code, diagnosis_date

# Parse death registry
df_death = parser.parse_death_registry()
# Returns: eid, death_date, primary_cause, secondary_causes
```

**Fields Extracted:**
```
Demographics:
├─ 31: Sex (0=F, 1=M)
├─ 21003: Age at assessment
├─ 21001: BMI
└─ 54: Assessment center

Accelerometry:
├─ 90006: Wear time
├─ 90007: Valid days
└─ 90009: Calibration error

Diagnoses:
├─ 41270: ICD-10 code (up to 213 instances)
└─ 41280: Diagnosis date

Death:
├─ 40000: Death date
├─ 40001: Primary cause
└─ 40002: Secondary causes
```

---

### 2. ICD-10 Outcome Processor (`src/clinical/icd10_outcomes.py`)

**Purpose:** Classify diseases and handle temporal alignment

**Disease Groups:**
```python
CVD:
├─ CHD: I20-I25
├─ Stroke: I60-I69
├─ Heart Failure: I50
└─ Atrial Fibrillation: I48

Diabetes:
├─ Type 1: E10
├─ Type 2: E11
└─ Other: E12-E14

Dementia:
├─ Alzheimer's: G30, F00
├─ Vascular: F01
└─ Other: F02-F03

Cancer:
├─ Lung: C34
├─ Breast: C50
├─ Prostate: C61
└─ Colorectal: C18-C20

COPD:
├─ COPD: J44
└─ Emphysema: J43
```

**Temporal Classification:**
```
Diagnosis before accelerometry → Prevalent (excluded)
Diagnosis 0-30 days after → Acute (excluded, too close)
Diagnosis >30 days after → Incident (included in analysis)
```

**Example Usage:**
```python
from src.clinical.icd10_outcomes import ICD10OutcomeProcessor

# Initialize
processor = ICD10OutcomeProcessor(min_incident_days=30)

# Classify diagnoses
df_classified = processor.classify_diagnoses(
    df_diagnoses,
    df_accelerometry_dates
)

# Create per-participant outcomes
df_outcomes = processor.create_outcome_summary(
    df_classified,
    include_death=df_death
)

# Outputs:
# cvd_incident: 0/1
# cvd_prevalent: 0/1
# cvd_days_to_event: float
# (same for diabetes, dementia, cancer, copd)
# died: 0/1
```

**Validation:**
```python
# Validate against expected rates
validation = processor.validate_outcomes(df_outcomes)

# Expected rates (5-year follow-up):
# CVD: 5-10%
# Diabetes: 2-5%
# Dementia: 0.5-2%
# Cancer: 1-5%
```

---

### 3. CAPTURE-24 Parser (`src/clinical/capture24_parser.py`)

**Purpose:** Ground truth activity labels from wrist-worn camera study

**Dataset:**
- 151 participants (P001-P151)
- 24-hour free-living
- 206 CPA activity codes
- Train: P001-P100, Test: P101-P151

**Intensity Levels (MET Thresholds):**
```
Sleep:     0.00-0.95 METs
Sedentary: 0.95-1.50 METs
Light PA:  1.50-3.00 METs
MVPA:      ≥3.00 METs
```

**Example Usage:**
```python
from src.clinical.capture24_parser import CAPTURE24Parser

# Initialize
parser = CAPTURE24Parser('capture24_annotations/')

# Parse participant
df_annotations = parser.parse_participant_annotations('P001')

# Parse all
df_all = parser.parse_all_annotations()

# Aggregate to 8.192s windows
df_windows = parser.aggregate_to_windows(df_all, window_size_sec=8.192)

# Train/test split
df_train, df_test = parser.create_train_test_split(df_windows)

# Label encoder
encoder = parser.get_label_encoder()
# Returns: {'sleep': 0, 'sedentary': 1, 'light': 2, 'mvpa': 3}
```

---

### 4. Clinical-Accelerometry Aligner (`src/clinical/alignment.py`)

**Purpose:** Merge data and apply quality control

**Quality Criteria:**
```
✓ Wear time ≥66% (16 hours/day minimum)
✓ Calibration error <10mg
✓ Valid wear days ≥3
✓ Proper temporal alignment
```

**Example Usage:**
```python
from src.clinical.alignment import ClinicalAccelerometryAligner

# Initialize
aligner = ClinicalAccelerometryAligner(
    min_wear_time=0.66,
    max_calib_error=10.0,
    min_valid_days=3
)

# Apply quality filters
df_filtered, qc_stats = aligner.apply_quality_filters(df_demographics)

# Merge clinical data
df_merged = aligner.merge_clinical_data(
    df_demographics,
    df_outcomes,
    df_death
)

# Create ML dataset
ml_stats = aligner.create_ml_dataset(
    df_merged,
    accelerometry_hdf5='acc_data.h5',
    output_path='ml_dataset.h5',
    split_by_center=True  # Geographic diversity
)

# Validate temporal alignment
validation = aligner.validate_temporal_alignment(df_merged)
```

---

## 🚀 Complete Pipeline

### Using the Integration Script

```bash
# Full pipeline
python scripts/integrate_clinical_data.py \
  --ukb-phenotype /path/to/ukb12345.csv \
  --ukb-withdrawn /path/to/withdrawn.csv \
  --capture24-dir /path/to/capture24/ \
  --output-dir ./clinical_data

# Test mode (1000 participants)
python scripts/integrate_clinical_data.py \
  --ukb-phenotype /path/to/ukb12345.csv \
  --output-dir ./clinical_data_test \
  --test
```

### Pipeline Steps

**Step 1: Parse UK Biobank**
- Reads 50GB+ CSV in chunks
- Extracts demographics, accelerometry QC, diagnoses, deaths
- Removes withdrawn participants

**Step 2: Process ICD-10 Outcomes**
- Classifies 213 diagnoses per participant
- Calculates days between accelerometry and diagnosis
- Creates per-participant outcome summary

**Step 3: Parse CAPTURE-24** (optional)
- Loads ground truth annotations
- Maps 206 CPA codes to 4 intensity levels
- Aggregates to 8.192s windows

**Step 4: Merge Clinical Data**
- Combines all data sources
- Aligns by participant ID

**Step 5: Quality Control**
- Applies wear time, calibration, valid days filters
- Typically excludes 30-40% of participants

**Step 6: Create ML Dataset**
- Splits by assessment center (geographic diversity)
- Creates HDF5 with train/val/test splits
- Includes demographics and outcomes

**Step 7: Validation**
- Checks temporal alignment
- Validates outcome rates
- Reports statistics

---

## 📊 Expected Output

### Files Created

```
clinical_data/
├── demographics.csv              # N×8 (eid, age, sex, bmi, etc.)
├── icd10_diagnoses.csv           # N×4 (eid, icd10, date, instance)
├── death_registry.csv            # N×4 (eid, date, primary, secondary)
├── clinical_outcomes.csv         # N×20 (incident, prevalent, days_to_event)
├── capture24_windows.csv         # N×7 (participant, window, intensity)
├── merged_clinical_data.csv      # N×30 (all data combined)
├── qc_filtered_data.csv          # N×30 (after QC)
├── ml_dataset.h5                 # HDF5 with train/val/test splits
└── pipeline_statistics.json      # Complete statistics
```

### Expected Cohort Size

```
Initial participants:     500,000
After withdrawn:          498,000
With accelerometry:       103,712
Passed QC filters:        ~96,000
    ├─ Train (70%):      ~67,000
    ├─ Val (15%):        ~14,000
    └─ Test (15%):       ~14,000
```

### Expected Outcome Rates (5-year follow-up)

```
CVD:        5-10% incident
Diabetes:   2-5% incident
Dementia:   0.5-2% incident
Cancer:     1-5% incident
Mortality:  1-3% overall
```

---

## 🎯 Use Cases

### 1. Train TTM on Labeled Data (CAPTURE-24)

```python
from src.clinical import CAPTURE24Parser
from src.data import StreamingDataset

# Get CAPTURE-24 labels
parser = CAPTURE24Parser('capture24_annotations/')
df_windows = parser.parse_all_annotations()
df_train, df_test = parser.create_train_test_split(df_windows)

# Create labeled dataset
# Use windows with high confidence (>0.8)
df_high_conf = df_train[df_train['confidence'] > 0.8]

# Train TTM classifier
# Target: >0.85 F1 on CAPTURE-24 test set
```

### 2. Predict Clinical Outcomes from Accelerometry

```python
from src.clinical import UKBiobankParser, ICD10OutcomeProcessor

# Get clinical outcomes
df_outcomes = load_outcomes('clinical_outcomes.csv')

# Match with accelerometry
df_matched = match_participants(df_outcomes, accelerometry_data)

# Train model to predict:
# - 5-year CVD risk
# - 10-year mortality risk
# - Diabetes onset
# - Frailty progression
```

### 3. Quality Control for Large Cohorts

```python
from src.clinical import ClinicalAccelerometryAligner

aligner = ClinicalAccelerometryAligner()

# Apply standard QC
df_filtered, qc_stats = aligner.apply_quality_filters(df_demographics)

print(f"Passed QC: {qc_stats['passed_all']:,} / {qc_stats['initial_n']:,}")
print(f"Exclusion rate: {qc_stats['excluded'] / qc_stats['initial_n'] * 100:.1f}%")
```

---

## 🐛 Troubleshooting

### Memory Issues

**Problem:** Out of memory with 50GB CSV

**Solution:**
```python
# Reduce chunk size
parser = UKBiobankParser(phenotype_path, chunk_size=5000)

# Or process specific fields only
# Edit FIELDS dict in ukb_parser.py to include only needed fields
```

### Missing Fields

**Problem:** Field not found in phenotype file

**Solution:**
```python
# Check available columns
df = pd.read_csv('ukb12345.csv', nrows=1)
print(df.columns.tolist())

# Update FIELDS dict to match your data version
```

### Low Outcome Rates

**Problem:** CVD rate is 2% (expected 5-10%)

**Possible Causes:**
1. Temporal misalignment (including prevalent cases)
2. Incomplete ICD-10 linkage
3. Short follow-up time
4. Wrong disease codes

**Solution:**
```python
# Check temporal classification
processor = ICD10OutcomeProcessor(min_incident_days=30)
df_classified = processor.classify_diagnoses(df_diagnoses, df_acc_dates)

# Check distribution
print(df_classified['temporal_class'].value_counts())
# Should see: prevalent, incident, acute

# Verify dates
print(f"Accelerometry: {df_acc_dates['acc_date'].min()} to {df_acc_dates['acc_date'].max()}")
print(f"Follow-up: {df_diagnoses['diagnosis_date'].max()}")
```

---

## 📚 References

### UK Biobank
- Main cohort: Sudlow et al. 2015, PLOS Medicine
- Accelerometry: Doherty et al. 2017, International Journal of Epidemiology
- Resource: https://biobank.ndph.ox.ac.uk/

### CAPTURE-24
- Benchmarkdataset: Willetts et al. 2018, Scientific Data
- 151 participants with wrist camera validation
- Download: https://ora.ox.ac.uk/objects/uuid:92650814-a209-4607-9fb5-921eab761c11

### Disease Classification
- ICD-10: WHO International Classification of Diseases
- MET values: Ainsworth et al. 2011, Medicine & Science in Sports & Exercise
- Quality control: van Hees et al. 2013, PLOS ONE

---

## ✅ Validation Checklist

Before using the data:

- [ ] Withdrawn participants removed
- [ ] Wear time ≥66% for all included participants
- [ ] Calibration error <10mg
- [ ] ≥3 valid wear days
- [ ] Temporal alignment correct (no prevalent cases as incident)
- [ ] Outcome rates match expected prevalence
- [ ] Train/test split has no overlap
- [ ] Geographic diversity in splits (by assessment center)
- [ ] Missing data <10% for key fields
- [ ] Class balance reasonable (not all one class)

---

**System ready for clinical data integration!** 🏥

All components tested and optimized for UK Biobank scale (500K+ participants).
