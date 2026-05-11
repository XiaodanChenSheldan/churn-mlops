
## 🚀 Quick Start

### Prerequisites

- Python 3.10
- Docker & Docker Compose
- Git
- 4GB RAM minimum (8GB recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/churn-mlops.git
cd churn-mlops
```

### 2. Create Virtual Environment (Conda)
```bash
# Create environment with Python 3.10
conda create -n mlops python=3.10 -c defaults

# Activate environment
conda init
source ~/.bashrc
conda activate mlops

# Verify Python version
python --version  # Should show Python 3.10.x
```

### 3. Install Dependencies
```shell
pip install -r requirements.txt
```

### 4. Data Processing (Optional)
```bash
# Get the Data
python download_data.py
```
You should see:
```bash
Downloaded 7043 rows, 21 columns
Columns: ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents']...
```
```bash
# Data Processing
python src/preprocess.py
```
```shell
Original shape: (7043, 21)
Processed shape: (7043, 19)
Churn rate: 26.54%

Preprocessing complete!
Features: ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure']...
```

