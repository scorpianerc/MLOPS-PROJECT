# Configuration Files

Folder ini berisi file-file konfigurasi untuk project MLOps.

## Files

### `params.yaml`
**Central configuration file** untuk semua parameters project.

**Sections:**
- **scraping**: Google Play Store scraping configuration
  - `max_reviews`: Maximum reviews to scrape
  - `app_id`: Target application ID
  - `sort_by`: Review sorting method
  
- **preprocessing**: Text preprocessing parameters
  - `max_length`: Maximum text length
  - `remove_stopwords`: Enable/disable stopword removal
  - `stemming`: Enable/disable stemming
  
- **training**: Model training parameters
  - `model_name`: Base model (IndoBERT)
  - `learning_rate`: Training learning rate
  - `batch_size`: Training batch size
  - `num_epochs`: Number of training epochs
  - `max_seq_length`: Maximum sequence length for tokenization
  
- **feature_engineering**: Feature extraction settings
  - List of 14 features to extract from text

**Usage:**
```python
import yaml

with open('config/params.yaml', 'r') as f:
    params = yaml.safe_load(f)
    
learning_rate = params['training']['learning_rate']
```

**Docker Mount:**
File ini di-mount ke container:
```yaml
volumes:
  - ./config/params.yaml:/app/config/params.yaml
```

---

### `dvc.yaml`
**DVC pipeline definition** - defines data processing stages and dependencies.

**Stages:**
1. **preprocess**: Data preprocessing stage
   - Input: `data/raw/`
   - Output: `data/processed/`
   - Command: `python src/preprocessing/preprocess.py`
   
2. **train**: Model training stage
   - Input: `data/processed/`, `config/params.yaml`
   - Output: `models/`
   - Command: `python src/training/train_bert.py`

**Usage:**
```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train
```

---

### `dvc.lock`
**DVC lock file** - tracks exact versions of data and model artifacts.

Generated automatically by DVC. **DO NOT edit manually**.

Contains:
- MD5 checksums of data files
- Dependencies between stages
- Output file hashes

---

## Configuration Best Practices

1. **params.yaml**
   - Keep all hyperparameters here (don't hardcode in Python)
   - Use descriptive parameter names
   - Add comments for complex parameters
   - Version control changes

2. **dvc.yaml**
   - Define clear stage dependencies
   - Use meaningful stage names
   - Specify all inputs and outputs
   - Keep commands simple and testable

3. **dvc.lock**
   - Commit to git after successful runs
   - Never manually edit
   - Use `dvc repro` to regenerate

---

## Related Documentation

- [SETUP.md](../docs/SETUP.md) - How to configure parameters
- [DVC Documentation](https://dvc.org/doc) - Official DVC guide
- [MLOps Guide](../docs/MLOPS_QUICK_REFERENCE.md) - MLOps best practices
