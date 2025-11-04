# SQLModel Integration - Quick Start

## What You Got

I've created a complete SQLModel database schema for your calcium imaging analysis data, along with migration tools and comprehensive documentation.

## 📁 Files Created

1. **`src/cali/models.py`** - Complete SQLModel schema (7 tables, 600+ lines)
2. **`migrate_json_to_db.py`** - CLI tool to import your existing JSON data
3. **`examples_sqlmodel_usage.py`** - Comprehensive working examples
4. **`SQLMODEL_GUIDE.md`** - Complete integration guide
5. **`IMPLEMENTATION_SUMMARY.md`** - Detailed overview
6. **`SCHEMA_DIAGRAM.md`** - Visual diagrams and examples
7. **`x.py`** - Quick reference example

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install sqlmodel
```

### Step 2: Import Your Data

```bash
# Import your existing JSON analysis data
python migrate_json_to_db.py \
    --json-dir tests/test_data/evoked/evk_analysis \
    --output my_analysis.db \
    --validate
```

This creates a database with all your:
- ROI analysis data
- Experimental conditions (genotype/treatment)
- Analysis settings
- Complete hierarchy (Experiment → Plate → Well → FOV → ROI)

### Step 3: Query Your Data

```python
from sqlmodel import Session, create_engine, select
from cali.models import ROI, Well, Condition, FOV

engine = create_engine("sqlite:///my_analysis.db")

with Session(engine) as session:
    # Get all active ROIs
    active_rois = session.exec(select(ROI).where(ROI.active)).all()
    print(f"Found {len(active_rois)} active cells")
    
    # Get ROIs by condition
    wt_rois = session.exec(
        select(ROI)
        .join(FOV).join(Well)
        .join(Condition, Well.condition_1_id == Condition.id)
        .where(Condition.name == "WT")
    ).all()
    
    # Export to pandas
    import pandas as pd
    df = pd.DataFrame([{
        "frequency": roi.dec_dff_frequency,
        "cell_size": roi.cell_size,
        "active": roi.active,
    } for roi in active_rois])
```

## 📊 Database Schema

```
Experiment
  └── Plate
      └── Well (with conditions: genotype, treatment)
          └── FOV (imaging position)
              └── ROI (individual cell with all analysis data)
```

Each **ROI** contains:
- ✅ Fluorescence traces (raw, corrected, dff, dec_dff)
- ✅ Calcium peaks (indices, amplitudes, frequency)
- ✅ Inferred spikes
- ✅ Cell metadata (size, activity, recording time)
- ✅ Evoked experiment data (stimulation info)
- ✅ Network analysis parameters
- ✅ Spatial masks

## 💡 Usage Ideas

### 1. Statistical Analysis

```python
# Compare conditions
with Session(engine) as session:
    wt = session.exec(select(ROI)...).all()
    ko = session.exec(select(ROI)...).all()
    
    wt_freqs = [r.dec_dff_frequency for r in wt]
    ko_freqs = [r.dec_dff_frequency for r in ko]
    
    # Run t-test, ANOVA, etc.
```

### 2. Plotting

```python
# Get data organized by condition
data_by_condition = {}
for roi in session.exec(select(ROI).join(FOV).join(Well)).all():
    condition = roi.fov.well.condition_1.name
    data_by_condition.setdefault(condition, []).append(roi.dec_dff_frequency)

# Plot with seaborn
import seaborn as sns
sns.boxplot(data=data_by_condition)
```

### 3. Export for External Tools

```python
# Export summary CSV
df = pd.DataFrame([...])
df.to_csv("summary.csv")

# Export traces
for roi in rois:
    trace_df = pd.DataFrame({
        "time": roi.elapsed_time_list_ms,
        "dff": roi.dff,
    })
    trace_df.to_csv(f"roi_{roi.id}.csv")
```

## 🔗 Integration with Your Code

### Option A: Parallel Storage (Recommended)

Keep JSON export, add database export:

```python
# In _analysis.py
def _save_analysis_data(self, fov_name: str):
    # Existing JSON save
    self._save_analysis_data_to_json(fov_name)
    
    # Also save to database
    if hasattr(self, 'db_engine'):
        self._save_analysis_data_to_db(fov_name)
```

### Option B: Replace Dict with Database Queries

```python
# Instead of:
analysis_data: dict[str, dict[str, ROIData]] = {}

# Use:
with Session(engine) as session:
    rois = session.exec(select(ROI).where(...)).all()
```

## 📚 Documentation

- **`SQLMODEL_GUIDE.md`** - Complete integration guide with best practices
- **`IMPLEMENTATION_SUMMARY.md`** - Detailed overview with workflows
- **`SCHEMA_DIAGRAM.md`** - Visual diagrams and query examples
- **`examples_sqlmodel_usage.py`** - Comprehensive code examples

## 🎯 Benefits

✅ **Powerful Queries**: SQL queries vs nested dictionary loops  
✅ **Relationships**: Navigate Experiment → Plate → Well → FOV → ROI  
✅ **Type Safety**: SQLModel validates data automatically  
✅ **Scalability**: Handles thousands of ROIs efficiently  
✅ **Statistics**: Direct pandas integration  
✅ **Versioning**: Track analysis parameters used  
✅ **Multi-user**: Upgrade to PostgreSQL for shared access  

## 🔍 Example Queries

```python
# All active cells
active = session.exec(select(ROI).where(ROI.active)).all()

# High frequency cells
high_freq = session.exec(
    select(ROI).where(ROI.dec_dff_frequency > 1.0)
).all()

# Cells in stimulated area
stimulated = session.exec(
    select(ROI).where(ROI.stimulated)
).all()

# Complex query: WT + Vehicle treated + active
results = session.exec(
    select(ROI)
    .join(FOV).join(Well)
    .join(Condition, Well.condition_1_id == Condition.id)
    .where(Condition.name == "WT", ROI.active)
).all()
```

## 🛠 Migration Path

**Week 1**: Import data, validate, experiment with queries  
**Week 2**: Add database export to analysis pipeline  
**Week 3**: Update plotting functions to use database  
**Week 4+**: Make database primary, add GUI integration  

## ❓ FAQ

**Q: Will this break my existing code?**  
A: No! You can use both JSON and database formats in parallel.

**Q: How do I handle large datasets?**  
A: SQLite works great for 1000s of ROIs. For 100k+, use PostgreSQL.

**Q: Can I modify the schema?**  
A: Yes! Just edit `models.py` and regenerate tables.

**Q: How do I backup the database?**  
A: SQLite is a single file - just copy `analysis.db`.

## 🚦 Next Steps

1. ✅ Review `src/cali/models.py` to understand the schema
2. ✅ Run `migrate_json_to_db.py` to import your data
3. ✅ Try queries in `examples_sqlmodel_usage.py`
4. ✅ Read `SQLMODEL_GUIDE.md` for integration ideas
5. ✅ Experiment with your own queries

## 📧 Questions?

The schema maps 1:1 with your existing `ROIData` dataclass, making migration straightforward. Every field is preserved with proper relationships.

See the comprehensive guides for detailed integration strategies!
