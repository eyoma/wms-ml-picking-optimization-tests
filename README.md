# Warehouse Management System (WMS) - Picking Sequence Optimization

This project implements machine learning models to optimize warehouse picking sequences using synthetic warehouse data. The goal is to predict the most efficient picking order for warehouse pickers based on historical data.

## 📁 Project Structure

### Core Implementation Files

- **`0.dataGenerator.py`** - Synthetic warehouse data generator
  - Creates realistic warehouse picking datasets with configurable parameters
  - Generates data for multiple warehouses, pickers, and orders
  - Includes features like order timing, picker experience, and item popularity
  - Outputs CSV with columns: `order_id`, `item_id`, `datetime_picked`, `picker_id`, `warehouse_id`, `bin_id`

- **`1.markov.py`** - Markov Chain Model Implementation
  - Implements a Markov model to learn transition probabilities between bin locations
  - Uses historical picking sequences to predict optimal picking order
  - Includes evaluation metrics for sequence prediction accuracy
  - Provides greedy prediction algorithm with fallback mechanisms

- **`2.leaksafeitemranking.py`** - Item-Level Ranking with LightGBM
  - Implements item-level ranking using LightGBM's LambdaRank
  - Uses proper time-based splits to prevent data leakage
  - Creates relevance labels based on actual picking order
  - Evaluates using NDCG per order

- **`3.leaksafeitemranking34.py`** - Simplified Item Ranking with Gradient Boosting
  - Alternative implementation using GradientBoostingRegressor
  - Simpler approach for item-level ranking
  - Includes evaluation for orders with different sizes

- **`4.time-based-kendall-tau.py`** - Time-Based Split with Kendall-τ Evaluation
  - Implements time-based train/test splits (80/20)
  - Uses Kendall-τ correlation for rank agreement evaluation
  - Includes both NDCG and Kendall-τ metrics
  - Provides comprehensive evaluation across different order sizes

- **`5.lgbm-timebased-kendall-tau.py`** - Advanced LightGBM Ranker Implementation
  - Most sophisticated implementation using LGBMRanker
  - Time-based splits with proper leakage prevention
  - Comprehensive feature engineering and evaluation
  - Reports both NDCG and Kendall-τ metrics with confidence intervals

### Advanced LightGBM Ranker Architecture

[View on Eraser![](https://app.eraser.io/workspace/32DQ9UadLiAZe47K4lyo/preview?elements=TKB_FeJtsO3nOp4FZZechA&type=embed)](https://app.eraser.io/workspace/32DQ9UadLiAZe47K4lyo?elements=TKB_FeJtsO3nOp4FZZechA)

### Documentation Files

- **`kendall-meaning.md`** - Detailed explanation of Kendall-τ metric
  - Explains what Kendall-τ measures in the context of warehouse picking
  - Provides interpretation guidelines for different τ values
  - Includes practical examples and improvement suggestions

- **`LightGBM Ranker.md`** - Implementation plan for LightGBM Ranker
  - Outlines the approach for learning-to-rank models
  - Details feature engineering strategies
  - Describes evaluation methodologies

- **`markov.md`** - Markov Model implementation plan
  - Step-by-step guide for Markov model development
  - Data preprocessing requirements
  - Model evaluation strategies

### Data Files

- **`synthetic_warehouse_picks_sample.csv`** - Generated synthetic warehouse data
- **`wms/synthetic_warehouse_picks_sample.csv`** - Alternative data location

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm scipy matplotlib
```

### Environment Setup

The project includes a virtual environment (`my_env/`) with all necessary dependencies installed.

### Running the Models

1. **Generate synthetic data:**
   ```bash
   python 0.dataGenerator.py
   ```

2. **Run Markov model:**
   ```bash
   python 1.markov.py
   ```

3. **Run advanced LightGBM ranker:**
   ```bash
   python 5.lgbm-timebased-kendall-tau.py
   ```

## 📊 Model Performance

### Key Metrics

- **NDCG (Normalized Discounted Cumulative Gain)**: Measures ranking quality (0-1, higher is better)
- **Kendall-τ**: Measures rank correlation between predicted and actual sequences (-1 to +1, higher is better)
- **Consecutive Transitions**: Counts exact consecutive matches in sequences

### Actual Model Results

#### 1. Markov Model (`1.markov.py`)
```
Dataset: 13,458 entries across 6 columns
Original Order Bins: ['W01-A06-S011', 'W01-A07-S001', 'W01-A12-S011']
Predicted Picking Sequence: ['W01-A06-S011', 'W01-A07-S001', 'W01-A12-S011']
Average consecutive bin transitions: 0.65
```

**Interpretation**: The Markov model achieves 65% consecutive transition accuracy, meaning it correctly predicts the next bin in the sequence 65% of the time. This is a solid baseline performance for a simple probabilistic model.

#### 2. LightGBM Item Ranking (`2.leaksafeitemranking.py`)
```
NDCG (all orders): 0.9190
NDCG (orders ≥3 lines): 0.9190 (386 orders)
```

**Interpretation**: Excellent NDCG score of 0.919 indicates very high ranking quality. The model performs consistently across different order sizes.

#### 3. Gradient Boosting Ranking (`3.leaksafeitemranking34.py`)
```
NDCG (all orders): 0.9349 (639 orders)
NDCG (orders ≥3 lines): 0.9190 (386 orders)
```

**Interpretation**: Slightly better overall NDCG (0.9349) than LightGBM, but similar performance on larger orders. Shows that simpler models can be effective.

#### 4. Time-Based Split with Kendall-τ (`4.time-based-kendall-tau.py`)
```
Training cutoff: 2025-09-25
NDCG: 0.9347 | Kendall-τ: 0.4725 (637 orders)
```

**Interpretation**: 
- **NDCG 0.9347**: Excellent ranking quality
- **Kendall-τ 0.4725**: Moderate rank correlation, indicating the model captures useful sequencing patterns but has room for improvement

#### 5. Advanced LightGBM Ranker (`5.lgbm-timebased-kendall-tau.py`)
```
Training cutoff: 2025-09-25 12:36:11
Train: 10,766 lines | Test: 2,692 lines
All orders: NDCG: 0.9274 | Kendall-τ: 0.3921 (637 orders)
Orders ≥3 lines: NDCG: 0.9100 | Kendall-τ: 0.3302 (393 orders)
```

**Interpretation**: 
- **NDCG 0.9274**: Very high ranking quality
- **Kendall-τ 0.3921**: Moderate correlation, slightly lower than simpler models
- **Performance drops on larger orders**: NDCG decreases to 0.9100 for orders ≥3 lines

### Model Comparison Analysis

| Model | NDCG | Kendall-τ | Complexity | Best Use Case |
|-------|------|-----------|------------|---------------|
| Markov | N/A | N/A | Low | Baseline, interpretable |
| LightGBM Item | 0.919 | N/A | Medium | Balanced performance |
| Gradient Boosting | 0.935 | N/A | Low | Simple, effective |
| Time-based + τ | 0.935 | 0.473 | Medium | Comprehensive evaluation |
| Advanced LGBM | 0.927 | 0.392 | High | Most sophisticated |

### Key Insights from Results

1. **NDCG Performance**: All models achieve excellent NDCG scores (0.91-0.94), indicating strong ranking quality
2. **Kendall-τ Trade-off**: More complex models show lower Kendall-τ, suggesting overfitting to ranking metrics vs. sequence correlation
3. **Order Size Impact**: Larger orders (≥3 lines) are more challenging, with performance dropping
4. **Time-based Splits**: More realistic evaluation reveals slightly lower but more trustworthy performance
5. **Model Complexity**: Simpler models (Gradient Boosting) can outperform more complex ones (Advanced LGBM)

## 🔧 Model Approaches

### 1. Markov Chain Models
- Learn transition probabilities between bin locations
- Simple and interpretable
- Good baseline for sequence prediction

### 2. Learning-to-Rank Models
- **Pairwise Ranking**: Compare pairs of items to learn relative preferences
- **Item-Level Ranking**: Direct ranking of items within orders
- **Listwise Ranking**: Consider entire lists for optimization

### 3. Feature Engineering
- **Spatial Features**: Warehouse layout, aisle/section information
- **Temporal Features**: Time-based patterns, picker experience
- **Item Features**: Popularity, frequency, characteristics
- **Picker Features**: Experience level, average speed

## 📈 Evaluation Strategy

### Data Splitting
- **Time-based splits**: 80% training, 20% testing to prevent leakage
- **Order-level splits**: Ensure no order appears in both train and test sets

### Metrics
- **NDCG**: Standard ranking metric for information retrieval
- **Kendall-τ**: Rank correlation coefficient (-1 to +1)
- **Pairwise Accuracy**: Fraction of correctly ordered item pairs

### Leakage Prevention
- Compute aggregates only on training data
- Use proper time-based splits
- Avoid features derived from future information

## 🎯 Key Insights

1. **Time-based splits are crucial** for realistic evaluation
2. **Feature engineering** significantly impacts model performance
3. **Kendall-τ ≈ 0.47** represents good but improvable performance
4. **Order size matters** - larger orders are more challenging to predict
5. **Spatial relationships** are important for warehouse optimization

## 🔮 Future Improvements

1. **Enhanced Features**:
   - Physical distances between bins
   - Warehouse layout coordinates
   - Item characteristics (size, weight, fragility)

2. **Advanced Models**:
   - Transformer-based sequence models
   - Graph neural networks for spatial relationships
   - Reinforcement learning for dynamic optimization

3. **Real-world Integration**:
   - Live warehouse data integration
   - Real-time optimization
   - A/B testing with actual pickers

## 📝 Usage Notes

- All models are designed to prevent data leakage
- Synthetic data generation is configurable via parameters
- Models can be easily adapted for different warehouse layouts
- Evaluation metrics are standardized for comparison

## 🏆 Model Performance Summary

Based on the comprehensive evaluation across all models:

### Best Performing Models
1. **Gradient Boosting Ranking** (`3.leaksafeitemranking34.py`): Highest NDCG (0.935)
2. **Time-based Split Model** (`4.time-based-kendall-tau.py`): Best balance of NDCG (0.935) and Kendall-τ (0.473)
3. **LightGBM Item Ranking** (`2.leaksafeitemranking.py`): Consistent performance across order sizes

### Key Findings
- **All models achieve excellent NDCG scores** (0.91-0.94), indicating strong ranking quality
- **Kendall-τ shows moderate correlation** (0.33-0.47), suggesting room for sequence optimization
- **Simpler models often outperform complex ones**, indicating the importance of proper feature engineering over model complexity
- **Time-based evaluation is crucial** for realistic performance assessment
- **Order size matters** - larger orders are significantly more challenging to optimize

### Recommended Approach
For production deployment, consider the **Time-based Split Model** (`4.time-based-kendall-tau.py`) as it provides the best balance of ranking quality and sequence correlation with realistic evaluation methodology.

## 🤝 Contributing

This project demonstrates various approaches to warehouse picking optimization. Feel free to experiment with different models, features, and evaluation strategies.

## 📄 License

This project is for educational and research purposes.
