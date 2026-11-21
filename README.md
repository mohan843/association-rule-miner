# Association Rule Miner

A dependency-free data-mining mini-project that discovers useful item co-occurrence patterns from transaction data with the Apriori algorithm. It ranks association rules using support, confidence, and lift.

## Overview

Association-rule mining helps uncover relationships such as products that frequently appear together in shopping baskets, learning resources selected together, or features that co-occur in event logs. This project implements the complete frequent-itemset and rule-generation workflow from scratch.

## Problem statement

Given a collection of transactions, identify frequent item combinations and produce interpretable directional rules without relying on a prebuilt machine-learning library.

## Features

- Apriori frequent-itemset discovery with candidate pruning.
- Configurable minimum support and confidence.
- Support, confidence, and lift for every rule.
- CSV transaction loader and command-line interface.
- Deterministic ranking of the highest-lift rules.
- Unit tests for itemsets, metrics, loading, validation, and sorting.

## Technology stack

- Python 3.10+
- Data mining and unsupervised learning
- Apriori algorithm
- Association-rule metrics
- Python `csv`, `argparse`, `dataclasses`, and `unittest`
- Standard library only

## Installation

```bash
git clone https://github.com/mohan843/association-rule-miner.git
cd association-rule-miner
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party dependencies are required.

## Input format

The CSV must contain one row per item in a transaction:

```csv
transaction_id,item
T01,bread
T01,milk
```

## Execution

Run the sample market-basket analysis:

```bash
python miner.py data/transactions.csv --min-support 0.25 --min-confidence 0.6 --top 5
```

Use the library API:

```python
from miner import apriori, generate_rules

frequent = apriori(transactions, min_support=0.25)
rules = generate_rules(frequent, min_confidence=0.6)
```

## Sample output

```text
Transactions analyzed: 12
Frequent itemsets: 7
Rules generated: 5
  {eggs} -> {milk} (support=0.50, confidence=0.75, lift=1.00)
  {milk} -> {eggs} (support=0.50, confidence=0.67, lift=1.00)
  {milk} -> {bread} (support=0.50, confidence=0.67, lift=0.89)
  {bread} -> {milk} (support=0.50, confidence=0.67, lift=0.89)
  {eggs} -> {bread} (support=0.42, confidence=0.63, lift=0.83)
```

The exact number and ranking of displayed rules depend on the thresholds and `--top` value.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Metric interpretation

- **Support:** fraction of transactions containing the complete itemset.
- **Confidence:** fraction of antecedent transactions that also contain the consequent.
- **Lift:** confidence divided by the consequent's baseline frequency. Lift above 1 suggests positive association; values near 1 may be explained by independence.

## Limitations and responsible use

Apriori can become expensive as item vocabulary and transaction density grow. Association does not imply causation, and high-lift rules based on very few transactions can be unstable. Production use should apply minimum-count safeguards, time-window validation, privacy controls, and domain review before changing recommendations or prices.

## Future improvements

- Add FP-Growth for larger transaction collections.
- Add minimum support-count and statistical significance filters.
- Support time windows and incremental streaming updates.
- Export rules to JSON for recommendation services.
- Add rule diversity and redundancy reduction.
