"""Dependency-free Apriori association-rule mining for transaction data."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AssociationRule:
    antecedent: frozenset[str]
    consequent: frozenset[str]
    support: float
    confidence: float
    lift: float


def load_transactions(path: str | Path) -> list[frozenset[str]]:
    """Load one item per row from a CSV with transaction_id and item columns."""
    grouped: dict[str, set[str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"transaction_id", "item"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain 'transaction_id' and 'item' columns")
        for row_number, row in enumerate(reader, start=2):
            transaction_id = (row.get("transaction_id") or "").strip()
            item = (row.get("item") or "").strip()
            if not transaction_id or not item:
                raise ValueError(f"Missing transaction or item on row {row_number}")
            grouped.setdefault(transaction_id, set()).add(item)
    transactions = [frozenset(items) for items in grouped.values() if items]
    if not transactions:
        raise ValueError("CSV contains no transactions")
    return transactions


def _support_counts(transactions: Sequence[frozenset[str]], candidates: Iterable[frozenset[str]]) -> dict[frozenset[str], int]:
    counts = {candidate: 0 for candidate in candidates}
    for transaction in transactions:
        for candidate in counts:
            if candidate.issubset(transaction):
                counts[candidate] += 1
    return counts


def apriori(transactions: Sequence[frozenset[str]], min_support: float = 0.2) -> dict[frozenset[str], float]:
    """Return all frequent itemsets and their support values."""
    if not transactions:
        raise ValueError("transactions must not be empty")
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1]")
    minimum_count = max(1, int(min_support * len(transactions) + 0.999999))
    frequent: dict[frozenset[str], float] = {}
    items = sorted(set().union(*transactions))
    candidates = [frozenset((item,)) for item in items]
    size = 1
    while candidates:
        counts = _support_counts(transactions, candidates)
        current = [itemset for itemset, count in counts.items() if count >= minimum_count]
        if not current:
            break
        for itemset in current:
            frequent[itemset] = counts[itemset] / len(transactions)
        size += 1
        previous = {itemset for itemset in current}
        unions = {
            frozenset(left | right)
            for left in previous
            for right in previous
            if len(left | right) == size
        }
        candidates = [
            candidate for candidate in unions
            if all(frozenset(subset) in previous for subset in combinations(candidate, size - 1))
        ]
    return frequent


def generate_rules(
    frequent_itemsets: dict[frozenset[str], float], min_confidence: float = 0.5
) -> list[AssociationRule]:
    """Generate rules from frequent itemsets, sorted by lift then confidence."""
    if not 0 < min_confidence <= 1:
        raise ValueError("min_confidence must be in (0, 1]")
    rules: list[AssociationRule] = []
    for itemset, itemset_support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        items = tuple(itemset)
        for size in range(1, len(items)):
            for antecedent_tuple in combinations(items, size):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                antecedent_support = frequent_itemsets.get(antecedent)
                consequent_support = frequent_itemsets.get(consequent)
                if antecedent_support is None or consequent_support is None:
                    continue
                confidence = itemset_support / antecedent_support
                if confidence >= min_confidence:
                    rules.append(AssociationRule(
                        antecedent, consequent, itemset_support, confidence,
                        confidence / consequent_support,
                    ))
    return sorted(rules, key=lambda rule: (-rule.lift, -rule.confidence, rule.antecedent))


def _format_items(items: frozenset[str]) -> str:
    return "{" + ", ".join(sorted(items)) + "}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV transaction file")
    parser.add_argument("--min-support", type=float, default=0.25)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--top", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    transactions = load_transactions(args.input)
    frequent = apriori(transactions, args.min_support)
    rules = generate_rules(frequent, args.min_confidence)[: args.top]
    print(f"Transactions analyzed: {len(transactions)}")
    print(f"Frequent itemsets: {len(frequent)}")
    print(f"Rules generated: {len(rules)}")
    for rule in rules:
        print(f"  {_format_items(rule.antecedent)} -> {_format_items(rule.consequent)} "
              f"(support={rule.support:.2f}, confidence={rule.confidence:.2f}, lift={rule.lift:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
