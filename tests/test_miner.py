import tempfile
import unittest
from pathlib import Path

from miner import apriori, generate_rules, load_transactions


class AprioriTests(unittest.TestCase):
    def setUp(self):
        self.transactions = [
            frozenset({"bread", "milk"}),
            frozenset({"bread", "milk", "eggs"}),
            frozenset({"bread", "eggs"}),
            frozenset({"milk", "eggs"}),
        ]

    def test_frequent_itemsets_include_expected_pair(self):
        frequent = apriori(self.transactions, min_support=0.5)
        self.assertAlmostEqual(frequent[frozenset({"bread"})], 0.75)
        self.assertAlmostEqual(frequent[frozenset({"bread", "milk"})], 0.5)
        self.assertNotIn(frozenset({"bread", "milk", "eggs"}), frequent)

    def test_rule_metrics(self):
        frequent = apriori(self.transactions, min_support=0.5)
        rules = generate_rules(frequent, min_confidence=0.6)
        rule = next(rule for rule in rules if rule.antecedent == {"bread"} and rule.consequent == {"milk"})
        self.assertAlmostEqual(rule.support, 0.5)
        self.assertAlmostEqual(rule.confidence, 2 / 3)
        self.assertAlmostEqual(rule.lift, 8 / 9)

    def test_csv_loader_groups_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transactions.csv"
            path.write_text("transaction_id,item\nA,x\nA,y\nB,x\n", encoding="utf-8")
            self.assertEqual(load_transactions(path), [frozenset({"x", "y"}), frozenset({"x"})])

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            apriori([], 0.5)
        with self.assertRaises(ValueError):
            apriori(self.transactions, 0)
        with self.assertRaises(ValueError):
            generate_rules({}, 1.1)

    def test_rules_are_sorted_by_lift(self):
        frequent = apriori(self.transactions, min_support=0.5)
        rules = generate_rules(frequent, min_confidence=0.5)
        self.assertEqual(rules, sorted(rules, key=lambda rule: (-rule.lift, -rule.confidence, rule.antecedent)))


if __name__ == "__main__":
    unittest.main()
