import argparse

from experiments.classpect.decision_tree import load_feature_records, train_tree


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train class/aspect decision trees from extracted features."
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum tree depth. Use 0 for unlimited depth.",
    )
    args = parser.parse_args()
    max_depth = None if args.max_depth == 0 else args.max_depth

    records = load_feature_records()
    for target in ("class", "aspect"):
        result = train_tree(records, target, max_depth=max_depth)
        print(f"{target.upper()} TREE")
        print(f"training_accuracy={result.training_accuracy:.3f}")
        print(f"leave_one_out_accuracy={result.leave_one_out_accuracy:.3f}")
        print(result.tree_text)


if __name__ == "__main__":
    main()
