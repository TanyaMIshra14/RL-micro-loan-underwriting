import argparse
from pathlib import Path

import numpy as np

from Inference import UnderwritingPredictor


def main():
    workspace = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Smoke-test a saved underwriting model on an unlabeled CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=workspace / "Dataset1" / "test.csv",
        help="Unlabeled application CSV.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=workspace / "outputs" / "checkpoint",
        help="Directory containing the model checkpoint and feature scaler.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace / "outputs" / "unseen_predictions.csv",
        help="Where to write prediction results.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Optionally limit the smoke test to the first N applications.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input CSV does not exist: {args.input}")
    if not (args.checkpoint / "underwriting_model.pt").is_file():
        parser.error(
            f"No checkpoint at {args.checkpoint}. Run Main.py to train and save one first."
        )
    if args.rows is not None and args.rows < 1:
        parser.error("--rows must be a positive integer.")

    predictor = UnderwritingPredictor.load_checkpoint(args.checkpoint)
    import pandas as pd

    applications = pd.read_csv(args.input, nrows=args.rows)
    predictions, _ = predictor.predict_frame(applications)

    if len(predictions) != len(applications):
        raise AssertionError("The prediction count does not match the input row count.")
    if not predictions["DECISION"].isin(["APPROVE", "DECLINE"]).all():
        raise AssertionError("Unexpected decision label returned.")
    if not np.isfinite(predictions[["DECLINE_Q_SCORE", "APPROVE_Q_SCORE"]]).all().all():
        raise AssertionError("The model returned a non-finite Q-score.")
    if not np.allclose(
        predictions["DECISION_MARGIN"],
        predictions["APPROVE_Q_SCORE"] - predictions["DECLINE_Q_SCORE"],
    ):
        raise AssertionError("The decision margin is inconsistent with the Q-scores.")

    approved_apr = predictions.loc[
        predictions["DECISION"] == "APPROVE", "PROPOSED_APR_PERCENT"
    ]
    declined_apr = predictions.loc[
        predictions["DECISION"] == "DECLINE", "PROPOSED_APR_PERCENT"
    ]
    if not approved_apr.between(8.0, 36.0).all():
        raise AssertionError("An approved APR is outside the configured 8%-36% bounds.")
    if declined_apr.notna().any():
        raise AssertionError("Declined applications should not have a proposed APR.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    counts = predictions["DECISION"].value_counts().to_dict()
    print(f"Validated {len(predictions)} unseen applications: {counts}")
    print(f"Predictions written to: {args.output}")
    print("Note: the unlabeled test file supports inference checks, not accuracy scoring.")


if __name__ == "__main__":
    main()