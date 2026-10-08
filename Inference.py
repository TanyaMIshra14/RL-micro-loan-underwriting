from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from Ensemble import Gatekeeper, PricingActorCritic
from Mapping import Dataset, STATE_FEATURE_NAMES


class UnderwritingPredictor:
    def __init__(self, gatekeeper, pricing_actor, scaler, background_samples=None):
        self.gatekeeper = gatekeeper.eval()
        self.pricing_actor = pricing_actor.eval()
        self.scaler = scaler
        self.background_samples = background_samples

    @staticmethod
    def save_checkpoint(
        output_dir, gatekeeper, pricing_actor, scaler, background_samples=None
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        state_dim = gatekeeper.net[0].in_features
        checkpoint_path = output_dir / "underwriting_model.pt"
        scaler_path = output_dir / "feature_scaler.joblib"

        torch.save(
            {
                "schema_version": 1,
                "state_dim": state_dim,
                "gatekeeper_state_dict": gatekeeper.state_dict(),
                "pricing_actor_state_dict": pricing_actor.state_dict(),
                "feature_names": STATE_FEATURE_NAMES + ["Portfolio GNPA", "Portfolio PSL share"],
                "background_samples": (
                    torch.as_tensor(np.asarray(background_samples), dtype=torch.float32)
                    if background_samples is not None
                    else None
                ),
            },
            checkpoint_path,
        )
        joblib.dump(scaler, scaler_path)
        return checkpoint_path, scaler_path

    @classmethod
    def load_checkpoint(cls, output_dir, device="cpu"):
        output_dir = Path(output_dir)
        checkpoint_path = output_dir / "underwriting_model.pt"
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        if checkpoint.get("schema_version") != 1:
            raise ValueError("Unsupported underwriting checkpoint schema.")
        state_dim = int(checkpoint["state_dim"])
        gatekeeper = Gatekeeper(state_dim=state_dim).to(device)
        pricing_actor = PricingActorCritic(state_dim=state_dim).to(device)
        gatekeeper.load_state_dict(checkpoint["gatekeeper_state_dict"])
        pricing_actor.load_state_dict(checkpoint["pricing_actor_state_dict"])
        scaler = joblib.load(output_dir / "feature_scaler.joblib")
        return cls(
            gatekeeper,
            pricing_actor,
            scaler,
            checkpoint.get("background_samples"),
        )

    def _augment_portfolio_state(self, states, portfolio_gnpa, psl_share):
        portfolio = np.column_stack(
            [
                np.full(len(states), portfolio_gnpa, dtype=np.float32),
                np.full(len(states), psl_share, dtype=np.float32),
            ]
        )
        return np.column_stack([states, portfolio]).astype(np.float32)

    def predict_csv(
        self,
        input_path,
        portfolio_gnpa=0.02,
        psl_share=0.0,
        batch_size=4096,
    ):
        dataset = Dataset(str(input_path))
        dataset.scaler = self.scaler
        frame, states, _ = dataset.preprocess_unseen()
        return self.predict_frame(frame, states, portfolio_gnpa, psl_share, batch_size)

    def predict_frame(
        self,
        frame,
        states=None,
        portfolio_gnpa=0.02,
        psl_share=0.0,
        batch_size=4096,
    ):
        if states is None:
            dataset = Dataset("")
            dataset.scaler = self.scaler
            states, _ = dataset.preprocess_frame(frame)
        observations = self._augment_portfolio_state(states, portfolio_gnpa, psl_share)
        if not len(observations):
            raise ValueError("No application rows were provided for prediction.")

        q_batches = []
        apr_batches = []
        for start in range(0, len(observations), batch_size):
            batch = torch.as_tensor(observations[start : start + batch_size], dtype=torch.float32)
            with torch.no_grad():
                q_values = self.gatekeeper(batch)
                apr_mean, _, _ = self.pricing_actor(batch)
            q_batches.append(q_values.cpu().numpy())
            apr_batches.append(apr_mean[:, 0].cpu().numpy())

        q_values = np.concatenate(q_batches, axis=0)
        apr_mean = np.concatenate(apr_batches, axis=0)
        decisions = q_values[:, 1] > q_values[:, 0]
        applicant_ids = (
            frame["UNIQUEID"].to_numpy()
            if "UNIQUEID" in frame.columns
            else np.arange(len(frame))
        )

        predictions = pd.DataFrame(
            {
                "UNIQUEID": applicant_ids,
                "DECISION": np.where(decisions, "APPROVE", "DECLINE"),
                "DECLINE_Q_SCORE": q_values[:, 0],
                "APPROVE_Q_SCORE": q_values[:, 1],
                "DECISION_MARGIN": q_values[:, 1] - q_values[:, 0],
                "PROPOSED_APR_PERCENT": np.where(decisions, apr_mean * 100.0, np.nan),
                "PORTFOLIO_GNPA_INPUT": portfolio_gnpa,
                "PORTFOLIO_PSL_SHARE_INPUT": psl_share,
            }
        )
        return predictions, observations