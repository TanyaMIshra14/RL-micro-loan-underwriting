import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from Mapping import Dataset
from Enviroment import CustomEnv
from QLearning import OfflineBuffer, QNetwork, train_cql
from Ensemble import (
    Gatekeeper,
    PricingActorCritic,
    OnlineBuffer,
    CascadingUnderwritingEnsemble,
    explain_underwriting_shap,
)
from Inference import UnderwritingPredictor

# Set seeds for deterministic verification
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    workspace = Path(__file__).resolve().parent
    dataset_path = workspace / "Dataset1" / "train.csv"
    unseen_dataset_path = workspace / "Dataset1" / "test.csv"
    output_dir = workspace / "outputs"

    # =========================================================================
    # PHASE 1: DATA INGESTION & ENVIRONMENT INITIALIZATION
    # =========================================================================
    print("=" * 75)
    print("PHASE 1: INGESTING LTFS DATASET & CREATING GYM ENVIRONMENT")
    print("=" * 75)

    data_loader = Dataset(csv_path=str(dataset_path))
    states, default_labels, context_labels = data_loader.preprocess()

    env = CustomEnv(states=states, default=default_labels, contexts=context_labels)
    state_dim = env.state_space  # 17 base + 2 portfolio tracking = 19
    print(f">> Environment Ready | Total State Dimensions: {state_dim}")

    # =========================================================================
    # PHASE 2: OFFLINE CONSERVATIVE Q-LEARNING (CQL) PRE-TRAINING
    # =========================================================================
    print("\n" + "=" * 75)
    print("PHASE 2: POPULATING BUFFER & OFFLINE CQL WARM-START")
    print("=" * 75)

    # Use a random sample of rows (not just the first N) and store BOTH actions per applicant.
    n_rows = min(30000, len(states))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(states))[:n_rows]

    offline_buffer = OfflineBuffer(capacity=2 * n_rows, state_dim=state_dim)

    # Augment base states with portfolio tracking features (GNPA and PSL)
    augmented_states = np.array([env._get_observation(i) for i in idx])
    repo_rates = states[idx, 14]

    offline_buffer.populate(
        states=augmented_states,
        defaults=default_labels[idx],
        repo_rates=repo_rates,
        assumed_apr=0.18,
        decline_reward=0.0,
    )

    cql_q_net = QNetwork(state_dim=state_dim, num_actions=2)
    cql_target_net = QNetwork(state_dim=state_dim, num_actions=2)
    cql_target_net.load_state_dict(cql_q_net.state_dict())

    train_cql(
        q_net=cql_q_net,
        target_net=cql_target_net,
        buffer=offline_buffer,
        steps=3000,
        batch_size=256,
        alpha_cql=0.1,
        lr=3e-4,
        gamma=0.0,
    )

    # Sanity check: the policy must use BOTH actions.
    with torch.no_grad():
        q_chk = cql_q_net(torch.tensor(augmented_states, dtype=torch.float32))
    approve_mask = (q_chk[:, 1] > q_chk[:, 0]).numpy()
    chk_defaults = default_labels[idx]
    print(f">> Gatekeeper approval rate on training sample: {approve_mask.mean():.1%}")
    if approve_mask.any() and (~approve_mask).any():
        print(f">> Default rate | approved: {chk_defaults[approve_mask].mean():.1%} "
              f"| declined: {chk_defaults[~approve_mask].mean():.1%}")
    if approve_mask.mean() > 0.98 or approve_mask.mean() < 0.02:
        print("!! WARNING: policy is (almost) constant - check rewards/assumed_apr before continuing.")
    print(">> CQL Warm-Start Complete. Historical actions stabilized.")

    # =========================================================================
    # PHASE 3: CASCADING ENSEMBLE INITIALIZATION & INTERACTIVE PPO TRAINING
    # =========================================================================
    print("\n" + "=" * 75)
    print("PHASE 3: ENSEMBLE INITIALIZATION & PPO DYNAMIC APR TUNING")
    print("=" * 75)

    # Initialize DQN Gatekeeper and transfer CQL weights
    dqn_gatekeeper = Gatekeeper(state_dim=state_dim)
    dqn_gatekeeper.load_state_dict(cql_q_net.state_dict())

    ppo_engine = PricingActorCritic(state_dim=state_dim)
    ppo_optimizer = optim.Adam(ppo_engine.parameters(), lr=3e-4)
    ppo_buffer = OnlineBuffer()

    ensemble = CascadingUnderwritingEnsemble(dqn_gate=dqn_gatekeeper, ppo_engine=ppo_engine)

    # Interaction Loop: DQN handles qualification; PPO tunes continuous APR
    obs, _ = env.reset(seed=SEED)
    total_steps = 20000

    print(">> Running Interactive Training on Approved Stream...")
    for t in range(total_steps):
        action, log_prob, val = ensemble.predict(
            obs, epsilon=0.1, deterministic=False
        )
        
        step_result = env.step(action)
        
        # Support both 4-tuple and 5-tuple Gym returns
        if len(step_result) == 5:
            next_obs, reward, term, trunc, info = step_result
        else:
            next_obs, reward, trunc, info = step_result
            term = env.current_idx >= env.max_steps

        # Buffer only records approved loans (Kiatsupaibul et al., 2024 feedback loop)
        if action[0] == 1:
            ppo_buffer.states.append(obs)
            ppo_buffer.actions.append(action[1][0])
            ppo_buffer.rewards.append(reward)
            ppo_buffer.dones.append(float(term or trunc))
            ppo_buffer.log_probs.append(log_prob)
            ppo_buffer.values.append(val)

        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()

        # Update PPO policy parameters
        if len(ppo_buffer.states) >= 128:
            OnlineBuffer.train_step(
                engine=ppo_engine,
                optimizer=ppo_optimizer,
                buffer=ppo_buffer,
                epochs=4
            )
            ppo_buffer.clear()

    print(">> PPO Pricing Policy Successfully Optimized.")

    # Save and reload the complete model plus the training-fitted feature scaler.
    checkpoint_dir = output_dir / "checkpoint"
    UnderwritingPredictor.save_checkpoint(
        checkpoint_dir,
        dqn_gatekeeper,
        ppo_engine,
        data_loader.scaler,
        background_samples=augmented_states[: min(50, len(augmented_states))],
    )
    predictor = UnderwritingPredictor.load_checkpoint(checkpoint_dir)

    # =========================================================================
    # PHASE 4: STRESS TESTING, FAIRNESS AUDIT, AND SHAP EXPLAINABILITY
    # =========================================================================
    print("\n" + "=" * 75)
    print("PHASE 4: REGULATORY VALIDATION, AUDITS & EXPLAINABILITY")
    print("=" * 75)

    # 4A: Macroeconomic Regime Stress Test (125 bps Repo Spike)
    CascadingUnderwritingEnsemble.execute_macro_stress_test(env=env, agent=ensemble, steps=300)

    # 4B: Demographic Parity Audit across Context Groups
    CascadingUnderwritingEnsemble.audit_demographic_parity(env=env, agent=ensemble, eval_records=600)

    # 4C: Score unlabeled applicants and explain the first decision.
    print("\n" + "=" * 75)
    print("PHASE 4C: UNSEEN APPLICANT DECISIONS & SHAP EXPLANATION")
    print("=" * 75)
    predictions, unseen_observations = predictor.predict_csv(unseen_dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "unseen_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    print(f">> Saved {len(predictions)} applicant decisions to {predictions_path}")

    if len(predictions):
        first = predictions.iloc[0]
        apr_text = (
            f"{first['PROPOSED_APR_PERCENT']:.2f}%"
            if first["DECISION"] == "APPROVE"
            else "not applicable (declined)"
        )
        print(
            f">> Example applicant {first['UNIQUEID']}: {first['DECISION']} | "
            f"Proposed APR: {apr_text} | "
            f"Decision margin: {first['DECISION_MARGIN']:+.4f}"
        )
        background_samples = augmented_states[: min(50, len(augmented_states))]
        q_values = np.array(
            [first["DECLINE_Q_SCORE"], first["APPROVE_Q_SCORE"]], dtype=np.float32
        )
        explain_underwriting_shap(
            dqn_gate=dqn_gatekeeper,
            background_samples=background_samples,
            applicant_state=unseen_observations[0],
            applicant_id=first["UNIQUEID"],
            q_values=q_values,
            output_path=output_dir / f"shap_explanation_{first['UNIQUEID']}.html",
        )
    else:
        print(">> No unseen applicants found; skipped prediction explanation.")

    print("\n>> All Underwriting Pipeline Phases Finished Successfully.")

if __name__ == "__main__":
    main()