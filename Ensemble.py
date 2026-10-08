import html
from pathlib import Path

import torch
import torch.nn as nn
import shap
import numpy as np
import plotly.graph_objects as go

class Gatekeeper(nn.Module):
    def __init__(self,state_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        return self.net(x)

class PricingActorCritic(nn.Module):
    def __init__(self,state_dim):
        super().__init__()
        self.backbone=nn.Sequential(
            nn.Linear(state_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh()
        )
        self.actor_mu=nn.Linear(64,1)
        self.actor_log=nn.Parameter(torch.ones(1)*-1.2)
        self.critic=nn.Linear(64,1)

    def forward(self,x):
        features=self.backbone(x)
        mu=torch.sigmoid(self.actor_mu(features))*0.28+0.08
        std=torch.exp(self.actor_log)
        value=self.critic(features)
        return mu,std,value

    def evaluate(self,x,action):
        mu,std,value=self.forward(x)
        dist=torch.distributions.Normal(mu,std)
        log_prob=dist.log_prob(action).sum(axis=-1,keepdim=True)
        entropy=dist.entropy().sum(axis=-1,keepdim=True)
        return log_prob,value,entropy


class OnlineBuffer:
    def __init__(self):
        self.states=[]
        self.actions=[]
        self.rewards=[]
        self.dones=[]
        self.log_probs=[]
        self.values=[]

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def train_step(engine,optimizer,buffer,gamma=0.99,gae_lambda=0.95,clip_eps=0.2,epochs=4):
        states=torch.tensor(np.array(buffer.states),dtype=torch.float32)
        actions=torch.tensor(np.array(buffer.actions),dtype=torch.float32).unsqueeze(1)
        old_log_probs = torch.tensor(np.array(buffer.log_probs), dtype=torch.float32).unsqueeze(1)
        rewards = buffer.rewards
        dones = buffer.dones
        values = buffer.values
        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1)
        returns = advantages + torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs):
            log_probs, state_vals, entropy = engine.evaluate(states, actions)
            ratios = torch.exp(log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - state_vals).pow(2).mean()
            loss = policy_loss + value_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

class CascadingUnderwritingEnsemble:
    def __init__(self, dqn_gate, ppo_engine):
        self.dqn = dqn_gate
        self.ppo = ppo_engine

    def predict(self, state_np, epsilon=0.0, deterministic=True):
        s_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.dqn(s_tensor)

        if epsilon > 0.0 and np.random.rand() < epsilon:
            approval = np.random.choice([0, 1])
        else:
            approval = int(torch.argmax(q_vals, dim=1).item())

        if approval == 0:
            return (0, np.array([0.0], dtype=np.float32)), 0.0, 0.0
        with torch.no_grad():
            mu, std, val = self.ppo(s_tensor)
            if deterministic:
                rate_action = mu
            else:
                dist = torch.distributions.Normal(mu, std)
                rate_action = torch.clamp(dist.sample(), 0.08, 0.36)
            dist = torch.distributions.Normal(mu, std)
            log_prob = dist.log_prob(rate_action).item()

        return (1, rate_action.squeeze(0).cpu().numpy()), log_prob, val.item()

    def execute_macro_stress_test(env, agent, steps=400):
        print("\n" + "="*70)
        print("PHASE 4A: MACROECONOMIC REGIME STRESS TEST (RBI REPO HIKE)")
        print("="*70)
        obs, _ = env.reset()
        total_reward = 0.0
        gnpa_records = []

        for t in range(steps):
            if t == 150:
                print(">> [STOCK SHOCK] RBI hikes Policy Repo by 125 bps; tightening liquidity.")
                env.states[:, 14] += 0.0125
            action, _, _ = agent.predict(obs, epsilon=0.0)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            gnpa_records.append(info.get("gnpa", 0.0))

            if term or trunc:
                obs, _ = env.reset()

        print(f"Stress Test Completed | Total Reward: {total_reward:.2f}")
        print(f"Max Portfolio GNPA Reached: {max(gnpa_records):.2%}")
        return gnpa_records

    def audit_demographic_parity(env, agent, eval_records=1000):
        print("\n" + "="*70)
        print("PHASE 4B: FAIR LENDING & DEMOGRAPHIC PARITY AUDIT (M=4 GROUPS)")
        print("="*70)
        approvals = {i: 0 for i in range(4)}
        totals = {i: 0 for i in range(4)}

        obs, _ = env.reset()
        for _ in range(eval_records):
            group_id = int(np.argmax(obs[10:14]))
            totals[group_id] += 1

            action, _, _ = agent.predict(obs, epsilon=0.0)
            if action[0] == 1:
                approvals[group_id] += 1

            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs,_=env.reset()

        labels = ["Salaried Established", "Salaried New-to-Credit", "Self-Emp Established", "Kirana/Informal NTC"]
        for g in range(4):
            rate = approvals[g] / max(1, totals[g])
            print(f"Segment [{g}]: {labels[g]:<24} | Total: {totals[g]:<4} | Approval Rate: {rate:.2%}")

def explain_underwriting_shap(
    dqn_gate,
    background_samples,
    applicant_state,
    applicant_id,
    q_values,
    output_path,
):
    feature_names = [
        "Disbursed amount (standardized)",
        "Asset cost (standardized)",
        "Loan-to-value (standardized)",
        "New-to-credit flag",
        "CIBIL score (standardized)",
        "Active accounts (standardized)",
        "Overdue accounts (standardized)",
        "Inquiry velocity (standardized)",
        "Recent delinquency (standardized)",
        "Average account age (standardized)",
        "Salaried, established credit",
        "Self-employed, established credit",
        "Salaried, new-to-credit",
        "Self-employed, new-to-credit",
        "Repo rate",
        "Inflation assumption",
        "Quarter index",
        "Portfolio GNPA",
        "Portfolio PSL share",
    ]

    def model_decision_margin(x_numpy):
        tensor = torch.as_tensor(x_numpy, dtype=torch.float32)
        with torch.no_grad():
            q_values_batch = dqn_gate(tensor)
            return (q_values_batch[:, 1] - q_values_batch[:, 0]).cpu().numpy()

    background = np.asarray(background_samples, dtype=np.float32)
    applicant = np.asarray(applicant_state, dtype=np.float32).reshape(1, -1)
    explainer = shap.Explainer(model_decision_margin, background)
    explanation = explainer(applicant, max_evals=500)

    impacts = explanation.values[0]
    applicant_values = applicant[0]
    baseline_margin = float(np.asarray(explanation.base_values).reshape(-1)[0])
    explained_margin = baseline_margin + float(np.sum(impacts))
    q_reject, q_approve = np.asarray(q_values, dtype=float).reshape(-1)
    actual_margin = float(q_approve - q_reject)
    decision = "APPROVE" if actual_margin > 0 else "DECLINE"

    sorted_indices = np.argsort(np.abs(impacts))[::-1]
    plot_indices = sorted_indices[:12]
    plot_features = [feature_names[i] for i in plot_indices]
    plot_impacts = [float(impacts[i]) for i in plot_indices]
    plot_values = [float(applicant_values[i]) for i in plot_indices]
    colors = ["#168B73" if impact >= 0 else "#D45444" for impact in plot_impacts]

    fig = go.Figure(
        go.Bar(
            x=plot_impacts,
            y=plot_features,
            orientation="h",
            marker_color=colors,
            customdata=plot_values,
            hovertemplate=(
                "<b>%{y}</b><br>Impact on decision margin: %{x:+.4f}<br>"
                "Model input value: %{customdata:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Feature contributions to this lending decision",
        xaxis_title="SHAP impact on approval-minus-decline score",
        yaxis={"autorange": "reversed"},
        xaxis={"zeroline": True, "zerolinewidth": 2, "zerolinecolor": "#333"},
        template="plotly_white",
        height=620,
        margin={"l": 250, "r": 40, "t": 70, "b": 65},
    )

    adverse = [i for i in sorted_indices if impacts[i] < 0][:4]
    supportive = [i for i in sorted_indices if impacts[i] > 0][:4]

    def factor_list(indices):
        if not indices:
            return "<li>No material contributions in this direction.</li>"
        return "".join(
            "<li><b>{}</b>: moved the score by {:+.4f} (input value {:.3f})</li>".format(
                html.escape(feature_names[i]), impacts[i], applicant_values[i]
            )
            for i in indices
        )

    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    report = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Underwriting explanation</title>
<style>
body {{ max-width: 1100px; margin: 36px auto; padding: 0 20px; color: #202b2c; font: 16px/1.55 Segoe UI, sans-serif; }}
h1 {{ margin-bottom: 4px; }} .decision {{ font-size: 1.2rem; font-weight: 700; }}
.metrics {{ display: flex; gap: 28px; flex-wrap: wrap; padding: 14px 0; }}
.note {{ border-left: 4px solid #168B73; padding: 10px 14px; background: #f2f7f5; }}
li {{ margin: 6px 0; }}
</style>
<h1>Loan decision explanation</h1>
<p>Applicant <b>{html.escape(str(applicant_id))}</b></p>
<p class="decision">Model recommendation: {decision}</p>
<div class="metrics">
<span>Decline score: <b>{q_reject:.4f}</b></span>
<span>Approval score: <b>{q_approve:.4f}</b></span>
<span>Decision margin: <b>{actual_margin:+.4f}</b></span>
</div>
<p>A positive margin favors approval; a negative margin favors decline. These are learned Q-scores, not probabilities.</p>
<h2>Factors pulling toward decline</h2><ul>{factor_list(adverse)}</ul>
<h2>Factors supporting approval</h2><ul>{factor_list(supportive)}</ul>
<h2>How to read the chart</h2>
<p>Green contributions increase the approval-minus-decline score; red contributions lower it. The first ten continuous financial inputs are standardized model values. Hover over a bar for its input value.</p>
{plot_html}
<p class="note"><b>Use with care:</b> SHAP describes how this model used its inputs for this case. It does not establish causality, prove fairness, or by itself constitute a legally validated adverse-action reason. Validate explanations and reason-code mappings with lending, compliance, and risk teams.</p>
<p>SHAP baseline margin: {baseline_margin:+.4f}; reconstructed margin: {explained_margin:+.4f}.</p>
</html>"""

    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Applicant {applicant_id}: {decision}")
    print(f"  Decline Q: {q_reject:.4f} | Approve Q: {q_approve:.4f} | Margin: {actual_margin:+.4f}")
    print("  Largest factors pulling toward decline:")
    for index in adverse:
        print(f"    {feature_names[index]}: {impacts[index]:+.4f}")
    print(f"  Human-readable SHAP report: {report_path}")
    return report_path, fig
            