# RL-micro-loan-underwriting

An end-to-end Reinforcement Learning (RL) framework designed to optimize credit decisioning, risk assessment, and dynamic capital allocation in micro-lending environments.

Traditional credit underwriting relies heavily on static classification models (e.g., Logistic Regression, GBDT) that score default probability at a single point in time. In real-world micro-finance, lending is inherently sequential: repayment behavior evolves, credit limits dynamically adjust, and portfolio balance requires managing interest yields against capital write-offs. 

This repository formulates micro-loan underwriting as a **Markov Decision Process (MDP)**, training autonomous agents (e.g., DQN, PPO, DDPG) to dynamically determine approval status, credit limits, and interest pricing under strict portfolio capital constraints.

---

## 📌 Key Features

* **Custom Gymnasium Environment:** Simulates applicant arrival, multi-feature risk profiles, dynamic repayments, interest accruals, and default events.
* **Cost-Sensitive Multi-Objective Reward:** Balances net interest margins, principal loss penalties, customer lifetime value (LTV), and portfolio-level risk caps.
* **RL Policy Implementations:** Supports discrete action spaces (Approve/Reject/Tiered Limits via DQN/Double-DQN) and continuous action spaces (Exact Credit Limit & Pricing via PPO/SAC/DDPG).
* **Baseline Benchmarks:** Built-in comparisons against standard heuristics (rule-based thresholds) and supervised baselines (XGBoost/LightGBM credit scoring).
* **Backtesting & Stress Testing:** Evaluates policy resilience under sudden macroeconomic shocks, elevated default regimes, and liquidity crunches.

---

## 🏗️ MDP Formulation

| Component | Description |
| :--- | :--- |
| **State ($S_t$)** | Applicant feature vector (income, alternative data score, debt-to-income, repayment history) + Portfolio status (current default rate, available liquidity). |
| **Action ($A_t$)** | **Discrete:** `0: Reject`, `1: Approve Tier 1 ($)`, `2: Approve Tier 2 ($$)`, etc. <br> **Continuous:** Tuple `(Credit Limit, Interest Rate)`. |
| **Reward ($R_t$)** | $R_t = \text{Interest Earned} - \text{Default Loss} - \lambda \cdot (\text{Portfolio Risk Penalty})$. |
| **Transition ($P$)** | Borrowers transition through repayment states (on-time, late, default) based on underlying stochastic credit transition matrices. |

---

## 📂 Project Structure

```bash
RL-micro-loan-underwriting/
├── data/                      # Sample datasets / synthetic micro-loan data
│   └── raw/
├── envs/                      # Simulation environments
│   ├── __init__.py
│   └── microloan_env.py       # Custom Gymnasium environment
├── models/                    # RL agents & baseline models
│   ├── agents.py              # DQN / PPO wrappers
│   └── baselines.py           # XGBoost / Scorecard baselines
├── notebooks/                 # Exploratory data analysis & policy evaluation
│   └── exploration_and_eval.ipynb
├── utils/                     # Metrics, reward shapers, and plotting helpers
│   ├── metrics.py
│   └── visualizer.py
├── train.py                   # Model training script
├── evaluate.py                # Backtesting & benchmarking script
├── requirements.txt           # Project dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/TanyaMIshra14/RL-micro-loan-underwriting.git
cd RL-micro-loan-underwriting

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train an RL Agent

Train an agent on the simulated micro-lending environment:

```bash
python train.py --algo ppo --episodes 1000 --seed 42 --save_dir ./saved_models
```

Key arguments:
* `--algo`: RL algorithm (`dqn`, `ppo`, `sac`).
* `--episodes`: Total training episodes.
* `--capital_budget`: Initial portfolio capital constraint.

### 3. Evaluate and Compare with Baselines

Run policy evaluation against standard credit scorecards:

```bash
python evaluate.py --model_path ./saved_models/best_agent.pt --benchmark xgboost
```

This generates:
* Cumulative portfolio profit curves.
* Default rates across applicant risk deciles.
* Approval rates and policy distribution maps.

---

## 📊 Evaluation Metrics

* **Expected Portfolio Return (ROI):** Total interest yields minus cumulative write-offs over the deployment horizon.
* **Cumulative Default Rate (CDR):** Percentage of total approved micro-loans reaching 90+ days past due (DPD).
* **Approval Rate & Opportunity Cost:** Ratio of rejected solvent applicants vs. avoided bad loans.
* **Gini / KS Statistic (for baselines):** Discrimination metrics evaluated on synthetic or test default outcomes.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **RL Frameworks:** Gymnasium, PyTorch, Stable-Baselines3
* **Data & Machine Learning:** NumPy, Pandas, Scikit-learn, XGBoost / LightGBM
* **Visualization:** Matplotlib, Seaborn

---

## 📄 License

Distributed under the [MIT License](LICENSE).

---

## 👤 Author

* **Tanya Mishra** ([@TanyaMIshra14](https://github.com/TanyaMIshra14))