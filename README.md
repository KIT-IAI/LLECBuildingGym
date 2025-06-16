<p float="left">
    <img src="data/img/icon_kit.png" width="10%" hspace="20"/>
</p>

# LLECBuildingGym

This repository contains the Python implementation and resources for our paper: **Advanced Deep Reinforcement Learning for Heat Pump Control in Residential Buildings**, submitted to IEEE ISGT Europe 2025 in Malta. 

A Gymnasium Environment for Smart Heat Pump Control in Residential Buildings, Inspired by the Heat Pump House at the Living Lab Energy Campus (LLEC), KIT.

**⚠️ Note**: **Last update was on 16.06.2025**.

---

## Project Structure

```bash
LLECBuildingGym/                              # Root directory of the project
├── data/                                     # Input data (e.g., weather, pricing)
├── llec_building_gym/                        # Main Gymnasium environment implementations
│   ├── __init__.py                           # Registers environments
│   ├── envs/                                 # Submodule with environment definitions
│   │   ├── __init__.py                       # Exports environments for external use
│   │   └── base_building_gym.py              # Main environment logic and control integration
│   └── controllers/                          # Other controllers; Fuzzy, MPC, PI, PID
│       ├── fuzzy_controller.py               # Fuzzy controller 
│       ├── mpc_controller.py                 # MPC controller 
│       ├── pi_controller.py                  # PI controller 
│       ├── pid_controller.py                 # PID controller
│       └── README_MPC.md                     # MPC documentation and usage instructions 
├── models/                                   # Saved trained models (PPO, SAC, etc.)
├── plot-paper/                               # Notebooks to generate figures and tables for the paper
│   ├── check_envs_registration.ipynb         # Verifies registered Gymnasium environments
│   ├── generate_table03_summary_stats.ipynb  # Generate Table 03
│   ├── plot_fig03_temperature_data.ipynb     # Plots indoor/outdoor temperature data for Figure 03
│   ├── plot_fig04_price_data.ipynb           # Plots dynamic energy prices for Figure 04
│   ├── plot_fig05_indoor_temp_setpoint.ipynb # Plots dynamic indoor temp setpoints for Figure 05
│   └── preprocess_outdoor_temperature.ipynb  # Prepares outdoor temperature time series
├── slurm_logs_eval/                          # SLURM logs from evaluation jobs
├── slurm_logs_train/                         # SLURM logs from training jobs
├── slurm_script/                             # SLURM job submission scripts
├── results/                                  # Evaluation logs and result CSVs
├── .gitignore                                # Ignore in version control
├── LICENSE                                   # Licensing
├── README.md                                 # Repo documentation and usage instructions
├── pyproject.toml                            # Build system configuration
├── requirements.txt                          # Python dependencies
├── run_evaluation.py                         # Evaluate trained models
└── run_train_rl.py                           # Train RL models (PPO, SAC, DDPG,TD3, A2C)
```

## **1. Installation**
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/KIT-IAI/LLECBuildingGym
```

```bash
python3.9 -m venv llec_env
source llec_env/bin/activate

cd LLECBuildingGym
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```
##### Reinstallation (after code changes):
```bash
pip uninstall llec_building_gym -y
pip install -e .
```
##### Environment Check (verify that the environment is registered correctly):
```bash
python check_envs_registration.ipynb
```

##### For using Jupyter notebooks (Optional): 
```bash
pip install ipykernel
python -m ipykernel install --user --name=llec_env --display-name "Python (llec_env)"
jupyter kernelspec list
```

---
## 2. Environment LLECBuildingGym

##### 2.1 Description
The [base_building_gym.py](llec_building_gym/envs/base_building_gym.py) simulates thermal building dynamics with heat pump control in 5-minute intervals. 
It uses **[gymnasium](https://github.com/Farama-Foundation/Gymnasium)** library and is suitable for both RL agents and advanced control methods.

Key features:
- Single-zone indoor thermal model with electric heat pump control and heat loss dynamics.
- Dynamic energy pricing and weather inputs
- Configurable heat pump control every 5 minutes.
- Exogenous variables: outdoor temperature and dynamic energy prices.
- Modular design supporting custom reward modes and controllers (RL, PI, PID, Fuzzy, MPC)

##### 2.2 Uncertainty and Noise Modeling
To simulate real-world uncertainty, the environment includes:
- **Wiener Process Noise**: Adds random fluctuations in outdoor temperature.
- **Sensor Noise**: Simulates inaccurate indoor and outdoor temperature measurements.
- **Prediction Errors**: MPC can operate on noisy future weather forecasts.

These features support evaluation under uncertainty and help assess robustness of control strategies.

### **3.Training & Evaluation**

This repository supports both RL agent training and controller evaluation via script-based workflows. 
RL training is handled using **[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)** algorithms, while evaluation supports classical control strategies like PI, PID, Fuzzy Logic, and MPC Controllers.

#### **3.1 RL Training**:
Train RL agents using the script **[run_train_rl.py](run_train_rl.py)**:

Two reward modes and multiple observation variants are supported for flexible evaluation.
- `temperature`: Temperature-based reward (single-objective)
- `combined`: Temperature and energy cost combined (multi-objective)

##### **Command-line Arguments**:

| Argument        | Type   | Default Value  | Choices                          | Description                                                                 |
|-----------------|--------|----------------|----------------------------------|-----------------------------------------------------------------------------|
| `--algorithm`   | str    | `"ppo"`        | `ppo`, `sac`, `ddpg`,`td3`, `a2c`| RL algorithm to use (from Stable-Baselines3).                               |
| `--timesteps`   | float  | `1e6`          | Any positive float               | Total number of environment steps.                                          |
| `--num-envs`    | int    | `4`            | >= 1                             | Number of parallel environments (for vectorized training).                  |
| `--seed`        | int    | `42`           | Any integer                      | Random seed for reproducibility.                                            |
| `--eval-freq`   | int    | `5000`         | >= 1                             | Evaluation frequency (in timesteps).                                        |
| `--reward_mode` | str    | `"temperature"`| `temperature`, `combined`        | Reward mode:  temperature-only or combined (multi-objective).               |
| `--energy-price-path`    | str            | `"data/price_data_2025.csv"`     | Valid CSV path               | Path to normalized energy price CSV file.    |
| `--training`    | flag   | `False`        | `False`, `True`                  | Use training data for energy prices (default: `TOU Prices`).                |
| `--obs_variant` | str    | `T01`          | `T01`,`T02`.`T03`,`T04`,`C01`,`C02`.`C03`,`C04` | Select observation variant (see detailed list below).        |

#### **Observation Variants**

| Variant | Features Included                                           | Description                                |
|---------|-------------------------------------------------------------|--------------------------------------------|
| `T01`   | `noisy_temp_deviation`                                      | Temperature deviation only                 |
| `T02`   | `noisy_temp_deviation`, `time_of_day`                       | Add normalized time of day                 |
| `T03`   | `noisy_temp_deviation`, `prev_action`                       | Add previous normalized action             |
| `T04`   | `noisy_temp_deviation`, `time_of_day`, `prev_action`        | Full thermal state                         |
| `C01`   | `noisy_temp_deviation`, `energy_price`, `future_prices`     | Thermal + current and future energy prices |
| `C02`   | `C01` + `prev_action`                                       | C01 + previous action                      |
| `C03`   | `C01` + `time_of_day`                                       | C01 + time of day                          |
| `C04`   | `C01` + `time_of_day`, `prev_action`                        | Full combined state                        |

---
##### Example Usage
```bash
python run_train_rl.py --algorithm ppo --reward_mode temperature --training
```

#### **3.2 Evaluation**:

The evaluation supports both RL agents and advanced control strategies from control theory.  
These include:  
- **PI/PID Control** – widely used feedback controllers based on proportional, integral, and derivative action.  
- **Fuzzy Control** – heuristic rule-based controller using fuzzy logic for handling uncertainty.  
- **MPC Control** – Model Predictive Control with configurable prediction horizon.

##### **Command-line Arguments**:

| Argument          | Type   | Default Value       | Choices                                                                 | Description                                                                 |
|-------------------|--------|---------------------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `--algorithms`     | list   | `["ppo", "sac", "ddpg", "td3", "a2c", "PI Control", "PID Control", "Fuzzy Control", "MPC Control"]` | Any combination of supported controllers and RL models | List of algorithms or controllers to evaluate.                             |
| `--episodes`       | int    | `10`                | >= 1                                                             | Number of evaluation episodes per algorithm.                               |
| `--seed`           | int    | `42`                | Any integer                                                     | Random seed for reproducibility.                                           |
| `--mpc_horizon`    | int    | `72`                | >= 1 (typically multiples of 12)                                 | Prediction horizon for MPC (in 5-minute steps, e.g., 72 = 6 hours).        |
| `--reward_mode`    | str    | `"temperature"`     | `temperature`, `combined`                                       | Reward mode:  temperature-only or combined (multi-objective).              |
| `--obs_variant`    | str    | `T01`               | `T01`,`T02`.`T03`,`T04`,`C01`,`C02`.`C03`,`C04`                 | Select observation variant (see detailed list below).                      |
| `--prefer_best`    | flag   | `False`             | `False`,`True`                | If set, prefers loading `best_model.zip` instead of `<algorithm>_model_seed<seed>.zip` (e.g., `ppo_model_seed42.zip`) during evaluation. Supported algorithms: `ppo`, `sac`, `ddpg`,`td3`, `a2c`.             |

---

##### Example Usage
```bash
# Evaluate PPO agent for temperature based rewards
python run_evaluation.py --algorithms ppo --reward_mode temperature --obs_variant T01  

# Evaluate all available agents and controllers
chmod +x slurm_script/slurm_eval_rl_batch.sh
./slurm_script/slurm_eval_rl_batch.sh
```

The modular design allows users to plug in their own controllers or extend the environment with new features, e.g., building dynamics or pricing schemes.
## License

This code is licensed under the [MIT License](LICENSE).

## Citation 

If you use **LLECBuildingGym** in your research or project, please cite the following paper (in review):

- BibTeX format:
    ```bibtex
    @inproceedings{demirel2025_LLECBuildingGym,
      author    = {Gökhan Demirel and Ömer Ekin and Jianlei Liu and Luigi Spatafora and Kevin Förderer and Veit Hagenmeyer},
      title     = {Advanced Deep Reinforcement Learning for Heat Pump Control in Residential Buildings},
      booktitle = {Proceedings of the IEEE ISGT Europe 2025 (in review)},
      year      = {2025},
      address   = {Malta},
      institution = {Karlsruhe Institute of Technology, Institute for Automation and Applied Informatics (IAI)},
      url       = {https://github.com/KIT-IAI/LLECBuildingGym},
      note      = {Code and environment available at \url{https://github.com/KIT-IAI/LLECBuildingGym}},
    }
    ```