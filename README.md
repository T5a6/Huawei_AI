# Huawei_AI
# 🌱 Carbon-Aware Resource Allocation Brain (CARB)

## ⚠️ Important Note on Project Files

This repository contains **multiple experimental, supporting, and exploratory files**.  
❗ **Not all files are directly related to the implemented MindSpore AI model.**

👉 **Main implementation file:**  
🧠 **`train_ms_dqn.py`**

This file contains the **core MindSpore-based Deep Q-Network (DQN)** training logic for CARB.  
All other files should be considered **supporting utilities, experiments, or comparisons**.

---

## 🌍 Overview

Modern digital power systems and data centers are essential for cloud services, government platforms, and research workloads. While these systems monitor energy usage, cooling, and batteries, their decisions are mostly **rule-based and reactive**, focusing on cost or performance rather than environmental impact.

🚨 **Carbon emissions are rarely treated as a primary optimization objective**, leading to avoidable emissions and inefficient operations.

---

## 💡 Proposed Solution: CARB

**CARB (Carbon-Aware Resource Allocation Brain)** is an **AI-driven decision intelligence system** for digital power systems and data center operations.

CARB introduces a learning-based “brain” that continuously:
- Observes system states
- Predicts near-future conditions
- Selects actions that **minimize cumulative carbon emissions**
- Maintains performance and reliability

---

## ⚙️ Simple System Architecture

+---------------------------+
| Operational Data |
| (Workload, Cooling, |
| Battery, Carbon Int.) |
+-------------+-------------+
|
v
+---------------------------+
| Prediction Module |
| (Demand & Carbon Forecast|
+-------------+-------------+
|
v
+---------------------------+
| CARB AI Brain (DQN) |
| MindSpore RL Agent |
+-------------+-------------+
|
v
+---------------------------+
| Control Actions |
| (Scheduling, Cooling, |
| Battery Usage) |
+-------------+-------------+
|
v
+---------------------------+
| Feedback & Learning |
| (Reward: Carbon ↓ + SLA) |
+---------------------------+

yaml
Copy code

---

## 🔑 Key Capabilities

- 🔮 **Prediction:** Forecasts workload demand, cooling needs, and grid carbon intensity  
- 🎯 **Decision-Making:** Optimizes workload delays, cooling adjustments, and battery usage  
- 🔁 **Continuous Learning:** Reinforcement learning improves policies over time  
- 🧩 **System-Level Optimization:** Balances energy, performance, and carbon impact holistically  

---

## 🛠 Technology Stack

- 🧠 **AI Framework:** MindSpore  
- 📚 **Learning Method:** Deep Reinforcement Learning (DQN)  
- 🌱 **Domain:** Carbon-aware data center & digital power optimization 
