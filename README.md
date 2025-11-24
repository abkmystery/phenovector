# PhenoVector – Behavioural Genome Engine for Running Processes

PhenoVector is a fully local, lightweight behavioural‑genome engine that derives structured “genome vectors” for running system processes. It performs safe on‑device telemetry collection using `psutil`, computes >20 behaviour‑driven gene scores, assigns risk levels, and provides an optional Streamlit dashboard for visual exploration.

All capabilities described here match the actual source code: `features.py`, `genes.py`, `genome.py`, and `app.py`.

---
<img width="1886" height="903" alt="Img1" src="https://github.com/user-attachments/assets/fb797fc7-27d1-4a27-8083-1a0afbdcabe7" />
---

## 🔍 What PhenoVector Does

### **1. Collects Runtime Behaviour Features**
From each process (`features.py`):
- CPU percentage  
- RSS memory  
- Thread count  
- Open file handles  
- Network connections  
- Executable entropy (first 200kB)  
- Lifetime & CPU time  
- System process / temp executable identification  

### **2. Computes Behavioural Gene Scores**
Each behaviour feature is normalised via `PopulationStats` and transformed into 20+ genes (`genes.py`):
- resource_abuse  
- entropy  
- impersonation  
- exfiltration  
- tracking  
- persistence  
- mutation  
- stealth  
- latency  
- syscall_diversity  
- burst_density  
- thread_intensity  
- registry_touch  
- io_intensity  
- network_activity  
- file_entropy  
- handle_abuse  
- injection_sus  
- dll_sideload  

Every gene strictly returns a float in `[0,1]`.

### **3. Produces Per‑Process Genome Objects**
`genome.py` returns a structured `ProcessGenome`:
- identity: pid, name, exe  
- behaviour features  
- gene vector (dict)  
- risk_score (0–1)  
- risk_level (`benign`, `suspicious`, `high`)  

### **4. Optional Visual Dashboard**
`app.py` provides:
- PCA/t‑SNE process clustering  
- Radar charts of gene profiles  
- IsolationForest anomaly scoring  
- PID whitelisting  
- Raw genome table & JSON export  


---

## 📦 Installation

```bash
pip install phenovector
```


---

## 🚀 Quick Usage

```python
from phenovector.genome import analyze_system

genomes = analyze_system(limit=100)
for g in genomes:
    print(g.pid, g.name, g.risk_score, g.risk_level)
    print(g.genes)
```

---

## 📊 Run the Streamlit UI

```bash
streamlit run -m phenovector.app
```


---


## 📄 License

APACHE 2.0
