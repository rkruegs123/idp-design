# idp-design

This repository contains code corresponding to the paper titled **"[Generalized design of sequence-ensemble-function relationships for intrinsically disordered proteins](https://doi.org/10.1101/2024.10.10.617695)."**
We provide code for **two example optimizations**:
- **Radius of Gyration (Rg) optimization**
- **Salt sensor optimization**

Additional code is available upon request and will be made public given sufficient demand.

![Animation](img/pseq_animation.gif)

---

## **Installation**

### **1. Clone the Repository**
```sh
git clone https://github.com/YOUR-USERNAME/idp-design.git
cd idp-design
```

### **2. Install Dependencies**
To install all dependencies in **editable mode**, run:
```sh
pip install -e .
```
This allows you to modify the code and have changes reflected immediately without reinstalling.

---

## **Testing**
To ensure everything is working correctly, run:
```sh
pytest
```
This will execute all tests inside the `tests/` directory.

---

## **Usage**

All design scripts save results in a specified directory within the `output` folder.
**Before running any designs, create an output directory:**
```sh
mkdir output
```

### **Design an IDP with a Target Rg**
To design an IDP that **optimizes its radius of gyration (Rg)**:
```sh
python3 -m experiments.design_rg --run-name <RUN-NAME> --seq-length <LENGTH> --target-rg <TARGET-VALUE>
```
- `TARGET-VALUE`: The target Rg in **Angstroms**.
- `LENGTH`: The length of the IDP.
- **Results will be stored in** `output/RUN-NAME`.

---

### **Design an IDP as a Salt Sensor**
To design an IDP that **expands or contracts based on salt concentration**, run:
```sh
python3 -m experiments.design_rg_salt_sensor --run-name <RUN-NAME> --seq-length <LENGTH> --salt-lo 150 --salt-hi 450 --mode MODE
```
- `MODE`: Choose `"expander"` or `"contractor"`.
- `LENGTH`: The length of the IDP.
- **Results will be stored in** `output/RUN-NAME`.

🔹 **By default**, salt concentrations are:
  - **Low salt**: 150 mM (`--salt-lo 150`)
  - **High salt**: 450 mM (`--salt-hi 450`)
  You can adjust these values using the corresponding flags.

---

## **Contributing**
If you have suggestions, feel free to **open an issue** or **submit a pull request**.

---

# Citation

If you wish to cite this work, please cite the following .bib:
```
@article{krueger2024generalized,
  title={Generalized design of sequence-ensemble-function relationships for intrinsically disordered proteins},
  author={Krueger, Ryan and Brenner, Michael P and Shrinivas, Krishna},
  journal={bioRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
