# BPSPred

### Bioactive Peptide Safety Predictor


**BPSPred** is a web-based machine learning platform for the computational assessment of bioactive peptide safety through three complementary endpoints: **cytotoxicity, hemolysis, and cell death**.

The application provides an easy-to-use interface for **single peptide prediction** and **batch screening** of multiple peptide sequences.

---

## ✨ Key Features

* 🧬 **Single Peptide Prediction**
* 📊 **Batch Prediction using Excel**
* 🔬 **Three safety-related prediction endpoints**

  * Cytotoxicity
  * Hemolysis
  * Cell Death
* ⚡ Fast sequence-based computational screening
* 🌐 Interactive **Streamlit** web interface

---

## 🧠 Prediction Workflow

```text
Peptide Sequence
       ↓
Descriptor Generation
       ↓
Machine Learning Models
       ↓
Cytotoxicity | Hemolysis | Cell Death
       ↓
Prediction Results
```

BPSPred uses integrated **sequence-based and molecular descriptors** with endpoint-specific machine-learning models.

---

## 📥 Input

### Single Prediction

Enter a peptide sequence using standard one-letter amino-acid codes.

Example:

```text
GIGKFLHSAKKFGKAFVGEIMKS
```

### Batch Prediction

Upload an Excel file containing a column named:

```text
Sequence
```

A reference Excel template is provided within the application.

---

## 🛠️ Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/BPSPred.git
cd BPSPred
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run BPSPred.py
```

---

## 📦 Main Dependencies

* Python
* Streamlit
* NumPy
* Pandas
* RDKit
* Peptides
* OpenPyXL
* Plotly

---

## ⚠️ Disclaimer

BPSPred is intended for **research and preliminary computational screening**. Predictions should not be considered a substitute for experimental validation.

---


---

## 👨‍🔬 Developers

**BPSPred Development Team**
Prof. Kunal Roy and his team,
DTC Laboratory, 
Department of Pharmaceutical Technology,
Jadavpur University, Kolkata, India
