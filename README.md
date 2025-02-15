
# **Process Variant Identification (X-PVI)**
A cutting-edge tool for process analysis and variant detection, designed to help uncover control-flow variability, inefficiencies, and undesired behaviors in processes.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Run with Docker](#run-with-docker)
7. [Sample Event Log](#sample-event-log)
8. [Acknowledgments](#acknowledgments)
9. [Contact](#contact)

---

## **Introduction**
Traditional process mining techniques often focus on a single event log, limiting their ability to incorporate insights from undesirable behaviors. This tool bridges the gap by integrating desirable and undesirable event logs to improve:
- **Process discovery**: Capturing both ideal and problematic behaviors.
- **Rule-based constraints**: Enhancing interpretability and robustness.
- **Change detection**: Identifying shifts across performance dimensions.

This repository is associated with the implementation of our paper and its extended journal version, which is currently under review. The journal version includes exciting new features, such as enhanced explainability extraction, to provide deeper insights and improved usability.

```bash
@inproceedings{DBLP:conf/bpmds/NorouzifarRDA24,
  author       = {Ali Norouzifar and
                  Majid Rafiei and
                  Marcus Dees and
                  Wil M. P. van der Aalst},
  editor       = {Han van der Aa and
                  Dominik Bork and
                  Rainer Schmidt and
                  Arnon Sturm},
  title        = {Process Variant Analysis Across Continuous Features: {A} Novel Framework},
  booktitle    = {Enterprise, Business-Process and Information Systems Modeling - 25th
                  International Conference, {BPMDS} 2024, and 29th International Conference,
                  {EMMSAD} 2024, Limassol, Cyprus, June 3-4, 2024, Proceedings},
  series       = {Lecture Notes in Business Information Processing},
  volume       = {511},
  pages        = {129--142},
  publisher    = {Springer},
  year         = {2024},
  url          = {https://doi.org/10.1007/978-3-031-61007-3\_11},
  doi          = {10.1007/978-3-031-61007-3\_11},
}
```
---

## **Features**
- Detect process variability using control-flow change detection.
- Encode event logs into declarative constraints for intuitive analysis.
- Identify behavioral patterns using clustering.
- Visualize and analyze changes in performance metrics.

---

## **Prerequisites**
- **Python 3.10**
- **Dependencies**: Listed in `requirements.txt` (if not using Docker)

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/aliNorouzifar/X-PVI
   ```
2. Navigate to the project directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**
1. Run the tool:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8002/
   ```
3. Use the provided navigation links to explore the tool:
   - Upload an event log for analysis.
   - Download the sample event log for testing.

---

## **Run with Docker**
If you prefer to run the tool without installing dependencies, you can use the provided Docker image:

1. Pull the Docker image from the GitHub Container Registry:
   ```bash
   docker pull ghcr.io/alinorouzifar/x-pvi:latest
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8002:8002 ghcr.io/alinorouzifar/x-pvi:latest
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8002/
   ```

The tool will now be accessible without the need to install Python or dependencies on your system.

---

## **Sample Event Log**
You can download a sample event log (`test.xes`) generated from a BPMN model:
- [Download Sample File](./assets/test.xes)

This file allows you to explore the tool's features and functionality.

---

## **Acknowledgments**
- We leverage the Minerful Declarative Process Discovery Tool, developed by _Claudio Di Ciccio_, to extract features and corresponding evaluation metrics within sliding windows. This tool has been instrumental in ensuring robust and interpretable feature extraction.
- Special thanks to _Eduardo Goulart Rocha_ and _Tobias Brockhoff_ for their highly efficient implementation of distance matrix calculations. Their work has significantly reduced the computational cost of Earth Mover's Distance (EMD) calculations, enabling faster and more scalable performance analysis.
- Special thanks to _Majid Rafei_, _Marcus Dees_, and _Wil van der Aalst_, my co-authors of the original paper and its extension, for their invaluable support and constructive feedback throughout the process.
---

## **Contact**
For questions, feedback, or collaborations, feel free to reach out:

- 📧 **Email**: [ali.norouzifar@pads.rwth-aachen.de](mailto:ali.norouzifar@pads.rwth-aachen.de)
- 💼 **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/ali-norouzifar/)
