
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
8. [Contact](#contact)

---

## **Introduction**
Traditional process mining techniques often focus on a single event log, limiting their ability to incorporate insights from undesirable behaviors. This tool bridges the gap by integrating desirable and undesirable event logs to improve:
- **Process discovery**: Capturing both ideal and problematic behaviors.
- **Rule-based constraints**: Enhancing interpretability and robustness.
- **Change detection**: Identifying shifts across performance dimensions.

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

## **Contact**
For questions, feedback, or collaborations, feel free to reach out:

- 📧 **Email**: [ali.norouzifar@pads.rwth-aachen.de](mailto:ali.norouzifar@pads.rwth-aachen.de)
- 💼 **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/ali-norouzifar/)
