<h1 align="center">Akin: Generating UI Wireframes From UI Design Patterns Using Deep Learning</h1>
<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://api.metamorph.designwitheve.com/docs/" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="#" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <br/>
  <a href="#" target="_blank">
    <img alt="Python: 3.7" src="https://img.shields.io/badge/Python-3.7-important" />
  </a>
  <a href="#" target="_blank">
    <img alt="Dependency: Tensorflow 2.1" src="https://img.shields.io/badge/Tensorflow-2.1-important" />
  </a>
  <br/>
  <br/>
  <br/>
</p>

> Akin is a UI wireframe generator that allows designers to chose a UI design pattern and provides them with multiple UI wireframes for a given UI design pattern. Akin uses a fine-tuned Self-Attention Generative Adversarial Network trained with 500 UI wireframes of 5 android UI design patterns

---

## Dataset

Akin uses a manually annotated subset of RICO dataset

- [Dataset with annotations and semantic images](https://blackbox-toolkit.s3.us-east-2.amazonaws.com/datasets/Akin_SAGAN_500.tar.gz)

---

## Setup and usage

Akin uses Python 3.7 and Tensorflow 2.1.

To install and retrain Akin, follow the steps below

- Download and extract the dataset to the `data/train` directory

  - [Dataset with annotations and semantic images](https://blackbox-toolkit.s3.us-east-2.amazonaws.com/datasets/Akin_SAGAN_500.tar.gz)

- Download the following files to the `models/` directory

  - [Akin trained checkpoint](https://blackbox-toolkit.s3.us-east-2.amazonaws.com/models/akin_checkpoints.tar.gz)

- Install dependencies

  ```sh
  pip install -r requirements.txt
  ```

- To train run
  ```sh
  python main.py --train
  ```

---

## Authors

👤 **Nishit Gajjar**

- Github: [@nishit](https://github.com/nishit727)
- LinkedIn: [@nishit](https://www.linkedin.com/in/nishit-gajjar-6354a172/)

👤 **Vinoth Pandian**

- Website: [vinoth.info](https://vinoth.info)
- Github: [@vinothpandian](https://github.com/vinothpandian)
- LinkedIn: [@vinothpandian](https://linkedin.com/in/vinothpandian)

---

## Thanks to

- [Leafinity](https://github.com/leafinity): Code based on their implementation of [SAGAN](https://github.com/leafinity/SAGAN-tensorflow2.0)
