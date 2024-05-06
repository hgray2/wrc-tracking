# WRC (World Rally Championship) Tracking

## Introduction
This project explored ways to enhance the mean shift tracking algorithm to track race vehicles from the FIA World Rally Championship. While mean shift is usually a competent and efficient algorithm on its own, the challenges with tracking race vehicles include visual obstructions such as dust and terrain, as well as the issue of a moving camera tracking a moving target. This project seeked to overcome these issues by using optic flow techniques to ehnance the familiar mean shift algorithm.

## Usage

1. **Clone the Repository:**
   ```bash
   git clone git@github.com:hgray2/wrc-tracking.git
    ```

2. **Install Required Dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```

3. **Run the Enhanced MS Script** (currently, only **safari_4** is available)
    ```bash
    python3 EhnancedMeanShift.py 4
    ```
## Demo
https://github.com/hgray2/wrc-tracking/assets/70483261/1780fcc6-d7d0-4e03-ac0e-d58e01d4c142
