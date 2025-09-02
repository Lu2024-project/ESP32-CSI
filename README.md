# Respiration Monitoring with Wi-Fi CSI  

A noncontact respiration monitoring system based on **ESP32 Wi-Fi Channel State Information (CSI)**. It supports data collection, signal processing, respiration rate detection, and human presence and motion recognition.  

---

## Features
**CSI Data Collection and Visualization** : Capture CSI packets from ESP32 and display subcarriers, RSSI, and radar model in real time with GUI.  

**Respiration Signal Processing** : Median filtering and EMD decomposition for noise reduction.  

**Respiration Rate Detection** : Enhanced FFT spectral analysis and adaptive multi-scale peak detection for BPM estimation.  

**Data Fusion** : Robust respiration estimation through multi-subcarrier fusion.  

**Human State Detection** : Presence detection (none/someone) and motion recognition (static/move).  

**Flexible Configuration** : Customize Wi-Fi connection and display settings via `gui_config.json`.  

---

## Usage Examples

### 1. Start GUI for CSI Data Collection
```bash
python esp_csi_tool.py
```
### 2. Run Respiration Analysis
```bash
python esp32_respiration.py
```
