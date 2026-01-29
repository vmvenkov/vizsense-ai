# VizSense AI

VizSense AI is a course project in Data Mining that analyzes **how data is visualized**, 
not the data itself.

## Overview
The system evaluates data visualizations and:
- Scores quality from 0 to 100
- Detects visualization anti-patterns
- Explains weaknesses by criteria
- Automatically fixes Vega-Lite charts
- Compares Before vs After quality
- Suggests better visualizations based on data

## Input
- CSV dataset
- Chart specification (Vega-Lite / Plotly JSON) or image

## Output
- Quality score
- Detected issues with explanations
- Auto-fixed chart specification
- Suggested improved visualization

## Technology
- Python
- Streamlit
- Pandas / NumPy
- Altair (Vega-Lite)
- Rule-based analysis

## Live demo
https://vizsense-ai.streamlit.app
