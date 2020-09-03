# Singular Value Decomposition Demonstrations 
#### [Click here to view the web app!](http://svd-demo-icerm.herokuapp.com/)
A Streamlit web app demonstrating applications of the Singular Value Decomposition (SVD), a useful matrix decomposition from linear algbera.

## Current Features
- Compute and display low-rank approximations of:
  - Images
  - Audio
- Switch between:
  - Deterministic SVD
  - Randomized SVD

## Planned Features
- Low-rank approximations/background extraction for videos
- Support for compressed SVD algorithm
- Digital watermarking demonstrations
  - Image watermarking
  - Audio watermarking
  - Comparison of different watermarking schemes' robustness to attacks

## Usage
- Make sure you're in the project directory
- Run: `pipenv run streamlit run src/app.py`
- Open a web browser and navigate to the network URL given in the output
