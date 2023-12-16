#  Zipf’s Law and CCDF Analysis on Deep Reinforcement Learning
Supplementary code for final paper in CS6701-A: Principles of Complex Systems at University of Vermont

[Paper](project_proposal_10_08_23-revtex4.pdf)

## Get Started
Python and Anaconda are required. MacOS or Linux environment is recommended.

Execute the following to install all dependencies in a virtual environment.
```
conda env create -f environment.yml
conda activate rl_power_law
```

Add the current directory to Python Path.
```
export PYTHONPATH=$PWD
```

Generate dataset for OpenAI Gym "CartPole-v1" (This will overwright the preexisting dataset included in the repository at `results_cartpole-v1`).
```
python run.py -h
python run.py
```

Open `Power Law Analysis.ipynb` in Jupyter Notebook for Zipf distribution and CCDF analysis.
