# Traffic Flow Prediction System (TFPS)

This project implements a traffic flow prediction system using machine learning techniques 
COS30018-Intelligent Systems - Semester 2 2024

## Setup
### Prerequisite:
#### Installing `pyenv` - Python virtual environment:
- For Mac
` brew install pyenv`
- For Linux:
`sudo apt install pyenv`

#### Installing `graphviz` - Modelling graph:
- For Mac:
`brew install graphviz`
- For linux:
`sudo apt install graphviz`

1. Install pyenv (python virtual environment) python 3.9 or in my case I use 3.10.12 it works fine:
- `pyenv install <python version here>`
- `pyenv install 3.10.12`

3. Create Virtual Environment Directory and Requirements file for automatic installation - with Python version installed above:
First create a virtual environment folder (.venv - what ever name you like but `.venv` is the standard name)  in which you would put all your dependencies here:
- `mkdir .venv`
- `mkdir requirements.txt` copy the file down below to this & save it:
```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
tensorflow==2.15.0
keras==2.15.0
matplotlib==3.9.2
pydot==3.0.1
```

Then installing python into it:
Set the version you want to install
- `pyenv local 3.10.12`
Installing python onto that Virtual Env
- `pyenv -m venv .venv`
4. Installing all the dependencies -(requirements.txt) 
`pip install -r requirements.txt`
5. Activate virtual environment
`source .venv/bin/activate`


## Project Structure

- `data/`: Contains datasets
- `models/`: Stores trained models
- `src/`: Source code for the TFPS
- `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks for experimentation
- `docs/`: Project documentation

## Usage



## Contributing



## License

