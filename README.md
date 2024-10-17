# Traffic Flow Prediction System (TFPS)

This project implements a traffic flow prediction system using machine learning techniques 
COS30018-Intelligent Systems - Semester 2 2024

## Setup
### Prerequisite - Installation:
#### Installing `pyenv` - Python virtual environment:
 - `curl https://pyenv.run | bash`
#### Installing `graphviz` - Modelling graph:
- For Mac:
`brew install graphviz`
- For linux:
`sudo apt install graphviz`

## Installation
1. Install pyenv (python virtual environment) python 3.9 or in my case I use 3.10.12 it works fine:
- `pyenv install <python version here>`
- `pyenv install 3.10.12`

2. Create Virtual Environment Directory - with Python version installed above:
> [!IMPORTANT]  
> It is important that you activate your virtual environment first before installing dependencies `source .venv/bin/activate`

(.venv - what ever name you like I like .venv) in which you would put all your dependencies in this Directory here:
- `mkdir .venv`

- Set the version you want to install - recommend 3.10.12 it works on my PC:
`pyenv local 3.10.12`
- Installing Python Virtual Environment onto that `.venv` folder:
`pyenv exec python -m venv .venv`
- Activate virtual environment
`source .venv/bin/activate`
Installing all the dependencies from requirements.txt
`pip install -r requirements.txt`

## Graphs
### v0.2 12th Oct
![Screenshot 2024-10-17 at 14 39 55](https://github.com/user-attachments/assets/1eae5d94-8633-4079-aa73-705b5f64a7ab)

### v0.1 19th Sept
![image](https://github.com/user-attachments/assets/07dc703b-ee41-48f0-b28b-8e47fb54bfd0)






## Project Structure

- `data/`: Contains datasets
- `models/`: Stores trained models
- `src/`: Source code for the TFPS
- `tests/`: Unit tests
- `docs/`: Project documentation

## Usage



## Contributing



## License

