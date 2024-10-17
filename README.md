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

## Graphs
### v0.2 12th Oct
![Screenshot 2024-10-17 at 14 39 55](https://github.com/user-attachments/assets/1eae5d94-8633-4079-aa73-705b5f64a7ab)

### v0.1 19th Sept
![image](https://github.com/user-attachments/assets/07dc703b-ee41-48f0-b28b-8e47fb54bfd0)



#### Installing `graphviz` - Modelling graph:
- For Mac:
`brew install graphviz`
- For linux:
`sudo apt install graphviz`

1. Install pyenv (python virtual environment) python 3.9 or in my case I use 3.10.12 it works fine:
- `pyenv install <python version here>`
- `pyenv install 3.10.12`

3. Create Virtual Environment Directory - with Python version installed above:
(.venv - what ever name you like I like .venv) in which you would put all your dependencies in this Directory here:
- `mkdir .venv`


4. Then installing python dependencies into it:
4.1. Set the version you want to install
- `pyenv local 3.10.12`
4.2. Installing python onto that Virtual Env
- `pyenv exec python -m venv .venv`
4.3. Activate virtual environment
`source .venv/bin/activate`
4.4. Installing all the dependencies -(requirements.txt) 
`pip install -r requirements.txt`



## Project Structure

- `data/`: Contains datasets
- `models/`: Stores trained models
- `src/`: Source code for the TFPS
- `tests/`: Unit tests
- `docs/`: Project documentation

## Usage



## Contributing



## License

