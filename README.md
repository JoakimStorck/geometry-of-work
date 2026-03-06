# geometry-of-work

This repository contains the data files and code needed to replicate the
results presented in the paper *"A task-based geometry of work"* by
Joakim Storck and Jonatan Andersson (2026).

------------------------------------------------------------------------

## Setup

### 1. Clone the repository

Clone the repository to a local working directory:

``` bash
cd ~/code
git clone https://github.com/JoakimStorck/geometry-of-work.git
cd geometry-of-work
```

------------------------------------------------------------------------

### 2. Configure API keys

Create or edit the `.env` file in the repository root and provide your
personal OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key
```

An API key is required in order to use the OpenAI embedding model
`text-embedding-3-large`.

The code can also be configured to use other embedding models.\
See the documentation at the beginning of **Notebook 1:
`1_Embeddings.ipynb`**.

------------------------------------------------------------------------

### 2.5 Set the project root

Add the following line to the .env file. Replace the username 'joc' with your own username.
Adjust the path as needed.

PROJECT_ROOT=/home/joc/code/geometry-of-work

------------------------------------------------------------------------

### 3. Create a Python virtual environment

Create a virtual environment in the repository root:

``` bash
python3 -m venv .venv
```

Activate the environment:

``` bash
source .venv/bin/activate
```

------------------------------------------------------------------------

### 4. Install dependencies

Install the required Python packages:

``` bash
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel:

``` bash
python -m ipykernel install --user --name geometry-of-work
```

------------------------------------------------------------------------

### 5. Launch Jupyter Lab

Start Jupyter Lab:

``` bash
jupyter lab
```

If Jupyter Lab does not open automatically in your browser, a local URL will be displayed in the terminal, for example:

```
http://localhost:8888/lab
```

Open this address manually in your web browser.

You can now run the notebooks in the repository to reproduce the results
presented in the paper.

## Repository structure



## Replication workflow

Run the notebooks in the following order:

## License

The code in this repository is licensed under the MIT License.

Some data files originate from the O*NET database and are subject to the
terms specified by the O*NET program.
