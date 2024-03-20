# DR KNOW

Dr Know will create a persistent knowledge graph of your data, and allow you to query against a local Ollama instance to receive a natural language response to your queries.

Dr Know heavilly uses llama_index for all operations.

## Requirements

A functional local [Ollama](https://github.com/ollama/ollama) instance

Python 3.11.5 

## Installation

**Local Ollama instance must be running**
+ Recommended: create a virtual python environment
+ Clone the repository: `git clone https://github.com/theesfeld/drknow.git`
+ Install requirements: `pip install -r requirements.txt`

## Running

drknow is a command line program that depends on command line flags and arguments, as well as a configuration file (config.yml) for various static options.

+ Edit the config.yml to match your use case. Most of the settings will not need to be changed.
+ `python drknow.py -h` will give an explanitory help menu

```bash
usage: drknow.py [-h] [--config CONFIG] [--docs DOCS | --query QUERY] [--store STORE]

drknow Application

options:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Path to configuration file
  --docs DOCS, -d DOCS  Document path
  --query QUERY, -q QUERY
                        Query string
  --store STORE, -s STORE
                        Data store name
```

**Only the -d or the -q option can be used at one time**

### Creating a knowledge graph store

+ Run drknow.py with the -d option, specifying the location of your documents. The script will ingest all files in the path given.
+ Using -s to set the location of the persistent knowledge graph should be given at this time as well, using the default in the config.yml will overwrite when running a new -d instance

`python drknow.py -d /path/to/documents -s /path/to/knowledge/graph/store`

### Querying your persistent knowledge graphs

+ Run drknow.py with the -q option, specifying your query.
+ Include the -s option, to tell drknow which knowledge graph to search.

`python drknnow.py -q "what is the meaning of life?" -s /path/to/knowledge/graph/store`

## This is a work in progress

Things will change. 

**Pull requests are welcome.**
