#!.venv/bin/python
import logging
import sys
import yaml
import argparse

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import (
    StorageContext,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from pyvis.network import Network


class Configuration:
    def __init__(self, args):
        self.args = args

        with open(self.args.config, "r") as file:
            config = yaml.safe_load(file)

        self.llm_model = config["llm_settings"]["llm_model"]
        self.request_timeout = config["llm_settings"]["request_timeout"]
        self.max_retries = config["llm_settings"]["max_retries"]
        self.retry_delay = config["llm_settings"]["retry_delay"]
        self.retry_backoff = config["llm_settings"]["retry_backoff"]
        self.embed_model_name = config["llm_settings"]["embed_model_name"]
        self.chunk_size = config["llm_settings"]["chunk_size"]
        self.app_name = config["app_settings"]["app_name"]
        self.app_version = config["app_settings"]["app_version"]
        self.logfile = config["app_settings"]["logfile"]
        self.loglevel = config["app_settings"]["loglevel"]
        self.context = config["data_settings"]["context_path"]
        self.doc_path = config["data_settings"]["doc_path"]
        self.query = None
        self.docs = None

        if self.args.store:
            self.context = (
                config["data_settings"]["context_path"] + "/" + self.args.store
            )
        if self.args.query:
            self.query = self.args.query
        if self.args.docs:
            self.doc_path = self.args.docs

    def apply(self):
        Settings.llm = Ollama(
            model=self.llm_model,
            request_timeout=self.request_timeout,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            retry_backoff=self.retry_backoff,
        )
        Settings.embed_model = OllamaEmbedding(model_name=self.embed_model_name)
        Settings.chunk_size = self.chunk_size


class Logger:
    def __init__(self, config):
        self.config = config
        self.setup()

    def setup(self):
        try:
            formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handlers = [logging.StreamHandler(sys.stdout)]
            if self.config.logfile:
                handlers.append(logging.FileHandler(self.config.logfile, mode="w"))
            self.logger = logging.getLogger("citywide knowledge graph")
            self.logger.setLevel(
                getattr(logging, self.config.loglevel.upper(), logging.INFO)
            )
            for handler in handlers:
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        except Exception as e:
            sys.exit(f"Error setting up logging: {e}")

    def get_logger(self):
        return self.logger


class Application:
    def __init__(self, log, config):
        try:
            self.config = config
            self.store_path = self.config.context
            self.log = log
            self.context = self.config.context
            if self.config.doc_path:
                self.doc_path = self.config.doc_path
            if self.config.query:
                self.query = self.config.query
        except Exception as e:
            self.log.exception(f"Error initializing application: {e}")
            sys.exit()

    def store_data(self):
        self.log.info(f"Reading data from {self.config.doc_path} directory...")
        try:
            graph_store = SimpleGraphStore()
            storage_context = StorageContext.from_defaults(graph_store=graph_store)
            reader = SimpleDirectoryReader(input_dir=self.config.doc_path)
            documents = []
            for docs in reader.iter_data():
                self.log.info(f"Processing %s", docs[0].metadata["file_name"])
                for doc in docs:
                    self.log.info(
                        f"Processing page %s of %d",
                        doc.metadata["page_label"],
                        len(docs),
                    )
                documents.extend(docs)
                self.log.info(f"Processed {len(documents)} total pages")
            self.log.info(
                f"Storing data in {self.context} directory... This may take a while."
            )
            index = KnowledgeGraphIndex.from_documents(
                documents=documents,
                max_triplets_per_chunk=10,
                storage_context=storage_context,
                include_embeddings=True,
            )
            storage_context.persist(persist_dir=self.context)
            self.log.info(f"Data stored in {self.context} directory")
            self.log.info(f"Displaying graph...")
            self.display_graph(index, graph_store)
        except Exception as e:
            self.log.exception(f"Error storing data: {e}")
            sys.exit()
        return 0

    def query_data(self):
        self.log.info(f"Beginning query...")
        try:
            self.log.info(f"Loading existing data from {self.context} directory...")
            graph_store = SimpleGraphStore.from_persist_dir(persist_dir=self.context)
            storage_context = StorageContext.from_defaults(
                graph_store=graph_store,
                persist_dir=self.context,
            )
            loaded_index = load_index_from_storage(storage_context)

            self.log.info(f"Building query engine with query: {self.query}")
            query_engine = loaded_index.as_query_engine(
                storage_context=storage_context,
                include_text=True,
                response_mode="tree_summarize",
                embedding_mode="hybrid",
                similarity_top_k=20,
                include_embeddings=True,
                verbose=True,
                max_depth=5,
            )
            self.log.info(f"Running query engine...")
            response = query_engine.query(self.config.query)
        except Exception as e:
            self.log.exception(f"Error querying data: {e}")
            sys.exit()
        self.log.info(f"Query complete.")
        return response

    def display_graph(self, index, graph_store):
        self.log.info(f"Building knowledge tree graph...")
        try:
            g = index.get_networkx_graph()
            net = Network()
            net.from_nx(g)
            net.show("graph.html", notebook=False)
            graph_store.query
        except Exception as e:
            self.log.exception(f"Error displaying graph: {e}")
            return None
        self.log.info(f"Graph built and displayed. See graph.html for visualization")
        return 0


def main():
    parser = argparse.ArgumentParser(description="drknow Application")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file",
        action="store",
        default="config.yml",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--docs",
        "-d",
        type=str,
        help="Document path",
        action="store",
        default=None,
    )
    group.add_argument(
        "--query",
        "-q",
        type=str,
        help="Query string",
        action="store",
        default=None,
    )
    parser.add_argument(
        "--store",
        "-s",
        type=str,
        help="Data store name",
        action="store",
        default=None,
    )

    args = parser.parse_args()
    config = Configuration(args)
    config.apply()
    log = Logger(config).get_logger()
    log.info(f"{config.app_name} v{config.app_version} starting...")

    app = Application(log, config)

    if args.docs:
        app.store_data()
    elif args.query:
        response = app.query_data()
        print(response)


if __name__ == "__main__":
    main()
