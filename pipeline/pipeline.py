from climpyrical.cmd import preprocess_model

import yaml
from pkg_resources import resource_filename
import click

import logging


@click.command()
@click.option("-f", "--config-yml", help="Input config yml file.", required=True)
@click.option(
    "-l",
    "--log-level",
    help="Logging level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
)
def run_pipeline(config_yml, log_level):
    logging.basicConfig(level=log_level)
    with open(config_yml) as f:
        params = yaml.safe_load(f)

    for name in params["dvs"].keys():
        if "preprocess_model" in params["steps"]:
            out_path = resource_filename("climpyrical", params["paths"]["preprocessed_model_path"]+name+".nc")
            in_path = resource_filename("climpyrical", params["dvs"][name]["input_model_path"])
            fill_glaciers = params["dvs"][name]["fill_glaciers"]
            logging.info(f"Preprocessing Model for {name}")

            preprocess_model.downscale_and_fill(in_path=in_path, out_path=out_path, fill_glaciers=fill_glaciers, log_level=log_level)

if __name__ == "__main__":
    run_pipeline()

