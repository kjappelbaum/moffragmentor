# -*- coding: utf-8 -*-
import click

from ..harvester.harvest_building_blocks import harvest_directory


@click.command("cli")
@click.argument("indir", type=click.Path(exists=True))
@click.argument("dumpdir", default=None)
@click.option("--njobs", "-n", default=1, type=int)
def run_harvest(indir, dumpdir, njobs):
    harvest_directory(indir, njobs, dumpdir)
