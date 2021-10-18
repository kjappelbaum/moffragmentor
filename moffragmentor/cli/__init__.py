# -*- coding: utf-8 -*-
"""Command line interfaces"""
import click

from ..harvester.harvest_building_blocks import harvest_directory


@click.command("cli")
@click.argument("indir", type=click.Path(exists=True))
@click.argument("dumpdir", default=None)
@click.option("--njobs", "-n", default=1, type=int)
@click.option("--reverse", is_flag=True)
@click.option("--offset", "-o", default=0, type=int)
def run_harvest(indir, dumpdir, njobs, reverse, offset):
    harvest_directory(indir, njobs, dumpdir, reverse=reverse, offset=offset)
