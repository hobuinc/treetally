from click.testing import CliRunner
import os
from datetime import datetime

import numpy as np
from pytest import TempPathFactory

from silvimetric.cli import cli
from silvimetric.commands import shatter, info
from silvimetric import ShatterConfig, Bounds, StorageConfig, Storage, ExtractConfig

class TestCli(object):
    def test_cli_init(self, tmp_path_factory: TempPathFactory,
            runner: CliRunner, bounds: Bounds) -> None:
        path = tmp_path_factory.mktemp("test_tdb")
        p = os.path.abspath(path)
        res = runner.invoke(cli.cli, args=["-d", p, "--debug",
                "--scheduler", "single-threaded", "initialize", "--resolution",
                '10', '--crs', 'EPSG:3857', '--bounds', str(bounds)],
                catch_exceptions=False)
        assert res.exit_code == 0

    def test_cli_metric_groups(self, tmp_path_factory: TempPathFactory,
            runner: CliRunner, bounds: Bounds) -> None:
        path = tmp_path_factory.mktemp("test_tdb")
        p = os.path.abspath(path)
        res = runner.invoke(cli.cli, args=["-d", p, "--debug", "--scheduler",
                "single-threaded", "initialize", "--resolution", "10", "--crs", "EPSG:3857",
                "--bounds", str(bounds), "-m", "stats,p_moments"], catch_exceptions=False)
        assert res.exit_code == 0

    def test_cli_metric_file(self, tmp_path_factory: TempPathFactory, alignment: str,
            runner: CliRunner, bounds: Bounds, date: datetime, copc_filepath: str) -> None:
        # test that we can pass in a file containing metrics and then run shatter over it
        path = tmp_path_factory.mktemp("test_tdb")
        p = os.path.abspath(path)
        fakes = os.path.abspath('tests/fixtures/fake_metrics.py')
        res = runner.invoke(cli.cli, args=["-d", p, "--debug",
                "--scheduler", "single-threaded", "initialize", "--alignment", alignment,
                "--resolution", "10", "--crs", "EPSG:3857", "--bounds",
                str(bounds), "-m", f"{fakes}"],
                catch_exceptions=False)
        assert res.exit_code == 0
        res2 = runner.invoke(cli.cli, args=["-d", p,
                "--scheduler", "single-threaded",
                "shatter", copc_filepath,
                "--date", date.isoformat()+'Z',
                "--tilesize", '10'], catch_exceptions=False)
        assert res2.exit_code == 0
        storage = Storage.from_db(p)
        with storage.open('r') as a:
            exists_vals = a[:,:]['m_Z_exists']
            assert all(exists_vals)
            count_vals = a[:,:]['m_Z_count']
            assert all(count_vals == 100)



    def test_cli_shatter(self, runner: CliRunner, date: datetime,
            tdb_filepath: str, copc_filepath: str, storage: shatter.Storage,
            test_point_count) -> None:

        from test_shatter import confirm_one_entry
        base = 11 if storage.config.alignment == 'pixelispoint' else 10
        maxy = storage.config.root.maxy

        res = runner.invoke(cli.cli, args=["-d", tdb_filepath,
                "--scheduler", "single-threaded",
                "shatter", copc_filepath,
                "--date", date.isoformat()+'Z',
                "--tilesize", '10'], catch_exceptions=False)
        assert res.exit_code == 0
        confirm_one_entry(storage, maxy, base, test_point_count, 1)

    def test_cli_scan(self, runner: CliRunner, copc_filepath: str,
            storage_config: StorageConfig) -> None:
        tdb_dir = storage_config.tdb_dir
        res = runner.invoke(cli.cli, args=['-d', tdb_dir, '--scheduler',
            'single-threaded', 'scan', copc_filepath])
        print(res.output)
        assert res.exit_code == 0

    def test_cli_info(self, tdb_filepath: str, runner: CliRunner,
            shatter_config: ShatterConfig) -> None:
        shatter.shatter(shatter_config)
        res = runner.invoke(cli.cli, args=['-d', tdb_filepath,
            '--scheduler', 'single-threaded', 'info'])
        assert res.exit_code == 0

    def test_cli_extract(self, runner: CliRunner, extract_config: ExtractConfig,
            storage: Storage):
        atts = []
        for a in extract_config.attrs:
            atts.append('-a')
            atts.append(a.name)
        ms = []
        for m in extract_config.metrics:
            ms.append('-m')
            ms.append(m.name)
        out_dir = extract_config.out_dir
        tdb_dir = extract_config.tdb_dir

        res = runner.invoke(cli.cli, args=['-d', tdb_dir, '--scheduler',
            'single-threaded', 'extract', *atts, *ms, '--outdir' ,out_dir])

        assert res.exit_code == 0

    def test_cli_delete(self, runner: CliRunner, shatter_config: ShatterConfig,
            pre_shatter: int) -> None:
        i = shatter_config.name
        res = runner.invoke(cli.cli, ['-d', shatter_config.tdb_dir,
                '--scheduler', 'single-threaded', 'delete', '--id', i])
        assert res.exit_code == 0

        i = info.info(shatter_config.tdb_dir)
        assert len(i['history']) == 0

    def test_cli_restart(self, runner: CliRunner, shatter_config: ShatterConfig,
            pre_shatter: int) -> None:

        i = shatter_config.name
        res = runner.invoke(cli.cli, ['-d', shatter_config.tdb_dir,
                '--scheduler', 'single-threaded', 'restart', '--id', i])
        assert res.exit_code == 0
        i = info.info(shatter_config.tdb_dir)
        assert len(i['history']) == 1
        assert ShatterConfig.from_dict(i['history'][0])

    def test_cli_resume(self, runner: CliRunner, shatter_config: ShatterConfig,
            pre_shatter: int) -> None:

        i = shatter_config.name
        res = runner.invoke(cli.cli, ['-d', shatter_config.tdb_dir,
                '--scheduler', 'single-threaded', 'restart', '--id', i])
        assert res.exit_code == 0
        i = info.info(shatter_config.tdb_dir)
        assert len(i['history']) == 1
        assert ShatterConfig.from_dict(i['history'][0])