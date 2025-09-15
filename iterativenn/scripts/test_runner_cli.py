from click.testing import CliRunner
from runner_cli import cli

def test_runner_allvlast():
    runner = CliRunner()
    result = runner.invoke(cli, ['allvlast'])
    assert result.exit_code == 0

if __name__ == '__main__':
    test_runner_allvlast()