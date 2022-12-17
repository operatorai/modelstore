#    Copyright 2022 Neal Lathia
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import tempfile
import subprocess


def _run_cli_command(args: list) -> str:
    """Runs a modelstore CLI command"""
    command = ["python", "-m", "modelstore"] + args
    print(f"⏱  Running: {command}")
    return (
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            check=True,
        )
        .stdout.decode("utf-8")
        .replace("\n", "")
    )


def assert_upload_runs(domain: str, model_path: str) -> str:
    """Runs the 'python -m modelstore upload' command"""
    assert os.path.exists(model_path)
    model_id = _run_cli_command(
        [
            "upload",
            domain,
            model_path,
        ]
    )
    assert model_id is not None
    assert model_id != ""
    print(f"✅  CLI command uploaded model={model_id}")
    return model_id


def assert_download_runs(domain: str, model_id: str):
    """Runs the 'python -m modelstore download' command
    in a temporary directory"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        _run_cli_command(["download", domain, model_id, str(tmp_dir)])
        model_dir = os.path.join(tmp_dir, domain, model_id)
        assert os.path.exists(model_dir)
        assert len(os.listdir(model_dir)) != 0
        print(f"✅  CLI command downloaded model={model_id}")
