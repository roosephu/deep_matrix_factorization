from typing import Optional
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import fasteners
import toml


class FileStorage:
    log_dir: Optional[Path]
    exp_dir: Optional[Path]

    def __init__(self):
        self.log_dir = None
        self.exp_dir = None
        self.run_id = None
        self._exp = {}

    def init(self, exp_dir):
        self.exp_dir = Path(exp_dir)
        with fasteners.InterProcessLock(self.exp_dir / '.lock'):
            self._exp = self._read_exp_status()
            self.run_id = str(self._exp['id'])
            self._exp['id'] += 1

            self.log_dir = Path(exp_dir).expanduser() / self.run_id
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.git_backup()

            toml.dump(self._exp, open(self.exp_dir / '.status.toml', 'w'))

    def _read_exp_status(self):
        status_path = self.exp_dir / '.status.toml'
        if status_path.exists():
            return toml.load(status_path)
        else:
            run_id = 0
            while (self.exp_dir / str(run_id)).exists():
                run_id += 1
            return {'id': run_id}

    def git_backup(self):
        try:
            from git import Repo
            from git.exc import InvalidGitRepositoryError
        except ImportError as e:
            print(f"Can't import `git`: {e}")
            return

        try:
            repo = Repo('.')
            pkg = ZipFile(self.log_dir / 'source.zip', 'w')

            for file_name in repo.git.ls_files().split():
                pkg.write(file_name)

        except InvalidGitRepositoryError as e:
            print(f"Can't use git to backup files: {e}")
        except FileNotFoundError as e:
            print(f"Can't find file {e}. Did you delete a file and forget to `git add .`")

    def resolve(self, file_name: str):
        if '$LOGDIR' in file_name:
            file_name = file_name.replace('$LOGDIR', str(self.log_dir))
        return Path(file_name).expanduser()

    def save(self, file_name: str, array: np.ndarray):
        resolved = self.resolve(file_name)
        if resolved:
            np.save(str(resolved), array)

    def load(self, file_name: str):
        return np.load(self.resolve(file_name))
