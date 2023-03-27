import launch
from pathlib import Path
from modules import paths
from modules.scripts import basedir

if not launch.is_installed("clip"):
    launch.run_pip(
        "install git+https://github.com/openai/CLIP.git",
        "requirements for Video Extractor",
    )

extension_path =  Path(paths.script_path, "extensions") / "stable-diffusion-webui-video-extractor"
tagger_dir = extension_path / "tagger"
preload_py = extension_path / "preload.py"

# if preload.py is does not exist, copy from /tagger/preload.py and other .py from /tagger/tagger to /tagger
if not preload_py.exists():
    import shutil

    shutil.copyfile(
        tagger_dir / "preload.py", preload_py,
    )

    pyfiles = list(tagger_dir.glob("tagger/*.py"))
    for pyfile in pyfiles:
        shutil.copyfile(pyfile, tagger_dir / pyfile.name)
