# ai-internal_datenauswertung-praktikum_sw

## Install
```bash
sudo apt install lzma
sudo apt install liblzma-dev
sudo apt install libbz2-dev
sudo apt install ffmpeg

python3.10 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Upgrade Analysis Library
[Documentation](https://ai-gruppe.github.io/ai_internal_anAIlysis-rg_sw/)
```bash
pip install --upgrade git+ssh://git@github.com/AI-Gruppe/ai_internal_anAIlysis-rg_sw.git@rg/dev#egg=anAIlysis
```

### anAIlysis with GitHub Codespaces
0. Open the repository in GitHub Codespaces
1. Create a new SSH key `ssh-keygen -t rsa -b 4096` or use an existing one
2. Copy the public key to your clipboard `cat ~/.ssh/id_rsa.pub`
3. Give the key to the repository owner
4. Let the key be added to the repository deploy keys
