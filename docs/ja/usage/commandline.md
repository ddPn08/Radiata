# 環境変数とコマンドライン引数

## 環境変数

| 名前                   | 説明                                                                | 
| ---------------------- | ------------------------------------------------------------------- | 
| PYTHON                 | python.exeへのパス                                                  | 
| VENV_DIR               | venvのパス                                                          | 
| GIT                    | gitへのパス                                                         | 
| INDEX_URL              | pipの--index_url                                                    | 
| COMMANDLINE_ARGS       | radiata本体へのコマンドライン引数                                   | 
| TENSORRT_LINUX_COMMAND | linuxでのtensorrtのインストールコマンド `e.g. pip install tensorrt` | 
| TORCH_COMMAND          | pytorchのインストールコマンド `e.g. pip install torch`              | 
| XFORMERS_COMMAND       | xformersのインストールコマンド `e.g. pip install xformers`          | 


## コマンドライン引数

| 名前                 | 説明                               | 規定値      | 
| -------------------- | ---------------------------------- | ----------- | 
| --config-file        | config.tomlへのパス                | config.toml | 
| --host               | webuiが起動するIP                  | -        | 
| --port               | webuiが起動するポート              | 7860        | 
| --share              | webuiの共有                        | -        | 
| --model-dir          | モデルのディレクトリ               | -           | 
| --hf-token           | Hugging Faceのトークン             | -           | 
| --skip-install       | パッケージのインストールをスキップ | -           | 
| --reinstall-torch    | pytorchを再インストール            | -        | 
| --reinstall-xformers | xformersを再インストール           | -        | 
| --xformers           | xformersを有効化                   | -        | 
| --tensorrt           | tensorrtを有効化                   | -        | 
| --deepfloyd_if       | deepfloydを有効化                  | -        | 